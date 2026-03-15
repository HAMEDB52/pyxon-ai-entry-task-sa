# ============================================================
# agents/agent1_research.py
# وكيل البحث — يبحث في المستندات ويجمع السياق
# هو المسؤول عن إيجاد المعلومات ذات الصلة بالسؤال
# ============================================================

import os
from loguru import logger

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from src.agents.state import GraphState, AgentStage, RetrievedChunk
from src.reasoning_engine.planner import Planner
from src.reasoning_engine.tool_execution import ToolExecutor
from src.database.retrieval_enhancer import RetrievalEnhancer
from src.database.knowledge_graph    import KnowledgeGraph


# ============================================================
# وكيل البحث
# ============================================================

class ResearchAgent:
    """
    وكيل البحث — Agent 1

    المهام:
        1. يأخذ الخطة من Planner
        2. ينفذ البحث في المستندات عبر ToolExecutor
        3. يجمع النتائج في الحالة المشتركة
        4. يولّد إجابة مسودة أولية

    التدفق:
        state → plan → search → collect chunks → draft answer → state

    مثال الاستخدام:
        agent  = ResearchAgent()
        state  = GraphState.create("ما إجمالي الفاتورة؟")
        state  = agent.run(state)
        print(state.draft_answer)
    """

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY غير موجود في .env")

        self.client = genai.Client(
            api_key      = api_key,
            http_options = types.HttpOptions(api_version="v1beta"),
        )
        self.model_name  = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.planner     = Planner()
        # 🚀 إعدادات محسّنة: توسيع حقيقي + Reranker + ضغط خفيف
        self.enhancer    = RetrievalEnhancer(
            n_expansions=3,    # توسيع بـ 3 صياغات بديلة
            candidate_k=20,    # جلب 20 مرشح للـ Reranker
            final_k=8,         # إرجاع أفضل 8 نتائج
            rerank_weight=0.7, # وزن قوي للـ Reranker
            compress=True,     # ضغط سياقي خفيف
        )
        self.kg_enabled = True  # تفعيل Knowledge Graph
        self.executor    = ToolExecutor()
        
        # تحميل RAPTOR للاسترجاع الهرمي
        try:
            from src.data_processing.raptor_engine import RaptorEngine
            self.raptor = RaptorEngine()
            self.use_raptor = True
        except Exception as e:
            logger.warning(f"⚠️ RAPTOR غير متاح: {e}")
            self.raptor = None
            self.use_raptor = False

        logger.info(f"✅ ResearchAgent (Agent 1) جاهز | نموذج: {self.model_name} | RetrievalEnhancer مفعّل بالكامل")

    # ============================================================
    # الدالة الرئيسية
    # ============================================================

    def run(self, state: GraphState) -> GraphState:
        """
        تشغيل وكيل البحث

        Args:
            state: الحالة المشتركة

        Returns:
            GraphState: الحالة المحدّثة بنتائج البحث
        """
        logger.info(f"🔍 Agent 1 | بحث عن: {state.query[:60]}")
        state.set_stage(AgentStage.RESEARCHING)

        try:
            # --- 1. بناء خطة البحث ---
            plan = self.planner.plan(state.query)
            state["plan"] = plan
            logger.info(f"  📋 الخطة: {plan}")

            # --- 2. تنفيذ البحث (Enhanced) ---
            execution = self.executor.execute_plan(plan)
            base_chunks = self._convert_to_chunks(execution)
            logger.info(f"  📦 البحث الأساسي: {len(base_chunks)} قطعة")

            # --- 3. RAPTOR: بحث هرمي في الملخصات ---
            raptor_chunks = []
            if self.use_raptor and self.raptor:
                try:
                    raptor_results = self.raptor.search(state.query, top_k=3)
                    if raptor_results:
                        from src.agents.state import RetrievedChunk
                        for r in raptor_results:
                            raptor_chunks.append(RetrievedChunk(
                                chunk_id       = f"raptor_{r['node_id']}",
                                content        = r['content'],
                                source_file    = r['source_file'],
                                page_number    = 0,
                                parent_heading = f"RAPTOR L{r['level']} Summary",
                                score          = r.get('score', 0.7),
                            ))
                        logger.info(f"  🌳 RAPTOR: {len(raptor_chunks)} ملخصات هرمية")
                except Exception as raptor_err:
                    logger.debug(f"  ⚠️ RAPTOR: {raptor_err}")

            # --- 4. RetrievalEnhancer: توسيع + Reranker + ضغط ---
            enhanced_chunks = []
            try:
                enhanced = self.enhancer.search(state.query, top_k=8)
                if enhanced:
                    from src.agents.state import RetrievedChunk
                    enhanced_chunks = [
                        RetrievedChunk(
                            chunk_id       = r.chunk_id,
                            content        = r.best_content,
                            source_file    = r.source_file,
                            page_number    = r.page_number,
                            parent_heading = r.parent_heading,
                            score          = r.final_score,
                        )
                        for r in enhanced
                    ]
                    logger.info(f"  🚀 RetrievalEnhancer: {len(enhanced_chunks)} قطعة محسّنة")
            except Exception as re_err:
                logger.warning(f"  ⚠️ RetrievalEnhancer فشل ({re_err}) → البحث الأساسي")
                enhanced_chunks = []

            # --- 5. KnowledgeGraph (اختياري) ---
            kg_context = ""
            if self.kg_enabled:
                try:
                    chunk_ids  = [c.chunk_id for c in (enhanced_chunks or base_chunks)[:5]]
                    if chunk_ids:
                        from src.database.knowledge_graph import KnowledgeGraph
                        kg = KnowledgeGraph()
                        kg_ctx = kg.get_context_for_query(state.query, chunk_ids)
                        if kg_ctx and kg_ctx.entities:
                            kg_context = kg_ctx.to_text()
                            state["kg_context"] = kg_context
                            logger.info(f"  🕸️ KG: {len(kg_ctx.entities)} كيانات")
                except Exception as kg_err:
                    logger.debug(f"  ⚠️ KG: {kg_err}")

            # --- 6. دمج كل المصادر مع إزالة التكرار ---
            all_chunks = []
            seen_ids = set()
            
            # الأولوية للنتائج المحسّنة (Enhanced)
            for c in enhanced_chunks:
                if c.chunk_id not in seen_ids:
                    all_chunks.append(c)
                    seen_ids.add(c.chunk_id)
            
            # ثم RAPTOR
            for c in raptor_chunks:
                if c.chunk_id not in seen_ids:
                    all_chunks.append(c)
                    seen_ids.add(c.chunk_id)
            
            # ثم البحث الأساسي
            for c in base_chunks:
                if c.chunk_id not in seen_ids:
                    all_chunks.append(c)
                    seen_ids.add(c.chunk_id)

            # ترتيب حسب الدرجة
            all_chunks.sort(key=lambda c: c.score, reverse=True)
            
            # إبقاء أفضل 10 قطع فقط
            all_chunks = all_chunks[:10]
            
            state.set_chunks(all_chunks)
            logger.info(f"  ✅ إجمالي القطع بعد الدمج: {len(all_chunks)}")

            # --- 7. توليد إجابة مسودة ---
            if all_chunks:
                # إضافة سياق KG للـ context
                context = state.build_context(max_chunks=8)
                if kg_context:
                    context = kg_context + "\n\n" + context
                draft = self._generate_draft(state.query, context)
                state["draft_answer"] = draft
                logger.info(f"  📝 مسودة: {draft[:80]}...")
            else:
                state["draft_answer"] = ""
                logger.warning("  ⚠️ لم تُسترجع أي قطع")

            state.set_stage(AgentStage.VERIFYING)

        except Exception as e:
            logger.error(f"❌ Agent 1 فشل: {e}")
            import traceback
            traceback.print_exc()
            state.set_error(str(e))

        return state

    # ============================================================
    # توليد الإجابة المسودة
    # ============================================================

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=5, max=20),
    )
    def _generate_draft(self, query: str, context: str) -> str:
        """
        توليد إجابة مسودة بناءً على السياق المسترجع

        Args:
            query  : سؤال المستخدم
            context: السياق المسترجع

        Returns:
            str: الإجابة المسودة
        """
        prompt = f"""أنت نظام RAG متخصص في استخراج المعلومات من المستندات.
مهمتك: أجب من النص المُقدَّم فقط — لا تستخدم معرفتك العامة أبداً.

السؤال: {query}

{context}

قواعد صارمة:
1. أجب فقط من النص أعلاه
2. إذا السؤال بالعربي → أجب بالعربي | If question in English → answer in English
3. ابحث عن الأسماء والأرقام والتواريخ كما هي في النص (عربي أو إنجليزي)
4. إذا لم تجد الإجابة → قل بالضبط: "هذه المعلومة غير موجودة في المستندات المتاحة"
5. لا تقل "لم يتم ذكر" إذا كانت المعلومة موجودة بلغة مختلفة

الإجابة:"""

        try:
            response = self.client.models.generate_content(
                model    = self.model_name,
                contents = prompt,
            )
            return response.text.strip()

        except Exception as e:
            logger.error(f"❌ فشل توليد المسودة: {e}")
            raise

    # ============================================================
    # تحويل نتائج البحث
    # ============================================================

    def _convert_to_chunks(self, execution) -> list[RetrievedChunk]:
        """
        تحويل نتائج ToolExecutor إلى قائمة RetrievedChunk

        Args:
            execution: نتيجة تنفيذ الخطة من ToolExecutor

        Returns:
            list[RetrievedChunk]
        """
        chunks     : list[RetrievedChunk] = []
        seen_ids   : set[str]             = set()

        for tool_result in execution.tool_results:
            if not tool_result.has_results:
                continue

            for item in tool_result.data:
                # دعم SearchResult objects و dicts
                if isinstance(item, dict):
                    chunk_id = item.get("chunk_id", "")
                    content  = item.get("content", "")
                    source   = item.get("source_file", "")
                    page     = item.get("page_number", 0)
                    heading  = item.get("parent_heading", "")
                    score    = item.get("rrf_score",
                               item.get("similarity_score", 0.0))
                else:
                    chunk_id = getattr(item, "chunk_id", "")
                    content  = getattr(item, "content", "")
                    source   = getattr(item, "source_file", "")
                    page     = getattr(item, "page_number", 0)
                    heading  = getattr(item, "parent_heading", "")
                    score    = getattr(item, "rrf_score",
                               getattr(item, "similarity_score", 0.0))

                # تجنب التكرار
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)

                chunks.append(RetrievedChunk(
                    chunk_id       = chunk_id,
                    content        = content,
                    source_file    = source,
                    page_number    = page,
                    parent_heading = heading,
                    score          = float(score),
                ))

        # ترتيب تنازلياً حسب الدرجة
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks