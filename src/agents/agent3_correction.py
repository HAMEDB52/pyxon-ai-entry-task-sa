# ============================================================
# agents/agent3_correction.py
# وكيل التصحيح — يصحح الإجابات المرفوضة من وكيل التحقق
# يحاول تحسين الإجابة أو إعادة البحث إذا لزم الأمر
# ============================================================

import os
from loguru import logger

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from src.agents.state import GraphState, AgentStage


# ============================================================
# وكيل التصحيح
# ============================================================

class CorrectionAgent:
    """
    وكيل التصحيح — Agent 3

    المهام:
        1. يقرأ الإجابة المرفوضة ومشاكل التحقق
        2. يحاول تصحيح الإجابة بناءً على المشاكل المحددة
        3. إذا كانت المشكلة في السياق → يطلب إعادة البحث
        4. إذا كانت المشكلة في الصياغة → يعيد الصياغة مباشرة
        5. يُرجع إجابة محسّنة أو يُشير لإعادة البحث

    استراتيجيات التصحيح:
        rephrase      : إعادة صياغة الإجابة فقط
        filter_halluc : حذف المعلومات المخترعة
        re_search     : إعادة البحث بكلمات أخرى
        fallback      : الإجابة بـ "لا أعلم" بشكل مهذب

    مثال الاستخدام:
        agent = CorrectionAgent()
        state = agent.run(state)
    """

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY غير موجود في .env")

        self.client = genai.Client(
            api_key      = api_key,
            http_options = types.HttpOptions(api_version="v1beta"),
        )
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        logger.info(f"✅ CorrectionAgent (Agent 3) جاهز | نموذج: {self.model_name}")

    # ============================================================
    # الدالة الرئيسية
    # ============================================================

    def run(self, state: GraphState) -> GraphState:
        """
        تشغيل وكيل التصحيح

        Args:
            state: الحالة المشتركة

        Returns:
            GraphState: الحالة المحدّثة بإجابة مصحّحة
        """
        logger.info(f"🔧 Agent 3 | تصحيح الإجابة | محاولة {state.retry_count + 1}/{state.max_retries}")
        state.set_stage(AgentStage.CORRECTING)

        verification = state.get("verification")
        issues       = verification.issues      if verification else []
        suggestions  = verification.suggestions if verification else []

        # --- تحديد استراتيجية التصحيح ---
        strategy = self._decide_strategy(
            issues      = issues,
            retry_count = state.retry_count,
            has_context = state.has_context,
        )
        logger.info(f"  🎯 الاستراتيجية: {strategy}")

        try:
            if strategy == "re_search":
                # إعادة البحث بكلمات مختلفة
                state = self._handle_re_search(state)

            elif strategy == "rephrase":
                # إعادة الصياغة فقط
                corrected = self._rephrase_answer(
                    query       = state.query,
                    draft       = state.draft_answer,
                    context     = state.build_context(),
                    issues      = issues,
                    suggestions = suggestions,
                )
                state["draft_answer"] = corrected
                state.increment_retry()
                state.set_stage(AgentStage.VERIFYING)

            elif strategy == "filter_halluc":
                # حذف المعلومات المخترعة
                corrected = self._filter_hallucinations(
                    query   = state.query,
                    draft   = state.draft_answer,
                    context = state.build_context(),
                )
                state["draft_answer"] = corrected
                state.increment_retry()
                state.set_stage(AgentStage.VERIFYING)

            else:  # fallback
                # إجابة "لا أعلم" مهذبة
                state = self._handle_fallback(state, issues)

        except Exception as e:
            logger.error(f"❌ Agent 3 فشل: {e}")
            state = self._handle_fallback(state, [str(e)])

        return state

    # ============================================================
    # تحديد الاستراتيجية
    # ============================================================

    def _decide_strategy(
        self,
        issues      : list[str],
        retry_count : int,
        has_context : bool,
    ) -> str:
        """
        تحديد أفضل استراتيجية تصحيح

        المنطق:
            - إذا استُنفدت المحاولات → fallback
            - إذا لا يوجد سياق → re_search
            - إذا كانت المشكلة هلوسة → filter_halluc
            - غير ذلك → rephrase
        """
        if retry_count >= 2:
            return "fallback"

        if not has_context:
            return "re_search"

        # فحص نوع المشكلة
        issues_text = " ".join(issues).lower()

        if any(kw in issues_text for kw in [
            "مخترع", "hallucin", "fabricat", "غير موجود", "not found",
            "لا يوجد في المصدر", "invented"
        ]):
            return "filter_halluc"

        if any(kw in issues_text for kw in [
            "بحث", "search", "مصادر أكثر", "more sources", "إعادة"
        ]):
            return "re_search"

        return "rephrase"

    # ============================================================
    # إعادة الصياغة
    # ============================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _rephrase_answer(
        self,
        query      : str,
        draft      : str,
        context    : str,
        issues     : list[str],
        suggestions: list[str],
    ) -> str:
        """إعادة صياغة الإجابة بناءً على ملاحظات التحقق"""

        issues_text      = "\n".join(f"- {i}" for i in issues)      if issues      else "لا توجد"
        suggestions_text = "\n".join(f"- {s}" for s in suggestions) if suggestions else "لا توجد"

        prompt = f"""أنت مساعد ذكي. صحّح الإجابة التالية بناءً على الملاحظات.

السؤال: {query}

الإجابة الحالية:
{draft}

السياق المتاح:
{context}

الملاحظات:
{issues_text}

الاقتراحات:
{suggestions_text}

التعليمات:
- أجب فقط بناءً على المعلومات في السياق
- لا تضف معلومات غير موجودة في السياق
- كن مختصراً ودقيقاً
- اذكر المصدر

الإجابة المصحّحة:"""

        response = self.client.models.generate_content(
            model    = self.model_name,
            contents = prompt,
        )

        corrected = response.text.strip()
        logger.info(f"  ✏️ إجابة مُعاد صياغتها: {corrected[:80]}...")
        return corrected

    # ============================================================
    # حذف الهلوسات
    # ============================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _filter_hallucinations(
        self,
        query  : str,
        draft  : str,
        context: str,
    ) -> str:
        """حذف المعلومات المخترعة وإبقاء المدعومة فقط"""

        prompt = f"""أنت محرر دقيق. راجع الإجابة التالية واحذف أي معلومة غير موجودة في السياق.

السؤال: {query}

السياق (المصادر الموثوقة):
{context}

الإجابة المراد تصفيتها:
{draft}

التعليمات:
- احذف أي معلومة لا تجدها في السياق
- أبقِ فقط ما هو مدعوم بالسياق
- إذا لم يبقَ شيء قل: "لا تتوفر معلومات كافية في المستندات للإجابة على هذا السؤال"
- لا تضف أي معلومة جديدة

الإجابة المُنقّحة:"""

        response = self.client.models.generate_content(
            model    = self.model_name,
            contents = prompt,
        )

        filtered = response.text.strip()
        logger.info(f"  🧹 إجابة بعد تصفية الهلوسات: {filtered[:80]}...")
        return filtered

    # ============================================================
    # إعادة البحث
    # ============================================================

    def _handle_re_search(self, state: GraphState) -> GraphState:
        """
        إعادة البحث بكلمات مفتاحية مختلفة

        يستخدم Gemini لتوليد استعلام بحث بديل
        ثم يُعيد البحث في قاعدة البيانات
        """
        logger.info("  🔄 إعادة البحث بكلمات مختلفة...")

        # توليد استعلام بديل
        alt_query = self._generate_alt_query(state.query)
        logger.info(f"  🔍 استعلام بديل: {alt_query}")

        try:
            from src.database.hybrid_search import HybridSearch
            search  = HybridSearch()
            results = search.search(alt_query, top_k=5)

            if results:
                from src.agents.state import RetrievedChunk
                new_chunks = [
                    RetrievedChunk(
                        chunk_id       = getattr(r, "chunk_id", ""),
                        content        = getattr(r, "content", ""),
                        source_file    = getattr(r, "source_file", ""),
                        page_number    = getattr(r, "page_number", 0),
                        parent_heading = getattr(r, "parent_heading", ""),
                        score          = getattr(r, "rrf_score", 0.0),
                    )
                    for r in results
                ]
                # دمج مع القطع القديمة
                existing = state.retrieved_chunks
                all_chunks = existing + new_chunks

                # إزالة المكررات
                seen = set()
                unique = []
                for c in all_chunks:
                    if c.chunk_id not in seen:
                        seen.add(c.chunk_id)
                        unique.append(c)

                state.set_chunks(unique)
                logger.info(f"  📦 قطع جديدة: {len(new_chunks)} | إجمالي: {len(unique)}")

                # إعادة توليد الإجابة
                from src.agents.agent1_research import ResearchAgent
                research = ResearchAgent()
                draft = research._generate_draft(state.query, state.build_context())
                state["draft_answer"] = draft

            state.increment_retry()
            state.set_stage(AgentStage.VERIFYING)

        except Exception as e:
            logger.error(f"  ❌ فشل إعادة البحث: {e}")
            state = self._handle_fallback(state, [str(e)])

        return state

    # ============================================================
    # توليد استعلام بديل
    # ============================================================

    def _generate_alt_query(self, query: str) -> str:
        """توليد صياغة بديلة للاستعلام"""
        try:
            prompt = f"""أعِد صياغة السؤال التالي بكلمات مختلفة للبحث في قاعدة بيانات:

السؤال: {query}

أجب بجملة واحدة فقط — صياغة بديلة للسؤال:"""

            response = self.client.models.generate_content(
                model    = self.model_name,
                contents = prompt,
            )
            return response.text.strip()

        except Exception:
            return query  # إرجاع السؤال الأصلي عند الفشل

    # ============================================================
    # الإجابة الاحتياطية
    # ============================================================

    def _handle_fallback(
        self,
        state : GraphState,
        issues: list[str],
    ) -> GraphState:
        """إجابة مهذبة عند استنفاد جميع المحاولات"""

        logger.warning(f"  ⚠️ استنفاد المحاولات — إجابة احتياطية")

        sources = list({c.source_file for c in state.retrieved_chunks if c.source_file})
        sources_text = f" (تم البحث في: {', '.join(sources)})" if sources else ""

        fallback = (
            f"عذراً، لم أتمكن من إيجاد إجابة دقيقة وموثوقة لسؤالك"
            f"{sources_text}. "
            f"يُنصح بمراجعة المستندات الأصلية مباشرة."
        )

        state["draft_answer"] = fallback
        state["final_answer"] = fallback
        state.set_stage(AgentStage.DONE)

        return state