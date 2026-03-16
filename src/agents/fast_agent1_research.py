# ============================================================
# src/agents/fast_agent1_research.py
# Agent 1 المُحسّن للسرعة الفائقة
# تخطي الخطوات البطيئة غير الضرورية
# ============================================================

import os
import time
from loguru import logger

from google import genai
from google.genai import types

from src.agents.state import GraphState, AgentStage, RetrievedChunk
from src.reasoning_engine.planner import Planner
from src.database.retrieval_enhancer import RetrievalEnhancer


class FastResearchAgent:
    """
    Agent 1 المحسّن للسرعة:
    - قطع الخطوات غير الضرورية (RAPTOR، Knowledge Graph)
    - بحث مباشر محسّن فقط
    - توليد مسودة سريعة
    
    المدة المتوقعة: 25-30s بدلاً من 40+ s
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
        self.planner    = Planner()
        
        # ✨ Optimized RetrievalEnhancer: الحد الأدنى من الاستدعاءات
        self.enhancer = RetrievalEnhancer(
            n_expansions=1,      # Phase 6: بدون توسيع (استعلام واحد فقط)
            candidate_k=10,      # تقليل من 12
            final_k=5,          # تقليل من 5
            compress=False,     # بدون ضغط
            rrf_weight=0.3,
            rerank_weight=0.7,
        )
        
        logger.success("✅ FastResearchAgent جاهز (Phase 3-5 Optimized)")

    def run(self, state: GraphState) -> GraphState:
        """بحث سريع جداً"""
        logger.debug(f"🔍 Fast Agent 1 | {state.query[:50]}")  # ✨ تقليل logging overhead
        state.set_stage(AgentStage.RESEARCHING)
        
        start = time.time()

        try:
            # خطوة 1: بحث محسّن فقط (بدون RAPTOR/KG)
            enhanced = self.enhancer.search(state.query, top_k=3)  # ✨ تقليل من 5 إلى 3 للسرعة
            
            if not enhanced:
                logger.warning("⚠️ بدون نتائج → fallback")
                state["draft_answer"] = ""
                state.set_stage(AgentStage.ANSWERING)
                state["retrieved_chunks"] = []
                return state
            
            # تحويل النتائج
            chunks = [
                RetrievedChunk(
                    chunk_id=r.chunk_id,
                    content=r.best_content,
                    source_file=r.source_file,
                    page_number=r.page_number,
                    parent_heading=r.parent_heading,
                    score=r.final_score,
                )
                for r in enhanced
            ]
            state.set_chunks(chunks)
            
            # خطوة 2: مسودة سريعة
            context = state.build_context(max_chunks=4)  # أقل قطع
            draft = self._quick_answer(state.query, context)
            state["draft_answer"] = draft
            
            elapsed = time.time() - start
            logger.success(f"✅ FastAgent1 | {len(chunks)} قطع | {elapsed:.1f}s")
            state.set_stage(AgentStage.VERIFYING)
            
        except Exception as e:
            logger.error(f"❌ FastAgent1: {e}")
            state.set_error(str(e))

        return state

    def _quick_answer(self, query: str, context: str) -> str:
        """توليد إجابة بسرعة (بدون retry)"""
        if not context.strip():
            return ""
        
        try:
            response = self.client.models.generate_content(
                model    = self.model_name,
                contents = f"""أجب من النص فقط:

السؤال: {query}

{context}

إذا لم تجد الإجابة قل: "معلومة غير موجودة".""",
                config   = types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=200,  # قصر كتير
                ),
            )
            return response.text.strip()
        except Exception as e:
            logger.debug(f"⚠️ quick_answer: {e}")
            return ""
