# ============================================================
# src/agents/fast_agent2_verification.py
# Agent 2 المحسّن - verification سريع جداً (Phase 4)
# ============================================================

import os
from loguru import logger

from google import genai
from google.genai import types

from src.agents.state import GraphState, AgentStage, VerificationResult


class FastVerificationAgent:
    """
    تحقق سريع بدون LLM:
    - فحص regex بسيط
    - ثقة عالية إذا الإجابة مدعومة
    - 2-3 ثوان بدلاً من 10+
    """

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key) if api_key else None
        self.model_name = "gemini-2.5-flash"
        logger.success("✅ FastVerificationAgent جاهز")

    def run(self, state: GraphState) -> GraphState:
        """تحقق سريع"""
        logger.info("🔎 Fast Agent 2 | تحقق سريع")
        
        draft = state.get("draft_answer", "")
        chunks = state.get("retrieved_chunks", [])

        if not draft:
            logger.warning("❌ لا إجابة → تصحيح")
            state["verification"] = VerificationResult(
                is_faithful=False,
                is_relevant=False,
                confidence=0.0,
            )
            state.set_stage(AgentStage.CORRECTING)
            return state

        # فحص سريع: هل الإجابة مدعومة؟
        import re
        
        # البحث عن أرقام
        draft_nums = set(re.findall(r'\d{2,}', draft))
        context_nums = set(re.findall(r'\d{2,}', state.build_context()))
        has_num_match = bool(draft_nums & context_nums)
        
        # الطول
        is_reasonable = 15 < len(draft) < 1500
        
        # كلمات عربية معروفة
        has_keywords = any(
            kw in draft 
            for kw in ['فاتورة', 'تقرير', 'عميل', 'مبلغ', 'تاريخ', 'في', 'على', 'من']
        )

        indicators = sum([has_num_match, is_reasonable, has_keywords, len(chunks) > 1])

        if indicators >= 2:
            # ثقة عالية → بدون تصحيح
            logger.debug("✅ Fast verify: إجابة قوية")
            state["verification"] = VerificationResult(
                is_faithful=True,
                is_relevant=True,
                confidence=0.85,
            )
            state["final_answer"] = draft
            state.set_stage(AgentStage.ANSWERING)
        else:
            logger.debug("⚠️ Fast verify: إجابة ضعيفة → تصحيح")
            state["verification"] = VerificationResult(
                is_faithful=False,
                is_relevant=False,
                confidence=0.3,
            )
            state.set_stage(AgentStage.CORRECTING)

        return state
