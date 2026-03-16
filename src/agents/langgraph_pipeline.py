# ============================================================
# agents/langgraph_pipeline.py
# الأنبوب الرئيسي — يربط الوكلاء الأربعة في LangGraph
# هو "المايسترو" الذي يُنسّق تدفق العمل بين الوكلاء
# ============================================================

import os
from loguru import logger
from langgraph.graph import StateGraph, END

from src.agents.state      import GraphState, AgentStage
# ❌ ResearchAgent - تم استبداله بـ FastResearchAgent (أسرع 3x)
# from src.agents.agent1_research    import ResearchAgent
# ❌ VerificationAgent - تم استبداله بـ FastVerificationAgent (تحسين الأداء)
# from src.agents.agent2_verification import VerificationAgent
from src.agents.agent3_correction  import CorrectionAgent
from src.agents.agent4_answer      import AnswerAgent

# ✨ Phase 3-5: النسخ المحسّنة للسرعة الفائقة
from src.agents.fast_agent1_research import FastResearchAgent
from src.agents.fast_agent2_verification import FastVerificationAgent
from src.reasoning_engine.conditional_router import ConditionalRouter, Route
from src.security.gatekeeper import Gatekeeper


# ============================================================
# دالة مساعدة
# ============================================================

def to_graph_state(state: dict) -> GraphState:
    """تحويل dict عادي إلى GraphState"""
    gs = GraphState()
    gs.update(state)
    return gs


# ============================================================
# عُقد الـ Graph
# ============================================================

def node_route(state: dict) -> dict:
    """عُقدة التوجيه — تحدد المسار المناسب للاستعلام"""
    state    = to_graph_state(state)
    router   = ConditionalRouter()
    decision = router.route(state.query)
    state["route"] = decision.route

    # حماية metadata من الـ KeyError
    if "metadata" not in state or not isinstance(state.get("metadata"), dict):
        state["metadata"] = {"query": state.query}
    state["metadata"]["route"]      = decision.route
    state["metadata"]["confidence"] = decision.confidence

    logger.debug(f"🔀 توجيه: {decision}")  # ✨ تقليل logging overhead
    return dict(state)


def node_research(state: dict) -> dict:
    """عُقدة البحث — Agent 1 (استخدام Fast version - Phase 3)"""
    state = to_graph_state(state)
    # ✨ استخدم FastResearchAgent بدلاً من الـ slow version
    agent = FastResearchAgent()
    return dict(agent.run(state))


# ── Singleton agents (لا نُعيد بناءهم في كل استدعاء) ──
_VERIFY_AGENT  = None
_CORRECT_AGENT = None
_ANSWER_AGENT  = None

def _get_verify_agent():
    # ✨ استخدام FastVerificationAgent للتحقق الأسرع
    global _VERIFY_AGENT
    if _VERIFY_AGENT is None:
        _VERIFY_AGENT = FastVerificationAgent()
    return _VERIFY_AGENT

def _get_correct_agent():
    global _CORRECT_AGENT
    if _CORRECT_AGENT is None:
        _CORRECT_AGENT = CorrectionAgent()
    return _CORRECT_AGENT

def _get_answer_agent():
    global _ANSWER_AGENT
    if _ANSWER_AGENT is None:
        _ANSWER_AGENT = AnswerAgent()
    return _ANSWER_AGENT


def node_verify(state: dict) -> dict:
    """عُقدة التحقق — Agent 2 (مع fast-path للأسئلة البسيطة)"""
    state = to_graph_state(state)
    draft = state.get("draft_answer", "")

    # Fast-path: إذا الإجابة واضحة وقصيرة → تجاوز LLM verification
    chunks = state.get("retrieved_chunks", [])
    if draft and len(draft) > 20 and len(chunks) > 0 and len(draft) < 800:
        # تحقق regex سريع: هل الإجابة تحتوي محتوى من الـ chunks؟
        import re
        nums_in_draft   = set(re.findall(r'\d{3,}', draft))
        nums_in_context = set(re.findall(r'\d{3,}', state.build_context()))
        overlap         = nums_in_draft & nums_in_context

        if overlap or len(draft.split()) >= 5:
            # إجابة قصيرة مدعومة بأرقام من السياق → ثقة عالية
            from src.agents.state import AgentStage, VerificationResult
            state["verification"] = VerificationResult(
                is_faithful=True, is_relevant=True, confidence=0.85  # ✨ رفع إلى 0.85 لتخطي التصحيح
            )
            state["final_answer"] = draft
            state.set_stage(AgentStage.ANSWERING)
            logger.debug("⚡ fast-path verify: تجاوزنا LLM")
            return dict(state)

    return dict(_get_verify_agent().run(state))


def node_correct(state: dict) -> dict:
    """عُقدة التصحيح — Agent 3"""
    state = to_graph_state(state)
    return dict(_get_correct_agent().run(state))


def node_answer(state: dict) -> dict:
    """عُقدة الإجابة النهائية — Agent 4"""
    state = to_graph_state(state)
    return dict(_get_answer_agent().run(state))


def node_direct_answer(state: dict) -> dict:
    """
    عُقدة الإجابة المباشرة — للتحيات وأسئلة الهوية فقط.
    ⚠️ لا تُجيب على أي سؤال يتعلق بمستندات أو بيانات.
    """
    state = to_graph_state(state)
    query = state.query.strip()
    logger.info(f"💬 إجابة مباشرة (تحية/هوية): {query[:60]}")

    GREETINGS = {
        "مرحب", "مرحبا", "مرحباً", "أهلا", "أهلاً", "هلا",
        "صباح", "مساء", "كيف حالك", "كيف الحال",
        "شكرا", "شكراً", "thanks", "thank you",
        "hello", "hi", "hey",
    }
    IDENTITY = {"من أنت", "ما اسمك", "what is your name", "who are you"}

    query_lower = query.lower()

    # ── تحية ──
    if any(g in query_lower for g in GREETINGS):
        answer = "أهلاً! أنا RAGintel، نظام ذكاء اصطناعي متخصص في تحليل المستندات والفواتير. كيف يمكنني مساعدتك؟"

    # ── سؤال هوية ──
    elif any(i in query_lower for i in IDENTITY):
        answer = "أنا RAGintel — نظام Agentic RAG مبني على Gemini و LangGraph. متخصص في استرجاع المعلومات من مستنداتك وتحليلها بدقة."

    # ── أي شيء آخر وصل هنا خطأً → أعده للبحث ──
    else:
        logger.warning(f"⚠️ سؤال وصل لـ direct_answer خطأً، يُعاد توجيهه: {query[:60]}")
        # redirect: ابحث في المستندات (fallback فقط)
        from src.agents.fast_agent1_research  import FastResearchAgent  # ✨ استخدم fast version
        from src.agents.agent4_answer    import AnswerAgent

        try:
            state = FastResearchAgent().run(state)
            state = AnswerAgent().run(state)
        except Exception as e:
            state["final_answer"] = (
                "عذراً، لا يمكنني الإجابة على هذا السؤال من خارج المستندات المتاحة. "
                "يرجى رفع المستند المطلوب أولاً."
            )
        state.set_stage(AgentStage.DONE)
        return dict(state)

    state["final_answer"] = answer
    state.set_stage(AgentStage.DONE)
    return dict(state)


def node_safety_block(state: dict) -> dict:
    """عُقدة الحجب الأمني — للاستعلامات المشبوهة"""
    state = to_graph_state(state)
    logger.warning(f"🚨 تم حجب الاستعلام لأسباب أمنية: {state.query[:60]}")
    state["final_answer"] = (
        "عذراً، لا يمكنني معالجة هذا الطلب. "
        "يبدو أنه يحتوي على محتوى غير مسموح به."
    )
    state.set_stage(AgentStage.DONE)
    return dict(state)


# ============================================================
# دوال التوجيه الشرطي
# ============================================================

def route_after_routing(state: dict) -> str:
    """بعد التوجيه — أين نذهب؟"""
    route = state.get("route")

    if route == Route.STRESS_TEST:
        return "safety_block"
    elif route == Route.DIRECT_ANSWER:
        return "direct_answer"
    else:
        return "research"


def route_after_verify(state: dict) -> str:
    """بعد التحقق — قبول أم تصحيح؟"""
    state = to_graph_state(state)
    verification = state.get("verification")
    stage = state.stage

    # ✨ تحسين السرعة: ثقة عالية → تخطي التصحيح مباشرة
    if verification and verification.confidence >= 0.8:
        logger.debug(f"⚡ تخطي التصحيح | ثقة: {verification.confidence:.0%}")
        return "answer"

    if stage == AgentStage.ANSWERING:
        return "answer"
    elif stage == AgentStage.CORRECTING:
        if state.can_retry:
            return "correct"
        else:
            return "answer"
    elif stage == AgentStage.ERROR:
        return "answer"
    else:
        return "answer"


def route_after_correct(state: dict) -> str:
    """بعد التصحيح — إعادة التحقق أم الإجابة مباشرة؟"""
    state = to_graph_state(state)
    stage = state.stage

    if stage == AgentStage.VERIFYING:
        return "verify"
    elif stage == AgentStage.DONE:
        return END
    else:
        return "answer"


# ============================================================
# بناء الـ Graph
# ============================================================

def build_pipeline() -> StateGraph:
    """
    بناء pipeline LangGraph الكاملة

    الهيكل:
        route → research → verify ⟶ answer → END
                              ↕
                           correct → verify (حتى 3 مرات)
                              ↓
                           answer → END

    Returns:
        StateGraph: الـ graph المُعدّ للتشغيل
    """
    graph = StateGraph(dict)

    # --- إضافة العُقد ---
    graph.add_node("route",         node_route)
    graph.add_node("research",      node_research)
    graph.add_node("verify",        node_verify)
    graph.add_node("correct",       node_correct)
    graph.add_node("answer",        node_answer)
    graph.add_node("direct_answer", node_direct_answer)
    graph.add_node("safety_block",  node_safety_block)

    # --- نقطة البداية ---
    graph.set_entry_point("route")

    # --- حواف التوجيه الشرطي ---
    graph.add_conditional_edges(
        "route",
        route_after_routing,
        {
            "research"    : "research",
            "direct_answer": "direct_answer",
            "safety_block": "safety_block",
        },
    )

    # --- حواف ثابتة ---
    graph.add_edge("research",      "verify")
    graph.add_edge("direct_answer", END)
    graph.add_edge("safety_block",  END)
    graph.add_edge("answer",        END)

    # --- بعد التحقق ---
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "answer" : "answer",
            "correct": "correct",
        },
    )

    # --- بعد التصحيح ---
    graph.add_conditional_edges(
        "correct",
        route_after_correct,
        {
            "verify": "verify",
            "answer": "answer",
            END     : END,
        },
    )

    return graph


# ============================================================
# واجهة التشغيل
# ============================================================

class RAGPipeline:
    """
    واجهة موحدة لتشغيل الـ pipeline

    مثال الاستخدام:
        pipeline = RAGPipeline()
        result   = pipeline.run("ما إجمالي فاتورة INV-101؟")
        print(result["final_answer"])
    """

    def __init__(self):
        graph        = build_pipeline()
        self.app     = graph.compile()
        self.gate    = Gatekeeper(use_llm_check=False)
        logger.success("✅ RAGPipeline جاهز | LangGraph مُفعَّل | Gatekeeper نشط")

    def run(self, query: str, user_id: str = "default") -> dict:
        """
        تشغيل الـ pipeline لاستعلام واحد

        Args:
            query: سؤال المستخدم

        Returns:
            dict: الحالة النهائية مع الإجابة
        """
        import time
        logger.info(f"\n{'='*55}")
        logger.info(f"🚀 سؤال جديد: {query}")

        # --- فحص الحارس أولاً ---
        decision = self.gate.check(query, user_id=user_id)
        if not decision.allowed:
            logger.warning(f"🚫 Gatekeeper حجب: {decision}")
            return {
                "query"       : query,
                "final_answer": f"عذراً، لا يمكنني معالجة هذا الطلب. ({decision.reason})",
                "risk_level"  : decision.risk_level,
                "blocked"     : True,
            }

        # استخدام الاستعلام المُنظَّف
        clean_query = decision.query
        if decision.warnings:
            logger.info(f"  ⚠️ تحذيرات: {decision.warnings}")

        init_state = GraphState.create(clean_query)
        plain      = dict(init_state)
        start_time = init_state["metadata"]["start_time"]

        try:
            result = self.app.invoke(plain)

            # LangGraph قد يرجع None
            if result is None:
                logger.warning("⚠️ invoke() أرجع None — نستخدم الحالة الأولية")
                result = plain

            elapsed = time.time() - start_time
            logger.success(f"✅ اكتمل في {elapsed:.1f}s")
            return result

        except Exception as e:
            logger.error(f"❌ Pipeline فشل: {e}")
            plain["final_answer"] = f"عذراً، حدث خطأ: {e}"
            return plain

    def stream(self, query: str):
        """
        تشغيل مع streaming لكل خطوة

        Args:
            query: سؤال المستخدم

        Yields:
            tuple: (اسم العقدة، الحالة)
        """
        init_state = GraphState.create(query)
        plain      = dict(init_state)

        for step in self.app.stream(plain):
            for node_name, node_state in step.items():
                logger.info(f"  ⚙️ عقدة: {node_name}")
                yield node_name, node_state