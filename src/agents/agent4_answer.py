# ============================================================
# agents/agent4_answer.py
# وكيل الإجابة النهائية — يُنتج الإجابة المنسّقة للمستخدم
# آخر وكيل في السلسلة — يجمع كل شيء في إجابة احترافية
# ============================================================

import os
from dataclasses import dataclass, field
from loguru import logger

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from src.agents.state import GraphState, AgentStage


# ============================================================
# الإجابة النهائية
# ============================================================

@dataclass
class FinalAnswer:
    """الإجابة النهائية المنسّقة للمستخدم"""
    answer        : str
    sources       : list[str]        = field(default_factory=list)
    confidence    : float            = 0.0
    answer_type   : str              = "factual"   # factual | analytical | unknown
    followup_qs   : list[str]        = field(default_factory=list)

    def format_with_sources(self) -> str:
        """تنسيق الإجابة مع المصادر"""
        text = self.answer

        if self.sources:
            unique = list(dict.fromkeys(self.sources))
            text += f"\n\n📎 **المصادر:** {' | '.join(unique)}"

        if self.followup_qs:
            text += "\n\n💡 **أسئلة مقترحة:**"
            for q in self.followup_qs[:3]:
                text += f"\n  • {q}"

        return text

    def __repr__(self):
        return (
            f"[إجابة نهائية] "
            f"ثقة: {self.confidence:.0%} | "
            f"مصادر: {len(self.sources)} | "
            f"{self.answer[:60]}..."
        )


# ============================================================
# وكيل الإجابة النهائية
# ============================================================

class AnswerAgent:
    """
    وكيل الإجابة النهائية — Agent 4

    المهام:
        1. يأخذ الإجابة المتحقق منها
        2. يُنسّقها بشكل احترافي مع المصادر
        3. يُضيف أسئلة متابعة مقترحة
        4. يحدد مستوى الثقة النهائي
        5. يُخزن الإجابة في الحالة

    مثال الاستخدام:
        agent  = AnswerAgent()
        state  = agent.run(state)
        print(state.final_answer)
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
        logger.info(f"✅ AnswerAgent (Agent 4) جاهز | نموذج: {self.model_name}")

    # ============================================================
    # الدالة الرئيسية
    # ============================================================

    def run(self, state: GraphState) -> GraphState:
        """
        تشغيل وكيل الإجابة النهائية

        Args:
            state: الحالة المشتركة

        Returns:
            GraphState: الحالة المحدّثة بالإجابة النهائية المنسّقة
        """
        logger.info(f"💬 Agent 4 | توليد الإجابة النهائية لـ: {state.query[:60]}")
        state.set_stage(AgentStage.ANSWERING)

        # الإجابة الأساسية
        base_answer = state.final_answer or state.draft_answer

        # ── فحص: هل لدينا سياق حقيقي؟ ──
        has_real_context = (
            len(state.retrieved_chunks) > 0 and
            base_answer and
            len(base_answer.strip()) > 10 and
            "غير موجودة" not in base_answer and
            "لا تتوفر" not in base_answer and
            "لم أتمكن" not in base_answer
        )

        if not has_real_context:
            # لا توجد إجابة حقيقية → رسالة مهذبة
            sources = list({c.source_file for c in state.retrieved_chunks if c.source_file})
            sources_text = f" ({len(sources)} ملفات)" if sources else ""
            
            state["final_answer"] = (
                f"عذراً، لم أتمكن من إيجاد إجابة دقيقة لسؤالك في المستندات المتاحة{sources_text}. "
                f"يرجى التحقق من رفع المستندات الصحيحة أو إعادة صياغة السؤال."
            )
            state["metadata"]["final_answer_obj"] = {
                "answer": state["final_answer"],
                "sources": sources,
                "confidence": 0.0,
                "answer_type": "unknown",
                "followup_qs": [],
            }
            state.set_stage(AgentStage.DONE)
            logger.warning("⚠️ لا توجد إجابة كافية في المستندات")
            return state

        try:
            # --- جمع المصادر ---
            sources = self._collect_sources(state)

            # --- درجة الثقة ---
            verification = state.get("verification")
            confidence   = verification.confidence if verification else 0.7

            # --- تحسين التنسيق (فقط للإجابات الطويلة) ---
            # ✨ إذا الإجابة قصيرة وواضحة → بدون polishing
            if len(base_answer) < 200 and len(base_answer.split()) >= 3:
                polished = base_answer
                logger.debug("⚡ إجابة قصيرة → بدون polishing")
            else:
                polished = self._polish_answer(
                    query   = state.query,
                    answer  = base_answer,
                    context = state.build_context(max_chunks=4),  # ✨ تقليل من 8 إلى 4
                    sources = sources,
                )

            # --- أسئلة متابعة ---
            followup_qs = self._generate_followup(
                query   = state.query,
                answer  = polished,
                sources = sources,
            )

            # --- بناء الإجابة النهائية ---
            final = FinalAnswer(
                answer      = polished,
                sources     = sources,
                confidence  = confidence,
                answer_type = self._detect_answer_type(state.query),
                followup_qs = followup_qs,
            )

            # --- تخزين في الحالة ---
            state["final_answer"] = final.format_with_sources()
            state["metadata"]["final_answer_obj"] = {
                "answer"     : final.answer,
                "sources"    : final.sources,
                "confidence" : final.confidence,
                "answer_type": final.answer_type,
                "followup_qs": final.followup_qs,
            }
            state.set_stage(AgentStage.DONE)

            logger.success(f"✅ {final}")

        except Exception as e:
            logger.error(f"❌ Agent 4 فشل: {e}")
            # إرجاع الإجابة الأساسية بدون تنسيق
            state["final_answer"] = base_answer
            state.set_stage(AgentStage.DONE)

        return state

    # ============================================================
    # تحسين التنسيق
    # ============================================================

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=5, max=20),
    )
    def _polish_answer(
        self,
        query  : str,
        answer : str,
        context: str,
        sources: list[str],
    ) -> str:
        """
        تحسين صياغة الإجابة وتنسيقها

        Args:
            query  : سؤال المستخدم
            answer : الإجابة الأولية
            context: السياق المسترجع
            sources: قائمة المصادر

        Returns:
            str: الإجابة المحسّنة
        """
        sources_str = ", ".join(sources) if sources else "غير محدد"

        # ── صارم: إذا لا يوجد سياق حقيقي → رفض الإجابة ──
        has_real_context = (
            context.strip()
            and context.strip() != "لا توجد مستندات ذات صلة."
            and len(context.strip()) > 30
        )

        if not has_real_context:
            return (
                "عذراً، لا تتوفر معلومات كافية في المستندات المتاحة للإجابة على هذا السؤال. "
                "يرجى التأكد من رفع المستند المناسب أولاً."
            )

        # Fast-path: لا تحسين للإجابات القصيرة والواضحة
        words = answer.split()
        if (
            5 <= len(words) <= 60 and 
            not any(w in answer for w in ["ربما","أظن","قد يكون","غير واضح","يبدو أن"]) and
            len(answer) < 400
        ):
            logger.debug("⚡ polish fast-path: إجابة واضحة، لا تحسين")
            return answer

        prompt = f"""أنت نظام RAG متخصص. مهمتك الوحيدة: تحسين صياغة الإجابة من السياق المُقدَّم فقط.

⚠️ قواعد صارمة لا تُخالَف:
1. أجب فقط من المعلومات الموجودة في السياق أدناه
2. إذا لم تجد الإجابة في السياق → قل بوضوح: "هذه المعلومة غير موجودة في المستندات المتاحة"
3. لا تستخدم معرفتك العامة أبداً
4. لا تخمّن أو تستنتج خارج نطاق النص
5. استخدم الأرقام والأسماء كما هي في المستندات دون تعديل
6. حسّن الصياغة فقط - لا تضف معلومات جديدة

السؤال: {query}

السياق المسترجع من المستندات:
{context}

المصادر: {sources_str}

الإجابة المبنية على المستندات فقط (حسّن الصياغة دون تغيير المعنى):"""

        try:
            response = self.client.models.generate_content(
                model    = self.model_name,
                contents = prompt,
                config   = types.GenerateContentConfig(
                    temperature=0.3,  # أقل عشوائية
                    top_p=0.8,
                ),
            )
            return response.text.strip()

        except Exception as e:
            logger.warning(f"⚠️ فشل تحسين الإجابة: {e} — سيُستخدم الأصل")
            return answer

    # ============================================================
    # أسئلة المتابعة
    # ============================================================

    def _generate_followup(
        self,
        query  : str,
        answer : str,
        sources: list[str],
    ) -> list[str]:
        """توليد أسئلة متابعة مقترحة"""

        prompt = f"""بناءً على السؤال والإجابة التالية، اقترح 3 أسئلة متابعة مفيدة.

السؤال: {query}
الإجابة: {answer[:300]}

اكتب 3 أسئلة قصيرة فقط، كل سؤال في سطر، بدون ترقيم أو نقاط:"""

        try:
            response = self.client.models.generate_content(
                model    = self.model_name,
                contents = prompt,
            )
            lines = [
                line.strip().lstrip("•-0123456789. ")
                for line in response.text.strip().split("\n")
                if line.strip() and len(line.strip()) > 5
            ]
            return lines[:3]

        except Exception:
            return []

    # ============================================================
    # دوال مساعدة
    # ============================================================

    def _collect_sources(self, state: GraphState) -> list[str]:
        """جمع المصادر الفريدة من القطع المسترجعة"""
        sources = []
        for chunk in state.retrieved_chunks:
            if chunk.source_file and chunk.source_file not in sources:
                sources.append(chunk.source_file)
        return sources

    def _detect_answer_type(self, query: str) -> str:
        """اكتشاف نوع الإجابة المطلوبة"""
        query_lower = query.lower()

        analytical_kw = ["قارن", "حلل", "فرق", "compare", "analyz", "differ"]
        if any(kw in query_lower for kw in analytical_kw):
            return "analytical"

        unknown_kw = ["لا أعلم", "غير موجود", "not found", "لا يوجد"]
        if any(kw in query_lower for kw in unknown_kw):
            return "unknown"

        return "factual"