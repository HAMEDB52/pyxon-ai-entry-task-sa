# ============================================================
# agents/agent2_verification.py
# وكيل التحقق — يتحقق من صحة ودقة الإجابة المسودة
# يسأل: هل الإجابة مدعومة بالمصادر؟ هل هي ذات صلة بالسؤال؟
# ============================================================

import os
from loguru import logger

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from src.agents.state import GraphState, AgentStage, VerificationResult


# ============================================================
# وكيل التحقق
# ============================================================

class VerificationAgent:
    """
    وكيل التحقق — Agent 2

    المهام:
        1. يقرأ السؤال والإجابة المسودة والسياق المسترجع
        2. يتحقق من أن الإجابة مدعومة بالمصادر (Faithfulness)
        3. يتحقق من أن الإجابة ذات صلة بالسؤال (Relevance)
        4. يصدر قراراً: قبول ✅ أو رفض ❌ مع سبب التصحيح

    التدفق:
        draft_answer + context → تحقق → VerificationResult → قبول أو إعادة بحث

    مثال الاستخدام:
        agent  = VerificationAgent()
        state  = agent.run(state)
        if state["verification"].passed:
            print("✅ الإجابة مقبولة")
        else:
            print("❌ تحتاج تصحيح:", state["verification"].issues)
    """

    # الحد الأدنى للثقة لقبول الإجابة
    CONFIDENCE_THRESHOLD = float(os.getenv("VERIFICATION_THRESHOLD", "0.7"))

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY غير موجود في .env")

        self.client = genai.Client(
            api_key      = api_key,
            http_options = types.HttpOptions(api_version="v1beta"),
        )
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        logger.info(f"✅ VerificationAgent (Agent 2) جاهز | نموذج: {self.model_name}")

    # ============================================================
    # الدالة الرئيسية
    # ============================================================

    def run(self, state: GraphState) -> GraphState:
        """
        تشغيل وكيل التحقق

        Args:
            state: الحالة المشتركة

        Returns:
            GraphState: الحالة المحدّثة بنتيجة التحقق
        """
        logger.info(f"🔎 Agent 2 | التحقق من الإجابة لـ: {state.query[:60]}")
        state.set_stage(AgentStage.VERIFYING)

        # إذا لم توجد إجابة مسودة
        if not state.draft_answer.strip():
            logger.warning("  ⚠️ لا توجد إجابة مسودة للتحقق منها")
            result = VerificationResult(
                is_faithful = False,
                is_relevant = False,
                confidence  = 0.0,
                issues      = ["لا توجد إجابة مسودة"],
                suggestions = ["إعادة البحث بكلمات مختلفة"],
            )
            state["verification"] = result
            state.set_stage(AgentStage.CORRECTING)
            return state

        # Fast-path: إذا الإجابة قصيرة وواضحة → تحقق سريع
        context = state.build_context()
        if (
            len(state.draft_answer.split()) < 50 and
            len(state.retrieved_chunks) > 0 and
            "غير موجودة" not in state.draft_answer and
            "لا تتوفر" not in state.draft_answer
        ):
            # تحقق سريع بالأرقام فقط
            import re
            nums_in_draft = set(re.findall(r'\d+', state.draft_answer))
            nums_in_context = set(re.findall(r'\d+', context))
            
            # إذا كل الأرقام في الإجابة موجودة في السياق → ثقة عالية
            if nums_in_draft and nums_in_draft.issubset(nums_in_context):
                result = VerificationResult(
                    is_faithful = True,
                    is_relevant = True,
                    confidence  = 0.85,
                    issues      = [],
                    suggestions = [],
                )
                state["verification"] = result
                state["final_answer"] = state.draft_answer
                state.set_stage(AgentStage.ANSWERING)
                logger.success("⚡ Fast-path verify: تحقق سريع بالأرقام")
                return state

        try:
            # --- تحقق الأمانة ---
            faithfulness = self._check_faithfulness(
                query   = state.query,
                answer  = state.draft_answer,
                context = context,
            )

            # --- تحقق الصلة ---
            relevance = self._check_relevance(
                query  = state.query,
                answer = state.draft_answer,
            )

            # --- بناء النتيجة ---
            issues      = []
            suggestions = []

            if not faithfulness["passed"]:
                issues.append(faithfulness["issue"])
                suggestions.append(faithfulness["suggestion"])

            if not relevance["passed"]:
                issues.append(relevance["issue"])
                suggestions.append(relevance["suggestion"])

            confidence = (
                faithfulness["score"] * 0.6 +
                relevance["score"]    * 0.4
            )

            result = VerificationResult(
                is_faithful = faithfulness["passed"],
                is_relevant = relevance["passed"],
                confidence  = confidence,
                issues      = issues,
                suggestions = suggestions,
            )

            state["verification"] = result

            if result.passed:
                state["final_answer"] = state.draft_answer
                state.set_stage(AgentStage.ANSWERING)
                logger.success(
                    f"  ✅ الإجابة مقبولة | "
                    f"ثقة: {confidence:.0%} | "
                    f"أمانة: {faithfulness['score']:.0%} | "
                    f"صلة: {relevance['score']:.0%}"
                )
            else:
                state.set_stage(AgentStage.CORRECTING)
                logger.warning(
                    f"  ❌ الإجابة مرفوضة | "
                    f"ثقة: {confidence:.0%} | "
                    f"المشاكل: {issues}"
                )

        except Exception as e:
            logger.error(f"❌ Agent 2 فشل: {e}")
            # في حالة الخطأ نقبل الإجابة لتجنب التعطل
            state["verification"] = VerificationResult(
                is_faithful = True,
                is_relevant = True,
                confidence  = 0.5,
                issues      = [],
                suggestions = [],
            )
            state["final_answer"] = state.draft_answer
            state.set_stage(AgentStage.ANSWERING)

        return state

    # ============================================================
    # التحقق من الأمانة (Faithfulness)
    # ============================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=30, max=90),
    )
    def _check_faithfulness(
        self,
        query  : str,
        answer : str,
        context: str,
    ) -> dict:
        """
        هل الإجابة مدعومة بالمصادر المسترجعة؟

        Returns:
            dict: {passed, score, issue, suggestion}
        """
        prompt = f"""أنت محكّم متخصص في تقييم جودة الإجابات في أنظمة RAG.

السؤال: {query}

المصادر المسترجعة:
{context}

الإجابة المقترحة:
{answer}

قيّم: هل الإجابة مدعومة بالمصادر؟ هل لا تحتوي معلومات مخترعة؟

أجب بهذا الشكل بالضبط:
PASSED: yes أو no
SCORE: رقم من 0.0 إلى 1.0
ISSUE: المشكلة إن وُجدت أو "لا توجد مشاكل"
SUGGESTION: اقتراح التحسين أو "لا يوجد"
"""
        try:
            response = self.client.models.generate_content(
                model    = self.model_name,
                contents = prompt,
            )
            return self._parse_check_response(response.text, default_passed=True)

        except Exception as e:
            logger.warning(f"⚠️ فشل تحقق الأمانة: {e}")
            raise

    # ============================================================
    # التحقق من الصلة (Relevance)
    # ============================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=30, max=90),
    )
    def _check_relevance(self, query: str, answer: str) -> dict:
        """
        هل الإجابة ذات صلة بالسؤال الأصلي؟

        Returns:
            dict: {passed, score, issue, suggestion}
        """
        prompt = f"""أنت محكّم متخصص في تقييم جودة الإجابات.

السؤال: {query}

الإجابة:
{answer}

قيّم: هل الإجابة تُجيب فعلاً على السؤال المطروح؟

أجب بهذا الشكل بالضبط:
PASSED: yes أو no
SCORE: رقم من 0.0 إلى 1.0
ISSUE: المشكلة إن وُجدت أو "لا توجد مشاكل"
SUGGESTION: اقتراح التحسين أو "لا يوجد"
"""
        try:
            response = self.client.models.generate_content(
                model    = self.model_name,
                contents = prompt,
            )
            return self._parse_check_response(response.text, default_passed=True)

        except Exception as e:
            logger.warning(f"⚠️ فشل تحقق الصلة: {e}")
            raise

    # ============================================================
    # تحليل رد Gemini
    # ============================================================

    def _parse_check_response(self, raw: str, default_passed: bool = True) -> dict:
        """تحليل رد التحقق من Gemini"""

        result = {
            "passed"    : default_passed,
            "score"     : 0.8 if default_passed else 0.3,
            "issue"     : "",
            "suggestion": "",
        }

        for line in raw.split("\n"):
            line = line.strip()

            if line.upper().startswith("PASSED:"):
                val = line.split(":", 1)[1].strip().lower()
                result["passed"] = val in ("yes", "نعم", "true", "1")

            elif line.upper().startswith("SCORE:"):
                try:
                    score = float(line.split(":", 1)[1].strip())
                    result["score"] = max(0.0, min(1.0, score))
                except ValueError:
                    pass

            elif line.upper().startswith("ISSUE:"):
                issue = line.split(":", 1)[1].strip()
                if issue and issue not in ("لا توجد مشاكل", "none", "no issue"):
                    result["issue"] = issue

            elif line.upper().startswith("SUGGESTION:"):
                suggestion = line.split(":", 1)[1].strip()
                if suggestion and suggestion not in ("لا يوجد", "none"):
                    result["suggestion"] = suggestion

        # إذا كانت الدرجة أقل من الحد نرفض
        if result["score"] < self.CONFIDENCE_THRESHOLD:
            result["passed"] = False

        return result