# ============================================================
# src/security/auditor.py
# المدقق — يفحص الإجابة بعد التوليد قبل إرسالها للمستخدم
#
# يفحص:
#   1. التحقق من المصادر (هل المعلومة موجودة فعلاً؟)
#   2. اكتشاف الهلوسة (hallucination detection)
#   3. التحقق من الأرقام (أرقام مالية وتواريخ)
#   4. فحص التحيز (هل الإجابة محايدة؟)
# ============================================================

import os
import re
import json
from dataclasses import dataclass, field
from loguru import logger

from google import genai
from google.genai import types


@dataclass
class AuditResult:
    passed           : bool
    hallucination    : bool   = False    # هل توجد معلومات مخترعة؟
    source_verified  : bool   = True     # هل المعلومات مدعومة؟
    numbers_accurate : bool   = True     # هل الأرقام صحيحة؟
    is_biased        : bool   = False    # هل توجد تحيزات؟
    confidence       : float  = 1.0
    issues           : list[str] = field(default_factory=list)
    redacted_answer  : str   = ""       # الإجابة بعد حذف المشكلات


class Auditor:
    """
    يفحص الإجابة النهائية قبل إرسالها.

    الاستخدام:
        auditor = Auditor()
        result  = auditor.audit(
            query="ما إجمالي INV-101؟",
            answer="الإجمالي 5500 ريال",
            context="...نص المستند..."
        )
        if result.passed:
            send(result.redacted_answer or answer)
    """

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        self._client = genai.Client(
            api_key      = api_key,
            http_options = types.HttpOptions(api_version="v1beta"),
        ) if api_key else None
        self._model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        logger.info("✅ Auditor جاهز")

    def audit(self, query: str, answer: str, context: str) -> AuditResult:
        """
        تدقيق الإجابة.

        Args:
            query  : السؤال الأصلي
            answer : الإجابة المولّدة
            context: السياق المسترجع (نص المستندات)

        Returns:
            AuditResult
        """
        issues = []

        # ── 1. فحص سريع بالقواعد (بدون LLM) ──
        rule_issues = self._rule_based_check(answer, context)
        issues.extend(rule_issues)

        # ── 2. فحص LLM للحالات الدقيقة ──
        if self._client and answer.strip():
            llm_result = self._llm_audit(query, answer, context)
            if llm_result.get("hallucination"):
                issues.append("⚠️ معلومات محتملة غير مدعومة بالسياق")
            if llm_result.get("numbers_issue"):
                issues.append("⚠️ تعارض في الأرقام")
            if llm_result.get("bias"):
                issues.append("⚠️ محتوى متحيز")

            hallucination   = llm_result.get("hallucination", False)
            source_verified = llm_result.get("source_verified", True)
            numbers_accurate= not llm_result.get("numbers_issue", False)
            is_biased       = llm_result.get("bias", False)
            confidence      = float(llm_result.get("confidence", 0.9))
            redacted        = llm_result.get("cleaned_answer", answer)
        else:
            hallucination   = False
            source_verified = len(rule_issues) == 0
            numbers_accurate= True
            is_biased       = False
            confidence      = 0.9 if not issues else 0.7
            redacted        = answer

        passed = (
            not hallucination and
            source_verified and
            numbers_accurate and
            len(issues) == 0
        )

        if issues:
            logger.warning(f"⚠️ Auditor: {len(issues)} مشكلة | {issues}")
        else:
            logger.success(f"✅ Auditor: إجابة نظيفة | ثقة={confidence:.0%}")

        return AuditResult(
            passed          = passed,
            hallucination   = hallucination,
            source_verified = source_verified,
            numbers_accurate= numbers_accurate,
            is_biased       = is_biased,
            confidence      = confidence,
            issues          = issues,
            redacted_answer = redacted,
        )

    # ════════════════════════════════════
    # فحص القواعد
    # ════════════════════════════════════

    def _rule_based_check(self, answer: str, context: str) -> list[str]:
        issues = []

        # فحص الأرقام الكبيرة (هل موجودة في السياق؟)
        numbers_in_answer = re.findall(r"\b[\d,]+(?:\.\d{1,2})?\b", answer)
        for num in numbers_in_answer:
            clean_num = num.replace(",", "")
            if len(clean_num) >= 4:   # أرقام ≥ 4 خانات فقط
                if clean_num not in context.replace(",", ""):
                    issues.append(f"رقم '{num}' غير موجود في السياق")

        # فحص الإجابة الفارغة
        if len(answer.strip()) < 10:
            issues.append("الإجابة قصيرة جداً")

        return issues

    # ════════════════════════════════════
    # فحص LLM
    # ════════════════════════════════════

    def _llm_audit(self, query: str, answer: str, context: str) -> dict:
        prompt = f"""أنت مدقق جودة لنظام RAG. افحص الإجابة مقارنةً بالسياق.

السؤال: {query}
السياق (المستند): {context[:800]}
الإجابة: {answer[:500]}

أجب بـ JSON فقط:
{{
  "hallucination": true/false,
  "source_verified": true/false,
  "numbers_issue": true/false,
  "bias": true/false,
  "confidence": 0.0-1.0,
  "cleaned_answer": "الإجابة بعد إزالة أي معلومات مشكوك فيها (أو نفس الإجابة إذا سليمة)"
}}"""

        try:
            resp = self._client.models.generate_content(
                model=self._model, contents=prompt
            )
            raw  = re.sub(r"```json|```", "", resp.text).strip()
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"⚠️ Auditor LLM: {e}")
            return {"source_verified": True, "confidence": 0.8, "cleaned_answer": answer}