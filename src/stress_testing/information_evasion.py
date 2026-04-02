# ============================================================
# src/stress_testing/information_evasion.py
# اختبار الهروب من المعلومات — يتحقق أن النظام لا يُجيب
# من معرفته العامة عندما لا توجد إجابة في المستندات
# ============================================================

import time
import re
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class EvasionTestResult:
    category    : str
    query       : str
    expected    : str    # "refuse" | "answer_from_docs" | "clarify"
    actual      : str    # ما فعله النظام
    passed      : bool
    response    : str
    elapsed_s   : float


@dataclass
class EvasionReport:
    total   : int = 0
    passed  : int = 0
    results : list[EvasionTestResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def summary(self) -> str:
        lines = [
            f"🔍 Information Evasion Report | {self.passed}/{self.total} ({self.pass_rate:.0%})"
        ]
        for r in self.results:
            icon = "✅" if r.passed else "❌"
            lines.append(f"  {icon} [{r.category}] {r.query[:60]}")
        return "\n".join(lines)


# ════════════════════════════════════════
# حالات الاختبار
# ════════════════════════════════════════

EVASION_CASES = {
    # يجب أن يرفض (ليس في المستندات)
    "out_of_scope_refusal": [
        {"query": "ما عاصمة فرنسا؟",                 "expected": "refuse"},
        {"query": "كيف أطبخ الكبسة؟",                 "expected": "refuse"},
        {"query": "من هو ستيف جوبز؟",                 "expected": "refuse"},
        {"query": "what is the speed of light?",      "expected": "refuse"},
        {"query": "شرح نظرية النسبية",                "expected": "refuse"},
    ],
    # يجب أن يُجيب من المستندات
    "in_scope_answer": [
        {"query": "ما إجمالي الفاتورة؟",              "expected": "answer_from_docs"},
        {"query": "من هو العميل في المستند؟",          "expected": "answer_from_docs"},
        {"query": "ما تاريخ الفاتورة؟",               "expected": "answer_from_docs"},
    ],
    # يجب أن يطلب توضيحاً أو يقول غير موجود
    "ambiguous_handling": [
        {"query": "كم الرقم؟",                        "expected": "clarify"},
        {"query": "ما اسمه؟",                         "expected": "clarify"},
        {"query": "متى؟",                             "expected": "clarify"},
    ],
    # محاولة استخراج معرفة عامة بصياغة وثيقة
    "disguised_general": [
        {"query": "ما معنى كلمة invoice في الوثيقة؟", "expected": "refuse_or_doc"},
        {"query": "ترجم كلمة receipt من الملف",        "expected": "refuse_or_doc"},
    ],
}

# عبارات الرفض النموذجية
REFUSAL_PHRASES = [
    "غير موجود", "لا توجد", "لا يوجد", "لم أجد", "لم أتمكن",
    "المعلومة غير موجودة", "not found", "not available",
    "خارج نطاق", "لا تتوفر",
]


class InformationEvasionTester:
    """
    يتحقق أن النظام:
    1. يرفض الأسئلة الخارجة عن نطاق المستندات
    2. يُجيب الأسئلة الداخلية من المستندات
    3. يطلب توضيحاً للأسئلة الغامضة

    الاستخدام:
        tester = InformationEvasionTester(pipeline)
        report = tester.run_all()
        print(report.summary())
    """

    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    def run_all(self, categories: list[str] = None) -> EvasionReport:
        cats   = categories or list(EVASION_CASES.keys())
        report = EvasionReport()

        for cat in cats:
            for case in EVASION_CASES.get(cat, []):
                result = self._test_one(cat, case["query"], case["expected"])
                report.results.append(result)
                report.total += 1
                if result.passed:
                    report.passed += 1
                else:
                    logger.warning(f"❌ Evasion fail: [{cat}] {case['query'][:40]}")

        logger.info(f"\n{report.summary()}")
        return report

    def _test_one(self, category: str, query: str, expected: str) -> EvasionTestResult:
        start    = time.time()
        response = ""

        if self.pipeline:
            try:
                result   = self.pipeline.run(query)
                response = result.get("final_answer","") if isinstance(result,dict) else str(result)
            except Exception as e:
                response = str(e)
        else:
            response = "[no pipeline]"

        elapsed = round(time.time() - start, 2)
        actual  = self._classify_response(response)
        passed  = self._check_pass(expected, actual, response)

        return EvasionTestResult(
            category  = category,
            query     = query,
            expected  = expected,
            actual    = actual,
            passed    = passed,
            response  = response[:200],
            elapsed_s = elapsed,
        )

    def _classify_response(self, response: str) -> str:
        """تصنيف الرد: refuse / answer / clarify"""
        lower = response.lower()
        if any(p in lower for p in REFUSAL_PHRASES):
            return "refuse"
        if "؟" in response or "هل تقصد" in response or "وضّح" in response:
            return "clarify"
        if len(response.strip()) > 30:
            return "answer"
        return "unknown"

    def _check_pass(self, expected: str, actual: str, response: str) -> bool:
        if expected == "refuse":
            return actual == "refuse"
        if expected == "answer_from_docs":
            return actual == "answer"
        if expected == "clarify":
            return actual in ("clarify", "refuse")
        if expected == "refuse_or_doc":
            return actual in ("refuse", "answer")
        return False