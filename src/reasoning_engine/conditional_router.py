# ============================================================
# reasoning_engine/conditional_router.py
# الموجّه الشرطي — يقرر أين يذهب كل استعلام
# هو "حارس البوابة" الذي يوجّه الاستعلام للمسار الصحيح
# ============================================================

import os
from dataclasses import dataclass, field
from enum import Enum

from google import genai
from google.genai import types
from loguru import logger


# ============================================================
# مسارات التوجيه
# ============================================================

class Route(str, Enum):
    DIRECT_ANSWER   = "direct_answer"    # إجابة مباشرة من الذاكرة
    DOCUMENT_SEARCH = "document_search"  # بحث في المستندات
    MULTI_AGENT     = "multi_agent"      # تفعيل نظام متعدد الوكلاء
    HUMAN_VALIDATION= "human_validation" # يحتاج تحقق بشري
    STRESS_TEST     = "stress_test"      # اكتُشف هجوم أو استعلام خبيث
    WEB_SEARCH      = "web_search"       # يحتاج بحثاً في الويب


# ============================================================
# قرار التوجيه
# ============================================================

@dataclass
class RoutingDecision:
    """
    قرار التوجيه لاستعلام واحد
    """
    query           : str
    route           : Route
    confidence      : float          = 1.0     # مستوى الثقة 0.0 → 1.0
    reason          : str            = ""      # سبب القرار
    is_safe         : bool           = True    # هل الاستعلام آمن؟
    risk_flags      : list[str]      = field(default_factory=list)
    metadata        : dict           = field(default_factory=dict)

    def __repr__(self):
        safety = "✅" if self.is_safe else "⚠️"
        return (
            f"{safety} [{self.route.value}] "
            f"ثقة: {self.confidence:.0%} | "
            f"{self.reason[:60]}"
        )


# ============================================================
# الموجّه الشرطي
# ============================================================

class ConditionalRouter:
    """
    يحلل كل استعلام ويقرر المسار الأنسب له

    المسارات الممكنة:
        direct_answer   : إجابة مباشرة بدون بحث
        document_search : بحث في المستندات المُدخلة
        multi_agent     : تفعيل نظام الوكلاء المتعدد
        human_validation: يحتاج مراجعة بشرية
        stress_test     : استعلام مشبوه أو هجوم
        web_search      : يحتاج معلومات من الإنترنت

    مثال الاستخدام:
        router   = ConditionalRouter()
        decision = router.route("ما هو إجمالي الفاتورة؟")
        print(decision)
    """

    # كلمات تدل على استعلام خبيث أو هجوم
    INJECTION_PATTERNS = [
        "ignore previous",
        "ignore all instructions",
        "forget your instructions",
        "you are now",
        "act as",
        "جاهل التعليمات",
        "تجاهل كل شيء",
        "أنت الآن",
    ]

    # أسئلة تحيّة أو هوية فقط — لا أسئلة عن المستندات أبداً
    DIRECT_PATTERNS = [
        "كيف حالك", "مرحبا", "مرحباً", "مساء الخير", "صباح الخير",
        "شكرا", "شكراً", "من أنت", "ما اسمك",
        "what is your name", "who are you",
        "hello", "hi", "hey", "thanks", "thank you",
    ]

    # كلمات تدل على سؤال عن المستندات — تمنع التوجيه المباشر دائماً
    DOCUMENT_KEYWORDS = [
        # عربي — وثائق وبيانات
        "ملف", "مستند", "فاتورة", "وثيقة", "سيرة", "عقد", "إيجار",
        "تقرير", "سجل", "صك", "ايصال", "إيصال", "كشف",
        # عربي — كيانات
        "اسم", "المؤجر", "المستأجر", "مؤجر", "مستأجر", "مالك", "مستأجر",
        "شركة", "عميل", "مورد", "بائع", "مشتري", "طرف",
        # عربي — أرقام ومبالغ
        "رقم", "قيمة", "مبلغ", "سعر", "إجمالي", "مجموع", "رصيد",
        "دفع", "دفعة", "ريال", "دولار",
        # عربي — تواريخ ومواعيد
        "تاريخ", "موعد", "مدة", "سنة", "شهر",
        # عربي — أسئلة
        "وش", "ايش", "أيش", "شو", "كم", "من", "منو", "متى",
        "ابحث", "أخبرني", "اعطني", "وين", "أين",
        "cv", "pdf", "موجود", "يوجد",
        # إنجليزي
        "file", "document", "invoice", "receipt", "contract",
        "name", "landlord", "tenant", "owner", "party",
        "amount", "total", "price", "payment", "date",
        "who", "what", "when", "how much", "find",
    ]

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY غير موجود في .env")

        self.client = genai.Client(
            api_key      = api_key,
            http_options = types.HttpOptions(api_version="v1beta"),
        )
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        logger.info(f"✅ ConditionalRouter جاهز | نموذج: {self.model_name}")

    # ============================================================
    # الدالة الرئيسية
    # ============================================================

    def route(self, query: str) -> RoutingDecision:
        """
        تحليل الاستعلام وإصدار قرار التوجيه

        Args:
            query: استعلام المستخدم

        Returns:
            RoutingDecision: قرار التوجيه
        """
        logger.info(f"🔀 توجيه الاستعلام: {query[:60]}")

        # --- 1. فحص الأمان أولاً ---
        safety_check = self._check_safety(query)
        if not safety_check["is_safe"]:
            return RoutingDecision(
                query      = query,
                route      = Route.STRESS_TEST,
                confidence = 1.0,
                reason     = "تم اكتشاف نمط هجوم محتمل",
                is_safe    = False,
                risk_flags = safety_check["flags"],
            )

        # --- 2. فحص الأسئلة المباشرة ---
        if self._is_direct_question(query):
            return RoutingDecision(
                query      = query,
                route      = Route.DIRECT_ANSWER,
                confidence = 0.95,
                reason     = "سؤال بسيط لا يحتاج بحثاً",
                is_safe    = True,
            )

        # --- 3. التوجيه الذكي باستخدام Gemini ---
        decision = self._llm_route(query)
        logger.success(f"✅ قرار التوجيه: {decision}")
        return decision

    # ============================================================
    # فحص الأمان
    # ============================================================

    def _check_safety(self, query: str) -> dict:
        """
        فحص الاستعلام بحثاً عن أنماط الهجوم

        Returns:
            dict: {"is_safe": bool, "flags": list}
        """
        query_lower = query.lower()
        flags       = []

        for pattern in self.INJECTION_PATTERNS:
            if pattern.lower() in query_lower:
                flags.append(f"نمط حقن: '{pattern}'")

        is_safe = len(flags) == 0

        if not is_safe:
            logger.warning(f"🚨 استعلام مشبوه: {flags}")

        return {"is_safe": is_safe, "flags": flags}

    # ============================================================
    # فحص الأسئلة المباشرة
    # ============================================================

    def _is_direct_question(self, query: str) -> bool:
        """التحقق من أن السؤال يمكن الإجابة عليه مباشرة"""
        query_lower = query.lower().strip()

        # أولاً: إذا يحتوي كلمة وثيقة → ابحث في المستندات دائماً
        for kw in self.DOCUMENT_KEYWORDS:
            if kw in query_lower:
                return False

        # ثانياً: تحية أو سؤال هوية فقط
        for pattern in self.DIRECT_PATTERNS:
            if pattern in query_lower:
                return True

        return False

    # ============================================================
    # التوجيه الذكي
    # ============================================================

    def _llm_route(self, query: str) -> RoutingDecision:
        """استخدام Gemini لتحديد المسار الأنسب"""

        prompt = f"""أنت موجّه ذكي لنظام RAG. حدد المسار الأنسب للسؤال التالي.

السؤال: {query}

المسارات المتاحة:
- document_search : يحتاج بحثاً في المستندات المُحمَّلة (فواتير، تقارير، وثائق)
- multi_agent     : سؤال معقد يحتاج تحليلاً متعدد الخطوات
- human_validation: يحتوي على قرارات حساسة تحتاج مراجعة بشرية
- web_search      : يحتاج معلومات حديثة من الإنترنت

أجب بهذا الشكل بالضبط:
ROUTE: [اسم المسار]
CONFIDENCE: [رقم من 0.0 إلى 1.0]
REASON: [سبب قصير]"""

        try:
            response = self.client.models.generate_content(
                model    = self.model_name,
                contents = prompt,
            )
            return self._parse_routing_response(query, response.text)

        except Exception as e:
            logger.warning(f"⚠️ فشل التوجيه الذكي: {e} — سيُستخدم document_search")
            return RoutingDecision(
                query      = query,
                route      = Route.DOCUMENT_SEARCH,
                confidence = 0.7,
                reason     = "توجيه افتراضي",
                is_safe    = True,
            )

    # ============================================================
    # تحليل رد Gemini
    # ============================================================

    def _parse_routing_response(self, query: str, raw: str) -> RoutingDecision:
        """تحليل رد Gemini واستخراج قرار التوجيه"""

        route      = Route.DOCUMENT_SEARCH
        confidence = 0.8
        reason     = ""

        for line in raw.split("\n"):
            line = line.strip()

            if line.upper().startswith("ROUTE:"):
                route_str = line.split(":", 1)[1].strip().lower()
                for r in Route:
                    if r.value in route_str:
                        route = r
                        break

            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass

            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        return RoutingDecision(
            query      = query,
            route      = route,
            confidence = confidence,
            reason     = reason,
            is_safe    = True,
        )

    # ============================================================
    # دوال مساعدة
    # ============================================================

    def should_use_multi_agent(self, decision: RoutingDecision) -> bool:
        """هل يجب تفعيل نظام الوكلاء المتعدد؟"""
        return decision.route in (
            Route.MULTI_AGENT,
            Route.DOCUMENT_SEARCH,
        ) and decision.confidence >= 0.7

    def needs_human_review(self, decision: RoutingDecision) -> bool:
        """هل يحتاج مراجعة بشرية؟"""
        return (
            decision.route == Route.HUMAN_VALIDATION
            or not decision.is_safe
            or decision.confidence < 0.5
        )