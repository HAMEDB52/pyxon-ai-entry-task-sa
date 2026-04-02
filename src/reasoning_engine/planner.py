# ============================================================
# reasoning_engine/planner.py
# المخطط — يحلل الاستعلام ويقسمه إلى خطوات قابلة للتنفيذ
# هو "عقل" النظام الذي يقرر كيف يتعامل مع كل سؤال
# ============================================================

import os
from dataclasses import dataclass, field
from enum import Enum

from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


# ============================================================
# أنواع الاستعلامات
# ============================================================

class QueryType(str, Enum):
    SIMPLE      = "simple"       # سؤال بسيط — إجابة مباشرة
    SEARCH      = "search"       # يحتاج بحثاً في المستندات
    MULTI_HOP   = "multi_hop"    # يحتاج خطوات متعددة
    ANALYTICAL  = "analytical"   # يحتاج تحليلاً ومقارنة
    UNKNOWN     = "unknown"      # غير محدد


# ============================================================
# خطة التنفيذ
# ============================================================

@dataclass
class SubQuestion:
    """سؤال فرعي واحد ضمن الخطة"""
    index       : int
    question    : str
    depends_on  : list[int] = field(default_factory=list)  # يعتمد على نتيجة أي سؤال فرعي

    def __repr__(self):
        deps = f" (يعتمد على: {self.depends_on})" if self.depends_on else ""
        return f"[{self.index}] {self.question}{deps}"


@dataclass
class ExecutionPlan:
    """
    خطة التنفيذ الكاملة للاستعلام
    تحتوي على:
        - نوع الاستعلام
        - الأسئلة الفرعية مرتبة
        - استراتيجية البحث
    """
    original_query  : str
    query_type      : QueryType
    sub_questions   : list[SubQuestion]   = field(default_factory=list)
    search_strategy : str                 = "hybrid"    # hybrid | vector | bm25
    needs_web_search: bool                = False
    reasoning_steps : list[str]           = field(default_factory=list)
    metadata        : dict                = field(default_factory=dict)

    @property
    def is_simple(self) -> bool:
        return self.query_type == QueryType.SIMPLE

    @property
    def num_steps(self) -> int:
        return len(self.sub_questions)

    def __repr__(self):
        return (
            f"📋 خطة: [{self.query_type.value}] "
            f"{self.num_steps} خطوة | "
            f"بحث: {self.search_strategy} | "
            f"ويب: {self.needs_web_search}"
        )


# ============================================================
# المخطط الرئيسي
# ============================================================

class Planner:
    """
    يحلل استعلام المستخدم ويبني خطة تنفيذ ذكية

    الخطوات:
        1. تصنيف نوع الاستعلام
        2. تقسيمه إلى أسئلة فرعية (إذا كان معقداً)
        3. تحديد استراتيجية البحث المناسبة
        4. تحديد ما إذا كان يحتاج بحثاً في الويب

    مثال الاستخدام:
        planner = Planner()
        plan = planner.plan("قارن بين أسعار المنتجات في الفاتورتين")
        print(plan)
    """

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY غير موجود في .env")

        self.client = genai.Client(
            api_key      = api_key,
            http_options = types.HttpOptions(api_version="v1beta"),
        )
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        logger.info(f"✅ Planner جاهز | نموذج: {self.model_name}")

    # ============================================================
    # الدالة الرئيسية
    # ============================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def plan(self, query: str, available_sources: list[str] = None) -> ExecutionPlan:
        """
        بناء خطة تنفيذ لاستعلام المستخدم مع التفكير المتسلسل (CoT)
        والرجوع للذاكرة طويلة المدى.
        """
        logger.info(f"🧠 تخطيط للاستعلام (CoT & Memory): {query[:80]}")

        # جلب الذاكرة طويلة المدى
        memory_context = ""
        try:
            from src.memory.long_term import default_lt_memory
            memory_context = default_lt_memory.get_all_memories()
        except Exception as e:
            logger.error(f"⚠️ فشل جلب الذاكرة: {e}")

        # بناء Prompt ذكي يفكر قبل أن يخطط
        prompt = f"""أنت مخطط ذكي لنظام استرجاع معلومات.
المستخدم سأل: {query}

معلومات قد تهمك عن المستخدم (ذاكرة طويلة المدى):
{memory_context}

التعليمات:
1. <thinking>فكر خطوة بخطوة في ما يحتاجه المستخدم حقاً وما هي أدوات البحث المناسبة (CoT). هل يحتاج لمعالجة ملف عبر إعطاء أمر parse_document؟ أم بحث عادي؟</thinking>
2. حدد الاستراتيجية المناسبة (hybrid, vector, bm25, parse_document).
3. قم بتبسيط السؤال إلى سؤال فرعي واحد أو أكثر.

أجب بالصيغة التالية تماماً:
STRATEGY: [اسم الاستراتيجية]
STEPS:
1. [سؤال البحث 1]
2. [سؤال البحث 2 إن لزم]
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            raw = response.text.strip()
            
            # استخراج التفكير (اختياري للعرض لاحقاً)
            thinking = ""
            if "<thinking>" in raw and "</thinking>" in raw:
                thinking = raw.split("<thinking>")[1].split("</thinking>")[0].strip()
                logger.debug(f"🤔 التفكير (CoT): {thinking}")
                
            return self._parse_plan_response(query, QueryType.SEARCH, raw)

        except Exception as e:
            logger.warning(f"⚠️ فشل المخطط الذكي: {e} — استخدام الخطة المباشرة كبديل احتياطي")
            return ExecutionPlan(
                original_query  = query,
                query_type      = QueryType.SEARCH,
                sub_questions   = [SubQuestion(index=1, question=query)],
                search_strategy = "hybrid",
                needs_web_search= False,
                reasoning_steps = ["خطة الطوارئ بسبب فشل الاتصال"]
            )

    # ============================================================
    # تصنيف الاستعلام
    # ============================================================

    def _classify_query(self, query: str) -> QueryType:
        """تصنيف نوع الاستعلام باستخدام Gemini"""

        prompt = f"""صنّف السؤال التالي إلى أحد هذه الأنواع فقط:
- simple: سؤال بسيط يحتاج إجابة واحدة مباشرة
- search: يحتاج بحثاً في مستندات للإجابة
- multi_hop: يحتاج خطوات متعددة أو مصادر متعددة
- analytical: يحتاج تحليلاً أو مقارنة أو حسابات

السؤال: {query}

أجب بكلمة واحدة فقط من الأنواع أعلاه:"""

        try:
            response = self.client.models.generate_content(
                model    = self.model_name,
                contents = prompt,
            )
            raw = response.text.strip().lower()

            for qt in QueryType:
                if qt.value in raw:
                    logger.debug(f"🏷️ نوع الاستعلام: {qt.value}")
                    return qt

            return QueryType.SEARCH  # الافتراضي

        except Exception as e:
            logger.warning(f"⚠️ فشل التصنيف: {e} — سيُستخدم SEARCH")
            return QueryType.SEARCH

    # ============================================================
    # بناء خطة معقدة
    # ============================================================

    def _build_complex_plan(
        self,
        query            : str,
        query_type       : QueryType,
        available_sources: list[str] = None,
    ) -> ExecutionPlan:
        """بناء خطة تفصيلية للاستعلامات المعقدة"""

        sources_text = ""
        if available_sources:
            sources_text = f"\nالمستندات المتاحة: {', '.join(available_sources)}"

        prompt = f"""أنت مساعد ذكي. حلل السؤال التالي وقسّمه إلى خطوات.

السؤال: {query}{sources_text}

أجب بهذا الشكل بالضبط:
STEPS:
1. [الخطوة الأولى]
2. [الخطوة الثانية]
3. [الخطوة الثالثة]
STRATEGY: hybrid
WEB: false

القواعد:
- STRATEGY يكون: hybrid أو vector أو bm25
- WEB يكون: true أو false
- لا تزيد عن 4 خطوات
- كل خطوة جملة واحدة واضحة"""

        try:
            response = self.client.models.generate_content(
                model    = self.model_name,
                contents = prompt,
            )
            raw = response.text.strip()

            return self._parse_plan_response(query, query_type, raw)

        except Exception as e:
            logger.warning(f"⚠️ فشل بناء الخطة: {e} — خطة بسيطة")
            return ExecutionPlan(
                original_query = query,
                query_type     = query_type,
                sub_questions  = [SubQuestion(index=1, question=query)],
                search_strategy = "hybrid",
            )

    # ============================================================
    # تحليل رد Gemini
    # ============================================================

    def _parse_plan_response(
        self,
        query     : str,
        query_type: QueryType,
        raw       : str,
    ) -> ExecutionPlan:
        """تحليل رد Gemini واستخراج الخطة"""

        sub_questions   : list[SubQuestion] = []
        search_strategy : str  = "hybrid"
        needs_web_search: bool = False
        reasoning_steps : list[str] = []

        lines = raw.split("\n")

        for line in lines:
            line = line.strip()

            # استخراج الخطوات
            if line and line[0].isdigit() and "." in line:
                parts = line.split(".", 1)
                if len(parts) == 2:
                    idx  = int(parts[0])
                    text = parts[1].strip()
                    sub_questions.append(SubQuestion(index=idx, question=text))
                    reasoning_steps.append(text)

            # استراتيجية البحث
            elif line.upper().startswith("STRATEGY:"):
                strategy = line.split(":", 1)[1].strip().lower()
                if strategy in ("hybrid", "vector", "bm25"):
                    search_strategy = strategy

            # هل يحتاج ويب
            elif line.upper().startswith("WEB:"):
                web_val = line.split(":", 1)[1].strip().lower()
                needs_web_search = web_val == "true"

        # إذا ما طلعت خطوات — نضيف السؤال الأصلي
        if not sub_questions:
            sub_questions = [SubQuestion(index=1, question=query)]

        return ExecutionPlan(
            original_query   = query,
            query_type       = query_type,
            sub_questions    = sub_questions,
            search_strategy  = search_strategy,
            needs_web_search = needs_web_search,
            reasoning_steps  = reasoning_steps,
        )