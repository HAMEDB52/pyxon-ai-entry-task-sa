# ============================================================
# agents/state.py
# الحالة المشتركة بين جميع الوكلاء في LangGraph
# هذا هو "ذاكرة" النظام التي تنتقل بين الوكلاء
# ============================================================

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Annotated
from langgraph.graph.message import add_messages


# ============================================================
# مراحل التنفيذ
# ============================================================

class AgentStage(str, Enum):
    START        = "start"         # بداية المحادثة
    ROUTING      = "routing"       # وكيل التوجيه يحلل الاستعلام
    PLANNING     = "planning"      # المخطط يبني الخطة
    RESEARCHING  = "researching"   # وكيل البحث يبحث في المستندات
    VERIFYING    = "verifying"     # وكيل التحقق يتحقق من الإجابة
    CORRECTING   = "correcting"    # وكيل التصحيح يصحح الإجابة
    ANSWERING    = "answering"     # توليد الإجابة النهائية
    DONE         = "done"          # اكتمل
    ERROR        = "error"         # خطأ


# ============================================================
# نتيجة البحث المبسطة للحالة
# ============================================================

@dataclass
class RetrievedChunk:
    """قطعة مسترجعة من قاعدة البيانات"""
    chunk_id       : str
    content        : str
    source_file    : str   = ""
    page_number    : int   = 0
    parent_heading : str   = ""
    score          : float = 0.0

    def to_context_string(self) -> str:
        """تحويل إلى نص سياق للنموذج اللغوي"""
        source = f"[{self.source_file}"
        if self.page_number:
            source += f", ص{self.page_number}"
        source += "]"

        heading = f" | {self.parent_heading}" if self.parent_heading else ""
        return f"{source}{heading}\n{self.content}"


# ============================================================
# نتيجة التحقق
# ============================================================

@dataclass
class VerificationResult:
    """نتيجة تحقق وكيل المراجعة"""
    is_faithful     : bool          # هل الإجابة مدعومة بالمصادر؟
    is_relevant     : bool          # هل الإجابة ذات صلة بالسؤال؟
    confidence      : float  = 0.0  # درجة الثقة 0.0 → 1.0
    issues          : list[str] = field(default_factory=list)
    suggestions     : list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.is_faithful and self.is_relevant and self.confidence >= 0.6


# ============================================================
# الحالة المشتركة الرئيسية
# ============================================================

class GraphState(dict):
    """
    الحالة المشتركة التي تنتقل بين جميع الوكلاء في LangGraph

    كل وكيل يقرأ ويكتب في هذه الحالة.
    LangGraph يتتبع التغييرات ويوجّه التدفق.

    الحقول:
        messages        : سجل المحادثة الكامل
        query           : السؤال الأصلي للمستخدم
        stage           : المرحلة الحالية
        route           : المسار المحدد من الموجّه
        plan            : خطة التنفيذ من المخطط
        retrieved_chunks: القطع المسترجعة من البحث
        draft_answer    : الإجابة المسودة قبل التحقق
        final_answer    : الإجابة النهائية المعتمدة
        verification    : نتيجة التحقق
        retry_count     : عدد محاولات إعادة البحث
        error_message   : رسالة الخطأ إن وُجد
        metadata        : بيانات إضافية
    """

    # ============================================================
    # مُنشئ الحالة الابتدائية
    # ============================================================

    @classmethod
    def create(cls, query: str) -> "GraphState":
        import time
        state = cls()
        state.update({
            "messages"        : [],
            "query"           : query,
            "stage"           : AgentStage.START,
            "route"           : None,
            "plan"            : None,
            "retrieved_chunks": [],
            "draft_answer"    : "",
            "final_answer"    : "",
            "verification"    : None,
            "retry_count"     : 0,
            "max_retries"     : int(3),
            "error_message"   : "",
            "metadata"        : {
                "query"       : query,
                "start_time"  : time.time(),
                "total_tokens": 0,
            },
        })
        return state

    # ============================================================
    # خصائص مساعدة
    # ============================================================

    @property
    def query(self) -> str:
        return self.get("query", "")

    @property
    def stage(self) -> AgentStage:
        return self.get("stage", AgentStage.START)

    @property
    def retrieved_chunks(self) -> list[RetrievedChunk]:
        return self.get("retrieved_chunks", [])

    @property
    def draft_answer(self) -> str:
        return self.get("draft_answer", "")

    @property
    def final_answer(self) -> str:
        return self.get("final_answer", "")

    @property
    def retry_count(self) -> int:
        return self.get("retry_count", 0)

    @property
    def max_retries(self) -> int:
        return self.get("max_retries", 3)

    @property
    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

    @property
    def has_answer(self) -> bool:
        return bool(self.final_answer.strip())

    @property
    def has_context(self) -> bool:
        return len(self.retrieved_chunks) > 0

    # ============================================================
    # دوال التحديث
    # ============================================================

    def set_stage(self, stage: AgentStage) -> "GraphState":
        """تحديث مرحلة التنفيذ"""
        self["stage"] = stage
        return self

    def add_chunk(self, chunk: RetrievedChunk) -> "GraphState":
        """إضافة قطعة مسترجعة"""
        chunks = self.get("retrieved_chunks", [])
        chunks.append(chunk)
        self["retrieved_chunks"] = chunks
        return self

    def set_chunks(self, chunks: list[RetrievedChunk]) -> "GraphState":
        """تعيين القطع المسترجعة"""
        self["retrieved_chunks"] = chunks
        return self

    def increment_retry(self) -> "GraphState":
        """زيادة عداد المحاولات"""
        self["retry_count"] = self.retry_count + 1
        return self

    def set_error(self, message: str) -> "GraphState":
        """تسجيل خطأ"""
        self["error_message"] = message
        self["stage"]         = AgentStage.ERROR
        return self

    # ============================================================
    # بناء السياق للنموذج اللغوي
    # ============================================================

    def build_context(self, max_chunks: int = 5) -> str:
        """
        بناء نص السياق من القطع المسترجعة
        لإرساله للنموذج اللغوي مع السؤال

        Args:
            max_chunks: أقصى عدد قطع في السياق

        Returns:
            str: نص السياق المنسق
        """
        chunks = self.retrieved_chunks[:max_chunks]

        if not chunks:
            return "لا توجد مستندات ذات صلة."

        parts = ["### المعلومات المسترجعة:\n"]
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"**[{i}]** {chunk.to_context_string()}\n")

        return "\n".join(parts)

    # ============================================================
    # ملخص الحالة
    # ============================================================

    def summary(self) -> str:
        """ملخص سريع للحالة الحالية"""
        return (
            f"📊 الحالة | "
            f"المرحلة: {self.stage.value} | "
            f"قطع: {len(self.retrieved_chunks)} | "
            f"محاولات: {self.retry_count}/{self.max_retries} | "
            f"إجابة: {'✅' if self.has_answer else '⏳'}"
        )