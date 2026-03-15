# ============================================================
# reasoning_engine/tool_execution.py
# منفذ الأدوات — ينفذ الأدوات بناءً على خطة المخطط
# يربط بين قرار الوكيل والتنفيذ الفعلي
# ============================================================

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from src.reasoning_engine.planner import ExecutionPlan, SubQuestion


# ============================================================
# أنواع الأدوات المتاحة
# ============================================================

class ToolType(str, Enum):
    HYBRID_SEARCH  = "hybrid_search"    # بحث هجين في قاعدة البيانات
    VECTOR_SEARCH  = "vector_search"    # بحث متجهي فقط
    BM25_SEARCH    = "bm25_search"      # بحث نصي فقط
    DOCUMENT_LOAD  = "document_load"    # تحميل مستند جديد
    PARSE_DOCUMENT = "parse_document"   # استخراج نص من مستند عبر MCP
    SUMMARIZE      = "summarize"        # تلخيص نص
    ANSWER         = "answer"           # توليد إجابة نهائية


# ============================================================
# نتيجة تنفيذ أداة
# ============================================================

@dataclass
class ToolResult:
    """نتيجة تنفيذ أداة واحدة"""
    tool_type    : ToolType
    success      : bool
    data         : Any                = None     # البيانات المُسترجعة
    error        : str                = ""       # رسالة الخطأ إن وُجد
    execution_ms : int                = 0        # وقت التنفيذ بالمللي ثانية
    metadata     : dict               = field(default_factory=dict)

    @property
    def has_results(self) -> bool:
        return self.success and self.data is not None

    def __repr__(self):
        status = "✅" if self.success else "❌"
        return f"{status} [{self.tool_type.value}] | {self.execution_ms}ms"


# ============================================================
# نتيجة تنفيذ الخطة كاملة
# ============================================================

@dataclass
class PlanExecutionResult:
    """نتيجة تنفيذ الخطة كاملة"""
    plan          : ExecutionPlan
    tool_results  : list[ToolResult]  = field(default_factory=list)
    final_context : str               = ""    # السياق المجمّع للإجابة
    success       : bool              = True

    @property
    def total_chunks_found(self) -> int:
        total = 0
        for r in self.tool_results:
            if r.has_results and isinstance(r.data, list):
                total += len(r.data)
        return total

    def __repr__(self):
        return (
            f"📊 تنفيذ الخطة | "
            f"أدوات: {len(self.tool_results)} | "
            f"قطع: {self.total_chunks_found} | "
            f"{'✅' if self.success else '❌'}"
        )


# ============================================================
# منفذ الأدوات الرئيسي
# ============================================================

class ToolExecutor:
    """
    ينفذ الأدوات بناءً على خطة المخطط (Planner)

    الأدوات المتاحة:
        - hybrid_search : بحث هجين (BM25 + Vector + RRF)
        - vector_search : بحث متجهي فقط
        - bm25_search   : بحث نصي فقط
        - document_load : تحميل مستند جديد
        - summarize     : تلخيص نص

    مثال الاستخدام:
        executor = ToolExecutor()
        result   = executor.execute_plan(plan)
        print(result.final_context)
    """

    def __init__(self):
        self._hybrid_search  = None   # تحميل كسول
        self._vector_store   = None
        self._loader         = None
        self._summarizer     = None
        logger.info("✅ ToolExecutor جاهز")

    # ============================================================
    # تنفيذ الخطة كاملة
    # ============================================================

    def execute_plan(self, plan: ExecutionPlan) -> PlanExecutionResult:
        """
        تنفيذ خطة التنفيذ كاملة

        Args:
            plan: الخطة من Planner

        Returns:
            PlanExecutionResult: نتائج جميع الأدوات مجمّعة
        """
        import time
        logger.info(f"⚙️ تنفيذ الخطة: {plan.query_type.value} | {plan.num_steps} خطوة")

        tool_results  : list[ToolResult] = []
        all_contexts  : list[str]        = []

        for sub_q in plan.sub_questions:
            logger.info(f"  🔧 خطوة {sub_q.index}: {sub_q.question[:60]}")

            start = time.time()
            result = self._execute_step(sub_q, plan.search_strategy)
            result.execution_ms = int((time.time() - start) * 1000)

            tool_results.append(result)

            # تجميع السياق من النتائج
            if result.has_results:
                context = self._format_results_as_context(
                    sub_q.question,
                    result.data,
                )
                all_contexts.append(context)

        # دمج كل السياقات
        final_context = "\n\n---\n\n".join(all_contexts)

        plan_result = PlanExecutionResult(
            plan          = plan,
            tool_results  = tool_results,
            final_context = final_context,
            success       = any(r.success for r in tool_results),
        )

        logger.success(f"✅ {plan_result}")
        return plan_result

    # ============================================================
    # تنفيذ خطوة واحدة
    # ============================================================

    def _execute_step(
        self,
        sub_q   : SubQuestion,
        strategy: str,
    ) -> ToolResult:
        """تنفيذ خطوة واحدة من الخطة"""

        try:
            if strategy == "hybrid":
                return self._run_hybrid_search(sub_q.question)
            elif strategy == "vector":
                return self._run_vector_search(sub_q.question)
            elif strategy == "bm25":
                return self._run_bm25_search(sub_q.question)
            elif strategy == "parse_document":
                # افتراض أن sub_q.question يحتوي على مسار الملف بوضوح للاستعانة بـ MCP
                file_path = sub_q.question.replace("اقرأ الملف: ", "").strip()
                return self._run_mcp_parse_document(file_path)
            else:
                return self._run_hybrid_search(sub_q.question)

        except Exception as e:
            logger.error(f"❌ فشل تنفيذ الخطوة: {e}")
            return ToolResult(
                tool_type = ToolType.HYBRID_SEARCH,
                success   = False,
                error     = str(e),
            )

    # ============================================================
    # البحث الهجين
    # ============================================================

    def _run_hybrid_search(self, query: str, top_k: int = 5) -> ToolResult:
        """تشغيل البحث الهجين"""
        try:
            search = self._get_hybrid_search()
            results = search.search(query, top_k=top_k)

            return ToolResult(
                tool_type = ToolType.HYBRID_SEARCH,
                success   = True,
                data      = results,
                metadata  = {"query": query, "top_k": top_k},
            )
        except Exception as e:
            logger.error(f"❌ البحث الهجين فشل: {e}")
            return ToolResult(
                tool_type = ToolType.HYBRID_SEARCH,
                success   = False,
                error     = str(e),
            )

    # ============================================================
    # البحث المتجهي
    # ============================================================

    def _run_vector_search(self, query: str, top_k: int = 5) -> ToolResult:
        """تشغيل البحث المتجهي فقط"""
        try:
            store   = self._get_vector_store()
            results = store.search(query, top_k=top_k)

            return ToolResult(
                tool_type = ToolType.VECTOR_SEARCH,
                success   = True,
                data      = results,
                metadata  = {"query": query, "top_k": top_k},
            )
        except Exception as e:
            logger.error(f"❌ البحث المتجهي فشل: {e}")
            return ToolResult(
                tool_type = ToolType.VECTOR_SEARCH,
                success   = False,
                error     = str(e),
            )

    # ============================================================
    # البحث النصي BM25
    # ============================================================

    def _run_bm25_search(self, query: str, top_k: int = 5) -> ToolResult:
        """تشغيل البحث النصي فقط"""
        try:
            search  = self._get_hybrid_search()
            results = search._bm25_search(query, top_k=top_k)

            return ToolResult(
                tool_type = ToolType.BM25_SEARCH,
                success   = True,
                data      = results,
                metadata  = {"query": query},
            )
        except Exception as e:
            logger.error(f"❌ البحث النصي فشل: {e}")
            return ToolResult(
                tool_type = ToolType.BM25_SEARCH,
                success   = False,
                error     = str(e),
            )

    # ============================================================
    # أدوات MCP
    # ============================================================

    def _run_mcp_parse_document(self, file_path: str) -> ToolResult:
        """تشغيل أداة parse_document عبر خادم MCP"""
        try:
            from src.reasoning_engine.mcp_client import default_mcp_client
            
            result_text = default_mcp_client.call_tool(
                tool_name="parse_document", 
                arguments={"file_path": file_path}
            )
            
            # نعامل النص الراجع كأنه قطعة نصية واحدة (Chunk) كبيرة
            data = [{"content": result_text, "source_file": file_path, "similarity_score": 1.0}]
            
            return ToolResult(
                tool_type = ToolType.PARSE_DOCUMENT,
                success   = not result_text.startswith("Error"),
                data      = data,
                metadata  = {"file_path": file_path},
            )
        except Exception as e:
            logger.error(f"❌ طلب MCP فشل: {e}")
            return ToolResult(
                tool_type = ToolType.PARSE_DOCUMENT,
                success   = False,
                error     = str(e),
            )

    # ============================================================
    # تنسيق النتائج كسياق
    # ============================================================

    def _format_results_as_context(
        self,
        question: str,
        results : list,
    ) -> str:
        """
        تحويل نتائج البحث إلى نص سياق مناسب للنموذج اللغوي

        Args:
            question: السؤال الفرعي
            results : نتائج البحث

        Returns:
            str: النص المنسق
        """
        if not results:
            return f"[لا توجد نتائج للسؤال: {question}]"

        parts = [f"### نتائج البحث عن: {question}\n"]

        for i, result in enumerate(results, 1):
            # دعم SearchResult objects و dicts
            if isinstance(result, dict):
                content        = result.get("content", "")
                source         = result.get("source_file", "")
                page           = result.get("page_number", 0)
                parent_heading = result.get("parent_heading", "")
                score          = result.get("rrf_score", result.get("similarity_score", 0))
            else:
                content        = getattr(result, "content", "")
                source         = getattr(result, "source_file", "")
                page           = getattr(result, "page_number", 0)
                parent_heading = getattr(result, "parent_heading", "")
                score          = getattr(result, "rrf_score", 0)

            citation = f"[{source}"
            if page:
                citation += f", ص{page}"
            citation += "]"

            parts.append(
                f"**المصدر {i}** {citation}"
                + (f" | القسم: {parent_heading}" if parent_heading else "")
                + f"\n{content}\n"
            )

        return "\n".join(parts)

    # ============================================================
    # تحميل كسول للمكونات
    # ============================================================

    def _get_hybrid_search(self):
        if self._hybrid_search is None:
            from src.database.hybrid_search import HybridSearch
            self._hybrid_search = HybridSearch()
        return self._hybrid_search

    def _get_vector_store(self):
        if self._vector_store is None:
            from src.database.vector_store import VectorStore
            self._vector_store = VectorStore()
        return self._vector_store