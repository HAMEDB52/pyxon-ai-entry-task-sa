# ============================================================
# src/data_processing/chunking/contextual_enricher.py
# إثراء السياق — يُضيف سياقاً لكل قطعة قبل التضمين
#
# المشكلة: قطعة معزولة مثل "المبلغ الإجمالي: 5500"
#          بدون سياق لا تعني شيئاً عند البحث.
#
# الحل (Anthropic Contextual Retrieval):
#   يُضاف: "هذا المقطع من فاتورة INV-101 الصادرة لشركة X
#           بتاريخ 2024-01-15. المبلغ الإجمالي: 5500 SAR"
#   → البحث يصبح أدق بكثير
# ============================================================

import os
from dataclasses import dataclass
from loguru import logger

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class EnrichedChunk:
    """قطعة مُثرَاة بالسياق"""
    chunk_id       : str
    original       : str     # النص الأصلي
    context_prefix : str     # السياق المُضاف
    enriched       : str     # context_prefix + original (يُستخدم للتضمين)
    source_file    : str
    page_number    : int   = 0
    chunk_type     : str   = "text"
    token_count    : int   = 0
    metadata       : dict  = None

    @property
    def content(self) -> str:
        """توافق مع نظام الـ chunks الأصلي"""
        return self.enriched

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ContextualEnricher:
    """
    يُضيف سياقاً نصياً لكل قطعة بناءً على محتوى المستند الكامل.

    الاستخدام:
        enricher = ContextualEnricher()
        enriched = enricher.enrich(chunks, full_document_text)

        # استخدم enriched_chunk.enriched للتضمين
        # استخدم enriched_chunk.original للعرض
    """

    CONTEXT_TOKENS  = 100   # أقصى طول للسياق المُضاف
    BATCH_SIZE      = 5     # عدد القطع لكل طلب LLM

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        api_key = os.getenv("GOOGLE_API_KEY")
        self._client = genai.Client(
            api_key      = api_key,
            http_options = types.HttpOptions(api_version="v1beta"),
        ) if api_key and use_llm else None
        self._model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
        logger.info(f"✅ ContextualEnricher | LLM={'✅' if self._client else '❌'}")

    # ════════════════════════════════════
    # الدالة الرئيسية
    # ════════════════════════════════════

    def enrich(
        self,
        chunks       : list,
        full_document: str,
        source_file  : str = "",
    ) -> list[EnrichedChunk]:
        """
        إثراء قائمة قطع بالسياق.

        Args:
            chunks       : قائمة dicts أو objects تحتوي content و chunk_id
            full_document: النص الكامل للمستند (أول 2000 حرف كافٍ)
            source_file  : اسم الملف

        Returns:
            list[EnrichedChunk]
        """
        logger.info(f"✨ إثراء {len(chunks)} قطعة | {source_file or 'unknown'}")

        # ملخص المستند (أول 1500 حرف)
        doc_summary = full_document[:1500] if full_document else ""

        enriched_chunks = []
        for chunk in chunks:
            content  = chunk.get("content","") if isinstance(chunk,dict) else getattr(chunk,"content","")
            chunk_id = chunk.get("chunk_id","") if isinstance(chunk,dict) else getattr(chunk,"chunk_id","")
            page     = chunk.get("page_number",0) if isinstance(chunk,dict) else getattr(chunk,"page_number",0)
            meta     = chunk.get("metadata",{}) if isinstance(chunk,dict) else getattr(chunk,"metadata",{})

            if not content.strip():
                continue

            # توليد السياق
            ctx = self._generate_context(content, doc_summary, source_file)

            # استخراج الحقول الإضافية للتوافق
            c_type = chunk.get("chunk_type", "text") if isinstance(chunk, dict) else getattr(chunk, "chunk_type", "text")
            t_count = chunk.get("token_count", 0) if isinstance(chunk, dict) else getattr(chunk, "token_count", 0)

            enriched_chunks.append(EnrichedChunk(
                chunk_id       = chunk_id,
                original       = content,
                context_prefix = ctx,
                enriched       = f"{ctx}\n\n{content}" if ctx else content,
                source_file    = source_file,
                page_number    = page,
                chunk_type     = c_type,
                token_count    = t_count,
                metadata       = meta or {},
            ))

        logger.success(f"✅ إثراء مكتمل | {len(enriched_chunks)} قطعة")
        return enriched_chunks

    # ════════════════════════════════════
    # توليد السياق
    # ════════════════════════════════════

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=15, max=60))
    def _generate_context(
        self,
        chunk_text  : str,
        doc_summary : str,
        source_file : str,
    ) -> str:
        """يولّد جملة سياق قصيرة للقطعة"""

        # بدون LLM → سياق regex أساسي
        if not self._client or not doc_summary:
            return self._regex_context(chunk_text, source_file)

        prompt = f"""اكتب جملة واحدة فقط (أقل من 50 كلمة) تصف السياق الذي يأتي منه هذا المقطع.

المستند: {source_file}
ملخص المستند: {doc_summary[:400]}

المقطع:
{chunk_text[:300]}

الجملة (بدون مقدمة، مباشرة):"""

        try:
            resp = self._client.models.generate_content(
                model=self._model, contents=prompt
            )
            ctx = resp.text.strip()
            # تأكد أنه قصير
            words = ctx.split()
            if len(words) > 50:
                ctx = " ".join(words[:50]) + "..."
            return ctx
        except Exception:
            return self._regex_context(chunk_text, source_file)

    def _regex_context(self, content: str, source_file: str) -> str:
        """سياق أساسي بدون LLM"""
        import re
        parts = []

        if source_file:
            parts.append(f"من مستند: {source_file}")

        # استخراج رقم فاتورة
        m = re.search(r"\b(INV|RCP|REF)[-\s]?\d{2,6}\b", content, re.I)
        if m:
            parts.append(f"رقم المرجع: {m.group().upper()}")

        # استخراج مبلغ
        m = re.search(r"[\d,]+(?:\.\d{1,2})?\s*(?:SAR|ريال|\$|USD)", content, re.I)
        if m:
            parts.append(f"المبلغ: {m.group()}")

        return " | ".join(parts) if parts else ""