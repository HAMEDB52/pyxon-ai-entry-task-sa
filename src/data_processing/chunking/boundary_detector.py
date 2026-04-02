# ============================================================
# data_processing/chunking/boundary_detector.py
# كاشف حدود التجزئة — HybridChunker الرئيسي
# يجمع بين الوعي الهيكلي وحدود الرموز (Tokens)
# ============================================================

import os
import re
from dataclasses import dataclass, field

from loguru import logger

from src.data_processing.restructuring.document_parser import (
    ParsedDocument,
    DocumentElement,
    ElementType,
    fix_disconnected_arabic,
)
from src.data_processing.chunking.table_preserver import TablePreserver, TableChunk
from src.data_processing.chunking.heading_detector import HeadingDetector


# ============================================================
# القطعة النهائية الجاهزة للتضمين
# ============================================================

@dataclass
class Chunk:
    """
    القطعة النهائية الجاهزة للتضمين (Embedding) والتخزين
    """
    chunk_id       : str
    content        : str                        # النص الكامل للقطعة
    chunk_type     : str = "text"               # text | table | image
    source_file    : str = ""                   # اسم الملف المصدر
    page_number    : int = 0                    # رقم الصفحة
    token_count    : int = 0                    # عدد الرموز التقريبي
    metadata       : dict = field(default_factory=dict)

    def __repr__(self):
        preview = self.content[:60].replace("\n", " ")
        return (
            f"[{self.chunk_type}] {self.chunk_id} | "
            f"~{self.token_count} token | {preview}..."
        )


# ============================================================
# HybridChunker الرئيسي
# ============================================================

class BoundaryDetector:
    """
    يجمع المعلومات من:
        - TablePreserver  → الجداول كقطع مستقلة
        - HeadingDetector → ربط النص بعناوينه
        - حدود الرموز    → لا تتجاوز MAX_TOKENS

    ثم ينتج قائمة Chunk جاهزة للتضمين والتخزين.

    مثال الاستخدام:
        detector = BoundaryDetector()
        chunks = detector.chunk(parsed_document)
    """

    def __init__(self):
        self.max_tokens     = int(os.getenv("CHUNK_MAX_TOKENS", 512))
        self.min_tokens     = int(os.getenv("CHUNK_MIN_TOKENS", 64))
        self.overlap        = int(os.getenv("CHUNK_OVERLAP", 50))
        self.table_preserver = TablePreserver()
        self.heading_detector = HeadingDetector()

        logger.info(
            f"⚙️ BoundaryDetector | "
            f"max={self.max_tokens} | min={self.min_tokens} | overlap={self.overlap}"
        )

    # ============================================================
    # الدالة الرئيسية
    # ============================================================

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        """
        تجزئة المستند المحلل إلى قطع جاهزة للتضمين

        Args:
            parsed: المستند المحلل من DocumentParser

        Returns:
            list[Chunk]: القطع النهائية
        """
        logger.info(f"✂️ بدء التجزئة: {parsed.file_name}")

        chunks     : list[Chunk] = []
        chunk_counter            = 0

        # --- 1. اكتشاف العناوين وإثراء العناصر ---
        headings = self.heading_detector.detect(parsed.elements)
        enriched = self.heading_detector.enrich_elements(parsed.elements, headings)

        # --- 2. استخراج الجداول كقطع مستقلة ---
        table_chunks = self.table_preserver.extract(enriched)
        for tc in table_chunks:
            chunk_counter += 1
            chunks.append(self._table_to_chunk(tc, chunk_counter, parsed.file_name))

        # --- 3. تجزئة النصوص (بدون الجداول) ---
        text_elements = [
            e for e in enriched
            if e.element_type not in (ElementType.TABLE, ElementType.IMAGE)
        ]

        # تنظيف إضافي للنص العربي قبل التجزئة
        for elem in text_elements:
            if elem.element_type == ElementType.PARAGRAPH:
                elem.content = fix_disconnected_arabic(elem.content)

        text_chunks = self._chunk_text_elements(
            text_elements,
            parsed.file_name,
            start_counter=chunk_counter,
        )
        chunks += text_chunks
        chunk_counter += len(text_chunks)

        # --- 4. دمج القطع الصغيرة جداً (min_tokens) ---
        chunks = self._merge_small_chunks(chunks)

        logger.success(
            f"✅ {parsed.file_name} → {len(chunks)} قطعة | "
            f"نص: {len(text_chunks)} | جداول: {len(table_chunks)}"
        )
        return chunks


    # ============================================================
    # تجزئة النصوص
    # ============================================================

    def _chunk_text_elements(
        self,
        elements     : list[DocumentElement],
        source_file  : str,
        start_counter: int = 0,
    ) -> list[Chunk]:
        """
        تجزئة العناصر النصية مع احترام حدود الرموز والعناوين
        """
        chunks          : list[Chunk] = []
        current_text    : list[str]   = []
        current_tokens  : int         = 0
        current_heading : str         = ""
        current_page    : int         = 0
        counter         : int         = start_counter

        def flush():
            """تفريغ المخزن المؤقت كقطعة جديدة"""
            nonlocal current_text, current_tokens, counter

            if not current_text:
                return

            content = "\n\n".join(current_text)
            counter += 1

            # إضافة Overlap من القطعة السابقة
            overlap_text = ""
            if chunks and self.overlap > 0:
                prev_words = chunks[-1].content.split()
                overlap_text = " ".join(prev_words[-self.overlap:]) + "\n\n"

            # توليد chunk_id فريد باسم الملف
            file_prefix = source_file.replace(".pdf", "").replace(".docx", "").replace(" ", "_").lower()[:20] if source_file else "doc"
            chunk_id = f"{file_prefix}_chunk_{counter:04d}"

            # تحسين breadcrumb: تجميع أول 2-3 أسطر كعنوان تسلسلي
            breadcrumb_parts = []
            if current_heading:
                breadcrumb_parts.append(current_heading.replace("## ", ""))
            # إضافة أول 1-2 عنصر من النص كـ breadcrumb
            for i, txt in enumerate(current_text[:2]):
                clean_txt = txt.replace("## ", "").strip()
                if clean_txt and len(clean_txt) > 5:
                    breadcrumb_parts.append(clean_txt[:100])  # حد أقصى 100 حرف
            
            chunk = Chunk(
                chunk_id    = chunk_id,
                content     = overlap_text + content,
                chunk_type  = "text",
                source_file = source_file,
                page_number = current_page,
                token_count = self._estimate_tokens(overlap_text + content),
                metadata    = {
                    "parent_heading": current_heading,
                    "breadcrumb"    : " > ".join(breadcrumb_parts) if breadcrumb_parts else "",
                    "source_file"   : source_file,
                    "page_number"   : current_page,
                },
            )
            chunks.append(chunk)
            current_text.clear()
            current_tokens = 0

        for element in elements:
            text   = element.content.strip()
            tokens = self._estimate_tokens(text)

            # عنوان جديد → نفرّغ المخزن دائماً
            if element.element_type == ElementType.HEADING:
                flush()
                current_heading = text
                current_page    = element.page_number or current_page
                current_text.append(f"## {text}")
                current_tokens += tokens
                continue

            # تحديث الصفحة
            if element.page_number:
                current_page = element.page_number

            # النص سيتجاوز الحد → نفرّغ أولاً
            if current_tokens + tokens > self.max_tokens and current_text:
                flush()

            current_text.append(text)
            current_tokens += tokens

        flush()  # تفريغ ما تبقى
        return chunks


    # ============================================================
    # دمج القطع الصغيرة
    # ============================================================

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        دمج القطع التي تقل عن min_tokens مع جارتها
        لتجنب قطع عديمة الفائدة عند البحث الدلالي
        """
        if not chunks:
            return chunks

        merged : list[Chunk] = []
        i = 0

        while i < len(chunks):
            chunk = chunks[i]

            # إذا كانت القطعة صغيرة ويوجد قطعة تالية من نفس النوع
            if (
                chunk.token_count < self.min_tokens
                and chunk.chunk_type == "text"
                and i + 1 < len(chunks)
                and chunks[i + 1].chunk_type == "text"
            ):
                # دمج مع القطعة التالية
                next_chunk = chunks[i + 1]
                merged_content = chunk.content + "\n\n" + next_chunk.content

                merged_chunk = Chunk(
                    chunk_id    = chunk.chunk_id,
                    content     = merged_content,
                    chunk_type  = "text",
                    source_file = chunk.source_file,
                    page_number = chunk.page_number,
                    token_count = self._estimate_tokens(merged_content),
                    metadata    = {
                        **chunk.metadata,
                        "merged": True,
                    },
                )
                merged.append(merged_chunk)
                i += 2  # تخطي القطعة التالية
                logger.debug(f"🔗 دُمجت قطعة صغيرة: {chunk.chunk_id}")
            else:
                merged.append(chunk)
                i += 1

        logger.info(f"🔗 بعد الدمج: {len(chunks)} → {len(merged)} قطعة")
        return merged


    # ============================================================
    # دوال مساعدة
    # ============================================================

    def _table_to_chunk(
        self,
        tc          : TableChunk,
        counter     : int,
        source_file : str,
    ) -> Chunk:
        """تحويل TableChunk إلى Chunk موحد"""
        return Chunk(
            chunk_id    = f"table_{counter:04d}",
            content     = tc.context_text,
            chunk_type  = "table",
            source_file = source_file,
            page_number = tc.page_number,
            token_count = self._estimate_tokens(tc.context_text),
            metadata    = {
                **tc.metadata,
                "source_file": source_file,
            },
        )

    def _estimate_tokens(self, text: str) -> int:
        """
        تقدير عدد الرموز بطريقة سريعة
        القاعدة التقريبية: كل 4 أحرف = رمز واحد
        """
        return max(1, len(text) // 4)