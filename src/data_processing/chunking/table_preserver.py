# ============================================================
# data_processing/chunking/table_preserver.py
# يحافظ على الجداول كاملة عند التجزئة — لا تُقطع الجداول أبداً
# ============================================================

import re
from dataclasses import dataclass, field

from loguru import logger


# ============================================================
# قطعة الجدول المحفوظة
# ============================================================

@dataclass
class TableChunk:
    """
    جدول مستخرج كقطعة مستقلة مع سياقه
    """
    content        : str              # محتوى الجدول كـ Markdown
    parent_heading : str  = ""        # العنوان الأب للجدول
    page_number    : int  = 0         # رقم الصفحة
    table_index    : int  = 0         # رقم الجدول في المستند
    num_rows       : int  = 0         # عدد الصفوف
    num_cols       : int  = 0         # عدد الأعمدة
    metadata       : dict = field(default_factory=dict)

    @property
    def context_text(self) -> str:
        """النص الكامل مع السياق للتضمين (Embedding)"""
        parts = []
        if self.parent_heading:
            parts.append(f"القسم: {self.parent_heading}")
        parts.append(self.content)
        return "\n\n".join(parts)

    def __repr__(self):
        return (
            f"[جدول {self.table_index}] "
            f"{self.num_rows} صف × {self.num_cols} عمود | "
            f"القسم: {self.parent_heading or 'بدون عنوان'}"
        )


# ============================================================
# محافظ الجداول
# ============================================================

class TablePreserver:
    """
    يستخرج الجداول من المستند ويحفظها كقطع مستقلة
    دون أي تقطيع — الجدول يبقى كاملاً دائماً

    القاعدة الذهبية:
        جدول = قطعة واحدة كاملة بغض النظر عن حجمه

    مثال الاستخدام:
        preserver = TablePreserver()
        table_chunks = preserver.extract(parsed_document)
    """

    def extract(self, elements: list) -> list[TableChunk]:
        """
        استخراج جميع الجداول من قائمة العناصر

        Args:
            elements: قائمة DocumentElement من DocumentParser

        Returns:
            list[TableChunk]: قائمة الجداول كقطع مستقلة
        """
        from src.data_processing.restructuring.document_parser import ElementType

        chunks      : list[TableChunk] = []
        table_index : int = 0

        for element in elements:
            if element.element_type != ElementType.TABLE:
                continue

            table_index += 1

            # حساب أبعاد الجدول من Markdown
            num_rows, num_cols = self._get_table_dimensions(element.content)

            chunk = TableChunk(
                content        = element.content,
                parent_heading = element.parent_heading or "",
                page_number    = element.page_number or 0,
                table_index    = table_index,
                num_rows       = num_rows,
                num_cols       = num_cols,
                metadata       = {
                    "type"           : "table",
                    "table_index"    : table_index,
                    "parent_heading" : element.parent_heading or "",
                    "page_number"    : element.page_number or 0,
                    **element.metadata,
                },
            )

            chunks.append(chunk)
            logger.debug(f"📊 جدول {table_index}: {num_rows}×{num_cols} في '{element.parent_heading}'")

        logger.info(f"✅ تم استخراج {len(chunks)} جدول")
        return chunks


    def _get_table_dimensions(self, markdown_table: str) -> tuple[int, int]:
        """
        حساب عدد الصفوف والأعمدة من Markdown

        Args:
            markdown_table: نص الجدول بصيغة Markdown

        Returns:
            tuple: (عدد الصفوف, عدد الأعمدة)
        """
        lines = [
            line.strip()
            for line in markdown_table.strip().split("\n")
            if line.strip().startswith("|")
        ]

        if not lines:
            return 0, 0

        # استبعاد سطر الفاصل (---|---)
        data_lines = [
            line for line in lines
            if not re.match(r"^\|[\s\-|:]+\|$", line)
        ]

        num_rows = max(0, len(data_lines) - 1)  # استبعاد سطر الرأس
        num_cols = lines[0].count("|") - 1 if lines else 0

        return num_rows, num_cols


    def validate_table_integrity(self, chunk: TableChunk) -> bool:
        """
        التحقق من أن الجدول مكتمل وغير مقطوع

        Args:
            chunk: قطعة الجدول للتحقق منها

        Returns:
            bool: True إذا كان الجدول مكتملاً
        """
        lines = chunk.content.strip().split("\n")
        table_lines = [l for l in lines if "|" in l]

        if len(table_lines) < 2:
            logger.warning(f"⚠️ جدول {chunk.table_index} قد يكون مقطوعاً")
            return False

        # التحقق من وجود سطر الفاصل
        has_separator = any(
            re.match(r"^\|[\s\-|:]+\|$", line.strip())
            for line in table_lines
        )

        if not has_separator:
            logger.warning(f"⚠️ جدول {chunk.table_index} بدون سطر فاصل")
            return False

        return True