# ============================================================
# data_processing/restructuring/document_parser.py
# تحليل المستند المحوّل من Docling واستخراج عناصره المهيكلة
# ============================================================

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger
from src.data_processing.metadata.keyword_extractor import normalize_arabic
from docling.datamodel.document import ConversionResult
from docling_core.types.doc import DoclingDocument


# ============================================================
# إصلاح النص العربي - معالجة مشكلة الأحرف المفككة
# ============================================================

def fix_disconnected_arabic(text: str) -> str:
    """
    إصلاح النص العربي المفكك من PDF
    
    المشكلة: Docling يستخرج الأحرف العربية بشكل منفصل
    مثال: 'ا ب ر ا ه ي م' → 'إبراهيم'
    
    الحل:
    1. NFKC Normalization
    2. إزالة المسافات الزائدة بين الأحرف العربية
    3. إعادة تجميع الكلمات العربية
    4. فصل الكلمات المتصلة بشكل خاطئ
    """
    if not text:
        return text
    
    # 1. NFKC Normalization - تحويل Arabic Presentation Forms
    text = unicodedata.normalize("NFKC", text)
    
    # 2. حذف التشكيل
    text = re.sub(r'[\u064B-\u065F\u0610-\u061A\u06D6-\u06DC\u0670]', '', text)
    
    # 3. RTL/LTR markers حذف
    text = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF]', '', text)
    
    # 4. توحيد الألف
    text = re.sub(r'[أإآ]', 'ا', text)
    
    # 5. توحيد التاء المربوطة والألف المقصورة
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    
    # 6. توحيد الأرقام
    def to_int_digit(m):
        """تحويل الأرقام العربية/الهندية إلى أرقام عربية قياسية"""
        cp = ord(m.group())
        if 0x0660 <= cp <= 0x0669:  # Arabic-Indic
            return str(cp - 0x0660)
        elif 0x06F0 <= cp <= 0x06F9:  # Extended Arabic-Indic
            return str(cp - 0x06F0)
        return m.group()
    
    text = re.sub(r'[٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹]', to_int_digit, text)
    
    # 7. إصلاح المسافات بين الأحرف العربية
    # نمط: حرف عربي + مسافات + حرف عربي → حرف عربي بدون مسافة
    arabic_pattern = re.compile(r'([\u0600-\u06FF])\s+([\u0600-\u06FF])')
    for _ in range(5):  # تكرار لضمان تنظيف جميع المسافات الزائدة
        text = arabic_pattern.sub(r'\1\2', text)
    
    # 8. تنظيف المسافات الزائدة العامة
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 9. فصل الكلمات المتصلة بشكل خاطئ (لـ Gemini Vision OCR)
    # إضافة مسافات بعد علامات الترقيم والكلمات الشائعة
    text = re.sub(r'([:؛،])', r'\1 ', text)
    text = re.sub(r'(\d)([\u0600-\u06FF])', r'\1 \2', text)  # رقم + حرف عربي
    text = re.sub(r'([\u0600-\u06FF])(\d)', r'\1 \2', text)  # حرف عربي + رقم
    text = re.sub(r'(/)([\u0600-\u06FF])', r'\1 \2', text)  # / + حرف عربي
    text = re.sub(r'([\u0600-\u06FF])(/)', r'\1 \2', text)  # حرف عربي + /
    
    # 10. إضافة مسافات قبل الكلمات المفتاحية الشائعة
    common_words = ['رقم', 'تاريخ', 'نوع', 'اسم', 'المالك', 'الحي', 'شارع', 'مساحة', 'قطعة', 'صك', 'مخطط', 'رخصة', 'بناء', 'سكني', 'ارضي', 'اول', 'ملاحق', 'اسوار', 'ملاحظات', 'القرار', 'المكتب', 'الهندسي', 'يلزم', 'يتحمل', 'تطبيق', 'الموقع', 'الجهد', 'وضع', 'عدم', 'تنفيذ', 'شركة', 'الكهرباء', 'المياه', 'الصرف', 'الصحي', 'العزل', 'الحراري', 'كود', 'البناء', 'السعودي', 'ريال', 'ايصال', 'هـ']
    for word in common_words:
        # إضافة مسافة قبل الكلمة إذا لم تكن في بداية السطر
        text = re.sub(r'(?<!^)(?<!\s)(' + word + r')', r' \1', text)
    
    # 11. تنظيف نهائي
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = text.strip()
    
    return text


# ============================================================
# أنواع العناصر المستخرجة
# ============================================================

class ElementType(str, Enum):
    HEADING    = "heading"     # عنوان رئيسي أو فرعي
    PARAGRAPH  = "paragraph"   # فقرة نصية
    TABLE      = "table"       # جدول
    IMAGE      = "image"       # صورة
    CODE       = "code"        # كتلة كود
    LIST       = "list"        # قائمة
    EQUATION   = "equation"    # معادلة رياضية
    CAPTION    = "caption"     # تسمية توضيحية


# ============================================================
# عنصر المستند الموحد
# ============================================================

@dataclass
class DocumentElement:
    """
    وحدة أساسية تمثل أي عنصر مستخرج من المستند
    """
    element_type : ElementType
    content      : str                        # النص أو Markdown للجدول
    page_number  : Optional[int]   = None     # رقم الصفحة
    level        : Optional[int]   = None     # مستوى العنوان (1, 2, 3...)
    parent_heading: Optional[str]  = None     # العنوان الأب لهذا العنصر
    metadata     : dict            = field(default_factory=dict)

    def __repr__(self):
        preview = self.content[:60].replace("\n", " ")
        return f"[{self.element_type.value}] (p.{self.page_number}) {preview}..."


# ============================================================
# نتيجة التحليل الكاملة
# ============================================================

@dataclass
class ParsedDocument:
    """
    نتيجة تحليل مستند كامل — تحتوي على جميع عناصره المهيكلة
    """
    file_name   : str
    elements    : list[DocumentElement] = field(default_factory=list)
    metadata    : dict                  = field(default_factory=dict)

    # إحصائيات سريعة
    @property
    def num_headings(self)   -> int:
        return sum(1 for e in self.elements if e.element_type == ElementType.HEADING)

    @property
    def num_paragraphs(self) -> int:
        return sum(1 for e in self.elements if e.element_type == ElementType.PARAGRAPH)

    @property
    def num_tables(self)     -> int:
        return sum(1 for e in self.elements if e.element_type == ElementType.TABLE)

    @property
    def num_images(self)     -> int:
        return sum(1 for e in self.elements if e.element_type == ElementType.IMAGE)

    def summary(self) -> str:
        return (
            f"📄 {self.file_name} | "
            f"عناوين: {self.num_headings} | "
            f"فقرات: {self.num_paragraphs} | "
            f"جداول: {self.num_tables} | "
            f"صور: {self.num_images}"
        )


# ============================================================
# محلل المستند الرئيسي
# ============================================================

class DocumentParser:
    """
    يحوّل نتيجة Docling (ConversionResult أو DoclingDocument)
    إلى قائمة مهيكلة من DocumentElement

    مثال الاستخدام:
        parser = DocumentParser()
        parsed = parser.parse(conversion_result, file_name="report.pdf")
        print(parsed.summary())
    """

    def parse(
        self,
        source: ConversionResult | DoclingDocument | str,
        file_name: str = "unknown"
    ) -> ParsedDocument:
        """
        تحليل مستند Docling واستخراج عناصره

        Args:
            source   : نتيجة التحويل من Docling أو DoclingDocument مباشرة أو نص خام
            file_name: اسم الملف للبيانات الوصفية

        Returns:
            ParsedDocument: المستند المحلل بجميع عناصره
        """
        logger.info(f"🔍 بدء تحليل: {file_name}")

        # حالة خاصة: نص خام من Gemini Vision OCR
        if isinstance(source, str):
            return self._parse_raw_text(source, file_name)
        
        # استخراج DoclingDocument من المصدر
        if isinstance(source, ConversionResult):
            doc = source.document
        else:
            doc = source

        elements : list[DocumentElement] = []
        current_heading : str = ""   # نتتبع العنوان الحالي لربط الفقرات به

        # ============================================================
        # المرور على جميع عناصر المستند بالترتيب
        # ============================================================
        for item, level in doc.iterate_items():

            item_type = type(item).__name__

            # --- العناوين ---
            if item_type == "SectionHeaderItem":
                heading_level = getattr(item, "level", 1)
                text = normalize_arabic(item.text.strip())
                # إصلاح إضافي للنص العربي المفكك
                text = fix_disconnected_arabic(text)
                current_heading = text

                elements.append(DocumentElement(
                    element_type   = ElementType.HEADING,
                    content        = text,
                    level          = heading_level,
                    parent_heading = None,
                    metadata       = {"level": heading_level},
                ))

            # --- الفقرات النصية ---
            elif item_type == "TextItem":
                text = normalize_arabic(item.text.strip())
                # إصلاح إضافي للنص العربي المفكك
                text = fix_disconnected_arabic(text)
                if not text:
                    continue

                elements.append(DocumentElement(
                    element_type   = ElementType.PARAGRAPH,
                    content        = text,
                    parent_heading = current_heading,
                    metadata       = {},
                ))

            # --- الجداول ---
            elif item_type == "TableItem":
                try:
                    # تصدير الجدول كـ Markdown للحفاظ على بنيته
                    table_md = item.export_to_markdown()
                    # إصلاح النص العربي في الجداول
                    table_md = fix_disconnected_arabic(table_md)
                except Exception:
                    table_md = str(item)

                elements.append(DocumentElement(
                    element_type   = ElementType.TABLE,
                    content        = table_md,
                    parent_heading = current_heading,
                    metadata       = {
                        "num_rows": getattr(item, "num_rows", None),
                        "num_cols": getattr(item, "num_cols", None),
                    },
                ))

            # --- الصور ---
            elif item_type == "PictureItem":
                caption = ""
                if hasattr(item, "captions") and item.captions:
                    caption = item.captions[0].text if item.captions else ""

                elements.append(DocumentElement(
                    element_type   = ElementType.IMAGE,
                    content        = caption or "[صورة بدون تسمية]",
                    parent_heading = current_heading,
                    metadata       = {
                        "has_caption": bool(caption),
                    },
                ))

            # --- المعادلات ---
            elif item_type == "EquationItem":
                elements.append(DocumentElement(
                    element_type   = ElementType.EQUATION,
                    content        = getattr(item, "text", "[معادلة]"),
                    parent_heading = current_heading,
                ))

            # --- الكود ---
            elif item_type == "CodeItem":
                elements.append(DocumentElement(
                    element_type   = ElementType.CODE,
                    content        = getattr(item, "text", ""),
                    parent_heading = current_heading,
                ))

            # --- القوائم ---
            elif item_type == "ListItem":
                elements.append(DocumentElement(
                    element_type   = ElementType.LIST,
                    content        = getattr(item, "text", ""),
                    parent_heading = current_heading,
                ))

        # ============================================================
        # بناء النتيجة
        # ============================================================
        parsed = ParsedDocument(
            file_name = file_name,
            elements  = elements,
            metadata  = {
                "source"        : file_name,
                "total_elements": len(elements),
            },
        )

        logger.success(f"✅ {parsed.summary()}")
        return parsed


    def _parse_raw_text(self, raw_text: str, file_name: str) -> ParsedDocument:
        """
        تحليل نص خام من Gemini Vision OCR
        
        يقسم النص إلى فقرات بناءً على الأسطر الفارغة
        """
        logger.info(f"📝 تحليل نص خام: {file_name}")
        
        elements: list[DocumentElement] = []
        current_heading: str = ""
        
        lines = raw_text.split("\n")
        current_paragraph: list[str] = []
        page_num = 1
        
        def flush_paragraph():
            nonlocal current_paragraph, current_heading
            if current_paragraph:
                text = "\n".join(current_paragraph).strip()
                if text and len(text) > 10:
                    # تحقق إذا كان عنوان
                    if text.startswith("---") or (len(text.split()) <= 5 and not any(c in text for c in ".,:؛،")):
                        # عنوان
                        elements.append(DocumentElement(
                            element_type=ElementType.HEADING,
                            content=text.replace("---", "").strip(),
                            level=1,
                            parent_heading=None,
                            metadata={"page": page_num},
                        ))
                        current_heading = text
                    else:
                        # فقرة عادية
                        elements.append(DocumentElement(
                            element_type=ElementType.PARAGRAPH,
                            content=fix_disconnected_arabic(text),
                            parent_heading=current_heading,
                            metadata={"page": page_num},
                        ))
                current_paragraph.clear()
        
        for line in lines:
            line_stripped = line.strip()
            
            # تتبع رقم الصفحة
            if line_stripped.startswith("--- صفحة"):
                flush_paragraph()
                try:
                    page_num = int(line_stripped.replace("--- صفحة", "").replace("---", "").strip())
                except:
                    pass
                continue
            
            # سطر فارغ → نهاية الفقرة
            if not line_stripped:
                flush_paragraph()
                continue
            
            current_paragraph.append(line_stripped)
        
        # تفريغ آخر فقرة
        flush_paragraph()
        
        # بناء النتيجة
        parsed = ParsedDocument(
            file_name=file_name,
            elements=elements,
            metadata={
                "source": file_name,
                "total_elements": len(elements),
                "ocr_method": "gemini_vision",
            },
        )
        
        logger.success(f"✅ {parsed.summary()}")
        return parsed


    def parse_to_markdown(
        self,
        source: ConversionResult | DoclingDocument,
    ) -> str:
        """
        تصدير المستند كاملاً كـ Markdown نظيف

        مفيد للمعالجة السريعة دون الحاجة لهيكلة كاملة

        Returns:
            str: نص Markdown للمستند بالكامل
        """
        if isinstance(source, ConversionResult):
            doc = source.document
        else:
            doc = source

        return doc.export_to_markdown()


    def get_tables_only(self, parsed: ParsedDocument) -> list[DocumentElement]:
        """استخراج الجداول فقط من المستند المحلل"""
        return [e for e in parsed.elements if e.element_type == ElementType.TABLE]


    def get_headings_tree(self, parsed: ParsedDocument) -> list[dict]:
        """
        بناء شجرة العناوين الهرمية للمستند
        مفيد لفهم هيكل المستند قبل التجزئة
        """
        tree = []
        for e in parsed.elements:
            if e.element_type == ElementType.HEADING:
                tree.append({
                    "level"  : e.level,
                    "heading": e.content,
                })
        return tree