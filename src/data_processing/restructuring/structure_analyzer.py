# ============================================================
# data_processing/restructuring/structure_analyzer.py
# تحليل البنية الهيكلية للمستند المحلل
# يكتشف: التسلسل الهرمي، الجداول، الأقسام، وجودة البنية
# ============================================================

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.data_processing.restructuring.document_parser import (
    ParsedDocument,
    DocumentElement,
    ElementType,
)


# ============================================================
# نتائج تحليل البنية
# ============================================================

@dataclass
class SectionNode:
    """
    يمثل قسماً واحداً في المستند مع محتوياته
    """
    heading      : str
    level        : int
    elements     : list[DocumentElement] = field(default_factory=list)
    subsections  : list["SectionNode"]  = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """النص الكامل للقسم مجمعاً"""
        parts = [self.heading]
        parts += [e.content for e in self.elements]
        return "\n\n".join(parts)

    @property
    def has_tables(self) -> bool:
        return any(e.element_type == ElementType.TABLE for e in self.elements)

    @property
    def has_images(self) -> bool:
        return any(e.element_type == ElementType.IMAGE for e in self.elements)

    def __repr__(self):
        return (
            f"{'  ' * (self.level - 1)}[H{self.level}] {self.heading} "
            f"({len(self.elements)} عناصر, {len(self.subsections)} أقسام فرعية)"
        )


@dataclass
class StructureAnalysisResult:
    """
    نتيجة التحليل الهيكلي الكامل للمستند
    """
    file_name       : str
    sections        : list[SectionNode]       = field(default_factory=list)
    orphan_elements : list[DocumentElement]   = field(default_factory=list)  # عناصر بدون عنوان
    quality_score   : float                   = 0.0   # جودة البنية 0.0 → 1.0
    quality_notes   : list[str]               = field(default_factory=list)
    metadata        : dict                    = field(default_factory=dict)

    @property
    def num_sections(self) -> int:
        return len(self.sections)

    @property
    def all_tables(self) -> list[DocumentElement]:
        """كل الجداول في جميع الأقسام"""
        tables = []
        for section in self.sections:
            tables += [e for e in section.elements if e.element_type == ElementType.TABLE]
        return tables

    def print_tree(self):
        """طباعة شجرة الأقسام بشكل مرئي"""
        logger.info(f"\n📑 هيكل المستند: {self.file_name}")
        for section in self.sections:
            print(section)
            for sub in section.subsections:
                print(sub)

    def summary(self) -> str:
        return (
            f"📊 {self.file_name} | "
            f"أقسام: {self.num_sections} | "
            f"جداول: {len(self.all_tables)} | "
            f"عناصر يتيمة: {len(self.orphan_elements)} | "
            f"جودة البنية: {self.quality_score:.0%}"
        )


# ============================================================
# محلل البنية الرئيسي
# ============================================================

class StructureAnalyzer:
    """
    يحوّل قائمة DocumentElement المسطحة إلى شجرة هرمية من SectionNode
    ويحسب درجة جودة بنية المستند

    مثال الاستخدام:
        analyzer = StructureAnalyzer()
        result = analyzer.analyze(parsed_document)
        result.print_tree()
        print(result.summary())
    """

    def analyze(self, parsed: ParsedDocument) -> StructureAnalysisResult:
        """
        تحليل المستند المحلل وبناء شجرته الهرمية

        Args:
            parsed: المستند المحلل من DocumentParser

        Returns:
            StructureAnalysisResult: نتيجة التحليل الهيكلي الكاملة
        """
        logger.info(f"🏗️ تحليل هيكل: {parsed.file_name}")

        sections        : list[SectionNode]     = []
        orphan_elements : list[DocumentElement] = []
        current_section : Optional[SectionNode] = None

        for element in parsed.elements:

            # --- عنوان رئيسي (H1 أو H2) → قسم جديد ---
            if element.element_type == ElementType.HEADING:
                level = element.level or 1

                new_section = SectionNode(
                    heading = element.content,
                    level   = level,
                )

                if level == 1:
                    # قسم رئيسي جديد
                    sections.append(new_section)
                    current_section = new_section

                else:
                    # قسم فرعي — نضيفه للقسم الحالي
                    if current_section:
                        current_section.subsections.append(new_section)
                    else:
                        # لا يوجد قسم أب → نعامله كقسم رئيسي
                        sections.append(new_section)
                        current_section = new_section

            # --- عنصر محتوى → نضيفه للقسم الحالي ---
            else:
                if current_section:
                    current_section.elements.append(element)
                else:
                    # عنصر يتيم قبل أي عنوان
                    orphan_elements.append(element)

        # ============================================================
        # حساب جودة البنية
        # ============================================================
        quality_score, quality_notes = self._evaluate_quality(
            parsed, sections, orphan_elements
        )

        result = StructureAnalysisResult(
            file_name       = parsed.file_name,
            sections        = sections,
            orphan_elements = orphan_elements,
            quality_score   = quality_score,
            quality_notes   = quality_notes,
            metadata        = {
                "source"         : parsed.file_name,
                "total_sections" : len(sections),
                "total_orphans"  : len(orphan_elements),
            },
        )

        logger.success(f"✅ {result.summary()}")
        return result


    def _evaluate_quality(
        self,
        parsed          : ParsedDocument,
        sections        : list[SectionNode],
        orphan_elements : list[DocumentElement],
    ) -> tuple[float, list[str]]:
        """
        تقييم جودة بنية المستند بناءً على عدة معايير

        Returns:
            tuple: (درجة الجودة 0.0-1.0, قائمة الملاحظات)
        """
        score  = 1.0
        notes  = []
        total  = len(parsed.elements)

        if total == 0:
            return 0.0, ["⚠️ المستند فارغ"]

        # --- معيار 1: وجود عناوين ---
        if parsed.num_headings == 0:
            score -= 0.3
            notes.append("⚠️ لا توجد عناوين — صعوبة في التجزئة الهرمية")
        elif parsed.num_headings < 2:
            score -= 0.1
            notes.append("💡 عدد العناوين قليل — قد تكون التجزئة محدودة")

        # --- معيار 2: نسبة العناصر اليتيمة ---
        orphan_ratio = len(orphan_elements) / total
        if orphan_ratio > 0.5:
            score -= 0.3
            notes.append(f"⚠️ {orphan_ratio:.0%} من العناصر بدون عنوان أب")
        elif orphan_ratio > 0.2:
            score -= 0.1
            notes.append(f"💡 {orphan_ratio:.0%} من العناصر بدون عنوان أب")

        # --- معيار 3: وجود فقرات ---
        if parsed.num_paragraphs == 0:
            score -= 0.2
            notes.append("⚠️ لا توجد فقرات نصية — قد يكون المستند صورة فقط")

        # --- معيار 4: الجداول بدون عناوين ---
        tables_without_heading = [
            e for e in parsed.elements
            if e.element_type == ElementType.TABLE
            and not e.parent_heading
        ]
        if tables_without_heading:
            score -= 0.1
            notes.append(f"💡 {len(tables_without_heading)} جدول بدون عنوان مرتبط")

        # --- معيار 5: بنية عميقة (مؤشر إيجابي) ---
        has_subsections = any(len(s.subsections) > 0 for s in sections)
        if has_subsections:
            notes.append("✅ يحتوي على أقسام فرعية — بنية هرمية جيدة")

        score = max(0.0, min(1.0, score))

        if score >= 0.8:
            notes.insert(0, "✅ جودة البنية ممتازة")
        elif score >= 0.5:
            notes.insert(0, "🟡 جودة البنية متوسطة")
        else:
            notes.insert(0, "🔴 جودة البنية ضعيفة — قد يؤثر على دقة الاسترجاع")

        return score, notes


    def get_section_by_heading(
        self,
        result  : StructureAnalysisResult,
        keyword : str,
    ) -> Optional[SectionNode]:
        """
        البحث عن قسم بالكلمة المفتاحية في عنوانه

        Args:
            result : نتيجة التحليل
            keyword: الكلمة المفتاحية للبحث

        Returns:
            SectionNode أو None
        """
        keyword_lower = keyword.lower()
        for section in result.sections:
            if keyword_lower in section.heading.lower():
                return section
            for sub in section.subsections:
                if keyword_lower in sub.heading.lower():
                    return sub
        return None


    def flatten_sections(
        self,
        result: StructureAnalysisResult,
    ) -> list[dict]:
        """
        تسطيح شجرة الأقسام إلى قائمة مسطحة
        مفيد لتمريرها لمرحلة التجزئة (Chunking)

        Returns:
            list[dict]: قائمة بالأقسام مع نصها الكامل وبياناتها الوصفية
        """
        flat = []

        for section in result.sections:
            flat.append({
                "heading"  : section.heading,
                "level"    : section.level,
                "text"     : section.full_text,
                "has_table": section.has_tables,
                "has_image": section.has_images,
            })
            for sub in section.subsections:
                flat.append({
                    "heading"       : sub.heading,
                    "level"         : sub.level,
                    "text"          : sub.full_text,
                    "parent_heading": section.heading,
                    "has_table"     : sub.has_tables,
                    "has_image"     : sub.has_images,
                })

        return flat