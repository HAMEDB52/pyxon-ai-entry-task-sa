# ============================================================
# data_processing/chunking/heading_detector.py
# يكتشف العناوين ويربط كل قطعة نصية بعنوانها الأب
# ============================================================

import re
from dataclasses import dataclass, field

from loguru import logger


# ============================================================
# معلومات العنوان
# ============================================================

@dataclass
class HeadingInfo:
    """معلومات عنوان واحد في المستند"""
    text         : str
    level        : int          # 1 = H1, 2 = H2 ...
    element_index: int          # موضعه في قائمة العناصر
    breadcrumb   : str = ""     # المسار الكامل: "الفصل 1 > القسم 2 > البند 3"

    def __repr__(self):
        return f"{'#' * self.level} {self.text} (idx={self.element_index})"


# ============================================================
# كاشف العناوين
# ============================================================

class HeadingDetector:
    """
    يكتشف ويصنّف العناوين في قائمة العناصر
    ويبني مسار التنقل (Breadcrumb) لكل عنصر

    مثال الاستخدام:
        detector = HeadingDetector()
        headings = detector.detect(parsed.elements)
        breadcrumb = detector.get_breadcrumb(element_index, headings)
    """

    def detect(self, elements: list) -> list[HeadingInfo]:
        """
        استخراج جميع العناوين من قائمة العناصر مع بناء Breadcrumb

        Args:
            elements: قائمة DocumentElement

        Returns:
            list[HeadingInfo]: قائمة العناوين المكتشفة
        """
        from src.data_processing.restructuring.document_parser import ElementType

        headings : list[HeadingInfo] = []

        # تتبع العناوين النشطة لكل مستوى
        active_headings : dict[int, str] = {}

        for idx, element in enumerate(elements):
            if element.element_type != ElementType.HEADING:
                continue

            level = element.level or self._guess_level(element.content)

            # تحديث العناوين النشطة وحذف المستويات الأعمق
            active_headings[level] = element.content
            keys_to_remove = [k for k in active_headings if k > level]
            for k in keys_to_remove:
                del active_headings[k]

            # بناء Breadcrumb
            breadcrumb = " > ".join(
                active_headings[lvl]
                for lvl in sorted(active_headings.keys())
            )

            heading = HeadingInfo(
                text          = element.content,
                level         = level,
                element_index = idx,
                breadcrumb    = breadcrumb,
            )

            headings.append(heading)
            logger.debug(f"🏷️ عنوان: {heading}")

        logger.info(f"✅ اكتشف {len(headings)} عنوان")
        return headings


    def get_breadcrumb(
        self,
        element_index : int,
        headings      : list[HeadingInfo],
    ) -> str:
        """
        الحصول على مسار التنقل (Breadcrumb) لعنصر معين

        Args:
            element_index: موضع العنصر في قائمة العناصر
            headings     : قائمة العناوين المكتشفة

        Returns:
            str: مسار التنقل مثل "الفصل 1 > القسم 2"
        """
        # أقرب عنوان قبل هذا العنصر
        relevant = [h for h in headings if h.element_index <= element_index]

        if not relevant:
            return ""

        return relevant[-1].breadcrumb


    def get_parent_heading(
        self,
        element_index : int,
        headings      : list[HeadingInfo],
        level         : int = 1,
    ) -> str:
        """
        الحصول على العنوان الأب المباشر لعنصر معين

        Args:
            element_index: موضع العنصر
            headings     : قائمة العناوين
            level        : مستوى العنوان المطلوب (1 = H1)

        Returns:
            str: نص العنوان الأب
        """
        relevant = [
            h for h in headings
            if h.element_index <= element_index and h.level == level
        ]

        return relevant[-1].text if relevant else ""


    def enrich_elements(self, elements: list, headings: list[HeadingInfo]) -> list:
        """
        إثراء العناصر بمعلومات العناوين (Breadcrumb + parent_heading)

        Args:
            elements: قائمة العناصر الأصلية
            headings: قائمة العناوين المكتشفة

        Returns:
            list: العناصر المُثراة بالبيانات الوصفية
        """
        for idx, element in enumerate(elements):
            breadcrumb = self.get_breadcrumb(idx, headings)

            if not element.metadata:
                element.metadata = {}

            element.metadata["breadcrumb"]     = breadcrumb
            element.metadata["parent_heading"] = self.get_parent_heading(idx, headings)

        logger.info(f"✅ تم إثراء {len(elements)} عنصر بمعلومات العناوين")
        return elements


    def _guess_level(self, text: str) -> int:
        """
        تخمين مستوى العنوان إذا لم يكن محدداً
        بناءً على علامات Markdown أو طول النص
        """
        # Markdown headings
        match = re.match(r"^(#{1,6})\s", text)
        if match:
            return len(match.group(1))

        # نص قصير جداً → عنوان رئيسي
        if len(text) < 30:
            return 1

        return 2