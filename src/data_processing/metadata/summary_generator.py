# ============================================================
# data_processing/metadata/summary_generator.py
# توليد ملخصات ذكية لكل قطعة نصية باستخدام Google Gemini
# ============================================================

import os
from dataclasses import dataclass

from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


# ============================================================
# نتيجة الملخص
# ============================================================

@dataclass
class SummaryResult:
    """نتيجة تلخيص قطعة نصية واحدة"""
    chunk_id      : str
    original_text : str
    summary       : str
    language      : str = "ar"   # اللغة المكتشفة

    def __repr__(self):
        return f"[ملخص {self.chunk_id}] {self.summary[:80]}..."


# ============================================================
# مولّد الملخصات
# ============================================================

class SummaryGenerator:
    """
    يولّد ملخصاً قصيراً لكل قطعة نصية باستخدام Gemini

    الهدف:
        - تحسين جودة التضمين (Embedding) بإضافة الملخص للبيانات الوصفية
        - تسريع البحث بتوفير نص أقصر وأكثر دلالة
        - دعم اللغتين العربية والإنجليزية تلقائياً

    مثال الاستخدام:
        generator = SummaryGenerator()
        result = generator.summarize("chunk_001", "النص الطويل هنا...")
        print(result.summary)
    """

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY غير موجود في .env")

        self.client    = genai.Client(
            api_key      = api_key,
            http_options = types.HttpOptions(api_version="v1beta"),
        )
        model_name     = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
        self.model_name = model_name
        logger.info(f"✅ SummaryGenerator جاهز | نموذج: {model_name}")


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def summarize(self, chunk_id: str, text: str) -> SummaryResult:
        """
        توليد ملخص لقطعة نصية واحدة

        Args:
            chunk_id: معرف القطعة
            text    : النص المراد تلخيصه

        Returns:
            SummaryResult: الملخص مع البيانات الوصفية
        """
        if not text.strip():
            return SummaryResult(
                chunk_id      = chunk_id,
                original_text = text,
                summary       = "",
            )

        # قطع النص إذا كان طويلاً جداً لتوفير الرموز
        truncated = text[:2000] if len(text) > 2000 else text

        prompt = f"""لخّص النص التالي في جملة واحدة أو جملتين فقط.
يجب أن يكون الملخص:
- موجزاً ودقيقاً
- يحافظ على المصطلحات التقنية
- بنفس لغة النص (عربي أو إنجليزي)
- بدون مقدمات أو تعليقات إضافية

النص:
{truncated}

الملخص:"""

        try:
            response = self.client.models.generate_content(
                model   = self.model_name,
                contents= prompt,
            )
            summary  = response.text.strip()

            logger.debug(f"📝 ملخص {chunk_id}: {summary[:60]}...")

            return SummaryResult(
                chunk_id      = chunk_id,
                original_text = text,
                summary       = summary,
            )

        except Exception as e:
            logger.error(f"❌ فشل تلخيص {chunk_id}: {e}")
            # في حالة الفشل → نعيد أول جملة من النص
            first_sentence = text.split(".")[0][:150]
            return SummaryResult(
                chunk_id      = chunk_id,
                original_text = text,
                summary       = first_sentence,
            )


    def summarize_batch(
        self,
        chunks     : list,
        text_field : str = "content",
        id_field   : str = "chunk_id",
    ) -> list[SummaryResult]:
        """
        تلخيص مجموعة من القطع دفعة واحدة

        Args:
            chunks    : قائمة القطع (Chunk objects أو dicts)
            text_field: اسم حقل النص
            id_field  : اسم حقل المعرف

        Returns:
            list[SummaryResult]: قائمة الملخصات
        """
        results = []
        total   = len(chunks)

        logger.info(f"📝 بدء تلخيص {total} قطعة...")

        for i, chunk in enumerate(chunks, 1):
            # دعم Chunk objects و dicts
            if isinstance(chunk, dict):
                chunk_id = chunk.get(id_field, f"chunk_{i}")
                text     = chunk.get(text_field, "")
            else:
                chunk_id = getattr(chunk, id_field, f"chunk_{i}")
                text     = getattr(chunk, text_field, "")

            result = self.summarize(chunk_id, text)
            results.append(result)

            if i % 5 == 0:
                logger.info(f"  ⏳ {i}/{total} تم تلخيصها...")

        logger.success(f"✅ اكتمل تلخيص {len(results)} قطعة")
        return results

    def summarize_document(self, file_name: str, full_text: str) -> str:
        """
        توليد ملخص شامل للمستند بالكامل
        """
        logger.info(f"📄 توليد ملخص شامل للمستند: {file_name}")
        
        # نأخذ أول 10 آلاف رمز تقريباً لعمل مسحة سريعة للهيكل
        sample = full_text[:10000]
        
        prompt = f"""قم بتحليل هذا المستند المسمى ({file_name}) وقدم ملخصاً شاملاً له في فقرة واحدة (30-50 كلمة).
ركز على: نوع المستند، الأطراف المعنية (شركات/أفراد)، والهدف الرئيسي منه.

المستند:
{sample}

الملخص الشامل:"""

        try:
            response = self.client.models.generate_content(
                model   = self.model_name,
                contents= prompt,
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"❌ فشل تلخيص المستند {file_name}: {e}")
            return f"مستند باسم {file_name}"