# ============================================================
# data_processing/metadata/question_generator.py
# توليد أسئلة افتراضية لكل قطعة نصية (Hypothetical Questions)
# تُستخدم لتحسين جودة الاسترجاع في RAG
# ============================================================

import os
from dataclasses import dataclass, field

from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


# ============================================================
# نتيجة توليد الأسئلة
# ============================================================

@dataclass
class QuestionsResult:
    """الأسئلة الافتراضية المولّدة لقطعة نصية"""
    chunk_id  : str
    questions : list[str] = field(default_factory=list)

    @property
    def questions_string(self) -> str:
        return "\n".join(f"- {q}" for q in self.questions)

    def __repr__(self):
        return f"[أسئلة {self.chunk_id}] {len(self.questions)} سؤال"


# ============================================================
# مولّد الأسئلة
# ============================================================

class QuestionGenerator:
    """
    يولّد أسئلة افتراضية لكل قطعة نصية باستخدام Gemini

    لماذا الأسئلة الافتراضية؟
        عند البحث، المستخدم يسأل سؤالاً لكن المستند يحتوي إجابة.
        بتوليد أسئلة تُجيب عنها القطعة، نحسّن مطابقة الاستعلامات
        مع المستندات (HyDE - Hypothetical Document Embeddings).

    مثال الاستخدام:
        generator = QuestionGenerator()
        result = generator.generate("chunk_001", "النص هنا...")
        print(result.questions_string)
    """

    def __init__(self, num_questions: int = 3):
        """
        Args:
            num_questions: عدد الأسئلة المولّدة لكل قطعة
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY غير موجود في .env")

        self.client        = genai.Client(
            api_key      = api_key,
            http_options = types.HttpOptions(api_version="v1beta"),
        )
        model_name         = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.model_name    = model_name
        self.num_questions = num_questions
        logger.info(f"✅ QuestionGenerator جاهز | {num_questions} أسئلة لكل قطعة")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate(self, chunk_id: str, text: str) -> QuestionsResult:
        """
        توليد أسئلة افتراضية لقطعة نصية

        Args:
            chunk_id: معرف القطعة
            text    : النص المراد توليد أسئلة عنه

        Returns:
            QuestionsResult: الأسئلة المولّدة
        """
        if not text or len(text.split()) < 10:
            return QuestionsResult(chunk_id=chunk_id)

        truncated = text[:1500] if len(text) > 1500 else text

        prompt = f"""بناءً على النص التالي، اكتب {self.num_questions} أسئلة يمكن أن يسألها شخص ما وتكون إجابتها موجودة في هذا النص.

القواعد:
- كل سؤال في سطر منفصل
- لا تضع أرقاماً أو رموزاً قبل الأسئلة
- الأسئلة يجب أن تكون محددة وعملية
- بنفس لغة النص (عربي أو إنجليزي)
- بدون أي مقدمات أو تعليقات

النص:
{truncated}

الأسئلة:"""

        try:
            response = self.client.models.generate_content(
                model   = self.model_name,
                contents= prompt,
            )
            raw_text  = response.text.strip()

            # تنظيف الأسئلة
            questions = [
                line.strip().lstrip("-•*0123456789). ")
                for line in raw_text.split("\n")
                if line.strip() and "?" in line or "؟" in line
            ]

            # إذا ما طلعت أسئلة بعلامة استفهام — خذ الأسطر مباشرة
            if not questions:
                questions = [
                    line.strip().lstrip("-•*0123456789). ")
                    for line in raw_text.split("\n")
                    if line.strip() and len(line.strip()) > 10
                ]

            questions = questions[:self.num_questions]

            logger.debug(f"❓ {chunk_id}: {len(questions)} سؤال مولّد")

            return QuestionsResult(
                chunk_id  = chunk_id,
                questions = questions,
            )

        except Exception as e:
            logger.error(f"❌ فشل توليد أسئلة {chunk_id}: {e}")
            return QuestionsResult(chunk_id=chunk_id)

    def generate_batch(
        self,
        chunks     : list,
        text_field : str = "content",
        id_field   : str = "chunk_id",
    ) -> list[QuestionsResult]:
        """
        توليد أسئلة لمجموعة من القطع

        Args:
            chunks    : قائمة القطع
            text_field: اسم حقل النص
            id_field  : اسم حقل المعرف

        Returns:
            list[QuestionsResult]
        """
        results = []
        total   = len(chunks)
        logger.info(f"❓ توليد أسئلة لـ {total} قطعة...")

        for i, chunk in enumerate(chunks, 1):
            if isinstance(chunk, dict):
                chunk_id = chunk.get(id_field, f"chunk_{i}")
                text     = chunk.get(text_field, "")
            else:
                chunk_id = getattr(chunk, id_field, f"chunk_{i}")
                text     = getattr(chunk, text_field, "")

            result = self.generate(chunk_id, text)
            results.append(result)

            if i % 5 == 0:
                logger.info(f"  ⏳ {i}/{total} تم معالجتها...")

        logger.success(f"✅ اكتمل توليد الأسئلة لـ {len(results)} قطعة")
        return results