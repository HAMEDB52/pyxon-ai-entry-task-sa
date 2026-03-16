# ============================================================
# data_processing/metadata/keyword_extractor.py
# استخراج الكلمات المفتاحية — ثنائي اللغة (عربي + إنجليزي)
#
# المشكلة القديمة:
#   KeyBERT + all-MiniLM-L6-v2 = نموذج إنجليزي فقط
#   → يُعطي كلمات مختلطة للعربية مثل "القسم أن إذا"
#
# الحل الجديد:
#   Layer 1: normalize_arabic() — يُحوّل Presentation Forms للنص العادي
#   Layer 2: Gemini LLM — استخراج ذكي بالعربية والإنجليزية
#   Layer 3: Regex fallback — بدون LLM
# ============================================================

import os
import re
import unicodedata
from collections import Counter
try:
    from src.data_processing.arabic_enhancer import normalize_arabic_text as _normalize
except ImportError:
    def _normalize(t): return unicodedata.normalize("NFKC", t) if t else t
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class KeywordsResult:
    chunk_id : str
    keywords : list[str] = field(default_factory=list)
    scores   : list[float] = field(default_factory=list)

    @property
    def top_keywords(self) -> list[str]:
        return self.keywords[:5]

    @property
    def keywords_string(self) -> str:
        return ", ".join(self.keywords)

    def __repr__(self):
        return f"[كلمات {self.chunk_id}] {self.keywords_string}"


# ════════════════════════════════════
# تطبيع النص العربي
# ════════════════════════════════════

def normalize_arabic(text: str) -> str:
    """
    تحويل Arabic Presentation Forms (FExx/FBxx) إلى Unicode المعياري.
    Docling + RapidOCR يُخرج النص بصيغة الحروف المرئية لا المعيارية.

    مثال: 'ﻣﺴﺘﻨﺪات' → 'مستندات'
    """
    if not text:
        return text
    # NFKC يُحوّل Presentation Forms تلقائياً
    text = unicodedata.normalize("NFKC", text)
    # حذف التشكيل
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # تطبيع الألف والهمزة والتاء المربوطة
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    # إزالة مسافات زائدة
    text = re.sub(r' {2,}', ' ', text).strip()
    return text


# ════════════════════════════════════
# المستخرج الرئيسي
# ════════════════════════════════════

class KeywordExtractor:
    """
    مستخرج كلمات مفتاحية ثنائي اللغة.
    يستخدم Gemini LLM مع fallback regex.
    """

    STOPWORDS_AR = {
        "في", "من", "إلى", "على", "عن", "مع", "هذا", "هذه", "ذلك",
        "التي", "الذي", "كان", "كانت", "يكون", "أن", "أو", "إذا",
        "لا", "ما", "لم", "قد", "وقد", "كما", "حيث", "بعد", "قبل",
        "إلا", "أي", "بين", "حتى", "عند", "منذ", "خلال", "بموجب",
        "القسم", "البند", "المادة", "الملحق", "وأن", "وأو", "وإذا",
    }
    STOPWORDS_EN = {
        "the", "and", "for", "with", "this", "that", "from", "are",
        "was", "has", "have", "will", "been", "they", "their", "not",
        "but", "can", "its", "may", "any", "all", "one", "also",
    }

    def __init__(self, top_n: int = 10):
        self.top_n    = top_n
        self._client  = None
        self._model   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                from google import genai
                from google.genai import types
                self._client = genai.Client(
                    api_key      = api_key,
                    http_options = types.HttpOptions(api_version="v1beta"),
                )
            except Exception as e:
                logger.warning(f"⚠️ Gemini: {e}")

        mode = "Gemini LLM" if self._client else "Regex"
        logger.info(f"✅ KeywordExtractor جاهز | top_n={top_n} | {mode}")

    def extract(self, chunk_id: str, text: str) -> KeywordsResult:
        if not text or len(text.split()) < 3:
            return KeywordsResult(chunk_id=chunk_id)

        # تطبيع النص أولاً دائماً
        clean = normalize_arabic(text)

        if self._client:
            try:
                return self._llm_extract(chunk_id, clean)
            except Exception as e:
                logger.warning(f"⚠️ LLM extract: {e}")

        return self._regex_extract(chunk_id, clean)

    def _llm_extract(self, chunk_id: str, text: str) -> KeywordsResult:
        snippet = text[:500]
        prompt = f"""استخرج {self.top_n} كلمات مفتاحية من النص التالي.

النص:
{snippet}

القواعد:
- ركّز على: الأسماء، الأرقام، التواريخ، المبالغ، المصطلحات القانونية
- احذف حروف الجر والكلمات الشائعة مثل (في، من، أن، أو)
- أبقِ كل كلمة بلغتها الأصلية (عربي أو إنجليزي)
- كلمة أو عبارتان لكل سطر فقط، بدون ترقيم:"""

        resp     = self._client.models.generate_content(model=self._model, contents=prompt)
        lines    = [l.strip().lstrip("-• ") for l in resp.text.strip().split("\n") if l.strip() and len(l.strip()) > 1]
        keywords = [kw for kw in lines if kw not in self.STOPWORDS_AR and kw.lower() not in self.STOPWORDS_EN][:self.top_n]

        logger.debug(f"🔑 {chunk_id}: {keywords[:5]}")
        return KeywordsResult(chunk_id=chunk_id, keywords=keywords,
                              scores=[round(1.0-i*0.05,2) for i in range(len(keywords))])

    def _regex_extract(self, chunk_id: str, text: str) -> KeywordsResult:
        ar_words  = [w for w in re.findall(r'[\u0600-\u06FF]{3,}', text)
                     if w not in self.STOPWORDS_AR]
        en_words  = [w for w in re.findall(r'[a-zA-Z]{3,}', text.lower())
                     if w not in self.STOPWORDS_EN]
        numbers   = re.findall(r'\b\d{4,}\b|\b\d+[.,]\d+\b', text)

        freq     = Counter(ar_words + en_words)
        top      = [w for w, _ in freq.most_common(self.top_n)]
        keywords = list(dict.fromkeys(numbers[:3] + top))[:self.top_n]

        logger.debug(f"🔑 regex {chunk_id}: {keywords[:5]}")
        return KeywordsResult(chunk_id=chunk_id, keywords=keywords,
                              scores=[round(1.0-i*0.05,2) for i in range(len(keywords))])

    def extract_batch(self, chunks: list, text_field: str = "content",
                      id_field: str = "chunk_id") -> list[KeywordsResult]:
        results = []
        logger.info(f"🔑 استخراج كلمات مفتاحية لـ {len(chunks)} قطعة...")
        for i, chunk in enumerate(chunks, 1):
            if isinstance(chunk, dict):
                chunk_id, text = chunk.get(id_field, f"chunk_{i}"), chunk.get(text_field, "")
            else:
                chunk_id, text = getattr(chunk, id_field, f"chunk_{i}"), getattr(chunk, text_field, "")
            results.append(self.extract(chunk_id, text))
        logger.success(f"✅ اكتمل استخراج الكلمات لـ {len(results)} قطعة")
        return results