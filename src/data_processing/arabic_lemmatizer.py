# ============================================================
# src/data_processing/arabic_lemmatizer.py
#
# معالج التصريف الشامل للعربية باستخدام FARASA
#
# FARASA = Fast and Accurate Retrieval and Summarization for Arabic
# - مُحلل صرفي دقيق
# - استخراج الجذور (lemmas)
# - تحليل الأجزاء اللغوية (POS tagging)
#
# الفائدة: فاتورة ← فاتورة (root) + variant "فواتير"، "الفاتورة"، إلخ
# ============================================================

import os
import re
from typing import Optional, List
from loguru import logger

try:
    from farasa.segmenter import FarasaSegmenter
    from farasa.lemmatizer import FarasaLemmatizer
    HAS_FARASA = True
except ImportError:
    HAS_FARASA = False
    logger.warning("⚠️ FARASA غير مثبت — استخدم معالجة بسيطة للعربية")


class ArabicLemmatizer:
    """
    يُحسّن استرجاع الاستعلامات العربية بـ:
    1. إزالة الدوال (ال، و، ب، ل)
    2. استخراج الجذور (lemmatization)
    3. معالجة الأحرف المختلفة (ة، ي، ا التمديد)
    4. دعم التشكيل العربي (harakat)

    الاستخدام:
        lemmatizer = ArabicLemmatizer()
        root = lemmatizer.get_root("فواتير")  → "فاتورة"
        normalized = lemmatizer.normalize_query("ابغى تقرير الهندسي")
    """
    
    def __init__(self):
        self.segmenter = None
        self.lemmatizer = None
        self.use_farasa = HAS_FARASA
        
        if HAS_FARASA:
            try:
                self.segmenter = FarasaSegmenter(interactive=False)
                self.lemmatizer = FarasaLemmatizer(interactive=False)
                logger.success("✅ FARASA Lemmatizer جاهز | morphological analysis")
            except Exception as e:
                logger.warning(f"⚠️ فشل تحميل FARASA: {e} → وسائل بسيطة")
                self.use_farasa = False
        else:
            logger.info("ℹ️ استخدام معالجة عربية بسيطة (بدون FARASA)")
        
        # قاموس مرادفات/جذور يدوي للحالات الشائعة
        self.roots_dict = {
            "فواتير": "فاتورة",
            "الفاتورة": "فاتورة",
            "فاتورة": "فاتورة",
            "تقارير": "تقرير",
            "التقرير": "تقرير",
            "تقارير": "تقرير",
            "الهندسي": "هندسي",
            "هندسية": "هندسي",
            "هندسيين": "هندسي",
            "التاريخ": "تاريخ",
            "التواريخ": "تاريخ",
            "مبالغ": "مبلغ",
            "المبلغ": "مبلغ",
            "دفعات": "دفعة",
            "الدفعة": "دفعة",
            "عملاء": "عميل",
            "العميل": "عميل",
            "الزبون": "زبون",
            "زبائن": "زبون",
        }
    
    def normalize_text(self, text: str) -> str:
        """
        تطبيع النص العربي:
        - إزالة الحروف الزائدة (التشكيل)
        - توحيد الحروف (ا → ا التمديد)
        - إزالة الفراغات الزائدة
        """
        if not text or not text.strip():
            return ""
        
        # إزالة الحروف العربية الإضافية والتشكيل
        text = self._remove_diacritics(text)
        
        # توحيد الحروف المختلفة
        text = text.replace("أ", "ا").replace("إ", "ا")
        text = text.replace("ة", "ه")  # الهاء المربوطة → هاء عادية
        text = text.replace("ؤ", "ء").replace("ئ", "ء")
        
        # إزالة الفراغات الزائدة
        text = " ".join(text.split())
        
        return text.strip()
    
    def _remove_diacritics(self, text: str) -> str:
        """
        إزالة التشكيل (الفتحة، الكسرة، الضمة، إلخ)
        
        التشكيل العربي يشمل:
        - الفتحة: َ
        - الضمة: ُ
        - الكسرة: ِ
        - الشدة: ّ
        - السكون: ْ
        - التنوين: ً ٌ ٍ
        - المد: ٓ
        """
        if not text:
            return ""
        
        # نمط شامل لجميع علامات التشكيل العربي
        diacritics_pattern = r'[\u064B-\u065F\u0670]'
        text = re.sub(diacritics_pattern, '', text)
        
        # إزالة التشكيل القديم كـ fallback
        arabic_diacritics = "ًٌٍَُِّْـ"
        return "".join(char for char in text if char not in arabic_diacritics)

    def preserve_diacritics(self, text: str) -> str:
        """
        الحفاظ على التشكيل العربي كما هو
        يُستخدم عند تخزين النصوص المهمة دينياً أو الشعر
        
        Args:
            text: النص العربي مع التشكيل
            
        Returns:
            النص مع الحفاظ على التشكيل
        """
        if not text:
            return ""
        
        # تطبيع Unicode مع الحفاظ على التشكيل
        import unicodedata
        normalized = unicodedata.normalize('NFC', text)
        return normalized

    def normalize_with_diacritics(self, text: str, remove_tashkeel: bool = False) -> str:
        """
        تطبيع النص العربي مع خيار إزالة/الحفاظ على التشكيل
        
        Args:
            text: النص العربي
            remove_tashkeel: إذا True، يزيل التشكيل
            
        Returns:
            النص المطبع
        """
        if not text:
            return ""
        
        if remove_tashkeel:
            text = self._remove_diacritics(text)
        
        # توحيد الحروف المختلفة
        text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        text = text.replace("ة", "ه")
        text = text.replace("ؤ", "ء").replace("ئ", "ء")
        text = text.replace("ى", "ي")
        
        # إزالة الفراغات الزائدة
        text = " ".join(text.split())
        
        return text.strip()

    def extract_diacritics_pattern(self, text: str) -> str:
        """
        استخراج نمط التشكيل من النص
        مفيد للتحقق من النصوص القرآنية أو الشعر
        
        Returns:
            سلسلة تمثل نمط التشكيل
        """
        if not text:
            return ""
        
        diacritics = []
        diacritics_pattern = r'[\u064B-\u065F\u0670]'
        
        for char in text:
            if re.match(diacritics_pattern, char):
                diacritics.append(char)
        
        return ''.join(diacritics)

    def compare_with_diacritics(self, text1: str, text2: str) -> bool:
        """
        مقارنة نصين عربيين مع تجاهل التشكيل
        
        Args:
            text1: النص الأول
            text2: النص الثاني
            
        Returns:
            True إذا كان النصان متطابقين بدون تشكيل
        """
        if not text1 or not text2:
            return text1 == text2
        
        norm1 = self.normalize_with_diacritics(text1, remove_tashkeel=True)
        norm2 = self.normalize_with_diacritics(text2, remove_tashkeel=True)
        
        return norm1 == norm2
    
    def get_root(self, word: str) -> Optional[str]:
        """
        استخراج جذر الكلمة العربية
        
        Args:
            word: الكلمة المراد استخراج جذرها
        
        Returns:
            الجذر أو الكلمة الأصلية إذا لم يتم إيجاد جذر
        """
        if not word or not word.strip():
            return None
        
        word = word.strip()
        
        # تحقق من القاموس الداخلي أولاً
        if word in self.roots_dict:
            return self.roots_dict[word]
        
        # إذا كان FARASA متاحاً، استخدمه
        if self.use_farasa and self.lemmatizer:
            try:
                # يعيد FARASA الكلمة + الجذر
                result = self.lemmatizer.lemmatize(word)
                if result and len(result) > 0:
                    # result عادة يكون [original, lemma]
                    lemma = result[0] if isinstance(result, (list, tuple)) else result
                    return lemma if lemma and lemma != word else word
            except Exception as e:
                logger.debug(f"⚠️ FARASA lemmatize فشل لـ '{word}': {e}")
        
        # Fallback: إزالة البادئات واللواحق الشائعة
        return self._simple_lemmatize(word)
    
    def _simple_lemmatize(self, word: str) -> str:
        """
        تصريف بسيط بدون FARASA:
        إزالة البادئات واللواحق الشائعة
        """
        # إزالة "ال" في البداية
        if word.startswith("ال"):
            word = word[2:]
        
        # إزالة البادئات الشائعة
        prefixes = ["و", "ف", "ب", "ل", "ك"]
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > 2:
                word = word[1:]
                break
        
        # إزالة اللواحق الشائعة
        suffixes = [
            "ات", "ان", "ين", "يه",  # جمع ومثنى
            "ها", "هم", "هن", "ني", "نا",  # ضمائر
            "ة", "ه"  # تأنيث
        ]
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                word = word[:-len(suffix)]
                break
        
        return word
    
    def normalize_query(self, query: str) -> List[str]:
        """
        تطبيع استعلام عربي لتحسين الاسترجاع
        
        Returns:
            قائمة بـ [الاستعلام الأصلي، صياغة معايرة، جذور]
        """
        if not query or not query.strip():
            return [query]
        
        results = [query]  # احفظ الأصلي دائماً
        
        # صيغة معايرة
        normalized = self.normalize_text(query)
        if normalized != query:
            results.append(normalized)
        
        # جذور الكلمات
        words = query.split()
        roots = []
        for word in words:
            root = self.get_root(word)
            if root and root != word:
                roots.append(root)
        if roots:
            rooted_query = " ".join(roots)
            if rooted_query not in results:
                results.append(rooted_query)
        
        return results
    
    def lemmatize_chunk(self, text: str) -> str:
        """
        يلمّع chunk بضيق الخيارات (lemmatization) لتحسين البحث BM25
        """
        if not text or not text.strip():
            return text
        
        # قسّم لكلمات
        words = text.split()
        lemmatized_words = []
        
        for word in words:
            # احتفظ بالكلمات القصيرة جداً كما هي
            if len(word) <= 2:
                lemmatized_words.append(word)
            else:
                root = self.get_root(word)
                lemmatized_words.append(root if root else word)
        
        return " ".join(lemmatized_words)
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        استخراج الكلمات المفتاحية من نص عربي
        (يُستخدم لتحسين البحث)
        """
        if not text or not text.strip():
            return []
        
        # قسّم لكلمات وأزل الكلمات الشائعة
        stopwords = {
            "في", "من", "إلى", "أو", "و", "ف", "هذا", "ذاك", "هو", "هي",
            "أن", "كان", "هناك", "هنا", "عندما", "لكن", "لأن", "إذا", "ما",
            "ا", "ان", "على", "عن", "بعد", "قبل", "خلال", "أمام", "خلف"
        }
        
        words = text.split()
        keywords = []
        
        for word in words:
            word = word.strip(".,;:!?\"'()[]{}").lower()
            if len(word) > 2 and word not in stopwords:
                root = self.get_root(word)
                if root and root not in keywords:
                    keywords.append(root)
        
        return keywords[:top_k]


# ════════════════════════════════════
# دالات مساعدة عامة
# ════════════════════════════════════

_lemmatizer_instance: Optional[ArabicLemmatizer] = None

def get_lemmatizer() -> ArabicLemmatizer:
    """احصل على singleton instance من Arabic Lemmatizer"""
    global _lemmatizer_instance
    if _lemmatizer_instance is None:
        _lemmatizer_instance = ArabicLemmatizer()
    return _lemmatizer_instance

def lemmatize(word: str) -> Optional[str]:
    """دالة مساعدة سريعة لاستخراج الجذر"""
    return get_lemmatizer().get_root(word)

def normalize_arabic_query(query: str) -> List[str]:
    """دالة مساعدة سريعة لتطبيع الاستعلام"""
    return get_lemmatizer().normalize_query(query)
