# ============================================================
# src/data_processing/arabic_enhancer.py
#
# محسّن النص العربي — ٣ طبقات:
#
# Layer 1: NFKC Normalization (دائماً، فوري)
#   → يُحوّل 74.9% من النص من Presentation Forms → Unicode
#
# Layer 2: Gemini Vision OCR (للجداول والصور)
#   → أفضل دقة للعربية: CER=0.13 vs EasyOCR CER=0.58
#   → يُعيد قراءة الصفحات التي تحتوي نصاً عربياً مشوهاً
#
# Layer 3: Text Cleanup (قواعد نصية)
#   → توحيد المسافات، الأرقام، علامات الترقيم
#
# المصدر: KITAB-Bench 2025 — Gemini-2.0-Flash الأفضل للعربية
# ============================================================

import os
import re
import base64
import unicodedata
from pathlib import Path
from loguru import logger


# ════════════════════════════════════════════════════════════
# Layer 1: NFKC Normalization (بدون LLM — فوري)
# ════════════════════════════════════════════════════════════

def normalize_arabic_text(text: str) -> str:
    """
    تحويل شامل للنص العربي:
    1. NFKC: يُحوّل Arabic Presentation Forms (FBxx/FExx) → Unicode المعياري
    2. حذف التشكيل (يُربك البحث)
    3. توحيد الألف والتاء المربوطة
    4. تنظيف المسافات والأحرف غير المرئية
    5. RTL/LTR markers حذف

    المشكلة التي يحلها:
        'ﻣﺴﺘﻨﺪات' (FExx) → 'مستندات' (قابل للبحث)
    """
    if not text:
        return text

    # ── 1. NFKC: الأقوى — يُعالج 74.9% من المشكلة ──
    text = unicodedata.normalize("NFKC", text)

    # ── 2. حذف RTL/LTR markers والأحرف التحكمية ──
    text = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF]', '', text)

    # ── 3. حذف التشكيل (يُربك BM25 و Embeddings) ──
    text = re.sub(r'[\u064B-\u065F\u0610-\u061A\u06D6-\u06DC\u0670]', '', text)

    # ── 4. توحيد الألف (أإآا → ا) ──
    text = re.sub(r'[أإآ]', 'ا', text)

    # ── 5. توحيد التاء المربوطة والألف المقصورة ──
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)

    # ── 6. تنظيف المسافات ──
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def separate_arabic_words(text: str) -> str:
    """
    فصل الكلمات العربية المتصلة تلقائياً باستخدام 3 طبقات:
    
    1. قاموس العبارات الشائعة (فوري)
    2. قواعد فصل البادئات (فوري)
    3. Gemini API (للنص الصعب جداً)
    
    المشكلة:
        OCR أحياناً يدمج الكلمات: "المملكةالعربيةالسعودية"
    
    الحل:
        استخدام قاموس الكلمات الشائعة + قواعد + LLM
    """
    if not text:
        return text
    
    # كلمات شائعة في الوثائق الرسمية (مرتبة حسب الأهمية)
    common_words = [
        # العبارات الثابتة
        'بسم الله الرحمن الرحيم',
        'المملكة العربية السعودية',
        'وزارة الشؤون البلدية والقروية',
        'أمانة منطقة الرياض',
        'بلدية عرقة',
        
        # كلمات مفردة
        'المملكة', 'العربية', 'السعودية', 'وزارة', 'الشئون', 'البلدية', 'القروية',
        'أمانة', 'منطقة', 'الرياض', 'بلدية', 'عرقة', 'حي', 'المهدية',
        'رخصة', 'بناء', 'رخصةبناء', 'فيلا', 'سكنية', 'سكنيه', 'الكتروني',
        'شارع', 'قطعة', 'مخطط', 'صك', 'رقم', 'تاريخ', 'الاثبات',
        'مالك', 'المالك', 'اسم', 'نوع', 'انتهاء', 'إصدار',
        'سكني', 'فيلا', 'ملحق', 'ملاحق', 'اسوار', 'أرضي', 'اول',
        'علويه', 'علوية', 'ملاحظات', 'قرار', 'مكتب', 'هندسي',
        'شركة', 'الكهرباء', 'المياه', 'الصرف', 'الصحي', 'العزل', 'الحراري',
        'كود', 'البناء', 'السعودي', 'ريال', 'ايصال', 'هـ',
        'لبناء', 'البناء', 'برخصة', 'الفرعية', 'الرئيسية',
        
        # أسماء
        'ابراهيم', 'عبدالرحمن', 'خلف', 'المطيري', 'دام', 'للاستشارات',
        'الهندسية', 'دخيل', 'الله', 'محمد', 'السبيعي', 'يحيي', 'موسى',
        'عثمان', 'العتيبي', 'عبدالله', 'هلال',
        
        # أرقام وهوية
        'هـ', 'م', 'كم', 'متر', 'متر مربع', 'فولت',
        
        # جهات
        'شمال', 'جنوب', 'شرق', 'غرب', 'القطعة', 'جزء',
        'الجهة', 'الحدود', 'الابعاد', 'الارتداد', 'المساحة',
        'مكونات', 'عدد', 'الوحدات', 'المساحة', 'الاستخدام',
    ]
    
    result = text
    
    # 1. فصل العبارات الطويلة أولاً
    for phrase in common_words:
        if ' ' in phrase:
            parts = phrase.split()
            joined = ''.join(parts)
            if joined in result:
                result = result.replace(joined, phrase)
    
    # 2. فصل الكلمات الفردية
    for word in common_words:
        if ' ' not in word and len(word) > 2:
            # البحث عن الكلمة متصلة بما قبلها
            pattern = rf'(?<!\s)({re.escape(word)})(?=[\u0600-\u06FF])'
            result = re.sub(pattern, r'\1 ', result)
            
            # البحث عن الكلمة متصلة بما بعدها
            pattern = rf'([\u0600-\u06FF])({re.escape(word)})(?!\s)'
            result = re.sub(pattern, r'\1 \2', result)
    
    # 3. قواعد عامة لفصل الكلمات
    # فصل "ال" عن ما بعدها
    result = re.sub(r'(ال)([\u0600-\u06FF]{3,})', r'\1 \2', result)
    
    # فصل "بـ" عن ما بعدها
    result = re.sub(r'(ب)([\u0600-\u06FF]{3,})', r'\1 \2', result)
    
    # فصل "لـ" عن ما بعدها
    result = re.sub(r'(ل)([\u0600-\u06FF]{3,})', r'\1 \2', result)
    
    # فصل "و" عن ما بعدها
    result = re.sub(r'(و)([\u0600-\u06FF]{3,})', r'\1 \2', result)
    
    # فصل "فـ" عن ما بعدها
    result = re.sub(r'(ف)([\u0600-\u06FF]{3,})', r'\1 \2', result)
    
    # تنظيف المسافات الزائدة
    result = re.sub(r'[ \t]{2,}', ' ', result)
    result = result.strip()
    
    return result


def separate_arabic_words_llm(text: str) -> str:
    """
    فصل الكلمات العربية باستخدام Gemini API
    يُستخدم للنص الصعب الذي فشل الفصل التلقائي
    
    Args:
        text: النص العربي المتصل
        
    Returns:
        النص مع الكلمات المفصولة
    """
    if not text or len(text) < 50:
        return text
    
    try:
        import os
        from google import genai
        from google.genai import types
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return text
        
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
        
        prompt = """فصل الكلمات في هذا النص العربي.
النص يحتوي على كلمات متصلة بشكل خاطئ.
المطلوب: أعد كتابة النص مع فصل كل كلمة بمسافة.

مثال:
المدخل: "المملكةالعربيةالسعودية"
المخرج: "المملكة العربية السعودية"

النص:"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, text],
            config=types.GenerateContentConfig(
                temperature=0.1,
            ),
        )
        
        separated = response.text.strip()
        if separated and len(separated) > len(text) * 0.5:
            logger.info(f"✅ Gemini فصل الكلمات: {len(text)} → {len(separated)} حرف")
            return separated
        else:
            return text
            
    except Exception as e:
        logger.warning(f"⚠️ Gemini word separation failed: {e}")
        return text


def has_presentation_forms(text: str) -> bool:
    """فحص سريع: هل النص يحتوي Arabic Presentation Forms؟"""
    for ch in text:
        cp = ord(ch)
        if 0xFB50 <= cp <= 0xFDFF or 0xFE70 <= cp <= 0xFEFF:
            return True
    return False


def presentation_forms_ratio(text: str) -> float:
    """نسبة Presentation Forms من إجمالي الأحرف العربية"""
    pf = ar = 0
    for ch in text:
        cp = ord(ch)
        if 0xFB50 <= cp <= 0xFDFF or 0xFE70 <= cp <= 0xFEFF:
            pf += 1
        elif 0x0600 <= cp <= 0x06FF:
            ar += 1
    total = pf + ar
    return pf / total if total > 0 else 0.0


# ════════════════════════════════════════════════════════════
# Layer 2: Gemini Vision OCR (للصفحات المشوهة)
# ════════════════════════════════════════════════════════════

class GeminiArabicOCR:
    """
    يستخدم Gemini Vision لإعادة قراءة صفحات PDF العربية.

    متى يُستخدم:
    - الجداول التي تحتوي نصاً عربياً مشوهاً
    - الصور الممسوحة ضوئياً
    - أي نص نسبة Presentation Forms فيه > 30%

    الدقة: CER=0.13 (الأفضل للعربية حسب KITAB-Bench 2025)
    """

    VISION_MODEL = "gemini-2.0-flash"

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        self._client = None
        if api_key:
            try:
                from google import genai
                from google.genai import types
                self._client = genai.Client(
                    api_key      = api_key,
                    http_options = types.HttpOptions(api_version="v1beta"),
                )
                self._types = types
                logger.success("✅ GeminiArabicOCR جاهز | أفضل دقة للعربية")
            except Exception as e:
                logger.warning(f"⚠️ Gemini Vision: {e}")

    def ocr_image(self, image_bytes: bytes, hint: str = "") -> str:
        """
        استخراج نص من صورة بـ Gemini Vision.

        Args:
            image_bytes: bytes الصورة (PNG/JPG)
            hint: تلميح عن نوع المحتوى

        Returns:
            str: النص المستخرج مع تطبيع تلقائي
        """
        if not self._client:
            return ""

        prompt = f"""استخرج كل النص من هذه الصورة بدقة تامة.

التعليمات:
- احتفظ بالنص كما هو (عربي أو إنجليزي)
- الجداول: احتفظ بالبنية باستخدام | للفصل
- لا تضف أي شرح أو تفسير
- الأرقام: اكتبها كما تظهر بالضبط
- النص العربي: يميناً لشمالاً{f' | السياق: {hint}' if hint else ''}

النص:"""

        try:
            image_b64 = base64.standard_b64encode(image_bytes).decode()
            response  = self._client.models.generate_content(
                model    = self.VISION_MODEL,
                contents = [
                    self._types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt,
                ],
            )
            extracted = response.text.strip()
            # طبّق normalization على نتيجة Gemini أيضاً
            return normalize_arabic_text(extracted)

        except Exception as e:
            logger.warning(f"⚠️ Gemini Vision OCR: {e}")
            return ""

    def is_available(self) -> bool:
        return self._client is not None


# ════════════════════════════════════════════════════════════
# Layer 3: Text Post-Processing
# ════════════════════════════════════════════════════════════

def clean_bilingual_table(text: str) -> str:
    """
    تنظيف الجداول الثنائية اللغة التي تأتي من Docling.

    المشكلة: Docling يُكرر محتوى الخلية في الجداول الثنائية
    مثال: | Contract Data | Contract Data | بيانات العقد |
    """
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        if '|' in line:
            cells = [c.strip() for c in line.split('|')]
            # احذف الخلايا المكررة (نفس المحتوى أو فارغة)
            seen = set()
            unique_cells = []
            for cell in cells:
                norm_cell = normalize_arabic_text(cell).lower()
                if norm_cell and norm_cell not in seen:
                    unique_cells.append(cell)
                    seen.add(norm_cell)
                elif not norm_cell:
                    unique_cells.append('')  # احتفظ بالأعمدة الفارغة للهيكل
            cleaned.append(' | '.join(unique_cells))
        else:
            cleaned.append(line)
    return '\n'.join(cleaned)


def fix_arabic_numbers(text: str) -> str:
    """
    توحيد الأرقام العربية والهندية-الشرقية.
    ٠١٢٣٤٥٦٧٨٩ → 0123456789
    """
    arabic_indic = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
    extended_arabic = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
    text = text.translate(arabic_indic)
    text = text.translate(extended_arabic)
    return text


def fix_arabic_spacing(text: str) -> str:
    """
    إصلاح المسافات الزائدة بين الأحرف العربية.
    
    المشكلة: Docling يفصل الأحرف العربية بمسافات
    مثال: 'ا ب ر ا ه ي م' → 'إبراهيم'
    """
    if not text:
        return text
    
    # نمط: حرف عربي + مسافات + حرف عربي
    arabic_pattern = re.compile(r'([\u0600-\u06FF])\s+([\u0600-\u06FF])')
    
    # تكرار حتى لا يتبقى مسافات زائدة
    prev_text = None
    while prev_text != text:
        prev_text = text
        text = arabic_pattern.sub(r'\1\2', text)
    
    return text


# ════════════════════════════════════════════════════════════
# الدالة الرئيسية الشاملة
# ════════════════════════════════════════════════════════════

def enhance_arabic_chunk(content: str, use_vision: bool = False,
                          image_bytes: bytes = None) -> str:
    """
    تطبيق جميع طبقات التحسين على chunk واحد.

    Args:
        content     : النص الخام من Docling
        use_vision  : استخدم Gemini Vision (للصور/الجداول المشوهة)
        image_bytes : bytes الصورة إذا use_vision=True

    Returns:
        str: النص المُحسَّن
    """
    if not content:
        return content

    # ── Layer 1: NFKC Normalization (دائماً) ──
    enhanced = normalize_arabic_text(content)

    # ── Layer 2: Gemini Vision (للمحتوى المشوه) ──
    if use_vision and image_bytes:
        ocr = GeminiArabicOCR()
        if ocr.is_available():
            vision_text = ocr.ocr_image(image_bytes, hint="عقد إيجار")
            if vision_text and len(vision_text) > len(enhanced) * 0.5:
                enhanced = vision_text

    # ── Layer 3: Post-processing ──
    enhanced = fix_arabic_numbers(enhanced)
    enhanced = fix_arabic_spacing(enhanced)  # إصلاح المسافات الزائدة
    enhanced = separate_arabic_words(enhanced)  # فصل الكلمات المتصلة (تلقائي)
    
    # ── Layer 4: Gemini LLM لفصل الكلمات (دائماً للنص من OCR) ──
    # النص من OCR يحتاج فصل كلمات بالـ LLM
    enhanced = separate_arabic_words_llm(enhanced)

    # تنظيف الجداول المكررة
    if '|' in enhanced:
        enhanced = clean_bilingual_table(enhanced)

    return enhanced


def enhance_chunks_batch(chunks: list) -> list:
    """
    تطبيق التحسين على مجموعة chunks.

    يُعدّل content في كل chunk مباشرة.
    """
    total  = len(chunks)
    fixed  = 0
    logger.info(f"🔤 تطبيع النص العربي لـ {total} قطعة...")

    for chunk in chunks:
        # دعم dict وobjects
        if isinstance(chunk, dict):
            content = chunk.get('content', '')
        else:
            content = getattr(chunk, 'content', '')

        if not content:
            continue

        # تحقق من الحاجة للتطبيع
        # 1. Presentation Forms
        pf_ratio = presentation_forms_ratio(content)
        # 2. نسبة مسافات منخفضة = كلمات متصلة
        space_ratio = content.count(' ') / len(content) if content else 0
        
        # إذا كان هناك Presentation Forms أو كلمات متصلة (مسافات < 20%)
        if pf_ratio > 0.05 or space_ratio < 0.20:
            enhanced = enhance_arabic_chunk(content)
            if isinstance(chunk, dict):
                chunk['content'] = enhanced
            else:
                chunk.content = enhanced
            fixed += 1

    logger.success(f"✅ تطبيع مكتمل | {fixed}/{total} قطعة حُسِّنت")
    return chunks
