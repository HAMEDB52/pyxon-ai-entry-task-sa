# ============================================================
# data_sources/loader.py
# نقطة دخول جميع أنواع المستندات إلى النظام
# يدعم: Documents, Code, Spreadsheets, Images
# ============================================================

import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult


# ============================================================
# أنواع المصادر المدعومة
# ============================================================

class SourceType(str, Enum):
    DOCUMENT = "document"       # PDF, DOCX, PPTX, TXT
    CODE     = "code"           # PY, JS, TS, JAVA, CPP ...
    SPREADSHEET = "spreadsheet" # XLSX, CSV
    IMAGE    = "image"          # PNG, JPG, JPEG, WEBP


# ============================================================
# نتيجة التحميل الموحدة
# ============================================================

@dataclass
class LoadedSource:
    """النتيجة الموحدة لأي مصدر بعد التحميل"""
    file_path: str
    source_type: SourceType
    file_name: str
    file_extension: str
    raw_text: str                        # النص الخام المستخرج
    conversion_result: Optional[ConversionResult] = None  # نتيجة Docling (للمستندات)
    metadata: dict = field(default_factory=dict)


# ============================================================
# خرائط الامتدادات
# ============================================================

DOCUMENT_EXTENSIONS    = {".pdf", ".docx", ".pptx", ".html", ".txt", ".md"}
CODE_EXTENSIONS        = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs", ".rb", ".php"}
SPREADSHEET_EXTENSIONS = {".xlsx", ".csv"}
IMAGE_EXTENSIONS       = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def detect_source_type(file_path: str) -> SourceType:
    """تحديد نوع المصدر تلقائياً بناءً على الامتداد"""
    ext = Path(file_path).suffix.lower()

    if ext in DOCUMENT_EXTENSIONS:
        return SourceType.DOCUMENT
    elif ext in CODE_EXTENSIONS:
        return SourceType.CODE
    elif ext in SPREADSHEET_EXTENSIONS:
        return SourceType.SPREADSHEET
    elif ext in IMAGE_EXTENSIONS:
        return SourceType.IMAGE
    else:
        logger.warning(f"امتداد غير معروف: {ext} — سيتم معاملته كمستند")
        return SourceType.DOCUMENT


# ============================================================
# بناء محول Docling
# ============================================================

def _build_converter() -> DocumentConverter:
    """
    بناء محول Docling مع الإعدادات المناسبة
    - تحليل الجداول بدقة عالية
    - استخراج الصور
    - دعم OCR
    """
    use_gpu = os.getenv("DOCLING_USE_GPU", "false").lower() == "true"
    table_mode = os.getenv("DOCLING_TABLE_MODE", "accurate")

    # ── إعدادات OCR للعربية والإنجليزية ──
    # RapidOCR الافتراضي مُدرَّب على الصينية → ضعيف للعربية
    # الحل: EasyOCR مع دعم العربية والإنجليزية
    try:
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
        from docling.datamodel.pipeline_options import EasyOcrOptions
        ocr_options = EasyOcrOptions(lang=["ar", "en"])
        pipeline_options = PdfPipelineOptions(
            do_ocr               = True,
            do_table_structure   = True,
            generate_page_images = False,
            generate_picture_images = True,
            ocr_options          = ocr_options,
        )
        logger.info("✅ EasyOCR مفعّل | لغات: عربي + إنجليزي")
    except Exception:
        # fallback للإعدادات الافتراضية
        pipeline_options = PdfPipelineOptions(
            do_ocr               = True,
            do_table_structure   = True,
            generate_page_images = False,
            generate_picture_images = True,
        )
        logger.warning("⚠️ EasyOCR غير متاح — OCR افتراضي")

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    logger.info(f"✅ Docling جاهز | GPU: {use_gpu} | وضع الجداول: {table_mode}")
    return converter


# ============================================================
# دوال التحميل لكل نوع
# ============================================================

def _build_fast_converter() -> DocumentConverter:
    """
    محول سريع للرفع الفوري — بدون OCR أو تحليل جداول
    أسرع بـ 10x من المحول الكامل
    """
    pipeline_options = PdfPipelineOptions(
        do_ocr              = False,   # ← أبطأ شيء، نوقفه
        do_table_structure  = False,   # ← ثاني أبطأ شيء
        generate_page_images= False,
        generate_picture_images = False,
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    logger.info("⚡ FastConverter جاهز | OCR=OFF | Tables=OFF")
    return converter


def _load_document(file_path: str, converter: DocumentConverter) -> LoadedSource:
    """تحميل المستندات (PDF, DOCX, PPTX...) عبر Docling

    للمستندات العربية: يستخدم نهج هجين لاستخراج النص
    1. استخراج النص الرقمي مباشرة من PDF
    2. استخدام OCR للنص الممسوح ضوئياً والكتابة اليدوية
    3. دمج النتائج للحصول على أفضل تغطية
    """
    logger.info(f"📄 تحميل مستند: {file_path}")

    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        # === نهج هجين: نص رقمي + OCR (يشمل الكتابة اليدوية) ===
        try:
            # 1. استخراج النص الرقمي مباشرة من PDF
            import fitz
            doc = fitz.open(file_path)
            digital_text_parts = []
            pages_with_digital = []
            total_pages = len(doc)

            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text("text", sort=True)
                if text and len(text.strip()) > 50:  # نص رقمي حقيقي
                    digital_text_parts.append(f"--- صفحة {page_num + 1} ---\n{text}")
                    pages_with_digital.append(page_num + 1)

            digital_text = "\n\n".join(digital_text_parts)
            
            # 2. استخدام OCR لجميع الصفحات لاستخراج الكتابة اليدوية
            # حتى لو كان هناك نص رقمي، الكتابة اليدوية تحتاج OCR
            logger.info(f"📊 نص رقمي في {len(pages_with_digital)}/{total_pages} صفحة")
            logger.info(f"✍️  تطبيق OCR شامل للكتابة اليدوية...")
            
            doc.close()
            
            # OCR شامل لجميع الصفحات (للكتابة اليدوية)
            ocr_text = _gemini_vision_ocr(file_path)
            
            # 3. دمج النص الرقمي وOCR
            if ocr_text and len(ocr_text.strip()) > 100:
                # استخدام OCR فقط لأنه يشمل الكتابة اليدوية
                raw_text = ocr_text
                logger.success(f"✅ OCR شامل: {len(ocr_text):,} حرف (يشمل الكتابة اليدوية)")
            else:
                raw_text = digital_text
                logger.success(f"✅ نص رقمي: {len(raw_text):,} حرف")

            # بناء النتيجة
            result = type('DummyResult', (), {'document': type('DummyDoc', (), {
                'export_to_markdown': lambda: raw_text,
                'num_pages': total_pages,
                'tables': [],
                'pictures': [],
            })()})()

            return LoadedSource(
                file_path         = file_path,
                source_type       = SourceType.DOCUMENT,
                file_name         = Path(file_path).name,
                file_extension    = ext,
                raw_text          = raw_text,
                conversion_result = result,
                metadata          = {
                    "num_pages"      : total_pages,
                    "num_tables"     : 0,
                    "num_images"     : 0,
                    "source"         : file_path,
                    "ocr_method"     : "handwriting_ocr" if ocr_text else "digital",
                    "digital_pages"  : pages_with_digital,
                },
            )

        except Exception as e:
            logger.warning(f"⚠️ الاستخراج فشل ({e}) — fallback إلى Gemini OCR الكامل")
            # Fallback إلى Gemini OCR الكامل
            raw_text = _gemini_vision_ocr(file_path)
            logger.success(f"✅ Gemini Vision OCR: {len(raw_text):,} حرف")

            result = type('DummyResult', (), {'document': type('DummyDoc', (), {
                'export_to_markdown': lambda: raw_text,
                'num_pages': 1,
                'tables': [],
                'pictures': [],
            })()})()

            return LoadedSource(
                file_path         = file_path,
                source_type       = SourceType.DOCUMENT,
                file_name         = Path(file_path).name,
                file_extension    = ext,
                raw_text          = raw_text,
                conversion_result = result,
                metadata          = {
                    "num_pages"  : 1,
                    "num_tables" : 0,
                    "num_images" : 0,
                    "source"     : file_path,
                    "ocr_method" : "gemini_vision",
                },
            )

    # Fallback إلى Docling للأنواع الأخرى
    result = converter.convert(file_path)
    doc = result.document

    # تحويل المستند إلى Markdown للحصول على نص منسق مع الجداول
    raw_text = doc.export_to_markdown()

    metadata = {
        "num_pages"  : getattr(doc, "num_pages", None),
        "num_tables" : len(doc.tables) if hasattr(doc, "tables") else 0,
        "num_images" : len(doc.pictures) if hasattr(doc, "pictures") else 0,
        "source"     : file_path,
    }

    return LoadedSource(
        file_path         = file_path,
        source_type       = SourceType.DOCUMENT,
        file_name         = Path(file_path).name,
        file_extension    = Path(file_path).suffix.lower(),
        raw_text          = raw_text,
        conversion_result = result,
        metadata          = metadata,
    )


def _gemini_vision_ocr(pdf_path: str) -> str:
    """
    استخراج النص من PDF باستخدام Gemini Vision OCR
    يدعم العربية والإنجليزية والكتابة اليدوية بدقة عالية
    """
    import fitz
    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY غير موجود")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta"),
    )

    # فتح PDF وتحويل الصفحات لصور
    doc = fitz.open(pdf_path)
    all_texts = []

    for page_num in range(min(len(doc), 10)):  # أول 10 صفحات
        page = doc[page_num]
        # تكبير 4x لتحسين الجودة للكتابة اليدوية
        mat = fitz.Matrix(4, 4)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        prompt = """أنت خبير في استخراج النصوص العربية من الوثائق الرسمية.

المطلوب:
1. استخرج **كل** النص من هذه الصورة - سواء كان:
   - نص مطبوع (كمبيوتر)
   - نص PDF رقمي
   - **كتابة بخط اليد** (مهم جداً!)
   - ختم رسمي
   - أرقام وتواقيع

2. ⚠️ تحذير هام جداً: فصل الكلمات
   - **يجب فصل كل كلمة عن الأخرى بمسافة واحدة صحيحة**
   - لا تدمج الكلمات مع بعضها أبداً
   - هذا أهم شرط في المهمة!
   
   ✅ أمثلة صحيحة (افعل هذا):
   - "المملكة العربية السعودية"
   - "بلدية عرقة"
   - "رخصة بناء"
   - "فيلا سكنية"
   - "عبدالرحمن بن خلف"
   
   ❌ أمثلة خاطئة (لا تفعل هذا):
   - "المملكةالعربيةالسعودية"
   - "بلديةعرقة"
   - "رخصةبناء"
   - "فيلاسكنية"
   - "عبدالرحمنبنخلف"

3. الأرقام والتواريخ:
   - الأرقام: اكتبها كما تظهر (عربية أو إنجليزية)
   - التواريخ الهجرية: احفظها كما هي (مثال: 1440-11-25 هـ)
   - أرقام الهواتف: افصل بين الأرقام بشرطة (مثال: 055-123-4567)

4. الجداول:
   - استخدم تنسيق Markdown مع | للفصل بين الأعمدة
   - افصل خلايا الجدول بمسافات

5. الكتابة اليدوية:
   - استخرجها بدقة حتى لو كانت غير واضحة
   - حاول تخمين الكلمات من السياق

6. لا تضف أي شرح أو تفسير - فقط النص الخام

استخرج النص الآن مع فصل جميع الكلمات بمسافات:"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",  # النموذج الأصلي - يدعم الكتابة اليدوية
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,  # أقل عشوائية لدقة أعلى
            ),
        )

        text = response.text.strip()
        if text:
            all_texts.append(f"--- صفحة {page_num + 1} ---\n{text}")
            logger.info(f"✅ صفحة {page_num + 1}: {len(text):,} حرف")

    doc.close()

    if all_texts:
        logger.success(f"✅ اكتمل OCR: {len(all_texts)} صفحة | إجمالي: {sum(len(t) for t in all_texts):,} حرف")

    return "\n\n".join(all_texts)


def _gemini_vision_ocr_selective(pdf_path: str, skip_pages: list[int]) -> str:
    """
    استخراج النص من صفحات محددة فقط (تخطي الصفحات التي تحتوي نص رقمي)
    يدعم الكتابة اليدوية
    
    Args:
        pdf_path: مسار ملف PDF
        skip_pages: قائمة أرقام الصفحات التي لا تحتاج OCR (1-based)
    
    Returns:
        النص المستخرج من الصفحات المحددة فقط
    """
    import fitz
    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY غير موجود")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta"),
    )

    doc = fitz.open(pdf_path)
    all_texts = []

    for page_num in range(len(doc)):
        page_number = page_num + 1
        
        # تخطي الصفحات التي تحتوي نص رقمي
        if page_number in skip_pages:
            continue

        page = doc[page_num]
        # تكبير 4x لتحسين الجودة للكتابة اليدوية
        mat = fitz.Matrix(4, 4)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        logger.info(f"📸 OCR صفحة {page_number} (يشمل الكتابة اليدوية)...")

        prompt = """أنت خبير في استخراج النصوص العربية من الوثائق الرسمية.

المطلوب:
1. استخرج **كل** النص من هذه الصورة - سواء كان:
   - نص مطبوع (كمبيوتر)
   - نص PDF رقمي
   - **كتابة بخط اليد** (مهم جداً!)
   - ختم رسمي
   - أرقام وتواقيع

2. تعليمات هامة جداً:
   - احتفظ بالنص كما هو (عربي فقط)
   - **الكتابة اليدوية**: استخرجها بدقة حتى لو كانت غير واضحة
   - الأرقام: اكتبها كما تظهر (عربية أو إنجليزية)
   - الجداول: استخدم تنسيق Markdown مع |
   - لا تضف أي شرح أو تفسير
   - **افصل بين الكلمات العربية بمسافات صحيحة**
   - إذا كان هناك نص مشطوب أو ممسوح، حاول قراءته من السياق

3. تنبيه مهم:
   - الكتابة بخط اليد مهمة جداً - ركز عليها
   - الأرقام التاريخية (هـ) مهمة
   - الأسماء والوظائف مهمة

مثال صحيح: "المملكة العربية السعودية"
مثال خاطئ: "المملكالعربيةلسعودية"

استخرج النص الآن:"""

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                ),
            )

            text = response.text.strip()
            if text:
                all_texts.append(f"--- صفحة {page_number} ---\n{text}")
                logger.success(f"✅ صفحة {page_number}: {len(text):,} حرف")
        except Exception as e:
            logger.warning(f"⚠️ OCR صفحة {page_number} فشل: {e}")

    doc.close()
    
    if all_texts:
        logger.success(f"✅ اكتمل OCR: {len(all_texts)} صفحة")
    
    return "\n\n".join(all_texts)


def _load_code(file_path: str) -> LoadedSource:
    """تحميل ملفات الكود كنص عادي مع الحفاظ على التنسيق"""
    logger.info(f"💻 تحميل كود: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        code_text = f.read()

    ext = Path(file_path).suffix.lstrip(".")

    # تغليف الكود في Markdown code block للحفاظ على التنسيق
    raw_text = f"```{ext}\n{code_text}\n```"

    metadata = {
        "language"  : ext,
        "num_lines" : code_text.count("\n"),
        "source"    : file_path,
    }

    return LoadedSource(
        file_path      = file_path,
        source_type    = SourceType.CODE,
        file_name      = Path(file_path).name,
        file_extension = Path(file_path).suffix.lower(),
        raw_text       = raw_text,
        metadata       = metadata,
    )


def _load_spreadsheet(file_path: str, converter: DocumentConverter) -> LoadedSource:
    """تحميل جداول البيانات عبر Docling للحفاظ على بنية الجداول"""
    logger.info(f"📊 تحميل جدول بيانات: {file_path}")

    ext = Path(file_path).suffix.lower()

    if ext == ".csv":
        # CSV: قراءة مباشرة وتحويل إلى Markdown table
        import csv
        rows = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)

        if rows:
            header = "| " + " | ".join(rows[0]) + " |"
            separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
            body = "\n".join("| " + " | ".join(row) + " |" for row in rows[1:])
            raw_text = f"{header}\n{separator}\n{body}"
        else:
            raw_text = ""

        metadata = {
            "num_rows"   : len(rows) - 1,
            "num_columns": len(rows[0]) if rows else 0,
            "source"     : file_path,
        }

    else:
        # XLSX: استخدام Docling
        result = converter.convert(file_path)
        raw_text = result.document.export_to_markdown()
        metadata = {"source": file_path}

    return LoadedSource(
        file_path      = file_path,
        source_type    = SourceType.SPREADSHEET,
        file_name      = Path(file_path).name,
        file_extension = ext,
        raw_text       = raw_text,
        metadata       = metadata,
    )


def _load_image(file_path: str, converter: DocumentConverter) -> LoadedSource:
    """تحميل الصور عبر Docling لاستخراج النص (OCR) والوصف"""
    logger.info(f"🖼️ تحميل صورة: {file_path}")

    result = converter.convert(file_path)
    raw_text = result.document.export_to_markdown()

    metadata = {
        "source": file_path,
    }

    return LoadedSource(
        file_path         = file_path,
        source_type       = SourceType.IMAGE,
        file_name         = Path(file_path).name,
        file_extension    = Path(file_path).suffix.lower(),
        raw_text          = raw_text,
        conversion_result = result,
        metadata          = metadata,
    )


# ============================================================
# الدالة الرئيسية - DataSourceLoader
# ============================================================

class DataSourceLoader:
    """
    محمّل موحد لجميع أنواع المصادر
    
    مثال الاستخدام:
        loader = DataSourceLoader()
        result = loader.load("report.pdf")
        print(result.raw_text)
    """

    def __init__(self, fast_mode: bool = False):
        """
        Args:
            fast_mode: True = بدون OCR/جداول (أسرع للرفع الفوري)
                       False = كامل الدقة (للـ ingest_all اليدوي)
        """
        self.fast_mode = fast_mode
        self.converter = _build_fast_converter() if fast_mode else _build_converter()
        mode_label = "⚡ سريع" if fast_mode else "🎯 كامل"
        logger.info(f"🚀 DataSourceLoader جاهز | وضع: {mode_label}")

    def load(self, file_path: str) -> LoadedSource:
        """
        تحميل أي ملف تلقائياً بناءً على نوعه
        
        Args:
            file_path: المسار الكامل للملف
            
        Returns:
            LoadedSource: النتيجة الموحدة تحتوي على النص والبيانات الوصفية
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"الملف غير موجود: {file_path}")

        source_type = detect_source_type(file_path)
        logger.info(f"📁 النوع المكتشف: {source_type.value} ← {path.name}")

        if source_type == SourceType.DOCUMENT:
            return _load_document(file_path, self.converter)

        elif source_type == SourceType.CODE:
            return _load_code(file_path)

        elif source_type == SourceType.SPREADSHEET:
            return _load_spreadsheet(file_path, self.converter)

        elif source_type == SourceType.IMAGE:
            return _load_image(file_path, self.converter)

    def load_directory(self, dir_path: str) -> list[LoadedSource]:
        """
        تحميل جميع الملفات في مجلد بشكل تلقائي
        
        Args:
            dir_path: مسار المجلد
            
        Returns:
            list[LoadedSource]: قائمة بجميع المستندات المحملة
        """
        directory = Path(dir_path)

        if not directory.is_dir():
            raise NotADirectoryError(f"المجلد غير موجود: {dir_path}")

        all_extensions = (
            DOCUMENT_EXTENSIONS |
            CODE_EXTENSIONS     |
            SPREADSHEET_EXTENSIONS |
            IMAGE_EXTENSIONS
        )

        files = [f for f in directory.rglob("*") if f.suffix.lower() in all_extensions]
        logger.info(f"📂 وجدت {len(files)} ملف في {dir_path}")

        results = []
        for file in files:
            try:
                loaded = self.load(str(file))
                results.append(loaded)
                logger.success(f"✅ تم تحميل: {file.name}")
            except Exception as e:
                logger.error(f"❌ فشل تحميل {file.name}: {e}")

        logger.info(f"✅ اكتمل التحميل: {len(results)}/{len(files)} ملف")
        return results