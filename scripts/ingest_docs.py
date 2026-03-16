# scripts/ingest_docs.py
# يحمّل جميع المستندات من مجلد All_Invoices_Files
# ويخزنها في قاعدة البيانات مع متجهاتها وبياناتها الوصفية
# ============================================================

import sys
sys.path.append(".")

from dotenv import load_dotenv
load_dotenv()

import os
import json
import time
from pathlib import Path
from loguru import logger

from src.data_sources.loader import DataSourceLoader
from src.data_processing.restructuring.document_parser import DocumentParser
from src.data_processing.chunking.boundary_detector import BoundaryDetector
from src.data_processing.metadata.summary_generator import SummaryGenerator
from src.data_processing.metadata.keyword_extractor import KeywordExtractor
from src.database.db_setup import setup_database, insert_chunks, register_source
from src.data_processing.arabic_enhancer import enhance_chunks_batch, normalize_arabic_text
from src.database.vector_store import VectorStore
from src.database.relational_db import RelationalDB


# ============================================================
# الإعدادات
# ============================================================

DOCS_FOLDER  = "All_Invoices_Files"  # مجلد يحتوي على المستندات

# نستوعب هذه الأنواع فقط — نتجنب التكرار و .txt لا يدعمها Docling
ALLOWED_EXTS = {".pdf", ".png", ".jpg", ".docx"}

# انتظار بين كل ملف (ثانية) لتجنب Rate Limit
SLEEP_BETWEEN = 5

# ⚙️ تحسين الأداء - عطّل الميزات البطيئة
ENABLE_ARABIC_ENHANCER = True   # ✅ مفعّل - ضروري للنص العربي (يأخذ 3-5 دقائق)
ENABLE_SUMMARY_GEN     = True   # توليد الملخصات
ENABLE_KEYWORD_EXT     = True   # استخراج الكلمات المفتاحية

# 🛠️ معالجة الأخطاء
SKIP_ON_ENHANCER_ERROR = False  # إذا True: تخطى الملف إذا فشل enhancer | إذا False: أكمل بدون enhancer
SKIP_ON_OCR_ERROR = False       # إذا True: تخطى الملف إذا فشل OCR | إذا False: أكمل بدون OCR


# ============================================================
# الدالة الرئيسية
# ============================================================

def ingest_all(path_override: str = None):
    """
    إدخال جميع المستندات من المجلد أو ملف مفرد
    
    Args:
        path_override: مسار بديل (ملف أو مجلد) بدلاً من DOCS_FOLDER
    """
    logger.info("🚀 بدء عملية إدخال المستندات...")

    # 1. إعداد قاعدة البيانات
    logger.info("⚙️ إعداد قاعدة البيانات...")
    setup_database()

    # 2. تهيئة المكونات
    loader     = DataSourceLoader()
    parser     = DocumentParser()
    chunker    = BoundaryDetector()
    summarizer = SummaryGenerator()
    extractor  = KeywordExtractor()
    store      = VectorStore()
    db         = RelationalDB()

    # 3. جمع الملفات (ملف واحد أو مجلد كامل)
    target_path = Path(path_override or DOCS_FOLDER)
    
    if target_path.is_file():
        # ملف مفرد
        if target_path.suffix.lower() in ALLOWED_EXTS:
            files = [target_path]
            logger.info(f"📄 معالجة ملف واحد: {target_path.name}")
        else:
            logger.error(f"❌ نوع ملف غير مدعوم: {target_path.suffix}")
            return
    elif target_path.is_dir():
        # مجلد كامل
        files = [
            f for f in target_path.iterdir()
            if f.suffix.lower() in ALLOWED_EXTS
        ]
        logger.info(f"📁 وجدت {len(files)} ملف في {target_path}")
    else:
        logger.error(f"❌ المسار غير موجود: {target_path}")
        return

    success_count = 0
    skip_count    = 0
    error_count   = 0

    for file_path in sorted(files):

        file_name = file_path.name
        logger.info(f"\n{'='*50}")
        logger.info(f"📄 معالجة: {file_name}")

        # تخطي الملفات المُدخلة مسبقاً
        if db.source_exists(file_name):
            logger.info(f"  ⏭️ تم إدخاله مسبقاً — تخطي")
            skip_count += 1
            continue
        
        # DEBUG: عرض جميع الملفات في قاعدة البيانات
        logger.debug(f"  🔍 التحقق من: {file_name}")

        try:
            # --- تحميل المستند ---
            logger.info(f"  1️⃣  تحميل المستند...")
            try:
                source = loader.load(str(file_path))
            except KeyboardInterrupt:
                logger.warning(f"  ⚠️ تم إلغاء الـ OCR (KeyboardInterrupt)")
                if SKIP_ON_OCR_ERROR:
                    skip_count += 1
                    continue
                logger.info(f"  ⚡ متابعة بدون OCR")
                continue
            except Exception as ocr_err:
                logger.error(f"  ❌ فشل الـ OCR: {ocr_err}")
                if SKIP_ON_OCR_ERROR:
                    skip_count += 1
                    continue
                logger.info(f"  ⚡ متابعة بدون OCR")
                continue

            if not source.conversion_result:
                logger.warning(f"  ⚠️ لا يوجد conversion_result — تخطي")
                skip_count += 1
                continue

            # --- تحليل المستند ---
            logger.info(f"  2️⃣  تحليل الهيكل...")
            # إذا كان النص من OCR (يشمل الكتابة اليدوية) أو نص رقمي، مرره كنص خام
            ocr_method = source.metadata.get("ocr_method", "")
            if ocr_method in ["gemini_vision", "hybrid", "digital", "handwriting_ocr"]:
                parsed = parser.parse(source.raw_text, file_name=file_name)
            else:
                parsed = parser.parse(source.conversion_result, file_name=file_name)
            logger.info(f"     {parsed.summary()}")

            # --- التجزئة ---
            logger.info(f"  3️⃣  تجزئة النص...")
            chunks = chunker.chunk(parsed)
            logger.info(f"     {len(chunks)} قطعة")

            if not chunks:
                logger.warning(f"  ⚠️ لا توجد قطع — تخطي")
                skip_count += 1
                continue

            # --- 3.5: تحسين النص العربي (NFKC + تطبيع) ---
            logger.info(f"  3️⃣½ تطبيع النص العربي...")
            if ENABLE_ARABIC_ENHANCER:
                try:
                    chunks = enhance_chunks_batch(chunks)
                    logger.info(f"     ✅ تم التطبيع")
                except KeyboardInterrupt:
                    logger.warning(f"     ⚠️ تم إلغاء التطبيع (KeyboardInterrupt)")
                    if SKIP_ON_ENHANCER_ERROR:
                        logger.info(f"  ⏭️ تخطى الملف")
                        skip_count += 1
                        continue
                    logger.info(f"     ⚡ متابعة بدون تطبيع")
                except Exception as enh_err:
                    logger.error(f"     ❌ تطبيع النص فشل: {enh_err}")
                    if SKIP_ON_ENHANCER_ERROR:
                        logger.info(f"  ⏭️ تخطى الملف")
                        skip_count += 1
                        continue
                    logger.info(f"     ⚡ متابعة بدون تطبيع")
            else:
                logger.info(f"     ⚡ تخطى التطبيع (يوفر 3-5 دقائق)")

            # --- توليد الملخصات والكلمات المفتاحية ---
            logger.info(f"  4️⃣  توليد الميتاداتا...")
            if ENABLE_SUMMARY_GEN:
                summaries = summarizer.summarize_batch(chunks)
            else:
                summaries = []
                logger.info(f"     ⚡ تخطى الملخصات")
            
            if ENABLE_KEYWORD_EXT:
                keywords  = extractor.extract_batch(chunks)
            else:
                keywords = []
                logger.info(f"     ⚡ تخطى الكلمات المفتاحية")

            # بناء قاموس الميتاداتا
            meta_map = {}
            for s in summaries:
                meta_map.setdefault(s.chunk_id, {})["summary"] = s.summary
            for k in keywords:
                meta_map.setdefault(k.chunk_id, {})["keywords"] = k.keywords

            # --- توليد المتجهات ---
            logger.info(f"  5️⃣  توليد المتجهات...")
            texts      = [c.content for c in chunks]
            embeddings = store.embed_batch(texts)

            # --- حفظ في قاعدة البيانات ---
            logger.info(f"  6️⃣  حفظ في قاعدة البيانات...")
            metadata_list = [
                meta_map.get(c.chunk_id, {})
                for c in chunks
            ]

            insert_chunks(chunks, embeddings, metadata_list)
            register_source(
                file_name   = file_name,
                file_path   = str(file_path),
                source_type = source.source_type.value,
                num_chunks  = len(chunks),
            )

            success_count += 1
            logger.success(f"  ✅ {file_name} → {len(chunks)} قطعة")

        except Exception as e:
            logger.error(f"  ❌ فشل معالجة {file_name}: {e}")
            error_count += 1
            continue

        # انتظار بين الملفات لتجنب Rate Limit
        logger.info(f"  ⏳ انتظار {SLEEP_BETWEEN}s قبل الملف التالي...")
        time.sleep(SLEEP_BETWEEN)

    # --- ملخص النهاية ---
    logger.info(f"\n{'='*50}")
    logger.success(f"✅ اكتملت عملية الإدخال!")
    logger.info(f"  نجح   : {success_count} ملف")
    logger.info(f"  تخطّى : {skip_count} ملف")
    logger.info(f"  فشل   : {error_count} ملف")

    # حالة قاعدة البيانات
    from src.database.db_setup import check_database_status
    status = check_database_status()
    logger.info(f"\n📊 قاعدة البيانات:")
    logger.info(f"  إجمالي القطع   : {status['total_chunks']}")
    logger.info(f"  إجمالي المصادر : {status['total_sources']}")
    logger.info(f"  قطع مع متجهات  : {status['chunks_with_embeddings']}")


def ingest_single(file_path: Path):
    """
    إدخال ملف واحد — كامل الإجراءات مثل ingest_all
    يُستدعى من api/main.py عبر subprocess --file <path>

    الإجراءات:
    1️⃣  تحميل المستند (Docling)
    2️⃣  تحليل الهيكل (عناوين، جداول، صور)
    3️⃣  تجزئة النص (Boundary Detection)
    3.5 تحسين النص العربي (NFKC Normalization)
    4️⃣  توليد الميتاداتا (ملخصات + كلمات مفتاحية)
    5️⃣  توليد المتجهات (Gemini Embeddings)
    6️⃣  حفظ في قاعدة البيانات
    """
    logger.info(f"🚀 بدء إدخال: {file_path.name}")
    logger.info(f"{'='*50}")

    if not file_path.exists():
        logger.error(f"❌ الملف غير موجود: {file_path}")
        return 0

    if file_path.suffix.lower() not in ALLOWED_EXTS:
        logger.warning(f"⚠️ امتداد غير مدعوم: {file_path.suffix}")
        return 0

    setup_database()

    # وضع ذكي: صغير < 500KB → سريع | كبير/ممسوح → كامل OCR
    file_size_kb = file_path.stat().st_size / 1024
    use_fast = file_size_kb < 500 and file_path.suffix.lower() == '.pdf'
    loader     = DataSourceLoader(fast_mode=use_fast)
    parser     = DocumentParser()
    chunker    = BoundaryDetector()
    summarizer = SummaryGenerator()
    extractor  = KeywordExtractor()
    store      = VectorStore()
    db         = RelationalDB()
    file_name  = file_path.name

    logger.info(f"  📋 الملف: {file_name}")
    logger.info(f"  📦 الحجم: {file_size_kb:.1f} KB")
    logger.info(f"  ⚙️  الوضع: {'⚡ سريع (بدون OCR)' if use_fast else '🎯 كامل (مع OCR)'}")
    logger.info(f"{'='*50}")

    try:
        # ══ 1. تحميل ══
        logger.info(f"  1️⃣  تحميل المستند...")
        source = loader.load(str(file_path))

        if not source or not source.conversion_result:
            logger.warning(f"  ⚠️ لا يوجد conversion_result")
            return 0

        raw_text_len = len(source.raw_text or "")
        logger.info(f"     ✓ تم التحميل | نص خام: {raw_text_len:,} حرف")

        # ══ 2. تحليل الهيكل ══
        logger.info(f"  2️⃣  تحليل الهيكل...")
        # إذا كان النص من OCR (يشمل الكتابة اليدوية) أو نص رقمي، مرره كنص خام
        ocr_method = source.metadata.get("ocr_method", "")
        if ocr_method in ["gemini_vision", "hybrid", "digital", "handwriting_ocr"]:
            parsed = parser.parse(source.raw_text, file_name=file_name)
        else:
            parsed = parser.parse(source.conversion_result, file_name=file_name)
        logger.info(f"     ✓ {parsed.summary()}")

        # ══ 3. تجزئة ══
        logger.info(f"  3️⃣  تجزئة النص...")
        chunks = chunker.chunk(parsed)
        logger.info(f"     ✓ {len(chunks)} قطعة ({sum(1 for c in chunks if c.chunk_type=='table')} جداول)")

        if not chunks:
            logger.warning(f"  ⚠️ لا توجد قطع")
            return 0

        # ══ 3.5. تحسين النص العربي ══
        logger.info(f"  3️⃣½ تحسين النص العربي (NFKC)...")
        if ENABLE_ARABIC_ENHANCER:
            try:
                from src.data_processing.arabic_enhancer import enhance_chunks_batch
                chunks = enhance_chunks_batch(chunks)
                logger.info(f"     ✅ تم التطبيع")
            except KeyboardInterrupt:
                logger.warning(f"     ⚠️ تم إلغاء التطبيع — متابعة بدون تطبيع")
            except Exception as ar_e:
                logger.warning(f"     ⚠️ arabic_enhancer: {ar_e}")
                logger.info(f"     ⚡ متابعة بدون تطبيع")
        else:
            logger.info(f"     ⚡ تخطى التطبيع (يوفر 3-5 دقائق)")

        # ══ 4. ميتاداتا ══
        logger.info(f"  4️⃣  توليد الميتاداتا ({len(chunks)} قطعة)...")
        try:
            if ENABLE_SUMMARY_GEN:
                summaries = summarizer.summarize_batch(chunks)
            else:
                summaries = []
                logger.info(f"     ⚡ تخطى الملخصات")
            
            if ENABLE_KEYWORD_EXT:
                keywords  = extractor.extract_batch(chunks)
            else:
                keywords = []
                logger.info(f"     ⚡ تخطى الكلمات المفتاحية")

            meta_map = {}
            for s in summaries:
                meta_map.setdefault(s.chunk_id, {})["summary"] = s.summary
            for k in keywords:
                meta_map.setdefault(k.chunk_id, {})["keywords"] = k.keywords

            logger.info(f"     ✓ ملخصات: {len(summaries)} | كلمات: {len(keywords)}")
        except Exception as meta_e:
            logger.warning(f"     ⚠️ ميتاداتا جزئية: {meta_e}")
            meta_map = {}

        # ══ 5. متجهات ══
        logger.info(f"  5️⃣  توليد المتجهات (Gemini Embeddings)...")
        texts      = [chunk.content for chunk in chunks]
        embeddings = store.embed_batch(texts)
        logger.info(f"     ✓ {len(embeddings)} متجه (3072 بُعد)")

        # ══ 6. حفظ ══
        logger.info(f"  6️⃣  حفظ في قاعدة البيانات...")
        metadata_list = [meta_map.get(c.chunk_id, {}) for c in chunks]

        insert_chunks(chunks, embeddings, metadata_list)
        register_source(
            file_name   = file_name,
            file_path   = str(file_path),
            source_type = source.source_type.value,
            num_chunks  = len(chunks),
        )

        logger.info(f"{'='*50}")
        logger.success(f"  ✅ اكتمل: {file_name}")
        logger.info(f"     قطع    : {len(chunks)}")
        logger.info(f"     نصية   : {sum(1 for c in chunks if c.chunk_type != 'table')}")
        logger.info(f"     جداول  : {sum(1 for c in chunks if c.chunk_type == 'table')}")
        logger.info(f"     متجهات : {len(embeddings)}")
        logger.info(f"{'='*50}")
        print(f"CHUNKS:{len(chunks)}")
        return len(chunks)

    except Exception as e:
        logger.error(f"  ❌ فشل: {e}")
        import traceback; traceback.print_exc()
        return 0



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="مسار ملف واحد أو مجلد كامل")
    args = ap.parse_args()

    if args.file:
        # إدخال ملف واحد أو مجلد
        ingest_all(path_override=args.file)
    else:
        # إدخال المجلد الافتراضي
        ingest_all()