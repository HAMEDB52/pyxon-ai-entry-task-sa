# ============================================================
# database/db_setup.py
# إعداد قاعدة البيانات PostgreSQL + pgvector على Neon
# ينشئ الجداول والفهارس اللازمة للبحث الهجين
# ============================================================

import os
import json
from loguru import logger
import psycopg2
from psycopg2.extras import execute_values


# ============================================================
# الاتصال بقاعدة البيانات
# ============================================================

def get_connection():
    """
    إنشاء اتصال بقاعدة بيانات Neon PostgreSQL

    Returns:
        psycopg2 connection object
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("❌ DATABASE_URL غير موجود في .env")

    try:
        conn = psycopg2.connect(database_url)
        logger.info("✅ اتصال ناجح بقاعدة البيانات")
        return conn
    except Exception as e:
        logger.error(f"❌ فشل الاتصال بقاعدة البيانات: {e}")
        raise


# ============================================================
# إنشاء الجداول والفهارس
# ============================================================

def setup_database():
    """
    إنشاء جميع الجداول والفهارس اللازمة

    الجداول:
        - doc_chunks : القطع النصية مع المتجهات والبحث النصي
        - doc_sources: مصادر المستندات المُدخلة
    """
    conn = get_connection()
    cur  = conn.cursor()

    try:
        # --- تفعيل ملحق pgvector ---
        logger.info("⚙️ تفعيل ملحق pgvector...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # --- جدول مصادر المستندات ---
        logger.info("📋 إنشاء جدول doc_sources...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS doc_sources (
                id           SERIAL PRIMARY KEY,
                file_name    TEXT NOT NULL,
                file_path    TEXT,
                source_type  TEXT,
                num_chunks   INTEGER DEFAULT 0,
                ingested_at  TIMESTAMP DEFAULT NOW(),
                metadata     JSONB DEFAULT '{}'
            );
        """)

        # --- جدول القطع الرئيسي ---
        logger.info("📋 إنشاء جدول doc_chunks...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS doc_chunks (
                id             SERIAL PRIMARY KEY,
                chunk_id       TEXT UNIQUE NOT NULL,
                source_id      INTEGER REFERENCES doc_sources(id),
                content        TEXT NOT NULL,
                chunk_type     TEXT DEFAULT 'text',
                source_file    TEXT,
                page_number    INTEGER DEFAULT 0,
                token_count    INTEGER DEFAULT 0,

                -- البحث المتجهي
                embedding      vector(768),

                -- البحث النصي (BM25) - يدعم العربية والإنجليزية
                fts_vector     tsvector GENERATED ALWAYS AS (
                    to_tsvector('simple',
                        coalesce(content,'') || ' ' ||
                        coalesce(summary, '') || ' ' ||
                        coalesce(parent_heading, '') || ' ' ||
                        coalesce(array_to_string(keywords, ' '), '')
                    )
                ) STORED,

                -- البيانات الوصفية
                parent_heading TEXT,
                summary        TEXT,
                keywords       TEXT[],
                questions      TEXT[],
                metadata       JSONB DEFAULT '{}',

                created_at     TIMESTAMP DEFAULT NOW()
            );
        """)

        # --- فهرس المتجهات (HNSW - الأسرع للبحث) ---
        # ملاحظة: HNSW في pgvector يدعم حتى 2000 بُعد فقط.
        # إذا كان البعد أكبر، نتجنب إنشاء الفهرس لتفادي الخطأ.
        logger.info("🔍 إنشاء فهرس المتجهات HNSW...")
        cur.execute("""
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'doc_chunks'::regclass
              AND attname  = 'embedding';
        """)
        row = cur.fetchone()
        # في pgvector: البُعد = atttypmod - 4 (عند وجود قيمة)
        dims = (row[0] - 4) if row and row[0] and row[0] > 4 else 768
        if dims > 2000:
            logger.warning(
                f"⚠️ تم تخطي فهرس HNSW لأن عدد الأبعاد ({dims}) أكبر من 2000."
            )
        else:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                ON doc_chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)

        # --- فهرس البحث النصي ---
        logger.info("🔍 إنشاء فهرس البحث النصي GIN...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_fts
            ON doc_chunks
            USING gin (fts_vector);
        """)

        # --- فهرس chunk_id ---
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id
            ON doc_chunks (chunk_id);
        """)

        # --- فهرس source_file ---
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_source_file
            ON doc_chunks (source_file);
        """)

        conn.commit()
        logger.success("✅ تم إعداد قاعدة البيانات بنجاح")

        # ── Migration: تحديث fts_vector لـ 'simple' إذا كانت 'english' ──
        try:
            cur2 = conn.cursor()
            cur2.execute("""
                SELECT column_default FROM information_schema.columns
                WHERE table_name='doc_chunks' AND column_name='fts_vector'
            """)
            row = cur2.fetchone()
            if row and 'english' in str(row[0]):
                logger.info("🔄 تحديث fts_vector من 'english' → 'simple'...")
                cur2.execute("ALTER TABLE doc_chunks DROP COLUMN fts_vector;")
                cur2.execute("""
                    ALTER TABLE doc_chunks
                    ADD COLUMN fts_vector tsvector GENERATED ALWAYS AS (
                        to_tsvector('simple',
                            coalesce(content,'') || ' ' ||
                            coalesce(metadata->>'keywords','')
                        )
                    ) STORED;
                """)
                cur2.execute("""
                    CREATE INDEX IF NOT EXISTS idx_fts_chunks
                    ON doc_chunks USING gin (fts_vector);
                """)
                conn.commit()
                logger.success("✅ Migration مكتمل — fts_vector يدعم العربية الآن")
            cur2.close()
        except Exception as mig_e:
            logger.debug(f"Migration: {mig_e}")
            try: conn.rollback()
            except: pass

    except Exception as e:
        conn.rollback()
        logger.error(f"❌ خطأ في إعداد قاعدة البيانات: {e}")
        raise
    finally:
        cur.close()
        conn.close()


# ============================================================
# إدراج القطع في قاعدة البيانات
# ============================================================

def insert_chunks(chunks: list, embeddings: list[list[float]], metadata_list: list[dict]) -> int:
    """
    إدراج قطع مع متجهاتها وبياناتها الوصفية

    Args:
        chunks        : قائمة Chunk objects
        embeddings    : قائمة المتجهات بنفس الترتيب
        metadata_list : قائمة البيانات الوصفية الإضافية (ملخص، كلمات، أسئلة)

    Returns:
        int: عدد القطع المُدرجة
    """
    if not chunks:
        return 0

    conn = get_connection()
    cur  = conn.cursor()

    try:
        rows = []
        for chunk, embedding, meta in zip(chunks, embeddings, metadata_list):
            rows.append((
                chunk.chunk_id,
                chunk.content,
                chunk.chunk_type,
                chunk.source_file,
                chunk.page_number,
                chunk.token_count,
                embedding,
                chunk.metadata.get("parent_heading", ""),
                meta.get("summary", ""),
                meta.get("keywords", []),
                meta.get("questions", []),
                json.dumps(chunk.metadata, ensure_ascii=False),
            ))

        execute_values(cur, """
            INSERT INTO doc_chunks (
                chunk_id, content, chunk_type, source_file,
                page_number, token_count, embedding,
                parent_heading, summary, keywords, questions, metadata
            )
            VALUES %s
            ON CONFLICT (chunk_id) DO UPDATE SET
                content        = EXCLUDED.content,
                embedding      = EXCLUDED.embedding,
                summary        = EXCLUDED.summary,
                keywords       = EXCLUDED.keywords,
                questions      = EXCLUDED.questions,
                metadata       = EXCLUDED.metadata
        """, rows)

        conn.commit()
        count = len(rows)
        logger.success(f"✅ تم إدراج {count} قطعة في قاعدة البيانات")
        return count

    except Exception as e:
        conn.rollback()
        logger.error(f"❌ خطأ في إدراج القطع: {e}")
        raise
    finally:
        cur.close()
        conn.close()


# ============================================================
# تسجيل المصدر
# ============================================================

def register_source(file_name: str, file_path: str, source_type: str, num_chunks: int) -> int:
    """
    تسجيل مصدر مستند جديد في قاعدة البيانات

    Returns:
        int: معرف المصدر المُسجَّل
    """
    conn = get_connection()
    cur  = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO doc_sources (file_name, file_path, source_type, num_chunks)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (file_name, file_path, source_type, num_chunks))

        source_id = cur.fetchone()[0]
        conn.commit()
        logger.info(f"✅ تم تسجيل المصدر: {file_name} (id={source_id})")
        return source_id

    except Exception as e:
        conn.rollback()
        logger.error(f"❌ خطأ في تسجيل المصدر: {e}")
        raise
    finally:
        cur.close()
        conn.close()


# ============================================================
# فحص حالة قاعدة البيانات
# ============================================================

def check_database_status() -> dict:
    """
    فحص حالة قاعدة البيانات وإحصائياتها

    Returns:
        dict: إحصائيات قاعدة البيانات
    """
    conn = get_connection()
    cur  = conn.cursor()

    try:
        cur.execute("SELECT COUNT(*) FROM doc_chunks;")
        total_chunks = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM doc_sources;")
        total_sources = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM doc_chunks WHERE embedding IS NOT NULL;")
        chunks_with_embeddings = cur.fetchone()[0]

        status = {
            "total_chunks"          : total_chunks,
            "total_sources"         : total_sources,
            "chunks_with_embeddings": chunks_with_embeddings,
            "status"                : "healthy",
        }

        logger.info(f"📊 حالة DB: {status}")
        return status

    except Exception as e:
        logger.error(f"❌ خطأ في فحص قاعدة البيانات: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        cur.close()
        conn.close()
