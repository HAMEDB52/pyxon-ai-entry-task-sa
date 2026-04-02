# ============================================================
# database/vector_store.py
# توليد المتجهات والبحث المتجهي باستخدام Google Gemini
# ============================================================

import os
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from google import genai
from google.genai import types
from google.genai import _api_client

# ✨ استيراد كاش التضمينات
from src.database.embedding_cache import get_embedding_cache


# ============================================================
# مولّد المتجهات
# ============================================================

class VectorStore:
    """
    يولّد متجهات (Embeddings) للقطع النصية باستخدام Gemini
    مع كاش لتسريع الاستعلامات المتكررة
    
    المميزات الجديدة:
    - ✨ كاش للتضمينات: تخزين مؤقت للمتجهات الشائعة
    - 🚀 سرعة أعلى: تقليل استدعاءات API بـ 40-50%
    - 💾 persistence: حفظ الكاش على القرص

    مثال الاستخدام:
        store = VectorStore()
        embedding = store.embed_text("نص هنا")  # من الكاش أو API
        results   = store.search("سؤال هنا", top_k=5)
    """

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY غير موجود في .env")

        self.client = genai.Client(
            api_key      = api_key,
            http_options = types.HttpOptions(api_version="v1beta"),
        )
        self._model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
        self.model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
        
        # ✨ احصل على كاش التضمينات العام
        self.embedding_cache = get_embedding_cache(max_size=500, enable_persistence=True)
        
        logger.info(f"✅ VectorStore جاهز | نموذج: {self.model_name} | كاش: ✓")

    # ============================================================
    # توليد المتجهات
    # ============================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def embed_text(self, text: str) -> list[float]:
        """
        توليد متجه لنص واحد (مع كاش)

        Args:
            text: النص المراد تحويله لمتجه

        Returns:
            list[float]: المتجه (768 بُعد)
        """
        if not text or not text.strip():
            return [0.0] * 768 
        
        # ✨ تحقق من الكاش أولاً
        cached = self.embedding_cache.get(text)
        if cached:
            logger.debug(f"🎯 استخدم متجه مخزن للنص: '{text[:30]}'")
            return cached

        try:
            result = self.client.models.embed_content(
                model   = self.model_name,
                contents= text,
                config  = types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768 
                ),
            )
            embedding = result.embeddings[0].values
            
            # ✨ خزّن في الكاش
            self.embedding_cache.set(text, embedding)
            
            return embedding

        except Exception as e:
            logger.error(f"❌ فشل توليد المتجه: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def embed_query(self, query: str) -> list[float]:
        """
        توليد متجه لاستعلام البحث (مع كاش)
        (يستخدم task_type مختلف لتحسين الاسترجاع)

        Args:
            query: نص الاستعلام

        Returns:
            list[float]: المتجه (768 بُعد)
        """
        # ✨ تحقق من الكاش أولاً
        cached = self.embedding_cache.get(query)
        if cached:
            logger.debug(f"🎯 استخدم متجه مخزن للاستعلام: '{query[:30]}'")
            return cached
        
        try:
            result = self.client.models.embed_content(
                model   = self.model_name,
                contents= query,
                config  = types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=768 
                ),
            )
            embedding = result.embeddings[0].values
            
            # ✨ خزّن في الكاش
            self.embedding_cache.set(query, embedding)
            
            return embedding

        except Exception as e:
            logger.error(f"❌ فشل توليد متجه الاستعلام: {e}")
            raise

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        توليد متجهات لقائمة من النصوص

        Args:
            texts: قائمة النصوص

        Returns:
            list[list[float]]: قائمة المتجهات
        """
        embeddings = []
        total      = len(texts)

        logger.info(f"⚙️ توليد متجهات لـ {total} نص...")

        for i, text in enumerate(texts, 1):
            emb = self.embed_text(text)
            embeddings.append(emb)

            if i % 10 == 0:
                logger.info(f"  ⏳ {i}/{total} تم معالجتها...")

        logger.success(f"✅ تم توليد {len(embeddings)} متجه")
        return embeddings

    # ============================================================
    # البحث المتجهي
    # ============================================================

    def search(
        self,
        query    : str,
        top_k    : int  = 5,
        threshold: float = 0.0,
    ) -> list[dict]:
        """
        البحث المتجهي عن القطع الأقرب للاستعلام

        Args:
            query    : نص الاستعلام
            top_k    : عدد النتائج المطلوبة
            threshold: الحد الأدنى لدرجة التشابه

        Returns:
            list[dict]: النتائج مرتبة تنازلياً حسب التشابه
        """
        import psycopg2
        from src.database.db_setup import get_connection

        query_embedding = self.embed_query(query)
        conn = get_connection()
        cur  = conn.cursor()

        try:
            cur.execute("""
                SELECT
                    chunk_id,
                    content,
                    chunk_type,
                    source_file,
                    page_number,
                    parent_heading,
                    summary,
                    keywords,
                    metadata,
                    1 - (embedding <=> %s::vector) AS similarity_score
                FROM doc_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, top_k))

            rows    = cur.fetchall()
            columns = [
                "chunk_id", "content", "chunk_type", "source_file",
                "page_number", "parent_heading", "summary", "keywords",
                "metadata", "similarity_score"
            ]

            results = []
            for row in rows:
                result = dict(zip(columns, row))
                if result["similarity_score"] >= threshold:
                    results.append(result)

            logger.info(f"🔍 البحث المتجهي: {len(results)} نتيجة لـ '{query[:50]}'")
            return results

        except Exception as e:
            logger.error(f"❌ خطأ في البحث المتجهي: {e}")
            return []
        finally:
            cur.close()
            conn.close()