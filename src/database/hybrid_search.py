# ============================================================
# database/hybrid_search.py
# البحث الهجين: BM25 (نصي) + Vector (دلالي) + RRF (دمج الرتب)
# ============================================================

import os
from dataclasses import dataclass, field
from loguru import logger

from src.database.vector_store import VectorStore
from src.database.db_setup import get_connection

# استخدم معالج التصريف العربي لتحسين البحث
try:
    from src.data_processing.arabic_lemmatizer import get_lemmatizer
    HAS_LEMMATIZER = True
except ImportError:
    HAS_LEMMATIZER = False
    logger.warning("⚠️ Arabic Lemmatizer غير متاح")


# ============================================================
# نتيجة البحث
# ============================================================

@dataclass
class SearchResult:
    """نتيجة بحث واحدة مع درجاتها"""
    chunk_id        : str
    content         : str
    source_file     : str
    page_number     : int   = 0
    parent_heading  : str   = ""
    summary         : str   = ""
    keywords        : list  = field(default_factory=list)

    # درجات البحث
    vector_score    : float = 0.0   # درجة البحث المتجهي
    bm25_score      : float = 0.0   # درجة البحث النصي
    rrf_score       : float = 0.0   # الدرجة المدمجة (RRF)

    metadata        : dict  = field(default_factory=dict)

    def __repr__(self):
        return (
            f"[{self.chunk_id}] RRF={self.rrf_score:.4f} | "
            f"Vector={self.vector_score:.3f} | BM25={self.bm25_score:.3f} | "
            f"{self.content[:60]}..."
        )


# ============================================================
# محرك البحث الهجين
# ============================================================

class HybridSearch:
    """
    يجمع بين البحث المتجهي والبحث النصي
    باستخدام خوارزمية RRF (Reciprocal Rank Fusion)

    المعادلة:
        score(d) = Σ 1 / (k + rank(r, d))
        حيث k = 60 (ثابت التنعيم)

    مثال الاستخدام:
        search = HybridSearch()
        results = search.search("ما هو إجمالي الفاتورة؟", top_k=5)
        for r in results:
            print(r)
    """

    def __init__(self, rrf_k: int = 60):
        """
        Args:
            rrf_k: ثابت التنعيم في معادلة RRF (الافتراضي 60)
        """
        self.rrf_k        = rrf_k
        self.vector_store = VectorStore()
        
        # استخدم Arabic Lemmatizer لتحسين BM25
        self.lemmatizer = None
        if HAS_LEMMATIZER:
            try:
                self.lemmatizer = get_lemmatizer()
            except Exception as e:
                logger.warning(f"⚠️ فشل تحميل Lemmatizer: {e}")
        
        logger.info(f"✅ HybridSearch جاهز | RRF k={rrf_k} | Lemmatizer={'✓' if self.lemmatizer else '✗'}")

    # ============================================================
    # البحث الرئيسي
    # ============================================================

    def search(
        self,
        query          : str,
        top_k          : int   = 5,
        vector_weight  : float = 0.5,
        bm25_weight    : float = 0.5,
        source_file    : str   = None,
    ) -> list[SearchResult]:
        """
        بحث هجين يجمع المتجهي والنصي

        Args:
            query         : نص الاستعلام
            top_k         : عدد النتائج النهائية
            vector_weight : وزن البحث المتجهي (0.0 - 1.0)
            bm25_weight   : وزن البحث النصي (0.0 - 1.0)
            source_file   : تصفية حسب ملف محدد (اختياري)

        Returns:
            list[SearchResult]: النتائج مرتبة بـ RRF
        """
        fetch_k = top_k * 3   # نجلب أكثر ثم نرتب

        # --- 1. البحث المتجهي ---
        vector_results = self._vector_search(query, fetch_k, source_file)

        # --- 2. البحث النصي BM25 ---
        bm25_results = self._bm25_search(query, fetch_k, source_file)

        # --- 3. دمج الرتب RRF ---
        merged = self._reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            vector_weight,
            bm25_weight,
        )

        # --- 4. إرجاع أفضل top_k نتيجة ---
        final = sorted(merged.values(), key=lambda x: x.rrf_score, reverse=True)[:top_k]

        logger.info(
            f"🔍 بحث هجين: '{query[:50]}' → "
            f"{len(final)} نتيجة | "
            f"متجهي: {len(vector_results)} | نصي: {len(bm25_results)}"
        )

        return final

    # ============================================================
    # البحث المتجهي
    # ============================================================

    def _vector_search(
        self,
        query      : str,
        top_k      : int,
        source_file: str = None,
    ) -> list[dict]:
        """بحث متجهي مع فلتر اختياري حسب المصدر"""
        query_embedding = self.vector_store.embed_query(query)
        conn = get_connection()
        cur  = conn.cursor()

        try:
            if source_file:
                cur.execute("""
                    SELECT
                        chunk_id, content, source_file, page_number,
                        parent_heading, summary, keywords, metadata,
                        1 - (embedding <=> %s::vector) AS score
                    FROM doc_chunks
                    WHERE embedding IS NOT NULL AND source_file = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, source_file, query_embedding, top_k))
            else:
                cur.execute("""
                    SELECT
                        chunk_id, content, source_file, page_number,
                        parent_heading, summary, keywords, metadata,
                        1 - (embedding <=> %s::vector) AS score
                    FROM doc_chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, top_k))

            rows = cur.fetchall()
            return [
                {
                    "chunk_id"      : r[0],
                    "content"       : r[1],
                    "source_file"   : r[2],
                    "page_number"   : r[3],
                    "parent_heading": r[4],
                    "summary"       : r[5],
                    "keywords"      : r[6] or [],
                    "metadata"      : r[7] or {},
                    "vector_score"  : float(r[8]),
                }
                for r in rows
            ]
        finally:
            cur.close()
            conn.close()

    # ============================================================
    # البحث النصي BM25
    # ============================================================

    def _bm25_search(
        self,
        query      : str,
        top_k      : int,
        source_file: str = None,
    ) -> list[dict]:
        """
        بحث نصي ثنائي اللغة (عربي + إنجليزي)
        المرحلة 1: tsvector بـ 'simple' (يعمل مع العربية والإنجليزية)
        المرحلة 2: ILIKE fallback للكلمات الفردية إذا فشل tsvector
        """
        conn = get_connection()
        cur  = conn.cursor()

        try:
            # ── تنظيف الاستعلام ──
            import re
            # أزل كلمات الاستفهام العربية الشائعة
            stop_ar = {"وش","ايش","شو","ما","من","متى","هل","كيف","كم","اين","أين"}
            words   = [w.strip("؟?.,!:") for w in query.split() if len(w.strip("؟?.,!:")) > 1]
            words   = [w for w in words if w not in stop_ar]

            if not words:
                return []
            
            # ✨ استخدم Arabic Lemmatizer لتحسين البحث
            if self.lemmatizer:
                lemmatized_words = []
                for word in words:
                    root = self.lemmatizer.get_root(word)
                    if root and root != word:
                        lemmatized_words.append(root)
                    lemmatized_words.append(word)  # احفظ الأصلي أيضاً
                # استخدم كلمات + جذورها للبحث
                enhanced_words = list(dict.fromkeys(lemmatized_words))[:15]  # أقصى 15 كلمة
            else:
                enhanced_words = words

            src_filter = "AND source_file = %s" if source_file else ""

            # ── المرحلة 1: tsvector 'simple' مع كلمات معززة ──
            tsq = " & ".join(enhanced_words)
            try:
                if source_file:
                    cur.execute(f"""
                        SELECT chunk_id, content, source_file, page_number,
                               parent_heading, summary, keywords, metadata,
                               ts_rank(fts_vector,
                                   to_tsquery('simple', %s)) AS score
                        FROM   doc_chunks
                        WHERE  fts_vector @@ to_tsquery('simple', %s)
                          AND  source_file = %s
                        ORDER  BY score DESC
                        LIMIT  %s
                    """, (tsq, tsq, source_file, top_k))
                else:
                    cur.execute("""
                        SELECT chunk_id, content, source_file, page_number,
                               parent_heading, summary, keywords, metadata,
                               ts_rank(fts_vector,
                                   to_tsquery('simple', %s)) AS score
                        FROM   doc_chunks
                        WHERE  fts_vector @@ to_tsquery('simple', %s)
                        ORDER  BY score DESC
                        LIMIT  %s
                    """, (tsq, tsq, top_k))
                rows = cur.fetchall()
            except Exception:
                rows = []

            # ── المرحلة 2: ILIKE fallback إذا لا نتائج ──
            if not rows:
                like_conditions = " OR ".join(["content ILIKE %s"] * len(enhanced_words))
                like_params     = [f"%{w}%" for w in enhanced_words]
                if source_file:
                    cur.execute(f"""
                        SELECT chunk_id, content, source_file, page_number,
                               parent_heading, summary, keywords, metadata,
                               0.5 AS score
                        FROM   doc_chunks
                        WHERE  ({like_conditions})
                          AND  source_file = %s
                        ORDER  BY chunk_id
                        LIMIT  %s
                    """, like_params + [source_file, top_k])
                else:
                    cur.execute(f"""
                        SELECT chunk_id, content, source_file, page_number,
                               parent_heading, summary, keywords, metadata,
                               0.5 AS score
                        FROM   doc_chunks
                        WHERE  ({like_conditions})
                        ORDER  BY chunk_id
                        LIMIT  %s
                    """, like_params + [top_k])
                rows = cur.fetchall()
                if rows:
                    logger.debug(f"  📝 BM25 ILIKE fallback: {len(rows)} نتيجة")

            return [
                {
                    "chunk_id"      : r[0],
                    "content"       : r[1],
                    "source_file"   : r[2],
                    "page_number"   : r[3],
                    "parent_heading": r[4],
                    "summary"       : r[5],
                    "keywords"      : r[6] or [],
                    "metadata"      : r[7] or {},
                    "bm25_score"    : float(r[8]),
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"⚠️ BM25 فشل: {e}")
            return []
        finally:
            cur.close()
            conn.close()

    # ============================================================
    # دمج الرتب المتبادلة (RRF)
    # ============================================================

    def _reciprocal_rank_fusion(
        self,
        vector_results : list[dict],
        bm25_results   : list[dict],
        vector_weight  : float,
        bm25_weight    : float,
    ) -> dict[str, SearchResult]:
        """
        دمج نتائج البحثين باستخدام RRF مع تحسينات إضافية

        التحسينات:
        1. تطبيع درجات VectorScore إلى 0-1
        2. تطبيع درجات BM25 إلى 0-1
        3. إضافة bonus score للنتائج التي تظهر في القائمتين
        4. خصم النتائج بدون محتوى ذي معنى

        المعادلة: score(d) = Σ weight / (k + rank) + bonus
        """
        merged: dict[str, SearchResult] = {}

        # --- تطبيع درجات البحث المتجهي ---
        vector_scores = [item.get("vector_score", 0.0) for item in vector_results]
        max_vector = max(vector_scores) if vector_scores else 1.0
        min_vector = min(vector_scores) if vector_scores else 0.0
        vector_range = max_vector - min_vector if max_vector != min_vector else 1.0

        # --- تطبيع درجات BM25 ---
        bm25_scores = [item.get("bm25_score", 0.0) for item in bm25_results]
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        min_bm25 = min(bm25_scores) if bm25_scores else 0.0
        bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0

        # --- معالجة نتائج البحث المتجهي ---
        for rank, item in enumerate(vector_results, start=1):
            cid   = item["chunk_id"]
            
            # تطبيع الدرجة
            raw_score = item.get("vector_score", 0.0)
            norm_score = (raw_score - min_vector) / vector_range if vector_range > 0 else 0.0
            
            # حساب درجة RRF
            score = vector_weight / (self.rrf_k + rank)
            
            # إضافة bonus للدرجة الطبيعية
            score += norm_score * 0.1  # 10% وزن للدرجة الطبيعية

            if cid not in merged:
                merged[cid] = SearchResult(
                    chunk_id       = cid,
                    content        = item["content"],
                    source_file    = item["source_file"],
                    page_number    = item["page_number"],
                    parent_heading = item["parent_heading"],
                    summary        = item["summary"],
                    keywords       = item["keywords"],
                    metadata       = item["metadata"],
                    vector_score   = float(raw_score),
                )
            else:
                # موجود مسبقاً → إضافة bonus
                merged[cid].rrf_score += score * 0.5  # 50% bonus للظهور في القائمتين
                merged[cid].vector_score = float(raw_score)
            
            merged[cid].rrf_score = score

        # --- معالجة نتائج البحث النصي ---
        for rank, item in enumerate(bm25_results, start=1):
            cid   = item["chunk_id"]
            
            # تطبيع الدرجة
            raw_score = item.get("bm25_score", 0.0)
            norm_score = (raw_score - min_bm25) / bm25_range if bm25_range > 0 else 0.0
            
            # حساب درجة RRF
            score = bm25_weight / (self.rrf_k + rank)
            
            # إضافة bonus للدرجة الطبيعية
            score += norm_score * 0.1

            if cid not in merged:
                merged[cid] = SearchResult(
                    chunk_id       = cid,
                    content        = item["content"],
                    source_file    = item["source_file"],
                    page_number    = item["page_number"],
                    parent_heading = item["parent_heading"],
                    summary        = item["summary"],
                    keywords       = item["keywords"],
                    metadata       = item["metadata"],
                    bm25_score     = float(raw_score),
                )
            else:
                # موجود مسبقاً → إضافة bonus كبير
                merged[cid].rrf_score += score * 0.5  # Bonus للظهور في القائمتين
                merged[cid].rrf_score += norm_score * 0.2  # 20% وزن للدرجة الطبيعية
                merged[cid].bm25_score = float(raw_score)
            
            merged[cid].rrf_score += score

        return merged