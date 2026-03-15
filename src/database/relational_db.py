# ============================================================
# database/relational_db.py
# عمليات CRUD الكاملة على قاعدة البيانات العلائقية
# قراءة، تحديث، حذف، إحصائيات
# ============================================================

import os
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from src.database.db_setup import get_connection


# ============================================================
# نماذج البيانات
# ============================================================

@dataclass
class ChunkRecord:
    """سجل قطعة من قاعدة البيانات"""
    id             : int
    chunk_id       : str
    content        : str
    chunk_type     : str   = "text"
    source_file    : str   = ""
    page_number    : int   = 0
    token_count    : int   = 0
    parent_heading : str   = ""
    summary        : str   = ""
    keywords       : list  = field(default_factory=list)
    questions      : list  = field(default_factory=list)
    metadata       : dict  = field(default_factory=dict)

    def __repr__(self):
        return f"[{self.chunk_id}] {self.source_file} | {self.content[:60]}..."


@dataclass
class SourceRecord:
    """سجل مصدر مستند"""
    id          : int
    file_name   : str
    file_path   : str   = ""
    source_type : str   = ""
    num_chunks  : int   = 0
    ingested_at : str   = ""
    metadata    : dict  = field(default_factory=dict)

    def __repr__(self):
        return f"[src:{self.id}] {self.file_name} | {self.num_chunks} قطعة"


# ============================================================
# عمليات القراءة
# ============================================================

class RelationalDB:
    """
    واجهة موحدة لجميع عمليات قاعدة البيانات العلائقية

    تشمل:
        - استعلامات القطع والمصادر
        - التحديث والحذف
        - الإحصائيات والتقارير
        - التصفية والفلترة

    مثال الاستخدام:
        db = RelationalDB()
        chunks = db.get_chunks_by_source("report.pdf")
        stats  = db.get_statistics()
    """

    # ============================================================
    # القطع (Chunks)
    # ============================================================

    def get_chunk_by_id(self, chunk_id: str) -> Optional[ChunkRecord]:
        """
        استرجاع قطعة واحدة بمعرفها

        Args:
            chunk_id: معرف القطعة

        Returns:
            ChunkRecord أو None
        """
        conn = get_connection()
        cur  = conn.cursor()

        try:
            cur.execute("""
                SELECT id, chunk_id, content, chunk_type, source_file,
                       page_number, token_count, parent_heading,
                       summary, keywords, questions, metadata
                FROM doc_chunks
                WHERE chunk_id = %s
            """, (chunk_id,))

            row = cur.fetchone()
            if not row:
                return None

            return self._row_to_chunk(row)

        finally:
            cur.close()
            conn.close()

    def get_chunks_by_source(self, source_file: str) -> list[ChunkRecord]:
        """
        استرجاع جميع قطع مستند معين

        Args:
            source_file: اسم الملف المصدر

        Returns:
            list[ChunkRecord]
        """
        conn = get_connection()
        cur  = conn.cursor()

        try:
            cur.execute("""
                SELECT id, chunk_id, content, chunk_type, source_file,
                       page_number, token_count, parent_heading,
                       summary, keywords, questions, metadata
                FROM doc_chunks
                WHERE source_file = %s
                ORDER BY id
            """, (source_file,))

            rows = cur.fetchall()
            chunks = [self._row_to_chunk(r) for r in rows]
            logger.info(f"📦 {len(chunks)} قطعة من {source_file}")
            return chunks

        finally:
            cur.close()
            conn.close()

    def get_chunks_by_type(self, chunk_type: str) -> list[ChunkRecord]:
        """
        استرجاع القطع حسب نوعها (text, table, image)

        Args:
            chunk_type: نوع القطعة

        Returns:
            list[ChunkRecord]
        """
        conn = get_connection()
        cur  = conn.cursor()

        try:
            cur.execute("""
                SELECT id, chunk_id, content, chunk_type, source_file,
                       page_number, token_count, parent_heading,
                       summary, keywords, questions, metadata
                FROM doc_chunks
                WHERE chunk_type = %s
                ORDER BY id
            """, (chunk_type,))

            rows = cur.fetchall()
            return [self._row_to_chunk(r) for r in rows]

        finally:
            cur.close()
            conn.close()

    def get_all_chunks(self, limit: int = 100, offset: int = 0) -> list[ChunkRecord]:
        """
        استرجاع جميع القطع مع pagination

        Args:
            limit : عدد القطع في كل صفحة
            offset: نقطة البداية

        Returns:
            list[ChunkRecord]
        """
        conn = get_connection()
        cur  = conn.cursor()

        try:
            cur.execute("""
                SELECT id, chunk_id, content, chunk_type, source_file,
                       page_number, token_count, parent_heading,
                       summary, keywords, questions, metadata
                FROM doc_chunks
                ORDER BY id
                LIMIT %s OFFSET %s
            """, (limit, offset))

            rows = cur.fetchall()
            return [self._row_to_chunk(r) for r in rows]

        finally:
            cur.close()
            conn.close()

    def get_chunks_without_embeddings(self) -> list[ChunkRecord]:
        """استرجاع القطع التي لم تُولَّد لها متجهات بعد"""
        conn = get_connection()
        cur  = conn.cursor()

        try:
            cur.execute("""
                SELECT id, chunk_id, content, chunk_type, source_file,
                       page_number, token_count, parent_heading,
                       summary, keywords, questions, metadata
                FROM doc_chunks
                WHERE embedding IS NULL
                ORDER BY id
            """)

            rows = cur.fetchall()
            logger.info(f"⚠️ {len(rows)} قطعة بدون متجهات")
            return [self._row_to_chunk(r) for r in rows]

        finally:
            cur.close()
            conn.close()

    # ============================================================
    # المصادر (Sources)
    # ============================================================

    def get_all_sources(self) -> list[SourceRecord]:
        """استرجاع جميع مصادر المستندات"""
        conn = get_connection()
        cur  = conn.cursor()

        try:
            cur.execute("""
                SELECT id, file_name, file_path, source_type,
                       num_chunks, ingested_at, metadata
                FROM doc_sources
                ORDER BY ingested_at DESC
            """)

            rows = cur.fetchall()
            return [self._row_to_source(r) for r in rows]

        finally:
            cur.close()
            conn.close()

    def source_exists(self, file_name: str) -> bool:
        """التحقق من إن المستند تم إدخاله مسبقاً"""
        conn = get_connection()
        cur  = conn.cursor()

        try:
            cur.execute(
                "SELECT COUNT(*) FROM doc_sources WHERE file_name = %s",
                (file_name,)
            )
            count = cur.fetchone()[0]
            return count > 0

        finally:
            cur.close()
            conn.close()

    # ============================================================
    # التحديث
    # ============================================================

    def update_chunk_summary(self, chunk_id: str, summary: str) -> bool:
        """تحديث ملخص قطعة معينة"""
        conn = get_connection()
        cur  = conn.cursor()

        try:
            cur.execute(
                "UPDATE doc_chunks SET summary = %s WHERE chunk_id = %s",
                (summary, chunk_id)
            )
            conn.commit()
            updated = cur.rowcount > 0
            if updated:
                logger.debug(f"✅ تم تحديث ملخص {chunk_id}")
            return updated

        except Exception as e:
            conn.rollback()
            logger.error(f"❌ خطأ في تحديث الملخص: {e}")
            return False
        finally:
            cur.close()
            conn.close()

    def update_chunk_keywords(self, chunk_id: str, keywords: list[str]) -> bool:
        """تحديث الكلمات المفتاحية لقطعة"""
        conn = get_connection()
        cur  = conn.cursor()

        try:
            cur.execute(
                "UPDATE doc_chunks SET keywords = %s WHERE chunk_id = %s",
                (keywords, chunk_id)
            )
            conn.commit()
            return cur.rowcount > 0

        except Exception as e:
            conn.rollback()
            logger.error(f"❌ خطأ في تحديث الكلمات: {e}")
            return False
        finally:
            cur.close()
            conn.close()

    def update_chunk_questions(self, chunk_id: str, questions: list[str]) -> bool:
        """تحديث الأسئلة الافتراضية لقطعة"""
        conn = get_connection()
        cur  = conn.cursor()

        try:
            cur.execute(
                "UPDATE doc_chunks SET questions = %s WHERE chunk_id = %s",
                (questions, chunk_id)
            )
            conn.commit()
            return cur.rowcount > 0

        except Exception as e:
            conn.rollback()
            logger.error(f"❌ خطأ في تحديث الأسئلة: {e}")
            return False
        finally:
            cur.close()
            conn.close()

    # ============================================================
    # الحذف
    # ============================================================

    def delete_source(self, file_name: str) -> int:
        """
        حذف مستند وجميع قطعه من قاعدة البيانات

        Args:
            file_name: اسم الملف المراد حذفه

        Returns:
            int: عدد القطع المحذوفة
        """
        conn = get_connection()
        cur  = conn.cursor()

        try:
            # حذف القطع أولاً
            cur.execute(
                "DELETE FROM doc_chunks WHERE source_file = %s",
                (file_name,)
            )
            deleted_chunks = cur.rowcount

            # حذف المصدر
            cur.execute(
                "DELETE FROM doc_sources WHERE file_name = %s",
                (file_name,)
            )

            conn.commit()
            logger.info(f"🗑️ تم حذف {file_name}: {deleted_chunks} قطعة")
            return deleted_chunks

        except Exception as e:
            conn.rollback()
            logger.error(f"❌ خطأ في الحذف: {e}")
            return 0
        finally:
            cur.close()
            conn.close()

    def clear_all_data(self) -> bool:
        """
        ⚠️ حذف جميع البيانات من قاعدة البيانات
        استخدم بحذر!
        """
        conn = get_connection()
        cur  = conn.cursor()

        try:
            cur.execute("TRUNCATE TABLE doc_chunks CASCADE;")
            cur.execute("TRUNCATE TABLE doc_sources CASCADE;")
            conn.commit()
            logger.warning("⚠️ تم حذف جميع البيانات من قاعدة البيانات")
            return True

        except Exception as e:
            conn.rollback()
            logger.error(f"❌ خطأ في حذف البيانات: {e}")
            return False
        finally:
            cur.close()
            conn.close()

    # ============================================================
    # الإحصائيات
    # ============================================================

    def get_statistics(self) -> dict:
        """
        إحصائيات شاملة عن قاعدة البيانات

        Returns:
            dict: الإحصائيات الكاملة
        """
        conn = get_connection()
        cur  = conn.cursor()

        try:
            # إحصائيات عامة
            cur.execute("""
                SELECT
                    COUNT(*)                                        AS total_chunks,
                    COUNT(*) FILTER (WHERE chunk_type = 'text')    AS text_chunks,
                    COUNT(*) FILTER (WHERE chunk_type = 'table')   AS table_chunks,
                    COUNT(*) FILTER (WHERE chunk_type = 'image')   AS image_chunks,
                    COUNT(*) FILTER (WHERE embedding IS NOT NULL)  AS embedded_chunks,
                    COUNT(*) FILTER (WHERE summary != '')          AS summarized_chunks,
                    AVG(token_count)                               AS avg_tokens,
                    SUM(token_count)                               AS total_tokens
                FROM doc_chunks;
            """)

            row = cur.fetchone()

            # إحصائيات المصادر
            cur.execute("SELECT COUNT(*) FROM doc_sources;")
            total_sources = cur.fetchone()[0]

            # أكثر المصادر قطعاً
            cur.execute("""
                SELECT source_file, COUNT(*) AS chunk_count
                FROM doc_chunks
                GROUP BY source_file
                ORDER BY chunk_count DESC
                LIMIT 5;
            """)
            top_sources = cur.fetchall()

            stats = {
                "total_chunks"      : row[0] or 0,
                "text_chunks"       : row[1] or 0,
                "table_chunks"      : row[2] or 0,
                "image_chunks"      : row[3] or 0,
                "embedded_chunks"   : row[4] or 0,
                "summarized_chunks" : row[5] or 0,
                "avg_tokens"        : round(float(row[6] or 0), 1),
                "total_tokens"      : row[7] or 0,
                "total_sources"     : total_sources,
                "top_sources"       : [
                    {"file": r[0], "chunks": r[1]}
                    for r in top_sources
                ],
            }

            logger.info(f"📊 إحصائيات DB: {stats['total_chunks']} قطعة | {stats['total_sources']} مصدر")
            return stats

        finally:
            cur.close()
            conn.close()

    # ============================================================
    # دوال مساعدة
    # ============================================================

    def _row_to_chunk(self, row: tuple) -> ChunkRecord:
        """تحويل صف قاعدة البيانات إلى ChunkRecord"""
        return ChunkRecord(
            id             = row[0],
            chunk_id       = row[1],
            content        = row[2],
            chunk_type     = row[3] or "text",
            source_file    = row[4] or "",
            page_number    = row[5] or 0,
            token_count    = row[6] or 0,
            parent_heading = row[7] or "",
            summary        = row[8] or "",
            keywords       = row[9] or [],
            questions      = row[10] or [],
            metadata       = row[11] or {},
        )

    def _row_to_source(self, row: tuple) -> SourceRecord:
        """تحويل صف قاعدة البيانات إلى SourceRecord"""
        return SourceRecord(
            id          = row[0],
            file_name   = row[1],
            file_path   = row[2] or "",
            source_type = row[3] or "",
            num_chunks  = row[4] or 0,
            ingested_at = str(row[5]) if row[5] else "",
            metadata    = row[6] or {},
        )