# ============================================================
# src/data_processing/raptor_engine.py
# محرك RAPTOR — تلخيص هرمي للمستندات
#
# RAPTOR = Recursive Abstractive Processing for Tree-Organized Retrieval
#
# الفكرة:
#   المستوى 0: القطع الأصلية
#   المستوى 1: ملخصات مجموعات القطع (clusters)
#   المستوى 2: ملخص المستند كله
#
# الفائدة: الأسئلة التي تحتاج فهم المستند ككل
#   مثال: "ما الاتجاه العام لفواتير هذا العميل؟"
#   → يجيب من المستوى 2 بدل البحث في 20 قطعة
# ============================================================

import os
import json
import uuid
from dataclasses import dataclass, field
from loguru import logger

from google import genai
from google.genai import types

from src.database.db_setup    import get_connection
from src.database.vector_store import VectorStore


@dataclass
class RaptorNode:
    """عقدة في شجرة RAPTOR"""
    node_id    : str
    level      : int          # 0=أصلي | 1=ملخص cluster | 2=ملخص كامل
    content    : str
    source_file: str
    children   : list[str] = field(default_factory=list)  # node_ids الأبناء
    embedding  : list[float] = field(default_factory=list)


class RaptorEngine:
    """
    يبني شجرة هرمية من ملخصات المستند.

    الاستخدام:
        raptor = RaptorEngine()
        raptor.build(source_file="invoice.pdf", chunks=[...])

        # عند البحث — يبحث في كل المستويات
        results = raptor.search(query="...", top_k=5)
    """

    CLUSTER_SIZE = 5   # عدد القطع في كل cluster

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        self._client = genai.Client(
            api_key      = api_key,
            http_options = types.HttpOptions(api_version="v1beta"),
        ) if api_key else None
        self._model      = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
        self._embed_model= "gemini-embedding-004"
        self._vector     = VectorStore()
        self._setup_table()
        logger.success("✅ RaptorEngine جاهز")

    # ════════════════════════════════════
    # بناء الشجرة
    # ════════════════════════════════════

    def build_tree(self, chunks: list, source_file: str, levels: int = 1) -> list:
        """
        بناء شجرة RAPTOR وإرجاع عقد الملخصات كقطع نصية (chunks).
        """
        if not self._client:
            logger.warning("⚠️ RAPTOR: لا يوجد LLM")
            return []

        logger.info(f"🌳 RAPTOR: بناء {levels} مستويات لـ {source_file}")
        all_new_nodes = []
        current_level_chunks = chunks

        # نكتفي بمستوى واحد حالياً كما في الاستدعاء
        chunk_list = list(current_level_chunks)
        for i in range(0, len(chunk_list), self.CLUSTER_SIZE):
            cluster = chunk_list[i:i + self.CLUSTER_SIZE]
            cluster_text = "\n\n".join([
                c.content if hasattr(c, "content") else c.get("content", "")
                for c in cluster
            ])
            child_ids = [
                c.chunk_id if hasattr(c, "chunk_id") else c.get("chunk_id", "")
                for c in cluster
            ]

            summary = self._summarize(cluster_text, level=1, source=source_file)
            if not summary:
                continue

            # إنشاء كائن متوافق مع نظام الـ chunks
            from src.agents.state import RetrievedChunk
            
            # محاكاة كائن Chunk
            class RaptorChunk:
                def __init__(self, cid, content, src):
                    self.chunk_id = cid
                    self.content = content
                    self.source_file = src
                    self.chunk_type = "raptor_summary"
                    self.page_number = 0
                    self.token_count = len(content.split())
                    self.metadata = {"is_raptor": True, "level": 1, "children": child_ids}

            new_chunk = RaptorChunk(
                cid=f"raptor_L1_{uuid.uuid4().hex[:8]}",
                content=summary,
                src=source_file
            )
            
            # حفظ في جدول RAPTOR الخاص (اختياري)
            node = RaptorNode(
                node_id=new_chunk.chunk_id,
                level=1,
                content=summary,
                source_file=source_file,
                children=child_ids,
            )
            node.embedding = self._embed(summary)
            self._save_node(node)
            
            all_new_nodes.append(new_chunk)

        logger.success(f"✅ RAPTOR: تم بناء {len(all_new_nodes)} عقدة ملخص")
        return all_new_nodes

    def build(self, source_file: str, chunks: list) -> int:
        """توافق مع النسخة القديمة"""
        nodes = self.build_tree(chunks, source_file)
        return len(nodes)

    # ════════════════════════════════════
    # بحث في الشجرة
    # ════════════════════════════════════

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """يبحث في كل مستويات الشجرة"""
        embedding = self._embed(query)
        if not embedding:
            # Fallback: بحث نصي بسيط
            logger.warning("⚠️ RAPTOR: لا يوجد embedding → بحث نصي")
            return self._text_search(query, top_k)
        
        conn = get_connection()
        cur  = conn.cursor()
        try:
            # التحقق من وجود embedding
            cur.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'raptor_nodes' AND column_name = 'embedding'
            """)
            has_embedding = cur.fetchone() is not None
            
            if not has_embedding:
                return self._text_search(query, top_k)
            
            cur.execute("""
                SELECT node_id, level, content, source_file,
                       1 - (embedding <=> %s::vector) AS score
                FROM   raptor_nodes
                WHERE  embedding IS NOT NULL
                ORDER  BY embedding <=> %s::vector
                LIMIT  %s
            """, (embedding, embedding, top_k))
            rows = cur.fetchall()
            return [
                {"node_id": r[0], "level": r[1], "content": r[2],
                 "source_file": r[3], "score": float(r[4])}
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"⚠️ RAPTOR search: {e}")
            return self._text_search(query, top_k)
        finally:
            cur.close(); conn.close()
    
    def _text_search(self, query: str, top_k: int = 3) -> list[dict]:
        """بحث نصي بسيط fallback"""
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT node_id, level, content, source_file, 0.5 AS score
                FROM raptor_nodes
                WHERE content ILIKE %s
                LIMIT %s
            """, (f"%{query}%", top_k))
            rows = cur.fetchall()
            return [
                {"node_id": r[0], "level": r[1], "content": r[2],
                 "source_file": r[3], "score": float(r[4])}
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"⚠️ RAPTOR text search: {e}")
            return []
        finally:
            cur.close(); conn.close()

    # ════════════════════════════════════
    # جدول DB
    # ════════════════════════════════════

    def _setup_table(self):
        conn = get_connection()
        cur  = conn.cursor()
        try:
            # استخدام 3072 بُعد لدعم gemini-embedding-001
            cur.execute("""
                CREATE TABLE IF NOT EXISTS raptor_nodes (
                    node_id     TEXT PRIMARY KEY,
                    level       INT  NOT NULL,
                    content     TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    children    JSONB DEFAULT '[]',
                    embedding   vector(3072),
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_raptor_embedding
                    ON raptor_nodes USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
            """)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"⚠️ RAPTOR table: {e}")
            # Fallback: جدول بدون embedding
            try:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS raptor_nodes (
                        node_id     TEXT PRIMARY KEY,
                        level       INT  NOT NULL,
                        content     TEXT NOT NULL,
                        source_file TEXT NOT NULL,
                        children    JSONB DEFAULT '[]',
                        created_at  TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                conn.commit()
            except Exception as e2:
                logger.error(f"❌ RAPTOR fallback table: {e2}")
        finally:
            cur.close(); conn.close()

    def _save_node(self, node: RaptorNode):
        conn = get_connection()
        cur  = conn.cursor()
        try:
            # التحقق من وجود عمود embedding
            cur.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'raptor_nodes' AND column_name = 'embedding'
            """)
            has_embedding = cur.fetchone() is not None
            
            if has_embedding and node.embedding:
                cur.execute("""
                    INSERT INTO raptor_nodes (node_id, level, content, source_file, children, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (node_id) DO UPDATE SET
                        level = EXCLUDED.level,
                        content = EXCLUDED.content,
                        children = EXCLUDED.children,
                        embedding = EXCLUDED.embedding
                """, (
                    node.node_id, node.level, node.content,
                    node.source_file, json.dumps(node.children),
                    node.embedding if node.embedding else None,
                ))
            else:
                # جدول بدون embedding
                cur.execute("""
                    INSERT INTO raptor_nodes (node_id, level, content, source_file, children)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (node_id) DO UPDATE SET
                        level = EXCLUDED.level,
                        content = EXCLUDED.content,
                        children = EXCLUDED.children
                """, (
                    node.node_id, node.level, node.content,
                    node.source_file, json.dumps(node.children),
                ))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"⚠️ RAPTOR save: {e}")
        finally:
            cur.close(); conn.close()

    # ════════════════════════════════════
    # Helpers
    # ════════════════════════════════════

    def _summarize(self, text: str, level: int, source: str) -> str:
        """توليد ملخص للمستوى المطلوب"""
        if not self._client:
            return ""

        level_desc = {
            1: "ملخص قصير (3-5 جمل) لمجموعة من الفقرات",
            2: "ملخص شامل (5-8 جمل) يلخص المستند بالكامل",
        }
        prompt = f"""اكتب {level_desc.get(level, 'ملخصاً')} لمستند: {source}

النص:
{text[:1500]}

الملخص:"""

        try:
            resp = self._client.models.generate_content(
                model=self._model, contents=prompt
            )
            return resp.text.strip()
        except Exception as e:
            logger.warning(f"⚠️ RAPTOR summarize L{level}: {e}")
            return ""

    def _embed(self, text: str) -> list[float]:
        if not self._client or not text.strip():
            return []
        try:
            result = self._client.models.embed_content(
                model   = self._embed_model,
                contents= text[:500],
                config  = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            # تحويل الـ embedding لأي dimension
            emb = result.embeddings[0].values
            # إذا البعد 768 (الافتراضي القديم) أو 3072 (الجديد)
            return emb
        except Exception as e:
            logger.warning(f"⚠️ RAPTOR embed error: {e}")
            return []