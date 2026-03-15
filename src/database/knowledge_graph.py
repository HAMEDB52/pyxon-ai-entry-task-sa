# ============================================================
# src/database/knowledge_graph.py
#
# الرابط بين البيانات — Knowledge Graph في PostgreSQL
#
# يحل المشكلة: chunk يعرف محتواه لكن لا يعرف علاقته
#              بالمستندات الأخرى
#
# ما يبنيه:
#   1. Entity Extraction  : شركات، أشخاص، مبالغ، تواريخ، أرقام فواتير
#   2. Entity Linking     : نفس الكيان في مستندات مختلفة → ربط تلقائي
#   3. Document Relations : INV-101 مرتبطة بـ RCP-101 (invoice→receipt)
#   4. Graph Retrieval    : ابدأ من chunk واسترجع الشبكة المرتبطة به
# ============================================================

import os
import re
import json
import time
from dataclasses import dataclass, field
from enum        import Enum
from loguru      import logger

from google import genai
from google.genai import types

from src.database.db_setup import get_connection


# ════════════════════════════════════════
# أنواع الكيانات
# ════════════════════════════════════════

class EntityType(str, Enum):
    PERSON          = "person"          # شخص / اسم
    ORGANIZATION    = "organization"    # شركة / جهة
    INVOICE         = "invoice"         # رقم فاتورة
    RECEIPT         = "receipt"         # رقم إيصال
    AMOUNT          = "amount"          # مبلغ مالي
    DATE            = "date"            # تاريخ
    CONTRACT        = "contract"        # رقم عقد
    PRODUCT         = "product"         # منتج / خدمة
    LOCATION        = "location"        # مكان
    ID_NUMBER       = "id_number"       # رقم هوية / ملف


class RelationType(str, Enum):
    INVOICE_OF      = "invoice_of"      # فاتورة خاصة بـ
    PAID_BY         = "paid_by"         # دُفعت بواسطة
    ISSUED_TO       = "issued_to"       # صُدرت لـ
    ISSUED_BY       = "issued_by"       # صُدرت من
    REFERENCES      = "references"      # تشير إلى
    SAME_ENTITY     = "same_entity"     # نفس الكيان
    RELATED_DOC     = "related_doc"     # مستند مرتبط
    AMOUNT_OF       = "amount_of"       # المبلغ المقابل
    DATE_OF         = "date_of"         # تاريخ مرتبط


# ════════════════════════════════════════
# نماذج البيانات
# ════════════════════════════════════════

@dataclass
class Entity:
    """كيان مُستخرَج من مستند"""
    entity_id   : str
    name        : str           # القيمة الفعلية (حامد الرويلي، INV-101)
    type        : EntityType
    source_file : str
    chunk_id    : str
    normalized  : str = ""      # شكل موحّد للمقارنة (lowercase, no spaces)
    confidence  : float = 0.9
    metadata    : dict  = field(default_factory=dict)

    def __post_init__(self):
        if not self.normalized:
            self.normalized = re.sub(r"\s+", "_", self.name.lower().strip())


@dataclass
class Relation:
    """علاقة بين كيانين"""
    relation_id : str
    source_id   : str   # entity_id المصدر
    target_id   : str   # entity_id الهدف
    type        : RelationType
    confidence  : float = 0.8
    evidence    : str   = ""    # الجملة التي استُنتجت منها العلاقة


@dataclass
class GraphContext:
    """سياق مُستخرَج من الـ Graph للإجابة على سؤال"""
    entities    : list[Entity]
    relations   : list[Relation]
    linked_docs : list[str]         # ملفات مرتبطة
    summary     : str = ""          # ملخص نصي للعلاقات

    def to_text(self) -> str:
        """تحويل لنص يُرسل للـ LLM"""
        lines = ["### 🔗 العلاقات بين البيانات:\n"]

        if self.entities:
            lines.append("**الكيانات المُكتشفة:**")
            by_type: dict[str, list] = {}
            for e in self.entities:
                by_type.setdefault(e.type.value, []).append(e.name)
            for t, names in by_type.items():
                lines.append(f"  - {t}: {', '.join(set(names))}")

        if self.linked_docs:
            lines.append(f"\n**مستندات مرتبطة:** {', '.join(self.linked_docs)}")

        if self.relations:
            lines.append("\n**العلاقات:**")
            for r in self.relations[:8]:
                lines.append(f"  - [{r.type.value}]: {r.evidence[:80]}")

        return "\n".join(lines)


# ════════════════════════════════════════
# Knowledge Graph الرئيسي
# ════════════════════════════════════════

class KnowledgeGraph:
    """
    يستخرج الكيانات والعلاقات من المستندات ويخزنها في PostgreSQL.
    يُستخدم لـ:
      1. تحسين الاسترجاع: ابدأ من chunk واجلب الشبكة المرتبطة
      2. الإجابة على أسئلة متعددة المستندات:
         "قارن بين INV-101 و INV-102" ← يعرف العلاقات تلقائياً

    الاستخدام:
        kg = KnowledgeGraph()

        # عند الإدخال
        kg.extract_and_store("path/to/invoice.pdf", chunks)

        # عند البحث
        context = kg.get_context_for_query("من دفع فاتورة INV-101؟")
        print(context.to_text())
    """

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        self._client = genai.Client(
            api_key      = api_key,
            http_options = types.HttpOptions(api_version="v1beta"),
        ) if api_key else None
        self._model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self._setup_tables()
        logger.success("✅ KnowledgeGraph جاهز")

    # ════════════════════════════════════
    # إنشاء الجداول
    # ════════════════════════════════════

    def _setup_tables(self):
        """إنشاء جداول الـ Knowledge Graph في PostgreSQL"""
        conn = get_connection()
        cur  = conn.cursor()
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kg_entities (
                    entity_id   TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    type        TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    chunk_id    TEXT,
                    normalized  TEXT,
                    confidence  FLOAT DEFAULT 0.9,
                    metadata    JSONB DEFAULT '{}',
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_kg_entities_normalized
                    ON kg_entities(normalized);
                CREATE INDEX IF NOT EXISTS idx_kg_entities_type
                    ON kg_entities(type);
                CREATE INDEX IF NOT EXISTS idx_kg_entities_source
                    ON kg_entities(source_file);

                CREATE TABLE IF NOT EXISTS kg_relations (
                    relation_id TEXT PRIMARY KEY,
                    source_id   TEXT REFERENCES kg_entities(entity_id) ON DELETE CASCADE,
                    target_id   TEXT REFERENCES kg_entities(entity_id) ON DELETE CASCADE,
                    type        TEXT NOT NULL,
                    confidence  FLOAT DEFAULT 0.8,
                    evidence    TEXT,
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_kg_relations_source
                    ON kg_relations(source_id);
                CREATE INDEX IF NOT EXISTS idx_kg_relations_target
                    ON kg_relations(target_id);
            """)
            conn.commit()
            logger.debug("✅ جداول KG جاهزة")
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ فشل إنشاء جداول KG: {e}")
        finally:
            cur.close(); conn.close()

    # ════════════════════════════════════
    # استخراج الكيانات من مستند
    # ════════════════════════════════════

    def extract_and_store(
        self,
        source_file : str,
        chunks      : list,
        overwrite   : bool = False,
    ) -> int:
        """
        استخراج كيانات وعلاقات من مستند وحفظها في DB.

        Args:
            source_file: اسم الملف
            chunks     : قائمة القطع (dicts أو objects)
            overwrite  : حذف الكيانات القديمة وإعادة الاستخراج

        Returns:
            int: عدد الكيانات المُستخرَجة
        """
        if overwrite:
            self._delete_source_entities(source_file)

        logger.info(f"🔍 KG: استخراج كيانات من {source_file} ({len(chunks)} قطعة)")

        all_entities : list[Entity]  = []
        all_relations: list[Relation] = []

        for chunk in chunks:
            # normalize chunk
            content  = chunk.get("content","") if isinstance(chunk, dict) else getattr(chunk,"content","")
            chunk_id = chunk.get("chunk_id","") if isinstance(chunk, dict) else getattr(chunk,"chunk_id","")

            if not content.strip():
                continue

            entities, relations = self._extract_from_chunk(content, chunk_id, source_file)
            all_entities.extend(entities)
            all_relations.extend(relations)

        # حفظ في DB
        if all_entities:
            self._save_entities(all_entities)
            self._auto_link_same_entities(source_file)

        if all_relations:
            self._save_relations(all_relations)

        logger.success(
            f"✅ KG: {source_file} → "
            f"{len(all_entities)} كيان | {len(all_relations)} علاقة"
        )
        return len(all_entities)

    # ════════════════════════════════════
    # استخراج من قطعة واحدة (LLM)
    # ════════════════════════════════════

    def _extract_from_chunk(
        self,
        content    : str,
        chunk_id   : str,
        source_file: str,
    ) -> tuple[list[Entity], list[Relation]]:
        """استخراج كيانات وعلاقات من قطعة نصية"""

        if not self._client:
            return self._regex_extract(content, chunk_id, source_file), []

        prompt = f"""استخرج الكيانات والعلاقات من النص التالي بتنسيق JSON دقيق.

النص:
{content[:600]}

أجب بـ JSON فقط بهذا الشكل (لا تكتب شيئاً آخر):
{{
  "entities": [
    {{"name": "...", "type": "person|organization|invoice|receipt|amount|date|contract|product|location|id_number"}}
  ],
  "relations": [
    {{"from": "entity_name", "to": "entity_name", "type": "invoice_of|paid_by|issued_to|issued_by|references|amount_of|date_of", "evidence": "الجملة الأصلية"}}
  ]
}}"""

        try:
            resp = self._client.models.generate_content(
                model    = self._model,
                contents = prompt,
            )
            raw  = resp.text.strip()
            raw  = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(raw)

            entities : list[Entity]   = []
            relations: list[Relation] = []
            name_to_id: dict[str, str] = {}

            for e in data.get("entities", []):
                if not e.get("name") or not e.get("type"):
                    continue
                try:
                    etype = EntityType(e["type"])
                except ValueError:
                    etype = EntityType.PERSON

                eid = f"e_{chunk_id}_{len(entities)}"
                entity = Entity(
                    entity_id   = eid,
                    name        = e["name"],
                    type        = etype,
                    source_file = source_file,
                    chunk_id    = chunk_id,
                )
                entities.append(entity)
                name_to_id[e["name"]] = eid

            for r in data.get("relations", []):
                src_name = r.get("from", "")
                tgt_name = r.get("to",   "")
                if src_name not in name_to_id or tgt_name not in name_to_id:
                    continue
                try:
                    rtype = RelationType(r["type"])
                except ValueError:
                    rtype = RelationType.REFERENCES

                rid = f"r_{chunk_id}_{len(relations)}"
                relations.append(Relation(
                    relation_id = rid,
                    source_id   = name_to_id[src_name],
                    target_id   = name_to_id[tgt_name],
                    type        = rtype,
                    evidence    = r.get("evidence", "")[:200],
                ))

            return entities, relations

        except Exception as e:
            logger.warning(f"⚠️ LLM extract فشل لـ {chunk_id}: {e}")
            return self._regex_extract(content, chunk_id, source_file), []

    # ════════════════════════════════════
    # استخراج نصي بـ Regex (Fallback)
    # ════════════════════════════════════

    def _regex_extract(
        self,
        content    : str,
        chunk_id   : str,
        source_file: str,
    ) -> list[Entity]:
        """استخراج كيانات أساسية بأنماط Regex بدون LLM"""
        entities = []
        idx = 0

        def add(name, etype):
            nonlocal idx
            entities.append(Entity(
                entity_id   = f"re_{chunk_id}_{idx}",
                name        = name,
                type        = etype,
                source_file = source_file,
                chunk_id    = chunk_id,
            ))
            idx += 1

        # أرقام فواتير
        for m in re.finditer(r"\b(INV|RCP|REF|ORD|CTR)[-\s]?\d{2,6}\b", content, re.I):
            t = EntityType.INVOICE if m.group(1).upper() in ("INV","ORD") else EntityType.RECEIPT
            add(m.group().upper(), t)

        # مبالغ مالية
        for m in re.finditer(r"\b[\d,]+(?:\.\d{1,2})?\s*(?:ريال|SAR|USD|EUR|\$)\b", content, re.I):
            add(m.group(), EntityType.AMOUNT)

        # تواريخ
        for m in re.finditer(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", content):
            add(m.group(), EntityType.DATE)

        # أسماء عربية (3 كلمات على الأقل تبدأ بكبيرة أو كلمة عربية)
        for m in re.finditer(r"[\u0600-\u06FF]{2,}\s+[\u0600-\u06FF]{2,}(?:\s+[\u0600-\u06FF]{2,})?", content):
            name = m.group().strip()
            if 5 < len(name) < 60:
                add(name, EntityType.PERSON)

        return entities[:20]   # حد معقول

    # ════════════════════════════════════
    # ربط نفس الكيانات عبر المستندات
    # ════════════════════════════════════

    def _auto_link_same_entities(self, new_source: str):
        """
        يبحث عن كيانات مشتركة بين المستند الجديد والمستندات الموجودة.
        مثال: "حامد الرويلي" في cv.pdf و invoice.pdf → يُنشئ علاقة SAME_ENTITY
        """
        conn = get_connection()
        cur  = conn.cursor()
        try:
            # اجلب كيانات المستند الجديد
            cur.execute("""
                SELECT entity_id, normalized, type
                FROM   kg_entities
                WHERE  source_file = %s
            """, (new_source,))
            new_entities = cur.fetchall()

            links_created = 0
            for new_id, normalized, etype in new_entities:
                if not normalized or len(normalized) < 4:
                    continue

                # ابحث عن نفس الكيان في مستندات أخرى
                cur.execute("""
                    SELECT entity_id, source_file
                    FROM   kg_entities
                    WHERE  normalized = %s
                      AND  type       = %s
                      AND  source_file != %s
                      AND  entity_id  != %s
                    LIMIT 10
                """, (normalized, etype, new_source, new_id))

                matches = cur.fetchall()
                for match_id, match_source in matches:
                    rid = f"link_{new_id}_{match_id}"
                    try:
                        cur.execute("""
                            INSERT INTO kg_relations
                                (relation_id, source_id, target_id, type, confidence, evidence)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (relation_id) DO NOTHING
                        """, (
                            rid, new_id, match_id,
                            RelationType.SAME_ENTITY.value, 0.85,
                            f"نفس الكيان في {new_source} و {match_source}",
                        ))
                        links_created += 1
                    except Exception:
                        pass

            conn.commit()
            if links_created:
                logger.info(f"🔗 KG: {links_created} رابط جديد بين المستندات")

        except Exception as e:
            conn.rollback()
            logger.warning(f"⚠️ auto_link فشل: {e}")
        finally:
            cur.close(); conn.close()

    # ════════════════════════════════════
    # استرجاع السياق من الـ Graph
    # ════════════════════════════════════

    def get_context_for_query(
        self,
        query          : str,
        chunk_ids      : list[str] | None = None,
        max_entities   : int = 20,
    ) -> GraphContext:
        """
        يسترجع الكيانات والعلاقات ذات الصلة بسؤال معين.

        Args:
            query     : سؤال المستخدم
            chunk_ids : قائمة الـ chunks التي أُرجعت من البحث (اختياري)

        Returns:
            GraphContext: سياق غني يُضاف لـ Agent 1
        """
        conn = get_connection()
        cur  = conn.cursor()

        try:
            # ── استخرج كلمات البحث من السؤال ──
            keywords = self._extract_query_keywords(query)

            # ── ابحث عن الكيانات ذات الصلة ──
            entities = []
            if chunk_ids:
                # من الـ chunks التي أرجعها البحث
                placeholders = ",".join(["%s"] * len(chunk_ids))
                cur.execute(f"""
                    SELECT entity_id, name, type, source_file, chunk_id, normalized, confidence
                    FROM   kg_entities
                    WHERE  chunk_id IN ({placeholders})
                    LIMIT  {max_entities}
                """, chunk_ids)
            elif keywords:
                # من الكلمات المفتاحية
                conditions = " OR ".join(["normalized ILIKE %s"] * len(keywords))
                params = [f"%{k}%" for k in keywords]
                cur.execute(f"""
                    SELECT entity_id, name, type, source_file, chunk_id, normalized, confidence
                    FROM   kg_entities
                    WHERE  ({conditions})
                    LIMIT  {max_entities}
                """, params)
            else:
                return GraphContext(entities=[], relations=[], linked_docs=[])

            rows = cur.fetchall()
            entities = [
                Entity(
                    entity_id   = r[0], name        = r[1],
                    type        = EntityType(r[2]) if r[2] in EntityType._value2member_map_ else EntityType.PERSON,
                    source_file = r[3], chunk_id    = r[4],
                    normalized  = r[5], confidence  = r[6] or 0.9,
                )
                for r in rows
            ]

            if not entities:
                return GraphContext(entities=[], relations=[], linked_docs=[])

            # ── اجلب العلاقات ──
            entity_ids = [e.entity_id for e in entities]
            placeholders = ",".join(["%s"] * len(entity_ids))
            cur.execute(f"""
                SELECT relation_id, source_id, target_id, type, confidence, evidence
                FROM   kg_relations
                WHERE  source_id IN ({placeholders})
                   OR  target_id IN ({placeholders})
                LIMIT  30
            """, entity_ids + entity_ids)

            relation_rows = cur.fetchall()
            relations = [
                Relation(
                    relation_id = r[0], source_id = r[1], target_id = r[2],
                    type        = RelationType(r[3]) if r[3] in RelationType._value2member_map_ else RelationType.REFERENCES,
                    confidence  = r[4] or 0.8,
                    evidence    = r[5] or "",
                )
                for r in relation_rows
            ]

            # ── المستندات المرتبطة ──
            linked_docs = list({e.source_file for e in entities})

            return GraphContext(
                entities    = entities,
                relations   = relations,
                linked_docs = linked_docs,
            )

        except Exception as e:
            logger.warning(f"⚠️ get_context فشل: {e}")
            return GraphContext(entities=[], relations=[], linked_docs=[])
        finally:
            cur.close(); conn.close()

    # ════════════════════════════════════
    # إحصاءات الـ Graph
    # ════════════════════════════════════

    def stats(self) -> dict:
        """إحصاءات الـ Knowledge Graph"""
        conn = get_connection()
        cur  = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM kg_entities")
            n_entities = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM kg_relations")
            n_relations = cur.fetchone()[0]
            cur.execute("SELECT COUNT(DISTINCT source_file) FROM kg_entities")
            n_docs = cur.fetchone()[0]
            cur.execute("""
                SELECT type, COUNT(*) FROM kg_entities GROUP BY type ORDER BY COUNT(*) DESC
            """)
            by_type = {r[0]: r[1] for r in cur.fetchall()}

            return {
                "entities"  : n_entities,
                "relations" : n_relations,
                "documents" : n_docs,
                "by_type"   : by_type,
            }
        finally:
            cur.close(); conn.close()

    # ════════════════════════════════════
    # Helpers
    # ════════════════════════════════════

    def _extract_query_keywords(self, query: str) -> list[str]:
        """استخراج كلمات مفتاحية من السؤال للبحث في الـ Graph"""
        stop = {"ما","من","متى","كيف","هل","في","على","عن","إلى","وش","ابغى","اعطني","قول"}
        words = [w.strip("؟?.,!") for w in query.split()]
        keywords = [w for w in words if len(w) > 2 and w not in stop]
        return keywords[:5]

    def _save_entities(self, entities: list[Entity]):
        """حفظ كيانات في DB"""
        conn = get_connection()
        cur  = conn.cursor()
        try:
            for e in entities:
                cur.execute("""
                    INSERT INTO kg_entities
                        (entity_id, name, type, source_file, chunk_id, normalized, confidence, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (entity_id) DO UPDATE
                        SET name = EXCLUDED.name, confidence = EXCLUDED.confidence
                """, (
                    e.entity_id, e.name, e.type.value,
                    e.source_file, e.chunk_id, e.normalized,
                    e.confidence, json.dumps(e.metadata),
                ))
            conn.commit()
        except Exception as ex:
            conn.rollback()
            logger.error(f"❌ save_entities: {ex}")
        finally:
            cur.close(); conn.close()

    def _save_relations(self, relations: list[Relation]):
        """حفظ علاقات في DB"""
        conn = get_connection()
        cur  = conn.cursor()
        try:
            for r in relations:
                cur.execute("""
                    INSERT INTO kg_relations
                        (relation_id, source_id, target_id, type, confidence, evidence)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (relation_id) DO NOTHING
                """, (
                    r.relation_id, r.source_id, r.target_id,
                    r.type.value, r.confidence, r.evidence,
                ))
            conn.commit()
        except Exception as ex:
            conn.rollback()
            logger.warning(f"⚠️ save_relations: {ex}")
        finally:
            cur.close(); conn.close()

    def _delete_source_entities(self, source_file: str):
        """حذف كيانات مستند قديم"""
        conn = get_connection()
        cur  = conn.cursor()
        try:
            cur.execute("DELETE FROM kg_entities WHERE source_file = %s", (source_file,))
            conn.commit()
        finally:
            cur.close(); conn.close()
