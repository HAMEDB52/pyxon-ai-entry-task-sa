# ============================================================
# src/database/retrieval_enhancer.py
#
# طبقة تحسين الاسترجاع — ٣ مراحل فوق HybridSearch الحالي:
#
#   Phase 1 — Query Expansion
#     السؤال الواحد → ٣ صياغات مختلفة → بحث بكل منها
#     يحل مشكلة: "وش الاسم" لا تجيب كـ "اسم صاحب الملف"
#
#   Phase 2 — Contextual Compression
#     كل chunk كبير → استخراج الجزء ذي الصلة فقط
#     يحل مشكلة: LLM يشتت في نص كثير
#
#   Phase 3 — LLM Reranker
#     أفضل 20 نتيجة → Gemini يرتبها حسب الصلة → أفضل 5
#     يحل مشكلة: vector score لا يعكس دائماً الإجابة الصحيحة
# ============================================================

import os
import re
import json
import time
from dataclasses import dataclass, field
from loguru import logger

from google import genai
from google.genai import types

from src.database.hybrid_search import HybridSearch, SearchResult


# ════════════════════════════════════════
# نموذج النتيجة المُحسَّنة
# ════════════════════════════════════════

@dataclass
class EnhancedResult:
    """نتيجة بحث مُحسَّنة بعد التوسيع وإعادة الترتيب"""
    chunk_id         : str
    content          : str           # المحتوى الأصلي
    compressed       : str           # الجزء المضغوط ذو الصلة
    source_file      : str
    page_number      : int   = 0
    parent_heading   : str   = ""
    rrf_score        : float = 0.0
    rerank_score     : float = 0.0   # درجة Gemini Reranker
    final_score      : float = 0.0   # الدرجة النهائية المدمجة
    matched_queries  : list[str] = field(default_factory=list)  # الصياغات التي وجدته

    @property
    def best_content(self) -> str:
        """أفضل محتوى للإرسال للـ LLM"""
        return self.compressed if self.compressed else self.content

    def __repr__(self):
        return (
            f"[{self.chunk_id}] final={self.final_score:.3f} | "
            f"rerank={self.rerank_score:.2f} | "
            f"{self.source_file} | "
            f"{self.content[:60]}…"
        )


# ════════════════════════════════════════
# محرك تحسين الاسترجاع
# ════════════════════════════════════════

class RetrievalEnhancer:
    """
    يُغلّف HybridSearch ويضيف فوقه ٣ طبقات تحسين.

    الاستخدام:
        enhancer = RetrievalEnhancer()
        results  = enhancer.search("وش الاسم الموجود بالـ cv؟", top_k=5)
        for r in results:
            print(r.best_content)
    """

    def __init__(
        self,
        n_expansions      : int   = 3,      # عدد صياغات توسيع الاستعلام
        candidate_k       : int   = 20,     # عدد المرشحين قبل Reranker
        final_k           : int   = 8,      # عدد النتائج النهائية
        rerank_weight     : float = 0.7,    # وزن Reranker في الدرجة النهائية
        rrf_weight        : float = 0.3,    # وزن RRF في الدرجة النهائية
        compress          : bool  = True,   # تفعيل الضغط السياقي
        min_expansion_chars: int = 4,      # الحد الأدنى لطول الاستعلام للتوسيع
    ):
        self.n_expansions   = n_expansions
        self.candidate_k    = candidate_k
        self.final_k        = final_k
        self.rerank_weight  = rerank_weight
        self.rrf_weight     = rrf_weight
        self.compress       = compress
        self.min_expansion_chars = min_expansion_chars

        # محرك البحث الأساسي
        self.hybrid = HybridSearch()

        # Gemini client
        api_key = os.getenv("GOOGLE_API_KEY")
        self._client = genai.Client(
            api_key      = api_key,
            http_options = types.HttpOptions(api_version="v1beta"),
        ) if api_key else None
        self._model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        logger.success(
            f"✅ RetrievalEnhancer | "
            f"expansions={n_expansions} | candidate_k={candidate_k} | "
            f"final_k={final_k} | compress={compress} | rerank_weight={rerank_weight}"
        )

    # ════════════════════════════════════
    # الدالة الرئيسية
    # ════════════════════════════════════

    def search(
        self,
        query      : str,
        top_k      : int | None = None,
        source_file: str | None = None,
    ) -> list[EnhancedResult]:
        """
        بحث مُحسَّن بثلاث مراحل مع fallback قوي.

        Args:
            query      : سؤال المستخدم الأصلي
            top_k      : عدد النتائج النهائية (يتجاوز الافتراضي)
            source_file: تصفية بملف محدد (اختياري)

        Returns:
            list[EnhancedResult]: نتائج مرتبة حسب الصلة
        """
        final_k = top_k or self.final_k
        start   = time.time()
        logger.info(f"🔍 RetrievalEnhancer | '{query[:60]}'")

        # ── Phase 0: التحقق من طول الاستعلام ──
        # الاستعلامات القصيرة جداً لا تحتاج توسيع
        should_expand = (
            len(query.strip()) >= self.min_expansion_chars and
            self.n_expansions > 1 and
            self._client is not None
        )
        
        if should_expand:
            queries = self._expand_query(query)
            logger.debug(f"   📝 صياغات: {queries}")
        else:
            queries = [query]
            logger.debug(f"   📝 استعلام واحد (قصير جداً)")

        # ── Phase 1: بحث هجين بكل صياغة ──
        merged = self._multi_query_search(queries, source_file)
        logger.debug(f"   🗃️ مرشحون: {len(merged)}")

        if not merged:
            logger.warning("⚠️ لا نتائج بعد البحث الهجين → fallback للبحث المباشر")
            # Fallback: بحث هجين مباشر بدون توسيع
            try:
                raw = self.hybrid.search(query, top_k=final_k, source_file=source_file)
                if raw:
                    return [
                        EnhancedResult(
                            chunk_id    = r.chunk_id,
                            content     = r.content,
                            compressed  = "",
                            source_file = r.source_file,
                            page_number = r.page_number,
                            parent_heading = r.parent_heading,
                            rrf_score   = r.rrf_score,
                            final_score = r.rrf_score,
                        )
                        for r in raw
                    ]
            except Exception as fb_err:
                logger.error(f"  ❌ Fallback فشل: {fb_err}")
            return []

        # ── Phase 2: Reranker ──
        reranked = self._rerank(query, list(merged.values()))
        logger.debug(f"   🏆 بعد Reranker: {len(reranked)}")

        # ── Phase 3: ضغط سياقي (اختياري) ──
        if self.compress:
            reranked = self._compress_results(query, reranked[:final_k])

        elapsed = round(time.time() - start, 2)
        logger.success(
            f"✅ RetrievalEnhancer | {len(reranked[:final_k])} نتيجة | {elapsed}s"
        )
        return reranked[:final_k]

    # ════════════════════════════════════
    # Phase 1 — توسيع الاستعلام
    # ════════════════════════════════════

    def _expand_query(self, query: str) -> list[str]:
        """
        يُنتج N صياغات مختلفة للاستعلام الأصلي.
        يعمل بدون LLM كـ fallback.
        يتخطى الصياغات المتشابهة جداً (جديد: smart similarity check).
        """
        queries = [query]   # ابدأ دائماً بالأصلي

        if not self._client:
            logger.debug("  ⚠️ لا يوجد LLM → توسيع بسيط")
            return queries

        # تحسين.prompt للتوسيع
        prompt = f"""أعد صياغة السؤال التالي بـ {self.n_expansions} طرق مختلفة للبحث في مستندات (فواتير، سير ذاتية، عقود).

السؤال الأصلي: {query}

القواعد:
- كل صياغة في سطر منفصل
- غيّر المفردات مع الحفاظ على المعنى
- استخدم مرادفات عربية وإنجليزية
- لا ترقيم ولا شرح
- اجعل كل صياغة واضحة ومباشرة

الصياغات:"""

        try:
            response = self._client.models.generate_content(
                model    = self._model,
                contents = prompt,
                config   = types.GenerateContentConfig(
                    temperature=0.7,  # تنويع أكبر
                    top_p=0.9,
                ),
            )
            expansions = [
                line.strip().lstrip("-•0123456789. ")
                for line in response.text.strip().split("\n")
                if line.strip() and len(line.strip()) > 3
            ]
            
            # ✨ جديد: تصفية الصياغات المتشابهة جداً (Jaccard similarity > 0.85 → تخطي)
            def jaccard_similarity(s1: str, s2: str) -> float:
                """حساب Jaccard similarity بسيط بين نصين"""
                words1 = set(s1.lower().split())
                words2 = set(s2.lower().split())
                if not words1 or not words2:
                    return 0.0
                intersection = len(words1 & words2)
                union = len(words1 | words2)
                return intersection / union if union > 0 else 0.0
            
            # أضف الصياغات الجديدة (بدون تكرار + بدون تشابه عالي جداً)
            for exp in expansions[:self.n_expansions]:
                # تجاهل إذا كانت نفس الأصلية تماماً
                if exp.lower() == query.lower():
                    continue
                # تجاهل إذا كانت متشابهة جداً (تكرار فعلي)
                if any(jaccard_similarity(exp, existing) > 0.85 for existing in queries):
                    logger.debug(f"   ⏭️ تخطي صياغة متشابهة: '{exp[:40]}'")
                    continue
                queries.append(exp)
                # إذا وصلنا للعدد المطلوب، توقف
                if len(queries) >= self.n_expansions + 1:
                    break

            logger.debug(f"   ✅ {len(queries)} صياغات (بعد تصفية التماثل)")
        except Exception as e:
            logger.warning(f"⚠️ فشل توسيع الاستعلام: {e}")
            # Fallback: توسيع بسيط بدون LLM
            queries.extend(self._simple_expansions(query))

        return queries

    def _simple_expansions(self, query: str) -> list[str]:
        """توسيع بسيط بدون LLM - إضافة مرادفات يدوية"""
        expansions = []
        
        # قاموس مرادفات بسيط
        synonyms = {
            "ما": ["وش", "ايش", "شو"],
            "هل": ["وش", "ايش"],
            "من": ["مين", "شكون"],
            "كم": ["قديش", "شحال"],
            "كيف": ["كيفاش", "شلون"],
            "إجمالي": ["مجموع", "كل", "كامل"],
            "فاتورة": ["فواتير", "bill", "invoice"],
            "عميل": ["زبون", "customer", "client"],
            "دفعة": ["دفع", "payment", "pay"],
            "مبلغ": ["قيمة", "amount", "value"],
            "تاريخ": ["date", "when", "متى"],
        }
        
        for word, syns in synonyms.items():
            if word in query:
                for syn in syns[:2]:  # أول مرادفين فقط
                    new_q = query.replace(word, syn, 1)
                    if new_q != query:
                        expansions.append(new_q)
                break  # استبدل كلمة واحدة فقط
        
        return expansions[:self.n_expansions - 1]  # ناقص الأصلي

    # ════════════════════════════════════
    # Phase 2 — بحث بكل صياغة + دمج
    # ════════════════════════════════════

    def _multi_query_search(
        self,
        queries    : list[str],
        source_file: str | None,
    ) -> dict[str, "EnhancedResult"]:
        """
        بحث هجين بكل صياغة ودمج النتائج.
        chunk يظهر في صياغات أكثر → درجة أعلى.
        """
        merged: dict[str, EnhancedResult] = {}

        for q in queries:
            try:
                raw = self.hybrid.search(
                    query       = q,
                    top_k       = self.candidate_k,
                    source_file = source_file,
                )
            except Exception as e:
                logger.warning(f"⚠️ فشل البحث لـ '{q[:40]}': {e}")
                continue

            for result in raw:
                cid = result.chunk_id
                if cid not in merged:
                    merged[cid] = EnhancedResult(
                        chunk_id       = cid,
                        content        = result.content,
                        compressed     = "",
                        source_file    = result.source_file,
                        page_number    = result.page_number,
                        parent_heading = result.parent_heading,
                        rrf_score      = result.rrf_score,
                    )
                else:
                    # تراكم الدرجات عبر الصياغات
                    merged[cid].rrf_score = max(merged[cid].rrf_score, result.rrf_score)

                # تتبع الصياغات التي وجدت هذا الـ chunk
                if q not in merged[cid].matched_queries:
                    merged[cid].matched_queries.append(q)
                    # bonus: كل صياغة إضافية تجد نفس الـ chunk → +5% درجة
                    merged[cid].rrf_score *= 1.05

        return merged

    # ════════════════════════════════════
    # Phase 3 — LLM Reranker
    # ════════════════════════════════════

    def _rerank(
        self,
        query      : str,
        candidates : list[EnhancedResult],
    ) -> list[EnhancedResult]:
        """
        يرتب المرشحين حسب الصلة باستخدام LLM + RRF.
        """
        if not self._client or not candidates:
            # بدون LLM → رتب حسب RRF فقط
            for r in candidates:
                r.final_score = r.rrf_score
            return sorted(candidates, key=lambda x: x.final_score, reverse=True)

        # اقتصر على أفضل 15 مرشح لتوفير tokens
        top_candidates = sorted(candidates, key=lambda x: x.rrf_score, reverse=True)[:15]

        # ابنِ قائمة مرقّمة للـ prompt
        numbered = "\n\n".join([
            f"[{i+1}] المصدر: {r.source_file}\n{r.content[:400]}"
            for i, r in enumerate(top_candidates)
        ])

        prompt = f"""أنت نظام ترتيب دقيق. رتّب القطع التالية من الأكثر إلى الأقل صلة بالسؤال.

السؤال: {query}

القطع المرشحة:
{numbered}

أجب بقائمة JSON فقط من أرقام الترتيب، من الأفضل للأسوأ.
مثال: [3, 1, 5, 2, 4]
لا تكتب أي شيء آخر:"""

        try:
            response = self._client.models.generate_content(
                model    = self._model,
                contents = prompt,
                config   = types.GenerateContentConfig(
                    temperature=0.3,  # أقل عشوائية للترتيب
                ),
            )
            raw = response.text.strip()

            # استخرج القائمة
            import re, json
            match = re.search(r"\[[\d,\s]+\]", raw)
            if not match:
                raise ValueError(f"لا يوجد JSON في الرد: {raw}")

            order = json.loads(match.group())
            order = [i for i in order if 1 <= i <= len(top_candidates)]

            # أضف rerank_score عكسية (الأول يأخذ أعلى درجة)
            n = len(order)
            for rank, idx in enumerate(order, start=1):
                cand = top_candidates[idx - 1]
                cand.rerank_score = (n - rank + 1) / n   # 1.0 → 0.0

            # للـ chunks التي لم يذكرها الـ LLM
            mentioned_ids = {top_candidates[i-1].chunk_id for i in order}
            for cand in top_candidates:
                if cand.chunk_id not in mentioned_ids:
                    cand.rerank_score = 0.2  # درجة منخفضة لكن ليست صفر

            logger.debug(f"   🏆 Reranker order: {order[:5]}")

        except Exception as e:
            logger.warning(f"⚠️ Reranker فشل: {e} → RRF فقط")
            for r in top_candidates:
                r.rerank_score = r.rrf_score  # fallback لـ RRF

        # دمج الدرجتين
        for r in candidates:
            r.final_score = (
                r.rrf_score     * self.rrf_weight +
                r.rerank_score  * self.rerank_weight
            )

        return sorted(candidates, key=lambda x: x.final_score, reverse=True)

    # ════════════════════════════════════
    # Phase 4 — ضغط سياقي
    # ════════════════════════════════════

    def _compress_results(
        self,
        query  : str,
        results: list[EnhancedResult],
    ) -> list[EnhancedResult]:
        """
        لكل chunk → استخرج الجملة أو الفقرة التي تجيب على السؤال فعلاً.
        يقلص الضوضاء ويركّز الـ LLM.
        """
        if not self._client:
            return results

        for r in results:
            # لا تضغط النصوص القصيرة أصلاً
            if len(r.content) < 200:
                r.compressed = r.content
                continue

            prompt = f"""استخرج الجملة أو الجمل القليلة من النص التالي التي تجيب مباشرة على السؤال.

السؤال: {query}

النص:
{r.content[:800]}

الاستخراج (بدون شرح، النص الأصلي فقط):"""

            try:
                resp = self._client.models.generate_content(
                    model    = self._model,
                    contents = prompt,
                )
                compressed = resp.text.strip()
                # فقط إذا الضغط فعلاً أقصر
                r.compressed = compressed if len(compressed) < len(r.content) * 0.8 else ""
            except Exception:
                r.compressed = ""

        return results

    # ════════════════════════════════════
    # استعلام بسيط بدون تحسين (للاختبار)
    # ════════════════════════════════════

    def search_basic(self, query: str, top_k: int = 5) -> list[EnhancedResult]:
        """بحث هجين بدون expansion أو reranking (للمقارنة)"""
        raw = self.hybrid.search(query, top_k=top_k)
        return [
            EnhancedResult(
                chunk_id    = r.chunk_id,
                content     = r.content,
                compressed  = "",
                source_file = r.source_file,
                rrf_score   = r.rrf_score,
                final_score = r.rrf_score,
            )
            for r in raw
        ]