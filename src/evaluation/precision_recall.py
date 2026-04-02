# ============================================================
# src/evaluation/precision_recall.py
# قياس دقة وجودة الاسترجاع
#
# المقاييس:
#   Precision@K  : من الـ K نتيجة المُسترجعة، كم منها صحيح؟
#   Recall@K     : من الإجابات الصحيحة الكلية، كم استُرجع؟
#   MRR          : Mean Reciprocal Rank (أين أول إجابة صحيحة؟)
#   NDCG         : Normalized Discounted Cumulative Gain
#   Context Score: هل السياق المُرسَل للـ LLM يحتوي الإجابة؟
# ============================================================

import re
import math
from dataclasses import dataclass, field
from typing import Callable
from loguru import logger


@dataclass
class RetrievalMetrics:
    precision_at_k  : float = 0.0
    recall_at_k     : float = 0.0
    f1_score        : float = 0.0
    mrr             : float = 0.0   # Mean Reciprocal Rank
    ndcg            : float = 0.0   # NDCG@K
    context_score   : float = 0.0   # هل السياق يحتوي الإجابة؟
    k               : int   = 5

    def summary(self) -> str:
        return (
            f"P@{self.k}={self.precision_at_k:.3f} | "
            f"R@{self.k}={self.recall_at_k:.3f} | "
            f"F1={self.f1_score:.3f} | "
            f"MRR={self.mrr:.3f} | "
            f"NDCG={self.ndcg:.3f} | "
            f"CTX={self.context_score:.3f}"
        )


@dataclass
class EvalSample:
    """عيّنة تقييم واحدة"""
    query          : str
    expected_answer: str          # الإجابة الصحيحة
    retrieved_ids  : list[str]    # chunk_ids المُسترجعة
    relevant_ids   : list[str]    # chunk_ids التي تحتوي الإجابة
    retrieved_text : list[str]    # نصوص القطع المُسترجعة
    generated      : str  = ""    # الإجابة المولّدة (اختياري)


class RetrievalEvaluator:
    """
    يُقيّم جودة الاسترجاع لنظام RAG.

    الاستخدام:
        evaluator = RetrievalEvaluator()

        sample = EvalSample(
            query           = "ما إجمالي INV-101؟",
            expected_answer = "5500 SAR",
            retrieved_ids   = ["c1","c2","c3"],
            relevant_ids    = ["c2"],
            retrieved_text  = ["...","5500 SAR...","..."],
        )
        metrics = evaluator.evaluate(sample, k=5)
        print(metrics.summary())

        # تقييم مجموعة
        report = evaluator.evaluate_batch(samples, k=5)
    """

    def __init__(self, relevance_fn: Callable = None):
        """
        Args:
            relevance_fn: دالة مخصصة لتحديد الصلة (اختياري)
                         signature: (chunk_text: str, expected: str) -> bool
        """
        self._relevance_fn = relevance_fn or self._default_relevance

    # ════════════════════════════════════
    # تقييم عيّنة واحدة
    # ════════════════════════════════════

    def evaluate(self, sample: EvalSample, k: int = 5) -> RetrievalMetrics:
        retrieved = sample.retrieved_ids[:k]
        relevant  = set(sample.relevant_ids)

        if not retrieved:
            return RetrievalMetrics(k=k)

        # ── Precision@K ──
        hits = sum(1 for cid in retrieved if cid in relevant)
        precision = hits / len(retrieved) if retrieved else 0.0

        # ── Recall@K ──
        recall = hits / len(relevant) if relevant else 0.0

        # ── F1 ──
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # ── MRR ──
        mrr = 0.0
        for rank, cid in enumerate(retrieved, start=1):
            if cid in relevant:
                mrr = 1.0 / rank
                break

        # ── NDCG@K ──
        ndcg = self._ndcg(retrieved, relevant, k)

        # ── Context Score ──
        ctx_score = self._context_score(
            sample.retrieved_text[:k],
            sample.expected_answer,
        )

        return RetrievalMetrics(
            precision_at_k = round(precision, 4),
            recall_at_k    = round(recall,    4),
            f1_score       = round(f1,        4),
            mrr            = round(mrr,       4),
            ndcg           = round(ndcg,      4),
            context_score  = round(ctx_score, 4),
            k              = k,
        )

    # ════════════════════════════════════
    # تقييم مجموعة
    # ════════════════════════════════════

    def evaluate_batch(
        self,
        samples : list[EvalSample],
        k       : int = 5,
    ) -> dict:
        """تقييم مجموعة عيّنات وحساب المتوسطات"""
        if not samples:
            return {}

        all_metrics = [self.evaluate(s, k) for s in samples]

        avg = lambda field: sum(getattr(m, field) for m in all_metrics) / len(all_metrics)

        report = {
            "n_samples"      : len(samples),
            "k"              : k,
            "mean_precision" : round(avg("precision_at_k"), 4),
            "mean_recall"    : round(avg("recall_at_k"),    4),
            "mean_f1"        : round(avg("f1_score"),       4),
            "mean_mrr"       : round(avg("mrr"),            4),
            "mean_ndcg"      : round(avg("ndcg"),           4),
            "mean_ctx_score" : round(avg("context_score"),  4),
            "per_sample"     : [m.summary() for m in all_metrics],
        }

        logger.info(
            f"📊 Retrieval Eval | n={report['n_samples']} | "
            f"P={report['mean_precision']:.3f} | "
            f"R={report['mean_recall']:.3f} | "
            f"MRR={report['mean_mrr']:.3f}"
        )
        return report

    # ════════════════════════════════════
    # Context Score (بدون LLM)
    # ════════════════════════════════════

    def _context_score(self, chunks: list[str], expected: str) -> float:
        """
        هل السياق المُسترجع يحتوي الإجابة الصحيحة؟
        0.0 = لا يوجد | 1.0 = موجود في أول قطعة
        """
        if not expected or not chunks:
            return 0.0

        # استخرج الأرقام والكيانات من الإجابة المتوقعة
        expected_nums = set(re.findall(r"\b[\d,]+(?:\.\d+)?\b", expected))
        expected_words = set(
            w.strip() for w in re.split(r"\s+|[،,.]", expected)
            if len(w.strip()) > 2
        )

        for rank, chunk in enumerate(chunks, start=1):
            # فحص الأرقام
            chunk_nums  = set(re.findall(r"\b[\d,]+(?:\.\d+)?\b", chunk))
            num_overlap = len(expected_nums & chunk_nums) / max(len(expected_nums), 1)

            # فحص الكلمات
            chunk_words = set(
                w.strip() for w in re.split(r"\s+|[،,.]", chunk)
                if len(w.strip()) > 2
            )
            word_overlap = len(expected_words & chunk_words) / max(len(expected_words), 1)

            score = (num_overlap * 0.6) + (word_overlap * 0.4)
            if score >= 0.5:
                # خصم بسيط بسبب الترتيب
                return round(score * (1.0 / rank ** 0.3), 4)

        return 0.0

    # ════════════════════════════════════
    # NDCG
    # ════════════════════════════════════

    def _ndcg(self, retrieved: list[str], relevant: set, k: int) -> float:
        """Normalized Discounted Cumulative Gain"""
        def dcg(results):
            return sum(
                (1 if cid in relevant else 0) / math.log2(i + 2)
                for i, cid in enumerate(results)
            )

        actual_dcg  = dcg(retrieved[:k])
        ideal_dcg   = dcg(sorted(retrieved[:k], key=lambda c: c in relevant, reverse=True))
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    # ════════════════════════════════════
    # دالة الصلة الافتراضية
    # ════════════════════════════════════

    def _default_relevance(self, chunk_text: str, expected: str) -> bool:
        """هل القطعة تحتوي الإجابة المتوقعة؟"""
        if not expected or not chunk_text:
            return False
        # مطابقة بسيطة
        expected_clean = expected.lower().strip()
        chunk_clean    = chunk_text.lower()
        # فحص الأرقام
        nums = re.findall(r"\b[\d,]+\b", expected_clean)
        if nums:
            return all(n.replace(",","") in chunk_clean.replace(",","") for n in nums)
        return expected_clean[:20] in chunk_clean