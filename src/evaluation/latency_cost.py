# ============================================================
# src/evaluation/latency_cost.py
# قياس الأداء والتكلفة — Latency & Cost Tracker
# يقيس زمن الاستجابة واستهلاك التوكنات وتكلفة API
# ============================================================

import os
import time
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
from loguru import logger


# ============================================================
# مراحل الـ Pipeline القابلة للقياس
# ============================================================

class PipelineStage(str, Enum):
    """مراحل الـ Pipeline التي نقيس أداءها"""
    GATEKEEPER    = "gatekeeper"
    ROUTING       = "routing"
    RETRIEVAL     = "retrieval"
    EMBEDDING     = "embedding"
    RERANKING     = "reranking"
    GENERATION    = "generation"
    VERIFICATION  = "verification"
    CORRECTION    = "correction"
    AUDITOR       = "auditor"
    TOTAL         = "total"


# ============================================================
# أسعار النماذج (لكل مليون توكن)
# ============================================================

MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.0-flash": {
        "input_per_million"  : 0.10,
        "output_per_million" : 0.40,
    },
    "gemini-2.5-flash": {
        "input_per_million"  : 0.15,
        "output_per_million" : 0.60,
    },
    "gemini-1.5-pro": {
        "input_per_million"  : 1.25,
        "output_per_million" : 5.00,
    },
    "gemini-2.0-pro": {
        "input_per_million"  : 1.25,
        "output_per_million" : 5.00,
    },
    "text-embedding-004": {
        "input_per_million"  : 0.00,
        "output_per_million" : 0.00,
    },
    "gemini-embedding-001": {
        "input_per_million"  : 0.00,
        "output_per_million" : 0.00,
    },
}


# ============================================================
# قياس مرحلة واحدة
# ============================================================

@dataclass
class StageMetric:
    """قياس أداء مرحلة واحدة"""
    stage          : PipelineStage
    latency_ms     : float = 0.0
    input_tokens   : int   = 0
    output_tokens  : int   = 0
    total_tokens   : int   = 0
    cost_usd       : float = 0.0
    model          : str   = ""
    success        : bool  = True
    error          : str   = ""

    def __repr__(self):
        icon = "✅" if self.success else "❌"
        cost_str = f"${self.cost_usd:.6f}" if self.cost_usd > 0 else "مجاني"
        return (
            f"{icon} {self.stage.value:14} | "
            f"{self.latency_ms:8.1f}ms | "
            f"توكن: {self.total_tokens:6} | "
            f"تكلفة: {cost_str}"
        )


# ============================================================
# قياس طلب كامل
# ============================================================

@dataclass
class RequestMetric:
    """قياس أداء طلب كامل عبر الـ Pipeline"""
    request_id     : str
    query          : str
    stages         : list[StageMetric] = field(default_factory=list)
    total_latency  : float = 0.0
    total_tokens   : int   = 0
    total_cost     : float = 0.0
    success        : bool  = True
    timestamp      : str   = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def stage_breakdown(self) -> dict[str, float]:
        if self.total_latency == 0:
            return {}
        return {
            s.stage.value: round(s.latency_ms / self.total_latency * 100, 1)
            for s in self.stages
        }

    @property
    def bottleneck(self) -> str:
        if not self.stages:
            return ""
        slowest = max(self.stages, key=lambda s: s.latency_ms)
        return slowest.stage.value

    def __repr__(self):
        return (
            f"📊 طلب {self.request_id[:8]} | "
            f"⏱️ {self.total_latency:.0f}ms | "
            f"🔤 {self.total_tokens} توكن | "
            f"💰 ${self.total_cost:.6f} | "
            f"🐌 عنق الزجاجة: {self.bottleneck}"
        )


# ============================================================
# تقرير الأداء والتكلفة
# ============================================================

@dataclass
class PerformanceReport:
    """تقرير أداء وتكلفة شامل"""
    requests         : list[RequestMetric] = field(default_factory=list)
    total_requests   : int   = 0
    avg_latency      : float = 0.0
    median_latency   : float = 0.0
    p95_latency      : float = 0.0
    p99_latency      : float = 0.0
    min_latency      : float = 0.0
    max_latency      : float = 0.0
    total_tokens     : int   = 0
    avg_tokens       : float = 0.0
    total_cost       : float = 0.0
    avg_cost         : float = 0.0
    stage_avg_latency: dict[str, float] = field(default_factory=dict)
    stage_avg_tokens : dict[str, float] = field(default_factory=dict)
    success_rate     : float = 0.0
    timestamp        : str   = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __repr__(self):
        return (
            f"📊 تقرير الأداء | طلبات: {self.total_requests}\n"
            f"   ⏱️  متوسط الزمن   : {self.avg_latency:.0f}ms\n"
            f"   ⏱️  الوسيط        : {self.median_latency:.0f}ms\n"
            f"   ⏱️  P95           : {self.p95_latency:.0f}ms\n"
            f"   ⏱️  P99           : {self.p99_latency:.0f}ms\n"
            f"   🔤 إجمالي التوكنات: {self.total_tokens:,}\n"
            f"   💰 إجمالي التكلفة : ${self.total_cost:.4f}\n"
            f"   💰 متوسط التكلفة  : ${self.avg_cost:.6f}/طلب\n"
            f"   ✅ نسبة النجاح    : {self.success_rate:.1%}"
        )


# ============================================================
# المتعقب الرئيسي
# ============================================================

class LatencyCostTracker:
    """
    متعقب الأداء والتكلفة

    مثال الاستخدام:
        tracker = LatencyCostTracker()
        req_id  = tracker.start_request("ما إجمالي فاتورة INV-101؟")

        with tracker.track_stage(req_id, PipelineStage.RETRIEVAL):
            results = search(...)

        metric = tracker.end_request(req_id)
        report = tracker.generate_report()
        print(report)
    """

    def __init__(
        self,
        model: str | None = None,
        custom_pricing: dict[str, dict[str, float]] | None = None,
        log_dir: str | None = None,
    ):
        self._model   = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._pricing = {**MODEL_PRICING}
        if custom_pricing:
            self._pricing.update(custom_pricing)

        self._active   : dict[str, dict]       = {}
        self._completed: list[RequestMetric]   = []
        self._current_request_id: str | None   = None

        self._log_dir = Path(log_dir) if log_dir else Path("logs/evaluation")
        self._log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"✅ LatencyCostTracker جاهز | "
            f"النموذج: {self._model} | "
            f"نماذج مُسعَّرة: {len(self._pricing)}"
        )

    # ──────────────────────────────────────────────
    # بدء وإنهاء الطلبات
    # ──────────────────────────────────────────────

    @contextmanager
    def track_request(self, query: str, request_id: str | None = None):
        """
        Context manager لتتبع طلب كامل.

        يدعم نمط الاستخدام في `src/agents/langgraph_pipeline.py`:
            with tracker.track_request(query):
                with tracker.track_stage(PipelineStage.GATEKEEPER):
                    ...
        """
        prev = self._current_request_id
        rid = self.start_request(query=query, request_id=request_id)
        self._current_request_id = rid
        try:
            yield rid
        finally:
            try:
                self.end_request(rid)
            finally:
                self._current_request_id = prev

    def start_request(self, query: str, request_id: str | None = None) -> str:
        if not request_id:
            import uuid
            request_id = str(uuid.uuid4())

        self._active[request_id] = {
            "query"      : query,
            "start_time" : time.time(),
            "stages"     : [],
        }
        logger.debug(f"⏱️ بدء تتبع: {request_id[:8]} | {query[:50]}")
        return request_id

    def end_request(self, request_id: str) -> RequestMetric:
        if request_id not in self._active:
            logger.warning(f"⚠️ طلب غير موجود: {request_id[:8]}")
            return RequestMetric(request_id=request_id, query="", success=False)

        data          = self._active.pop(request_id)
        total_latency = (time.time() - data["start_time"]) * 1000
        stages        = data["stages"]

        metric = RequestMetric(
            request_id    = request_id,
            query         = data["query"],
            stages        = stages,
            total_latency = round(total_latency, 2),
            total_tokens  = sum(s.total_tokens for s in stages),
            total_cost    = round(sum(s.cost_usd for s in stages), 8),
            success       = all(s.success for s in stages),
        )

        self._completed.append(metric)
        self._save_metric(metric)
        logger.info(f"⏱️ اكتمل: {metric}")
        return metric

    # ──────────────────────────────────────────────
    # تسجيل مراحل
    # ──────────────────────────────────────────────

    def record_stage(
        self,
        request_id   : str,
        stage        : PipelineStage,
        latency_ms   : float = 0.0,
        input_tokens : int   = 0,
        output_tokens: int   = 0,
        model        : str   = "",
        success      : bool  = True,
        error        : str   = "",
    ):
        if request_id not in self._active:
            return

        used_model   = model or self._model
        total_tokens = input_tokens + output_tokens
        cost         = self._calculate_cost(used_model, input_tokens, output_tokens)

        metric = StageMetric(
            stage         = stage,
            latency_ms    = round(latency_ms, 2),
            input_tokens  = input_tokens,
            output_tokens = output_tokens,
            total_tokens  = total_tokens,
            cost_usd      = cost,
            model         = used_model,
            success       = success,
            error         = error,
        )
        self._active[request_id]["stages"].append(metric)
        logger.debug(f"  📏 {metric}")

    @contextmanager
    def track_stage(
        self,
        request_id_or_stage: str | PipelineStage,
        stage: PipelineStage | None = None,
        model: str = "",
    ):
        """
        Context manager لقياس مرحلة تلقائياً.

        يدعم نمطين:
        - track_stage(request_id, PipelineStage.RETRIEVAL)
        - track_stage(PipelineStage.RETRIEVAL)  (يعتمد على track_request لتحديد request_id الحالي)
        """
        if stage is None:
            # تم الاستدعاء كـ track_stage(PipelineStage.X)
            if not isinstance(request_id_or_stage, PipelineStage):
                raise TypeError("track_stage(stage) يتطلب PipelineStage كوسيط أول")
            if not self._current_request_id:
                raise RuntimeError("لا يوجد request نشط. استخدم track_request() أو مرّر request_id صراحةً.")
            request_id = self._current_request_id
            stage = request_id_or_stage
        else:
            request_id = request_id_or_stage  # type: ignore[assignment]

        ctx = {"input_tokens": 0, "output_tokens": 0, "success": True, "error": ""}
        start = time.time()
        try:
            yield ctx
        except Exception as e:
            ctx["success"] = False
            ctx["error"]   = str(e)
            raise
        finally:
            self.record_stage(
                request_id    = request_id,
                stage         = stage,
                latency_ms    = (time.time() - start) * 1000,
                input_tokens  = ctx["input_tokens"],
                output_tokens = ctx["output_tokens"],
                model         = model,
                success       = ctx["success"],
                error         = ctx["error"],
            )

    # ──────────────────────────────────────────────
    # حساب التكلفة
    # ──────────────────────────────────────────────

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = self._pricing.get(model)
        if not pricing:
            return 0.0
        return round(
            (input_tokens  / 1_000_000) * pricing["input_per_million"] +
            (output_tokens / 1_000_000) * pricing["output_per_million"],
            8,
        )

    def estimate_cost(
        self,
        input_tokens : int,
        output_tokens: int,
        model        : str | None = None,
        num_requests : int = 1,
    ) -> dict:
        used_model  = model or self._model
        single_cost = self._calculate_cost(used_model, input_tokens, output_tokens)
        total_cost  = single_cost * num_requests
        return {
            "model"            : used_model,
            "input_tokens"     : input_tokens,
            "output_tokens"    : output_tokens,
            "cost_per_request" : round(single_cost, 8),
            "num_requests"     : num_requests,
            "total_cost"       : round(total_cost, 6),
            "total_cost_str"   : f"${total_cost:.6f}",
        }

    # ──────────────────────────────────────────────
    # توليد التقارير
    # ──────────────────────────────────────────────

    def generate_report(self, last_n: int | None = None) -> PerformanceReport:
        requests = self._completed
        if last_n:
            requests = requests[-last_n:]

        total = len(requests)
        if total == 0:
            return PerformanceReport()

        latencies        = [r.total_latency for r in requests]
        latencies_sorted = sorted(latencies)
        p95_idx          = min(int(total * 0.95), total - 1)
        p99_idx          = min(int(total * 0.99), total - 1)

        # ── لكل مرحلة ──
        stage_latencies: dict[str, list[float]] = {}
        stage_tokens:    dict[str, list[int]]   = {}
        for req in requests:
            for s in req.stages:
                name = s.stage.value
                stage_latencies.setdefault(name, []).append(s.latency_ms)
                stage_tokens.setdefault(name, []).append(s.total_tokens)

        total_req   = len(requests)
        total_tokens = sum(r.total_tokens for r in requests)
        total_cost   = sum(r.total_cost   for r in requests)
        success_count = sum(1 for r in requests if r.success)

        report = PerformanceReport(
            requests          = requests,
            total_requests    = total_req,
            avg_latency       = round(statistics.mean(latencies), 2),
            median_latency    = round(statistics.median(latencies), 2),
            p95_latency       = round(latencies_sorted[p95_idx], 2),
            p99_latency       = round(latencies_sorted[p99_idx], 2),
            min_latency       = round(min(latencies), 2),
            max_latency       = round(max(latencies), 2),
            total_tokens      = total_tokens,
            avg_tokens        = round(total_tokens / total_req, 1),
            total_cost        = round(total_cost, 6),
            avg_cost          = round(total_cost / total_req, 8),
            stage_avg_latency = {
                n: round(statistics.mean(v), 2)
                for n, v in stage_latencies.items()
            },
            stage_avg_tokens  = {
                n: round(statistics.mean(v), 1)
                for n, v in stage_tokens.items()
            },
            success_rate      = round(success_count / total_req, 4),
        )

        logger.success(f"\n{report}")
        return report

    # ──────────────────────────────────────────────
    # تحليل عنق الزجاجة
    # ──────────────────────────────────────────────

    def analyze_bottleneck(self, last_n: int | None = None) -> dict:
        requests = self._completed[-last_n:] if last_n else self._completed
        if not requests:
            return {"message": "لا توجد بيانات"}

        stage_totals: dict[str, float] = {}
        stage_counts: dict[str, int]   = {}

        for req in requests:
            for s in req.stages:
                name = s.stage.value
                stage_totals[name] = stage_totals.get(name, 0) + s.latency_ms
                stage_counts[name] = stage_counts.get(name, 0) + 1

        total_time = sum(stage_totals.values()) or 1

        analysis = sorted(
            [
                {
                    "stage"      : name,
                    "total_ms"   : round(stage_totals[name], 2),
                    "avg_ms"     : round(stage_totals[name] / stage_counts[name], 2),
                    "percentage" : round(stage_totals[name] / total_time * 100, 1),
                    "count"      : stage_counts[name],
                }
                for name in stage_totals
            ],
            key=lambda x: x["total_ms"],
            reverse=True,
        )

        bottleneck = analysis[0]["stage"] if analysis else ""
        if analysis:
            logger.info(
                f"🐌 عنق الزجاجة: {bottleneck} "
                f"({analysis[0]['percentage']:.1f}% من الوقت)"
            )

        return {
            "bottleneck"    : bottleneck,
            "stages"        : analysis,
            "total_time_ms" : round(total_time, 2),
        }

    # ──────────────────────────────────────────────
    # حفظ وتصدير
    # ──────────────────────────────────────────────

    def _save_metric(self, metric: RequestMetric):
        try:
            log_file = (
                self._log_dir
                / f"latency_cost_{datetime.now().strftime('%Y%m%d')}.jsonl"
            )
            entry = {
                "timestamp"     : metric.timestamp,
                "request_id"    : metric.request_id,
                "query"         : metric.query,
                "total_latency" : metric.total_latency,
                "total_tokens"  : metric.total_tokens,
                "total_cost"    : metric.total_cost,
                "success"       : metric.success,
                "bottleneck"    : metric.bottleneck,
                "stages"        : [
                    {
                        "stage"     : s.stage.value,
                        "latency_ms": s.latency_ms,
                        "tokens"    : s.total_tokens,
                        "cost"      : s.cost_usd,
                        "model"     : s.model,
                        "success"   : s.success,
                    }
                    for s in metric.stages
                ],
            }
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"⚠️ فشل حفظ القياسات: {e}")

    def get_stats(self) -> dict:
        total = len(self._completed)
        if total == 0:
            return {"total_requests": 0, "message": "لا توجد قياسات بعد"}
        latencies = [r.total_latency for r in self._completed]
        return {
            "total_requests"   : total,
            "avg_latency_ms"   : round(statistics.mean(latencies), 2),
            "median_latency_ms": round(statistics.median(latencies), 2),
            "total_tokens"     : sum(r.total_tokens for r in self._completed),
            "total_cost_usd"   : round(sum(r.total_cost for r in self._completed), 6),
            "success_rate"     : f"{sum(1 for r in self._completed if r.success)/total:.1%}",
            "active_requests"  : len(self._active),
        }

    def export_report(
        self,
        report  : PerformanceReport | None = None,
        filepath: str | None = None,
    ) -> str:
        if report is None:
            report = self.generate_report()

        if not filepath:
            filepath = str(
                self._log_dir
                / f"perf_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        data = {
            "generated_at"    : datetime.now(timezone.utc).isoformat(),
            "summary"         : {
                "total_requests" : report.total_requests,
                "avg_latency_ms" : report.avg_latency,
                "median_latency" : report.median_latency,
                "p95_latency"    : report.p95_latency,
                "p99_latency"    : report.p99_latency,
                "min_latency"    : report.min_latency,
                "max_latency"    : report.max_latency,
                "total_tokens"   : report.total_tokens,
                "avg_tokens"     : report.avg_tokens,
                "total_cost_usd" : report.total_cost,
                "avg_cost_usd"   : report.avg_cost,
                "success_rate"   : report.success_rate,
            },
            "stage_performance": {
                "avg_latency_ms": report.stage_avg_latency,
                "avg_tokens"    : report.stage_avg_tokens,
            },
            "bottleneck"      : self.analyze_bottleneck(),
            "per_request"     : [
                {
                    "request_id"  : r.request_id[:8],
                    "query"       : r.query[:80],
                    "total_latency": r.total_latency,
                    "total_tokens" : r.total_tokens,
                    "total_cost"   : r.total_cost,
                    "success"      : r.success,
                    "bottleneck"   : r.bottleneck,
                }
                for r in report.requests
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, ensure_ascii=False, indent=2, fp=f)

        logger.info(f"📄 تقرير الأداء مُصدَّر: {filepath}")
        return filepath