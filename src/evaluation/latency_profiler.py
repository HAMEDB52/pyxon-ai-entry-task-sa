# ============================================================
# src/evaluation/latency_profiler.py
#
# أداة تحليل الكمون (latency) لكل مرحلة من مراحل المعالجة
#
# تتبع:
# - Query Expansion (توسيع الاستعلام)
# - Hybrid Search (البحث الهجين)
# - Reranking (إعادة الترتيب)
# - Compression (الضغط السياقي)
# - Total (الإجمالي)
# ============================================================

import time
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
from loguru import logger


@dataclass
class LatencyStats:
    """إحصائيات الكمون لاستعلام واحد"""
    query: str
    timestamp: str = ""  # Default empty, will be set in __post_init__
    
    # الكمونات بالثواني
    expansion_time: float = 0.0
    search_time: float = 0.0
    reranking_time: float = 0.0
    compression_time: float = 0.0
    total_time: float = 0.0
    
    # النتائج
    num_candidates: int = 0
    num_final_results: int = 0
    
    # كلمات مفتاحية
    tags: List[str] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []
    
    @property
    def breakdown(self) -> Dict[str, float]:
        """نسبة الكمون لكل مرحلة"""
        total = self.total_time if self.total_time > 0 else 1e-6
        return {
            "expansion%": round(100 * self.expansion_time / total, 1),
            "search%": round(100 * self.search_time / total, 1),
            "reranking%": round(100 * self.reranking_time / total, 1),
            "compression%": round(100 * self.compression_time / total, 1),
        }


class LatencyProfiler:
    """
    أداة تحليل الكمون - تتبع أداء كل مرحلة من مراحل RAG
    
    الاستخدام:
        profiler = LatencyProfiler()
        
        with profiler.track_phase("search"):
            results = search_engine.search(query)
        
        with profiler.track_phase("reranking"):
            ranked = rerank(results)
        
        stats = profiler.get_stats()
        print(stats.breakdown)
    """
    
    def __init__(self, log_file: Optional[Path] = None):
        self.stats: Optional[LatencyStats] = None
        self.current_query: str = ""
        self.phase_times: Dict[str, float] = {}
        self.phase_start: Optional[float] = None
        self.current_phase: Optional[str] = None
        
        # ملف السجل
        self.log_file = log_file or Path("logs/latency_cost") / f"latency_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def start_query(self, query: str):
        """ابدأ تتبع استعلام جديد"""
        self.current_query = query
        self.phase_times.clear()
        self.phase_start = None
        self.current_phase = None
        
        self.stats = LatencyStats(query=query)
    
    def track_phase(self, phase_name: str):
        """Context manager لتتبع مرحلة معينة"""
        class PhaseTracker:
            def __init__(self, profiler, phase):
                self.profiler = profiler
                self.phase = phase
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, *args):
                elapsed = time.time() - self.start_time
                self.profiler.record_phase(self.phase, elapsed)
        
        return PhaseTracker(self, phase_name)
    
    def record_phase(self, phase_name: str, elapsed: float):
        """سجّل كمون مرحلة"""
        if self.stats is None:
            logger.warning("⚠️ ابدأ استعلام أولاً مع start_query()")
            return
        
        self.phase_times[phase_name] = elapsed
        
        # حدّث الـ stats حسب المرحلة
        if phase_name == "expansion":
            self.stats.expansion_time = elapsed
        elif phase_name == "search":
            self.stats.search_time = elapsed
        elif phase_name == "reranking":
            self.stats.reranking_time = elapsed
        elif phase_name == "compression":
            self.stats.compression_time = elapsed
        
        logger.debug(f"⏱️  {phase_name}: {elapsed:.3f}s")
    
    def finalize(self, num_candidates: int = 0, num_results: int = 0, tags: List[str] = None):
        """أنهِ تتبع الاستعلام"""
        if self.stats is None:
            return None
        
        self.stats.total_time = sum(self.phase_times.values())
        self.stats.num_candidates = num_candidates
        self.stats.num_final_results = num_results
        self.stats.tags = tags or []
        
        # سجّل في الملف
        self._save_stats()
        
        return self.stats
    
    def _save_stats(self):
        """احفظ الإحصائيات في ملف JSONL"""
        if self.stats is None:
            return
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(self.stats), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"❌ فشل حفظ latency stats: {e}")
    
    def get_stats(self) -> Optional[LatencyStats]:
        """احصل على إحصائيات الاستعلام الحالي"""
        return self.stats
    
    def print_breakdown(self):
        """اطبع نسبة الكمونات"""
        if self.stats is None:
            logger.warning("⚠️ لا توجد إحصائيات")
            return
        
        breakdown = self.stats.breakdown
        logger.info(
            f"⏱️  Latency Breakdown ({self.stats.total_time:.2f}s total):\n"
            f"  📝 Expansion: {breakdown['expansion%']}%\n"
            f"  🔍 Search: {breakdown['search%']}%\n"
            f"  🏆 Reranking: {breakdown['reranking%']}%\n"
            f"  📦 Compression: {breakdown['compression%']}%"
        )
    
    @staticmethod
    def summarize_log(log_file: Path, top_k: int = 10) -> Dict:
        """
        حلل ملف السجل واخرج إحصائيات مجملة
        
        Args:
            log_file: ملف JSONL به السجلات
            top_k: عدد أطول الاستعلامات
        
        Returns:
            dict بالإحصائيات المجملة
        """
        if not log_file.exists():
            return {"error": "Log file not found"}
        
        entries = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except:
                    pass
        
        if not entries:
            return {"error": "No entries found"}
        
        # احسب الإحصائيات
        total_times = [e.get("total_time", 0) for e in entries]
        expansion_times = [e.get("expansion_time", 0) for e in entries]
        search_times = [e.get("search_time", 0) for e in entries]
        reranking_times = [e.get("reranking_time", 0) for e in entries]
        
        summary = {
            "total_queries": len(entries),
            "avg_total_time": round(sum(total_times) / len(total_times), 3),
            "min_total_time": round(min(total_times), 3),
            "max_total_time": round(max(total_times), 3),
            "p95_total_time": round(sorted(total_times)[int(len(total_times) * 0.95)], 3),
            
            "avg_expansion_time": round(sum(expansion_times) / len(expansion_times), 3),
            "avg_search_time": round(sum(search_times) / len(search_times), 3),
            "avg_reranking_time": round(sum(reranking_times) / len(reranking_times), 3),
            
            "slowest_queries": [
                {
                    "query": e.get("query"),
                    "total_time": e.get("total_time"),
                    "breakdown": {
                        k: v for k, v in e.items()
                        if "time" in k and k != "total_time"
                    }
                }
                for e in sorted(entries, key=lambda x: x.get("total_time", 0), reverse=True)[:top_k]
            ]
        }
        
        return summary


# ════════════════════════════════════
# Singleton instance
# ════════════════════════════════════

_profiler_instance: Optional[LatencyProfiler] = None

def get_profiler() -> LatencyProfiler:
    """احصل على singleton instance"""
    global _profiler_instance
    if _profiler_instance is None:
        _profiler_instance = LatencyProfiler()
    return _profiler_instance
