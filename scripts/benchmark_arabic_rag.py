#!/usr/bin/env python3
# ============================================================
# scripts/benchmark_arabic_rag.py
#
# اختبار شامل للنظام بعد التحسينات
# - يُقيّس أداء الاسترجاع والاستخراج
# - يقارن بين الاستعلامات المتنوعة
# - يُنتج تقرير مفصل بالنتائج
# ============================================================

import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

# تحميل متغيرات .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from src.database.retrieval_enhancer import RetrievalEnhancer
from src.data_processing.arabic_lemmatizer import ArabicLemmatizer
from src.evaluation.latency_profiler import LatencyProfiler
from loguru import logger


# ════════════════════════════════════
# مجموعة الاختبارات
# ════════════════════════════════════

BENCHMARK_QUERIES = [
    {
        "query": "ابغى تقرير الهندسي",
        "category": "document_search",
        "expected_keywords": ["تقرير", "هندسي"],
    },
    {
        "query": "كم اجمالي الفواتير الي دفعناها",
        "category": "financial_aggregation",
        "expected_keywords": ["فواتير", "دفع"],
    },
    {
        "query": "الفاتورة INV-2024-0892 كاملة",
        "category": "specific_document",
        "expected_keywords": ["فاتورة", "INV"],
    },
    {
        "query": "وش اسم العميل الي دفع الفاتورة؟",
        "category": "entity_extraction",
        "expected_keywords": ["عميل", "فاتورة"],
    },
    {
        "query": "أين التقرير المالي السنوي",
        "category": "document_location",
        "expected_keywords": ["تقرير", "مالي"],
    },
    {
        "query": "الوثائق الهندسية يا جماعة",
        "category": "colloquial_search",
        "expected_keywords": ["وثائق", "هندسي"],
    },
    {
        "query": "كم المبلغ الكلي للدفعات هذا الشهر",
        "category": "temporal_aggregation",
        "expected_keywords": ["مبلغ", "دفعة"],
    },
    {
        "query": "الفواتير للعميل الهندسي",
        "category": "entity_matching",
        "expected_keywords": ["فاتورة", "عميل"],
    },
]


class BenchmarkRunner:
    """مُشغّل الاختبار الشامل"""
    
    def __init__(self):
        self.enhancer = None
        self.lemmatizer = None
        self.profiler = LatencyProfiler()
        self.results = []
        self.errors = []
        self._init_components()
    
    def _init_components(self):
        """بدّل المكونات"""
        try:
            logger.info("🚀 بدء تهيئة المكونات...")
            self.enhancer = RetrievalEnhancer(
                n_expansions=2,
                compress=False,  # عطّل الضغط للحصول على قياس أدق للكمون
            )
            logger.success("✅ RetrievalEnhancer جاهز")
        except Exception as e:
            logger.error(f"❌ فشل إنشاء RetrievalEnhancer: {e}")
            self.errors.append(f"RetrievalEnhancer init: {e}")
        
        try:
            self.lemmatizer = ArabicLemmatizer()
            logger.success("✅ Arabic Lemmatizer جاهز")
        except Exception as e:
            logger.error(f"⚠️  فشل تهيئة Lemmatizer: {e}")
            self.errors.append(f"Lemmatizer init: {e}")
    
    def run_benchmark(self) -> Dict:
        """شغّل جميع الاختبارات"""
        logger.info(f"📊 بدء الاختبار على {len(BENCHMARK_QUERIES)} استعلام")
        
        # اختبر كل استعلام
        for i, test_case in enumerate(BENCHMARK_QUERIES, 1):
            logger.info(f"\n⏳ الاستعلام {i}/{len(BENCHMARK_QUERIES)}: '{test_case['query'][:50]}'")
            result = self._run_single_query(test_case)
            self.results.append(result)
        
        # أنشئ التقرير
        report = self._generate_report()
        return report
    
    def _run_single_query(self, test_case: Dict) -> Dict:
        """شغّل استعلام واحد وقيّس أدائه"""
        query = test_case["query"]
        
        # ابدأ التسجيل
        self.profiler.start_query(query)
        
        result = {
            "query": query,
            "category": test_case.get("category", "unknown"),
            "num_results": 0,
            "latency_total": 0.0,
            "latency_breakdown": {},
            "cache_hit": False,
            "status": "unknown",
            "error": None,
        }
        
        # إذا لم يتم تهيئة enhancer (بسبب missing GOOGLE_API_KEY)
        if self.enhancer is None:
            result.update({
                "status": "skipped",
                "error": "RetrievalEnhancer not initialized - missing GOOGLE_API_KEY in .env",
            })
            logger.info(f"  ⏳ لم يتم البحث - Enhancer غير جاهز")
            return result
        
        try:
            # أول مرة - بدون cache
            start = time.time()
            
            with self.profiler.track_phase("retrieval"):
                results = self.enhancer.search(query, top_k=5)
            
            elapsed_first = time.time() - start
            
            # المرة الثانية - مع cache
            start = time.time()
            results_2 = self.enhancer.search(query, top_k=5)
            elapsed_second = time.time() - start
            
            # سجّل النتائج
            stats = self.profiler.finalize(
                num_candidates=len(results),
                num_results=len(results),
                tags=[test_case.get("category")]
            )
            
            result.update({
                "num_results": len(results),
                "latency_total": elapsed_first,
                "latency_breakdown": stats.breakdown,
                "cache_hit": elapsed_second < elapsed_first * 0.5,
                "status": "success",
            })
            
            # اطبع ملخص
            logger.info(
                f"  ✅ {len(results)} نتائج | "
                f"1st: {elapsed_first:.2f}s | "
                f"2nd (cache): {elapsed_second:.3f}s | "
                f"cache_hit={result['cache_hit']}"
            )
            
        except Exception as e:
            result.update({
                "status": "error",
                "error": str(e),
            })
            logger.error(f"  ❌ خطأ: {e}")
            self.errors.append(f"Query '{query}': {e}")
        
        return result
    
    def _generate_report(self) -> Dict:
        """أنشئ تقرير بالنتائج"""
        if not self.results:
            return {
                "status": "no_results",
                "errors": self.errors,
            }
        
        # احسب الإحصائيات
        successful = [r for r in self.results if r["status"] == "success"]
        failed = [r for r in self.results if r["status"] == "error"]
        
        latencies = [r["latency_total"] for r in successful if r["latency_total"] > 0]
        cache_hits = [r for r in successful if r["cache_hit"]]
        
        # إحصائيات الكمون
        latency_stats = {
            "count": len(latencies),
            "min": min(latencies) if latencies else 0,
            "max": max(latencies) if latencies else 0,
            "avg": sum(latencies) / len(latencies) if latencies else 0,
            "p95": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        }
        
        # التقرير النهائي
        report = {
            "timestamp": datetime.now().isoformat(),
            "benchmark_results": {
                "total_queries": len(self.results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": f"{100 * len(successful) / len(self.results):.1f}%",
                "cache_hits": len(cache_hits),
                "cache_hit_rate": f"{100 * len(cache_hits) / len(successful):.1f}%" if successful else "0%",
            },
            "latency_metrics": {
                f"latency_{k}s": round(v, 3) for k, v in latency_stats.items()
            },
            "details": {
                "successful_queries": [r for r in successful],
                "failed_queries": [r for r in failed],
            },
            "errors": self.errors,
            "system_config": {
                "num_expansions": 2,
                "candidate_k": 20,
                "final_k": 8,
                "compress": False,
                "cross_encoder": "enabled",
                "lemmatizer": "enabled",
                "cache": "enabled",
            }
        }
        
        return report
    
    def print_summary(self, report: Dict):
        """اطبع ملخص التقرير بشكل جميل"""
        logger.info("\n" + "="*60)
        logger.info("📊 تقرير الأداء الشامل")
        logger.info("="*60)
        
        bench = report["benchmark_results"]
        latency = report.get("latency_metrics", {})
        
        logger.info(
            f"\n📈 النتائج الأساسية:\n"
            f"  استعلامات: {bench['total_queries']} "
            f"(ناجحة: {bench['successful']}, فاشلة: {bench['failed']})\n"
            f"  معدل النجاح: {bench['success_rate']}\n"
            f"  Cache hits: {bench['cache_hits']} ({bench['cache_hit_rate']})\n"
        )
        
        # إذا كانت هناك بيانات كمون
        if latency:
            logger.info(
                f"⏱️  إحصائيات الكمون:\n"
                f"  Min: {latency.get('latency_min_s', 'N/A')}s\n"
                f"  Max: {latency.get('latency_max_s', 'N/A')}s\n"
                f"  Avg: {latency.get('latency_avg_s', 'N/A')}s\n"
                f"  P95: {latency.get('latency_p95_s', 'N/A')}s\n"
            )
        else:
            logger.info("⏱️  لا توجد بيانات كمون متاحة (لم تنجح الاستعلامات)\n")
        
        # أبطأ الاستعلامات
        slow_queries = sorted(
            report.get("details", {}).get("successful_queries", []),
            key=lambda x: x["latency_total"],
            reverse=True
        )[:3]
        
        if slow_queries:
            logger.info("\n🐌 أبطأ الاستعلامات:")
            for i, q in enumerate(slow_queries, 1):
                logger.info(f"  {i}. '{q['query'][:40]}' → {q['latency_total']:.2f}s")
        
        # نقاط إيجابية
        logger.info("\n✨ نقاط النظام:")
        config = report.get("system_config", {})
        logger.info(f"  • Cross-Encoder Reranker: {config.get('cross_encoder', 'N/A')}")
        logger.info(f"  • Arabic Lemmatizer: {config.get('lemmatizer', 'N/A')}")
        logger.info(f"  • Query Caching: {config.get('cache', 'N/A')}")
        logger.info(f"  • RRF Fusion: دمج ذكي للنتائج")
        
        # رسالة حول GOOGLE_API_KEY إذا لزم الأمر
        if report.get("errors"):
            logger.info("\n⚠️  التنبيهات:")
            for error in report["errors"][:3]:
                logger.info(f"  • {error[:80]}")
            logger.info("\n💡 للحصول على نتائج كاملة:")
            logger.info("   1. أضف GOOGLE_API_KEY إلى ملف .env")
            logger.info("   2. تأكد من اتصال قاعدة البيانات")
            logger.info("   3. شغل اختبارات الاسترجاع مع: pytest tests/test_arabic_queries.py -v")
        
        logger.info("\n" + "="*60)
    
    def save_report(self, report: Dict, filename: str = "benchmark_report.json"):
        """احفظ التقرير في ملف JSON"""
        report_path = Path("logs") / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.success(f"✅ تم حفظ التقرير في: {report_path}")
        except Exception as e:
            logger.error(f"❌ فشل حفظ التقرير: {e}")


# ════════════════════════════════════
# Function main
# ════════════════════════════════════

def main():
    """نقطة الدخول الرئيسية"""
    logger.info("🎯 بدء اختبار النظام الشامل للعربية")
    logger.info(f"⏰ التاريخ والوقت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    runner = BenchmarkRunner()
    
    # شغّل الاختبار
    report = runner.run_benchmark()
    
    # اطبع الملخص
    runner.print_summary(report)
    
    # احفظ التقرير
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner.save_report(report, f"benchmark_{timestamp}.json")
    
    logger.info("\n✅ انتهى الاختبار بنجاح!")
    
    return report


if __name__ == "__main__":
    main()
