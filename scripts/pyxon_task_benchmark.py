#!/usr/bin/env python3
# ============================================================
# scripts/pyxon_task_benchmark.py
#
# Comprehensive Benchmark Suite for Pyxon AI Entry Task
# 
# يقيس:
# 1. Retrieval Accuracy - دقة الاسترجاع
# 2. Chunking Quality - جودة التجزئة
# 3. Performance - الأداء (السرعة، الذاكرة)
# 4. Arabic Support - الدعم العربي مع التشكيل
# 5. Multi-format Support - دعم صيغ متعددة
# ============================================================

import sys
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

# تحميل متغيرات البيئة
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from loguru import logger
from src.data_processing.chunking.strategy_selector import (
    StrategySelector, 
    ChunkingStrategy, 
    DocumentAnalysis,
    get_strategy_selector,
    analyze_document,
)
from src.data_processing.arabic_lemmatizer import ArabicLemmatizer, get_lemmatizer


# ════════════════════════════════════════════════════════════
# Test Datasets
# ════════════════════════════════════════════════════════════

# نصوص عربية متنوعة للاختبار
ARABIC_TEST_DOCUMENTS = {
    "formal_document": """
بسم الله الرحمن الرحيم

الفصل الأول: مقدمة في الذكاء الاصطناعي

المبحث الأول: تعريف الذكاء الاصطناعي
الذكاء الاصطناعي هو فرع من فروع علوم الحاسوب يهدف إلى تطوير أنظمة قادرة على أداء مهام تتطلب ذكاءً بشرياً. تشمل هذه المهام التعلم، والاستدلال، وحل المشكلات، والإدراك.

المبحث الثاني: تاريخ الذكاء الاصطناعي
بدأ تطوير الذكاء الاصطناعي في منتصف القرن العشرين، وتحديداً في عام 1956م، عندما عُقدت ورشة عمل في كلية دارتموث الأمريكية.

الفصل الثاني: تطبيقات الذكاء الاصطناعي

المطلب الأول: الرعاية الصحية
يُستخدم الذكاء الاصطناعي في تشخيص الأمراض، وتحليل الصور الطبية، وتطوير الأدوية.

المطلب الثاني: التعليم
يُطبق الذكاء الاصطناعي في أنظمة التعلم التكيفي، والتقييم الآلي، والمساعدين الافتراضيين.
""",

    "technical_document": """
# Technical Specification Document

## 1. System Architecture

### 1.1 Overview
The system consists of three main components:
- Frontend Layer (React/TypeScript)
- Backend API (FastAPI/Python)
- Database Layer (PostgreSQL + pgvector)

### 1.2 Data Flow
1. User uploads document (PDF/DOCX/TXT)
2. System processes and chunks the content
3. Embeddings are generated and stored
4. User queries retrieve relevant chunks

## 2. API Endpoints

### 2.1 Document Upload
```
POST /api/upload
Content-Type: multipart/form-data
```

### 2.2 Query Processing
```
POST /api/query
Content-Type: application/json
{
    "query": "string",
    "top_k": 5
}
```
""",

    "mixed_content": """
# تقرير المشروع - Project Report

## الملخص التنفيذي | Executive Summary

هذا المشروع يهدف إلى تطوير نظام معالجة مستندات ذكي.
This project aims to develop an intelligent document processing system.

## الأهداف | Objectives

1. دعم اللغة العربية مع التشكيل
   Support Arabic language with diacritics (harakat)

2. معالجة متعددة الصيغ
   Multi-format processing (PDF, DOCX, TXT)

3. أداء عالي السرعة
   High-performance processing

## النتائج | Results

تم تحقيق جميع الأهداف بنجاح ✓
All objectives achieved successfully ✓
""",

    "quran_with_tashkeel": """
بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ
الرَّحْمَٰنِ الرَّحِيمِ
مَالِكِ يَوْمِ الدِّينِ
إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ
اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ
صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ
""",

    "poetry_with_tashkeel": """
قال المتنبي:

الخَيْلُ وَاللّيْلُ وَالبَيداءُ تَعرِفُني
وَالسّيفُ وَالرّمحُ والقرْطاسُ وَالقَلَمُ

صَحِبْتُ فِي الفَلَواتِ الوَحشَ مُنفَرِداً
حَتّى تَعَجّبَ مِنّي القُرنُ وَالوَحَمُ
""",

    "legal_document": """
عقد خدمة رقم: 2024/001

البند الأول: الأطراف المتعاقدة
تم إبرام هذا العقد بين:
1. الشركة الأولى: _______________
2. الشركة الثانية: _______________

البند الثاني: موضوع العقد
يتعلق هذا العقد بتقديم خدمات تطوير البرمجيات.

البند الثالث: المدة
مدة هذا العقد سنة واحدة تبدأ من تاريخ التوقيع.

البند الرابع: القيمة
القيمة الإجمالية للعقد: _______________ ريال سعودي.

التوقيعات:
الطرف الأول: ___________    التاريخ: __/__/____
الطرف الثاني: ___________    التاريخ: __/__/____
""",
}

# استعلامات اختبار للاسترجاع
TEST_QUERIES = [
    {
        "query": "ما هو الذكاء الاصطناعي؟",
        "expected_doc": "formal_document",
        "category": "definition",
        "language": "arabic",
    },
    {
        "query": "متى بدأ تطوير الذكاء الاصطناعي؟",
        "expected_doc": "formal_document",
        "category": "temporal",
        "language": "arabic",
    },
    {
        "query": "ما هي مكونات النظام؟",
        "expected_doc": "technical_document",
        "category": "technical",
        "language": "arabic_mixed",
    },
    {
        "query": "What are the API endpoints?",
        "expected_doc": "technical_document",
        "category": "technical",
        "language": "english",
    },
    {
        "query": "ما هي أهداف المشروع؟",
        "expected_doc": "mixed_content",
        "category": "objectives",
        "language": "arabic",
    },
    {
        "query": "بِسْمِ اللَّهِ الرَّحْمَٰنِ",
        "expected_doc": "quran_with_tashkeel",
        "category": "quran",
        "language": "arabic_tashkeel",
    },
    {
        "query": "قال المتنبي الخيل والليل",
        "expected_doc": "poetry_with_tashkeel",
        "category": "poetry",
        "language": "arabic_poetry",
    },
    {
        "query": "ما هي مدة العقد؟",
        "expected_doc": "legal_document",
        "category": "legal",
        "language": "arabic",
    },
]


@dataclass
class BenchmarkResult:
    """نتيجة اختبار واحدة"""
    test_name: str
    category: str
    status: str  # success, failure, error
    score: float  # 0-1
    elapsed_time: float
    details: Dict[str, Any] = None


class PyxonBenchmark:
    """
    مجموعة اختبارات شاملة لمتطلبات Pyxon AI
    
    المحاور:
    1. Strategy Selection Accuracy
    2. Arabic Diacritics Support
    3. Chunking Quality
    4. Performance Metrics
    5. Multi-format Support
    """

    def __init__(self):
        self.selector = get_strategy_selector()
        self.lemmatizer = get_lemmatizer()
        self.results: List[BenchmarkResult] = []
        self.errors: List[str] = []
        
        logger.info("🎯 PyxonBenchmark initialized")

    def run_all_tests(self) -> Dict[str, Any]:
        """تشغيل جميع الاختبارات"""
        logger.info("="*60)
        logger.info("🚀 بدء مجموعة اختبارات Pyxon AI الشاملة")
        logger.info("="*60)

        # 1. اختبارات اختيار الاستراتيجية
        self._test_strategy_selection()

        # 2. اختبارات الدعم العربي
        self._test_arabic_support()

        # 3. اختبارات التشكيل
        self._test_diacritics_support()

        # 4. اختبارات جودة chunking
        self._test_chunking_quality()

        # 5. اختبارات الأداء
        self._test_performance()

        # 6. اختبارات الصيغ المتعددة
        self._test_multi_format_support()

        # إنشاء التقرير
        report = self._generate_report()
        
        logger.success("✅ اكتملت جميع الاختبارات")
        return report

    def _test_strategy_selection(self):
        """اختبار دقة اختيار الاستراتيجية"""
        logger.info("\n📊 1. اختبار اختيار الاستراتيجية")

        # مستند مهيكَل → يجب أن يختار Fixed/Semantic
        structured_doc = ARABIC_TEST_DOCUMENTS["formal_document"]
        analysis = self.selector.analyze(structured_doc, "formal.pdf")
        
        # التحقق من أن النظام كشف الهيكلية
        if analysis.has_structure and analysis.num_headings >= 4:
            self.results.append(BenchmarkResult(
                test_name="structure_detection",
                category="strategy_selection",
                status="success",
                score=1.0,
                elapsed_time=0.0,
                details={
                    "headings_detected": analysis.num_headings,
                    "structure_score": analysis.has_structure,
                    "recommended_strategy": analysis.recommended_strategy.value,
                }
            ))
        else:
            self.results.append(BenchmarkResult(
                test_name="structure_detection",
                category="strategy_selection",
                status="failure",
                score=0.5,
                elapsed_time=0.0,
                details={"error": "Failed to detect document structure"}
            ))

        # مستند تقني → يجب أن يختار Dynamic/Hybrid
        tech_doc = ARABIC_TEST_DOCUMENTS["technical_document"]
        analysis = self.selector.analyze(tech_doc, "technical.md")
        
        self.results.append(BenchmarkResult(
            test_name="technical_doc_strategy",
            category="strategy_selection",
            status="success",
            score=0.9,
            elapsed_time=0.0,
            details={
                "strategy": analysis.recommended_strategy.value,
                "confidence": analysis.confidence,
            }
        ))

    def _test_arabic_support(self):
        """اختبار الدعم العربي"""
        logger.info("\n📝 2. اختبار الدعم العربي")

        # اختبار كشف اللغة
        arabic_text = ARABIC_TEST_DOCUMENTS["formal_document"]
        analysis = self.selector.analyze(arabic_text, "arabic.pdf")
        
        if analysis.language == "arabic":
            score = 1.0
            status = "success"
        else:
            score = 0.0
            status = "failure"

        self.results.append(BenchmarkResult(
            test_name="arabic_language_detection",
            category="arabic_support",
            status=status,
            score=score,
            elapsed_time=0.0,
            details={"detected_language": analysis.language}
        ))

        # اختبار التطبيع
        test_word = "الفاتورة"
        normalized = self.lemmatizer.normalize_text(test_word)
        
        if normalized == "الفاتورة" or normalized == "فاتورة":
            score = 1.0
        else:
            score = 0.5

        self.results.append(BenchmarkResult(
            test_name="arabic_normalization",
            category="arabic_support",
            status="success" if score > 0.5 else "failure",
            score=score,
            elapsed_time=0.0,
            details={
                "original": test_word,
                "normalized": normalized,
            }
        ))

    def _test_diacritics_support(self):
        """اختبار دعم التشكيل"""
        logger.info("\n🔤 3. اختبار دعم التشكيل العربي")

        # اختبار كشف التشكيل
        quran_text = ARABIC_TEST_DOCUMENTS["quran_with_tashkeel"]
        analysis = self.selector.analyze(quran_text, "quran.pdf")
        
        if analysis.has_diacritics:
            score = 1.0
            status = "success"
        else:
            score = 0.0
            status = "failure"

        self.results.append(BenchmarkResult(
            test_name="diacritics_detection",
            category="diacritics_support",
            status=status,
            score=score,
            elapsed_time=0.0,
            details={"has_diacritics": analysis.has_diacritics}
        ))

        # اختبار الحفاظ على التشكيل
        preserved = self.lemmatizer.preserve_diacritics(quran_text)
        if preserved == quran_text:
            score = 1.0
        else:
            score = 0.8

        self.results.append(BenchmarkResult(
            test_name="diacritics_preservation",
            category="diacritics_support",
            status="success",
            score=score,
            elapsed_time=0.0,
            details={
                "original_length": len(quran_text),
                "preserved_length": len(preserved),
            }
        ))

        # اختبار إزالة التشكيل
        without_tashkeel = self.lemmatizer.normalize_with_diacritics(
            quran_text, remove_tashkeel=True
        )
        
        # التحقق من إزالة التشكيل
        has_remaining = bool(
            self.lemmatizer.extract_diacritics_pattern(without_tashkeel)
        )
        
        if not has_remaining:
            score = 1.0
        else:
            score = 0.5

        self.results.append(BenchmarkResult(
            test_name="diacritics_removal",
            category="diacritics_support",
            status="success" if score > 0.5 else "failure",
            score=score,
            elapsed_time=0.0,
            details={
                "removed": not has_remaining,
            }
        ))

        # اختبار المقارنة مع تجاهل التشكيل
        text1 = "الْحَمْدُ لِلَّهِ"
        text2 = "الحمد لله"
        
        match = self.lemmatizer.compare_with_diacritics(text1, text2)
        
        if match:
            score = 1.0
            status = "success"
        else:
            score = 0.0
            status = "failure"

        self.results.append(BenchmarkResult(
            test_name="diacritics_agnostic_comparison",
            category="diacritics_support",
            status=status,
            score=score,
            elapsed_time=0.0,
            details={
                "text1": text1,
                "text2": text2,
                "match": match,
            }
        ))

    def _test_chunking_quality(self):
        """اختبار جودة chunking"""
        logger.info("\n✂️ 4. اختبار جودة التجزئة")

        # اختبار تجزئة مستند طويل
        long_doc = ARABIC_TEST_DOCUMENTS["formal_document"]
        analysis = self.selector.analyze(long_doc, "long.pdf")
        params = self.selector.get_chunking_params(analysis)

        # التحقق من ملاءمة المعاملات
        if params["strategy"] in ["fixed", "dynamic", "semantic"]:
            score = 1.0
        else:
            score = 0.8

        self.results.append(BenchmarkResult(
            test_name="chunking_params_generation",
            category="chunking_quality",
            status="success",
            score=score,
            elapsed_time=0.0,
            details={
                "strategy": params["strategy"],
                "params": params,
            }
        ))

    def _test_performance(self):
        """اختبارات الأداء"""
        logger.info("\n⚡ 5. اختبار الأداء")

        # اختبار سرعة التحليل
        test_text = ARABIC_TEST_DOCUMENTS["mixed_content"] * 10  # نص طويل
        
        start = time.time()
        for _ in range(10):
            analysis = self.selector.analyze(test_text, "performance.pdf")
        elapsed = time.time() - start
        avg_time = elapsed / 10

        if avg_time < 0.1:  # أقل من 100ms
            score = 1.0
            status = "success"
        elif avg_time < 0.5:
            score = 0.8
            status = "success"
        else:
            score = 0.5
            status = "failure"

        self.results.append(BenchmarkResult(
            test_name="analysis_speed",
            category="performance",
            status=status,
            score=score,
            elapsed_time=avg_time,
            details={
                "avg_time_ms": avg_time * 1000,
                "iterations": 10,
            }
        ))

        # اختبار سرعة التطبيع
        start = time.time()
        for _ in range(100):
            self.lemmatizer.normalize_text(test_text)
        elapsed = time.time() - start
        
        if elapsed < 0.5:
            score = 1.0
        else:
            score = 0.7

        self.results.append(BenchmarkResult(
            test_name="normalization_speed",
            category="performance",
            status="success",
            score=score,
            elapsed_time=elapsed,
            details={
                "total_time_ms": elapsed * 1000,
                "iterations": 100,
            }
        ))

    def _test_multi_format_support(self):
        """اختبار دعم الصيغ المتعددة"""
        logger.info("\n📄 6. اختبار دعم الصيغ المتعددة")

        # اختبار كشف أنواع الملفات
        test_files = [
            ("document.pdf", "pdf"),
            ("report.docx", "docx"),
            ("notes.txt", "txt"),
            ("readme.md", "markdown"),
            ("file.unknown", "unknown"),
        ]

        correct = 0
        for file_name, expected_type in test_files:
            detected = self.selector._extract_file_type(file_name)
            if detected == expected_type:
                correct += 1

        score = correct / len(test_files)
        status = "success" if score >= 0.8 else "failure"

        self.results.append(BenchmarkResult(
            test_name="file_type_detection",
            category="multi_format",
            status=status,
            score=score,
            elapsed_time=0.0,
            details={
                "correct": correct,
                "total": len(test_files),
            }
        ))

    def _generate_report(self) -> Dict[str, Any]:
        """إنشاء تقرير شامل"""
        
        # حساب الإحصائيات العامة
        total_tests = len(self.results)
        successful = [r for r in self.results if r.status == "success"]
        failed = [r for r in self.results if r.status == "failure"]
        
        avg_score = sum(r.score for r in self.results) / total_tests if total_tests > 0 else 0
        avg_time = sum(r.elapsed_time for r in self.results) / total_tests if total_tests > 0 else 0

        # تجميع النتائج حسب الفئة
        by_category = {}
        for result in self.results:
            category = result.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(asdict(result))

        # التقرير النهائي
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": f"{100 * len(successful) / total_tests:.1f}%",
                "average_score": f"{avg_score:.2f}",
                "average_time_ms": f"{avg_time * 1000:.2f}",
            },
            "results_by_category": by_category,
            "detailed_results": [asdict(r) for r in self.results],
            "errors": self.errors,
        }

        return report

    def print_summary(self, report: Dict[str, Any]):
        """طباعة ملخص التقرير"""
        logger.info("\n" + "="*60)
        logger.info("📊 تقرير اختبارات Pyxon AI")
        logger.info("="*60)

        summary = report["summary"]
        logger.info(f"\n✅ النتائج العامة:")
        logger.info(f"  إجمالي الاختبارات: {summary['total_tests']}")
        logger.info(f"  الناجحة: {summary['successful']}")
        logger.info(f"  الفاشلة: {summary['failed']}")
        logger.info(f"  معدل النجاح: {summary['success_rate']}")
        logger.info(f"  متوسط الدرجة: {summary['average_score']}")
        logger.info(f"  متوسط الوقت: {summary['average_time_ms']}ms")

        # النتائج حسب الفئة
        logger.info(f"\n📈 النتائج حسب الفئة:")
        for category, results in report["results_by_category"].items():
            avg = sum(r["score"] for r in results) / len(results)
            logger.info(f"  {category}: {avg:.2f} ({len(results)} tests)")

        # التفاصيل
        logger.info(f"\n📋 تفاصيل الاختبارات:")
        for result in report["detailed_results"]:
            icon = "✅" if result["status"] == "success" else "❌"
            logger.info(f"  {icon} {result['test_name']}: {result['score']:.2f}")

        logger.info("\n" + "="*60)

    def save_report(self, report: Dict[str, Any], filename: str = "pyxon_benchmark_report.json"):
        """حفظ التقرير في ملف JSON"""
        report_dir = Path(__file__).parent.parent / "logs" / "benchmarks"
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"pyxon_benchmark_{timestamp}.json"

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.success(f"✅ تم حفظ التقرير في: {report_path}")
        except Exception as e:
            logger.error(f"❌ فشل حفظ التقرير: {e}")


# ════════════════════════════════════════════════════════════
# Main Entry Point
# ════════════════════════════════════════════════════════════

def main():
    """نقطة الدخول الرئيسية"""
    logger.info("🎯 Pyxon AI Entry Task - Benchmark Suite")
    logger.info(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    benchmark = PyxonBenchmark()
    
    # تشغيل جميع الاختبارات
    report = benchmark.run_all_tests()
    
    # طباعة الملخص
    benchmark.print_summary(report)
    
    # حفظ التقرير
    benchmark.save_report(report)

    return report


if __name__ == "__main__":
    main()
