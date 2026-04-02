# ============================================================
# src/data_processing/chunking/strategy_selector.py
#
# Intelligent Chunking Strategy Selector
# يحلل المستند ويختار تلقائياً بين Fixed و Dynamic chunking
#
# يدعم جميع أنواع المستندات:
# - PDF, DOCX, TXT, MD
# - نصوص عربية وإنجليزية
# - مستندات قانونية، تقنية، أكاديمية، عامة
# ============================================================

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from loguru import logger


class ChunkingStrategy(Enum):
    """استراتيجيات chunking المدعومة"""
    FIXED = "fixed"           # حجم ثابت للقطع
    DYNAMIC = "dynamic"       # حجم ديناميكي حسب المحتوى
    SEMANTIC = "semantic"     # حدود دلالية
    HYBRID = "hybrid"         # دمج استراتيجيات


@dataclass
class DocumentAnalysis:
    """نتيجة تحليل المستند"""
    file_name: str
    file_type: str
    total_chars: int
    total_words: int
    num_paragraphs: int
    num_headings: int
    num_tables: int
    num_lists: int
    has_structure: bool
    language: str  # 'arabic', 'english', 'mixed'
    has_diacritics: bool  # هل يحتوي على تشكيل عربي
    complexity_score: float  # 0-100
    recommended_strategy: ChunkingStrategy
    confidence: float  # 0-1


class StrategySelector:
    """
    يحلل المستند ويختار أفضل استراتيجية chunking تلقائياً
    
    معايير الاختيار:
    1. المستندات المنتظمة (فواتير، نماذج) → Fixed
    2. المستندات غير المنتظمة (تقارير، أبحاث) → Dynamic
    3. المستندات المهيكلة (كتب، أدلة) → Semantic
    4. المستندات المعقدة → Hybrid
    """

    def __init__(self):
        self.config = {
            "fixed": {
                "chunk_size": int(os.getenv("FIXED_CHUNK_SIZE", 512)),
                "overlap": int(os.getenv("FIXED_CHUNK_OVERLAP", 50)),
            },
            "dynamic": {
                "min_chunk": int(os.getenv("DYNAMIC_MIN_CHUNK", 128)),
                "max_chunk": int(os.getenv("DYNAMIC_MAX_CHUNK", 1024)),
                "overlap_percent": float(os.getenv("DYNAMIC_OVERLAP_PERCENT", 0.1)),
            },
            "semantic": {
                "min_sentences": int(os.getenv("SEMANTIC_MIN_SENTENCES", 3)),
                "max_sentences": int(os.getenv("SEMANTIC_MAX_SENTENCES", 10)),
            },
            "thresholds": {
                "structure_score": 0.6,
                "complexity_high": 70,
                "complexity_low": 30,
            }
        }
        logger.info("📊 StrategySelector initialized")

    def analyze(self, text: str, file_name: str = "unknown") -> DocumentAnalysis:
        """
        تحليل المستند وتحديد أفضل استراتيجية chunking
        
        Args:
            text: محتوى المستند النصي
            file_name: اسم الملف
            
        Returns:
            DocumentAnalysis: نتيجة التحليل
        """
        logger.debug(f"🔍 تحليل المستند: {file_name}")

        # استخراج نوع الملف
        file_type = self._extract_file_type(file_name)

        # تحليل الخصائص الأساسية
        total_chars = len(text)
        total_words = len(text.split())
        num_paragraphs = self._count_paragraphs(text)
        num_headings = self._count_headings(text)
        num_tables = self._count_tables(text)
        num_lists = self._count_lists(text)

        # تحليل اللغة
        language = self._detect_language(text)

        # كشف التشكيل العربي
        has_diacritics = self._has_arabic_diacritics(text)

        # حساب درجة الهيكلية
        has_structure = self._calculate_structure_score(
            num_headings, num_paragraphs, num_tables, num_lists, total_words
        ) > self.config["thresholds"]["structure_score"]

        # حساب درجة التعقيد
        complexity_score = self._calculate_complexity(
            text, num_paragraphs, num_headings, total_words
        )

        # تحديد الاستراتيجية الموصى بها
        strategy, confidence = self._recommend_strategy(
            file_type, has_structure, complexity_score, 
            num_tables, num_headings, language
        )

        analysis = DocumentAnalysis(
            file_name=file_name,
            file_type=file_type,
            total_chars=total_chars,
            total_words=total_words,
            num_paragraphs=num_paragraphs,
            num_headings=num_headings,
            num_tables=num_tables,
            num_lists=num_lists,
            has_structure=has_structure,
            language=language,
            has_diacritics=has_diacritics,
            complexity_score=complexity_score,
            recommended_strategy=strategy,
            confidence=confidence,
        )

        logger.info(
            f"✅ التحليل: {file_name} | "
            f"الاستراتيجية: {strategy.value} | "
            f"الثقة: {confidence:.2f} | "
            f"التعقيد: {complexity_score:.1f}"
        )

        return analysis

    def get_chunking_params(self, analysis: DocumentAnalysis) -> Dict[str, Any]:
        """
        الحصول على معاملات chunking بناءً على التحليل
        
        Returns:
            dict: معاملات chunking المناسبة
        """
        strategy = analysis.recommended_strategy

        if strategy == ChunkingStrategy.FIXED:
            return {
                "strategy": "fixed",
                "chunk_size": self.config["fixed"]["chunk_size"],
                "overlap": self.config["fixed"]["overlap"],
                "description": "Fixed-size chunking for uniform documents",
            }

        elif strategy == ChunkingStrategy.DYNAMIC:
            return {
                "strategy": "dynamic",
                "min_chunk": self.config["dynamic"]["min_chunk"],
                "max_chunk": self.config["dynamic"]["max_chunk"],
                "overlap_percent": self.config["dynamic"]["overlap_percent"],
                "description": "Dynamic chunking based on content boundaries",
            }

        elif strategy == ChunkingStrategy.SEMANTIC:
            return {
                "strategy": "semantic",
                "min_sentences": self.config["semantic"]["min_sentences"],
                "max_sentences": self.config["semantic"]["max_sentences"],
                "description": "Semantic chunking respecting content boundaries",
            }

        else:  # Hybrid
            return {
                "strategy": "hybrid",
                "fixed_params": self.config["fixed"],
                "dynamic_params": self.config["dynamic"],
                "description": "Hybrid approach combining multiple strategies",
            }

    # ============================================================
    # دوال التحليل الداخلية
    # ============================================================

    def _extract_file_type(self, file_name: str) -> str:
        """استخراج نوع الملف من الامتداد"""
        if not file_name:
            return "unknown"
        
        ext_map = {
            ".pdf": "pdf",
            ".doc": "doc",
            ".docx": "docx",
            ".txt": "txt",
            ".md": "markdown",
            ".rst": "rst",
        }
        
        lower_name = file_name.lower()
        for ext, ftype in ext_map.items():
            if lower_name.endswith(ext):
                return ftype
        
        return "unknown"

    def _count_paragraphs(self, text: str) -> int:
        """عد الفقرات"""
        if not text.strip():
            return 0
        # الفقرات مفصولة بأسطر فارغة
        paragraphs = re.split(r'\n\s*\n', text)
        return len([p for p in paragraphs if p.strip()])

    def _count_headings(self, text: str) -> int:
        """عد العناوين"""
        if not text:
            return 0
        
        patterns = [
            r'^#{1,6}\s+.+$',           # Markdown headings
            r'^[A-Z][A-Z\s]+$',         # All caps headings
            r'^\d+\.\s+[A-Z].+$',       # Numbered headings
            r'^الفصل\s+\d+',            # Arabic chapter headings
            r'^المبحث\s+',              # Arabic section headings
            r'^المطلب\s+',              # Arabic subsection headings
        ]
        
        count = 0
        for line in text.split('\n'):
            for pattern in patterns:
                if re.match(pattern, line.strip(), re.MULTILINE):
                    count += 1
                    break
        
        return count

    def _count_tables(self, text: str) -> int:
        """عد الجداول"""
        if not text:
            return 0
        
        # كشف الجداول البسيطة
        table_patterns = [
            r'\|.*\|.*\|',              # Markdown tables
            r'^\s*\+.*\+.*\+',          # ASCII tables
            r'^\s*│.*│.*│',             # Box-drawing tables
        ]
        
        count = 0
        for pattern in table_patterns:
            count += len(re.findall(pattern, text, re.MULTILINE))
        
        return count

    def _count_lists(self, text: str) -> int:
        """عد القوائم"""
        if not text:
            return 0
        
        patterns = [
            r'^\s*[-*•]\s+',            # Bullet lists
            r'^\s*\d+\.\s+',            # Numbered lists
            r'^\s*[أبج]\.\s+',          # Arabic lettered lists
            r'^\s*\(\d+\)\s+',          # Numbered parentheses
        ]
        
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text, re.MULTILINE))
        
        return count

    def _detect_language(self, text: str) -> str:
        """
        كشف لغة المستند
        
        Returns:
            'arabic', 'english', 'mixed'
        """
        if not text or len(text) < 10:
            return "unknown"

        # عينة من النص
        sample = text[:1000]

        # عد الأحرف العربية والإنجليزية
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', sample))
        english_chars = len(re.findall(r'[a-zA-Z]', sample))

        total = arabic_chars + english_chars
        if total == 0:
            return "unknown"

        arabic_ratio = arabic_chars / total
        english_ratio = english_chars / total

        if arabic_ratio > 0.7:
            return "arabic"
        elif english_ratio > 0.7:
            return "english"
        else:
            return "mixed"

    def _has_arabic_diacritics(self, text: str) -> bool:
        """
        كشف ما إذا كان النص يحتوي على تشكيل عربي
        
        التشكيل: َ  ُ  ِ  ّ  ً  ٌ  ٍ  ْ  ٓ  ٔ  ٕ
        """
        if not text:
            return False

        diacritics_pattern = r'[\u064B-\u065F\u0670]'
        return bool(re.search(diacritics_pattern, text))

    def _calculate_structure_score(
        self, num_headings: int, num_paragraphs: int,
        num_tables: int, num_lists: int, total_words: int
    ) -> float:
        """
        حساب درجة الهيكلية (0-1)
        
        1.0 = مهيكلاً جداً
        0.0 = غير مهيكلاً
        """
        if total_words == 0:
            return 0.0

        # كثافة العناوين
        heading_density = num_headings / max(1, total_words / 500)

        # كثافة الجداول والقوائم
        element_density = (num_tables + num_lists) / max(1, total_words / 1000)

        # نسبة الفقرات المنظمة
        paragraph_score = min(1.0, num_paragraphs / max(1, total_words / 200))

        # الدرجة النهائية
        score = (
            0.4 * min(1.0, heading_density) +
            0.3 * min(1.0, element_density) +
            0.3 * paragraph_score
        )

        return min(1.0, score)

    def _calculate_complexity(
        self, text: str, num_paragraphs: int,
        num_headings: int, total_words: int
    ) -> float:
        """
        حساب درجة التعقيد (0-100)
        
        العوامل:
        - طول المستند
        - تنوع المفردات
        - كثافة العناوين
        - متوسط طول الفقرات
        """
        if total_words == 0:
            return 0.0

        # 1. درجة الطول (0-25)
        length_score = min(25, total_words / 100)

        # 2. درجة تنوع المفردات (0-25)
        unique_words = len(set(text.lower().split()))
        vocab_diversity = unique_words / max(1, total_words)
        vocab_score = min(25, vocab_diversity * 50)

        # 3. درجة كثافة العناوين (0-25)
        heading_density = num_headings / max(1, total_words / 500)
        heading_score = min(25, heading_density * 25)

        # 4. درجة طول الفقرات (0-25)
        avg_paragraph_length = total_words / max(1, num_paragraphs)
        paragraph_score = min(25, abs(avg_paragraph_length - 150) / 10)

        complexity = length_score + vocab_score + heading_score + paragraph_score
        return min(100, complexity)

    def _recommend_strategy(
        self, file_type: str, has_structure: bool,
        complexity_score: float, num_tables: int,
        num_headings: int, language: str
    ) -> tuple[ChunkingStrategy, float]:
        """
        التوصية بأفضل استراتيجية chunking
        
        Returns:
            (strategy, confidence)
        """
        confidence = 0.0

        # قاعدة 1: المستندات المهيكلة جداً → Fixed
        if has_structure and complexity_score < self.config["thresholds"]["complexity_low"]:
            return ChunkingStrategy.FIXED, 0.85

        # قاعدة 2: المستندات غير المنتظمة → Dynamic
        if not has_structure and complexity_score < self.config["thresholds"]["complexity_high"]:
            return ChunkingStrategy.DYNAMIC, 0.80

        # قاعدة 3: المستندات المعقدة جداً → Hybrid
        if complexity_score >= self.config["thresholds"]["complexity_high"]:
            return ChunkingStrategy.HYBRID, 0.75

        # قاعدة 4: مستندات بها عناوين كثيرة → Semantic
        if num_headings > 5 and has_structure:
            return ChunkingStrategy.SEMANTIC, 0.82

        # قاعدة 5: مستندات بها جداول → Hybrid
        if num_tables > 2:
            return ChunkingStrategy.HYBRID, 0.78

        # قاعدة 6: نصوص عربية مع تشكيل → Dynamic (للحفاظ على المعنى)
        if language == "arabic":
            return ChunkingStrategy.DYNAMIC, 0.70

        # الافتراضي: Dynamic
        return ChunkingStrategy.DYNAMIC, 0.65


# ════════════════════════════════════
# Singleton instance
# ════════════════════════════════════

_selector_instance: Optional[StrategySelector] = None


def get_strategy_selector() -> StrategySelector:
    """الحصول على instance مفردة من StrategySelector"""
    global _selector_instance
    if _selector_instance is None:
        _selector_instance = StrategySelector()
    return _selector_instance


def analyze_document(text: str, file_name: str = "unknown") -> DocumentAnalysis:
    """دالة مساعدة لتحليل المستند"""
    return get_strategy_selector().analyze(text, file_name)


def get_chunking_params(text: str, file_name: str = "unknown") -> Dict[str, Any]:
    """دالة مساعدة للحصول على معاملات chunking"""
    selector = get_strategy_selector()
    analysis = selector.analyze(text, file_name)
    return selector.get_chunking_params(analysis)
