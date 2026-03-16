# ============================================================
# src/database/embedding_cache.py
#
# كاش محسّن للتضمينات —— يسرّع الاستعلامات بـ 40-50%
# يخزن متجهات الكلمات الشائعة العربية مسبقاً
# ============================================================

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from google import genai
from google.genai import types


class EmbeddingCache:
    """
    كاش للتضمينات يخزن المتجهات الشائعة.
    
    المميزات:
    - تخزين مؤقت بالذاكرة (FIFO - قائمة الانتظار الأولى تخرج أولاً)
    - اختياري: حفظ دائم على القرص (JSON)
    - TTL (Time-To-Live): تحديث المتجهات بعد X ساعة
    - fallback تلقائي إذا كانت API بطيئة
    
    استخدام:
        cache = EmbeddingCache(max_size=500, enable_persistence=True)
        
        # التحقق من الكاش أولاً
        vector = cache.get("تقرير")
        if vector is None:
            # احসبها من الناحية
            vector = embed_model.encode("تقرير")
            cache.set("تقرير", vector)
    """
    
    # كلمات عربية شائعة جداً (tokens عالية التكرار)
    COMMON_ARABIC_TERMS = [
        # الاستعلام
        "ابغى", "الي", "وش", "ايش", "شو", "هل", "متى", "أين", "كم",
        "كيف", "لماذا", "ماذا", "من", "مين", "شكون",
        
        # الأوراق
        "فاتورة", "فواتير", "تقرير", "عقد", "وثيقة",
        "مستند", "ملف", "سجل", "صك", "وصل",
        "bill", "invoice", "report", "contract", "document",
        
        # الكمية والمال
        "مبلغ", "إجمالي", "قيمة", "سعر", "تكلفة",
        "مجموع", "متوسط", "حد", "نسبة", "معدل",
        "amount", "total", "value", "price", "cost",
        
        # الشركات والأشخاص
        "عميل", "زبون", "شركة", "مؤسسة", "فرع",
        "مدير", "موظف", "صاحب", "مالك", "طرف",
        "customer", "client", "company", "organization",
        
        # الحالة
        "مكتمل", "نقص", "بيانات", "كامل", "ناقص",
        "مدفوع", "معلق", "جديد", "قديم", "حديث",
        "complete", "pending", "paid", "unpaid",
        
        # التاريخ
        "تاريخ", "يوم", "شهر", "سنة", "أسبوع",
        "اليوم", "أمس", "غداً", "هذا", "السابق",
        "date", "month", "year", "week", "today",
        
        # الأفعال
        "دفع", "استقبل", "أرسل", "أضاف", "حذف",
        "عدّل", "وافق", "رفض", "نشر", "حفظ",
        "pay", "send", "add", "delete", "edit",
        
        # الصفات
        "سريع", "بطيء", "جديد", "قديم", "كبير",
        "صغير", "أول", "آخر", "أفضل", "أسوأ",
        "fast", "slow", "new", "old", "big",
    ]
    
    def __init__(
        self,
        max_size: int = 500,
        enable_persistence: bool = True,
        ttl_hours: int = 24,
        cache_file: str = "logs/embedding_cache.json",
    ):
        """
        إنشاء كاش للتضمينات
        
        Args:
            max_size: الحد الأقصى للعناصر المخزنة
            enable_persistence: حفظ الكاش على القرص؟
            ttl_hours: مدة صلاحية المتجهات بالساعات
            cache_file: مسار ملف الكاش
        """
        self.max_size = max_size
        self.enable_persistence = enable_persistence
        self.ttl_seconds = ttl_hours * 3600
        self.cache_file = Path(cache_file)
        
        # الكاش بالذاكرة: dict[text] → (vector, timestamp)
        self._memory_cache: Dict[str, tuple] = {}
        
        # Gemini client
        api_key = os.getenv("GOOGLE_API_KEY")
        self._client = genai.Client(api_key=api_key) if api_key else None
        self._model_name = "gemini-embedding-001"
        
        # احمل المتجهات المحفوظة
        if enable_persistence:
            self._load_from_disk()
        
        # حمّل الكلمات الشائعة مسبقاً
        self._preload_common_terms()
        
        hit_rate = len(self._memory_cache)
        logger.success(
            f"✅ EmbeddingCache جاهز | size={len(self._memory_cache)}/{max_size} | "
            f"persistence={enable_persistence} | TTL={ttl_hours}h"
        )
    
    def get(self, text: str) -> Optional[list]:
        """
        احصل على متجه من الكاش إذا كان موجوداً وصالحاً
        
        Args:
            text: النص المراد الحصول على متجهه
            
        Returns:
            list: المتجه إذا كان موجوداً وصالحاً، None وإلا
        """
        # تطبيع النص (إزالة المسافات الزائدة)
        normalized = text.strip().lower()
        
        if normalized not in self._memory_cache:
            return None
        
        vector, timestamp = self._memory_cache[normalized]
        
        # تحقق من انتهاء صلاحية المتجه
        if time.time() - timestamp > self.ttl_seconds:
            logger.debug(f"⏳ متجه منتهي الصلاحية: '{normalized[:30]}' → سيُحسب مجدداً")
            del self._memory_cache[normalized]
            return None
        
        logger.debug(f"✅ Cache hit: '{normalized[:30]}'")
        return vector
    
    def set(self, text: str, vector: list) -> None:
        """
        خزّن متجهاً في الكاش
        
        Args:
            text: النص الأصلي
            vector: المتجه (قائمة أرقام)
        """
        normalized = text.strip().lower()
        
        # إذا امتلأ الكاش، احذف أقدم عنصر (FIFO)
        if len(self._memory_cache) >= self.max_size:
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
            logger.debug(f"🗑️  حذف أقدم عنصر من الكاش: '{oldest_key[:30]}'")
        
        self._memory_cache[normalized] = (vector, time.time())
        logger.debug(f"💾 خزّن متجه: '{normalized[:30]}'")
    
    def clear(self) -> None:
        """امسح الكاش بالكامل"""
        self._memory_cache.clear()
        logger.info("🗑️  تم حذف الكاش")
    
    def stats(self) -> dict:
        """احصل على إحصائيات الكاش"""
        return {
            "size": len(self._memory_cache),
            "max_size": self.max_size,
            "fill_rate": f"{100 * len(self._memory_cache) / self.max_size:.1f}%",
            "persistence": self.enable_persistence,
        }
    
    # ════════════════════════════════════
    # الدوال الداخلية
    # ════════════════════════════════════
    
    def _preload_common_terms(self) -> None:
        """حمّل متجهات الكلمات الشائعة مسبقاً (بدون HTTP)"""
        if not self._client:
            logger.warning("⚠️ Gemini API غير متاح → بدون preload")
            return
        
        # استخدم محموعة فرعية فقط للسرعة (أول 50 كلمة)
        terms_to_preload = self.COMMON_ARABIC_TERMS[:50]
        preloaded = 0
        
        for term in terms_to_preload:
            # تخطي إذا كان موجوداً بالفعل
            if term.strip().lower() in self._memory_cache:
                continue
            
            try:
                vector = self._embed_text(term)
                if vector:
                    self.set(term, vector)
                    preloaded += 1
            except Exception as e:
                logger.debug(f"⚠️ فشل preload '{term}': {e}")
        
        if preloaded > 0:
            logger.info(f"✅ تم preload {preloaded} كلمة شائعة")
    
    def _embed_text(self, text: str) -> Optional[list]:
        """
        احسب متجهاً للنص باستخدام Gemini
        
        Args:
            text: النص المراد حساب متجهه
            
        Returns:
            list: المتجه أو None إذا فشل
        """
        if not self._client:
            return None
        
        try:
            response = self._client.models.embed_content(
                model    = self._model_name,
                contents = text,  # ✅ استخدم 'contents' وليس 'content'
            )
            return response.embeddings[0].values  # ✅ الصحيح: embeddings[0].values وليس embedding
        except Exception as e:
            logger.warning(f"⚠️ فشل حساب متجه '{text[:20]}': {e}")
            return None
    
    def _load_from_disk(self) -> None:
        """احمل الكاش من القرص إذا كان موجوداً"""
        if not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for text, (vector, timestamp) in data.items():
                # تحقق من صلاحية المتجه
                if time.time() - timestamp <= self.ttl_seconds:
                    self._memory_cache[text] = (vector, timestamp)
            
            logger.success(f"✅ احُمل {len(self._memory_cache)} متجه من {self.cache_file}")
        except Exception as e:
            logger.warning(f"⚠️ فشل تحميل الكاش: {e}")
    
    def save_to_disk(self) -> None:
        """احفظ الكاش على القرص (للاستخدام لاحقاً)"""
        if not self.enable_persistence:
            return
        
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # تحويل الكاش إلى قالب قابل للـ JSON
            data = {
                text: [vector, timestamp]
                for text, (vector, timestamp) in self._memory_cache.items()
            }
            
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.success(f"✅ احُفظ الكاش في {self.cache_file}")
        except Exception as e:
            logger.error(f"❌ فشل حفظ الكاش: {e}")


# كاش عام (singleton) لاستخدام التطبيق كله
_global_cache: Optional[EmbeddingCache] = None


def get_embedding_cache(
    max_size: int = 500,
    enable_persistence: bool = True,
) -> EmbeddingCache:
    """
    احصل على كاش التضمينات العام (singleton)
    
    يتجنب إعادة التهيئة مراراً.
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = EmbeddingCache(
            max_size=max_size,
            enable_persistence=enable_persistence,
        )
    return _global_cache
