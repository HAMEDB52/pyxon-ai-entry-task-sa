# ============================================================
# memory/long_term.py
# الذاكرة طويلة المدى (Long-term Memory)
# لحفظ التفضيلات والحقائق الهامة عن المستخدم عبر الجلسات
# ============================================================

import os
from datetime import datetime
from loguru import logger
import psycopg2

from src.database.db_setup import get_connection

class LongTermMemory:
    """
    تدير الذاكرة طويلة المدى للمستخدم بتخزين الحقائق في قاعدة البيانات
    واسترجاعها عند بداية كل جلسة لتكون جزءاً من سياق الوكيل (Context).
    """
    
    def __init__(self):
        self._ensure_table_exists()
        logger.info("🧠 LongTermMemory جاهزة")

    def _ensure_table_exists(self):
        """إنشاء جدول الذكريات إذا لم يكن موجوداً"""
        try:
            conn = get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_memories (
                        id SERIAL PRIMARY KEY,
                        fact TEXT NOT NULL,
                        category VARCHAR(50) DEFAULT 'general',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"❌ خطأ أثناء التأكد من جدول الذاكرة: {e}")

    def add_memory(self, fact: str, category: str = "general") -> bool:
        """
        إضافة حقيقة جديدة لذاكرة المستخدم
        مثال: add_memory("المستخدم يفضل الإجابات القصيرة جداً", "preference")
        """
        try:
            conn = get_connection()
            with conn.cursor() as cur:
                # التحقق من عدم وجود الحقيقة مسبقاً لتجنب التكرار
                cur.execute("SELECT id FROM user_memories WHERE fact = %s", (fact,))
                if cur.fetchone():
                    return True # موجودة مسبقاً
                    
                cur.execute(
                    "INSERT INTO user_memories (fact, category) VALUES (%s, %s)",
                    (fact, category)
                )
            conn.commit()
            conn.close()
            logger.info(f"💾 تم حفظ ذكرى جديدة: {fact}")
            return True
        except Exception as e:
            logger.error(f"❌ فشل حفظ الذكرى: {e}")
            return False

    def get_all_memories(self) -> str:
        """
        جلب كل الذكريات كَسياق نصي موحد يوضع في الـ Prompt للوكلاء
        """
        try:
            conn = get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT fact FROM user_memories ORDER BY created_at DESC LIMIT 20")
                rows = cur.fetchall()
            conn.close()
            
            if not rows:
                return "لا توجد ذكريات سابقة."
                
            memories = [f"- {row[0]}" for row in rows]
            return "\n".join(memories)
        except Exception as e:
            logger.error(f"❌ فشل جلب الذكريات: {e}")
            return "تعذر جلب الذكريات."

# كائن عالمي موحد للذاكرة
default_lt_memory = LongTermMemory()
