# ============================================================
# src/memory/short_term.py
# Short-Term Memory Manager
# يدير تاريخ المحادثة وسياق الجلسة الحالية
# ============================================================

from loguru import logger
from typing import List, Dict, Any

class ShortTermMemory:
    """
    مدير الذاكرة قصيرة المدى (Short-Term Memory).
    يتحكم في كمية السياق المرسلة للنموذج من تاريخ المحادثة.
    """

    def __init__(self, max_history: int = 10):
        self.max_history = max_history

    def get_context(self, messages: List[Any]) -> str:
        """
        تحويل رسائل الحالة (GraphState messages) إلى نص منظم للذاكرة
        """
        if not messages:
            return ""

        # نأخذ آخر N رسالة
        recent = messages[-self.max_history:]
        
        memory_lines = ["### سجل المحادثة القريب (ذاكرة قصيرة المدى):"]
        for msg in recent:
            # معالجة أنواع مختلفة من الرسائل (LangChain/LangGraph formats)
            role = "المستخدم" if getattr(msg, "type", "user") == "user" else "المساعد"
            content = getattr(msg, "content", str(msg))
            
            # تنظيف المحتوى إذا كان طويلاً جداً (Summary truncation)
            if len(content) > 500:
                content = content[:497] + "..."
                
            memory_lines.append(f"- {role}: {content}")

        return "\n".join(memory_lines)

    def summarize_history(self, messages: List[Any], client: Any, model_name: str) -> str:
        """
        تلخيص التاريخ القديم لضغط الذاكرة (Memory Compression)
        """
        if len(messages) <= self.max_history:
            return ""

        history_text = "\n".join([f"{getattr(m, 'type', 'user')}: {getattr(m, 'content', '')}" for m in messages[:-self.max_history]])
        
        prompt = f"""قم بتلخيص المحادثة التالية في 3 نقاط رئيسية لتعمل كذاكرة طويلة المدى للمساعد:
{history_text}

الخلاصة:"""

        try:
            response = client.models.generate_content(
                model    = model_name,
                contents = prompt,
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"❌ فشل تلخيص الذاكرة: {e}")
            return ""
