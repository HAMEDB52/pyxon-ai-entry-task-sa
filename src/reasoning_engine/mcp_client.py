# ============================================================
# src/reasoning_engine/mcp_client.py
# عميل MCP — تكامل مع أدوات خارجية عبر Model Context Protocol
#
# يُتيح للـ Agents استخدام أدوات مثل:
#   • حاسبة (calculator)
#   • تحويل العملات
#   • البحث في الويب (إذا احتاج)
#   • استدعاء APIs خارجية
# ============================================================

import os
import json
import time
from dataclasses import dataclass, field
from loguru import logger

from google import genai
from google.genai import types


@dataclass
class ToolCall:
    tool_name : str
    arguments : dict
    result    : str  = ""
    error     : str  = ""
    elapsed_s : float = 0.0

    @property
    def succeeded(self) -> bool:
        return not self.error and bool(self.result)


@dataclass
class MCPResponse:
    query      : str
    tool_calls : list[ToolCall] = field(default_factory=list)
    final_text : str = ""
    used_tools : list[str] = field(default_factory=list)


class MCPClient:
    """
    عميل MCP يمكّن الـ LLM من استخدام أدوات محلية.

    الأدوات المُدمجة:
        calculator     : عمليات حسابية دقيقة
        currency_convert: تحويل عملات (SAR↔USD↔EUR)
        date_diff      : حساب الفرق بين تاريخين
        doc_lookup     : البحث السريع في اسم مستند

    الاستخدام:
        client = MCPClient()
        resp   = client.run("ما مجموع 5500 + 3200 ريال؟")
        print(resp.final_text)
    """

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        self._client = genai.Client(
            api_key      = api_key,
            http_options = types.HttpOptions(api_version="v1beta"),
        ) if api_key else None
        self._model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

        # تعريف الأدوات المتاحة
        self._tool_definitions = self._define_tools()
        self._tool_handlers    = {
            "calculator"      : self._tool_calculator,
            "currency_convert": self._tool_currency,
            "date_diff"       : self._tool_date_diff,
            "doc_lookup"      : self._tool_doc_lookup,
        }
        logger.info(f"✅ MCPClient | {len(self._tool_handlers)} أداة")

    # ════════════════════════════════════
    # تشغيل مع الأدوات
    # ════════════════════════════════════

    def run(self, query: str, context: str = "") -> MCPResponse:
        """
        تشغيل الـ LLM مع دعم الأدوات.

        Args:
            query  : سؤال المستخدم
            context: سياق إضافي (من البحث)
        """
        if not self._client:
            return MCPResponse(query=query, final_text="MCP غير متاح")

        response = MCPResponse(query=query)
        messages = [{"role": "user", "content": f"{context}\n\nالسؤال: {query}"}]

        # محادثة متعددة الأدوار مع الأدوات (حتى 5 جولات)
        for _ in range(5):
            try:
                resp = self._client.models.generate_content(
                    model   = self._model,
                    contents= messages[-1]["content"],
                    config  = types.GenerateContentConfig(
                        tools = self._tool_definitions,
                    ),
                )
            except Exception as e:
                logger.warning(f"⚠️ MCP generate: {e}")
                break

            # لا أدوات → إجابة نهائية
            if not resp.candidates:
                break

            candidate = resp.candidates[0]
            part = candidate.content.parts[0] if candidate.content.parts else None

            if part and hasattr(part, "text") and part.text:
                response.final_text = part.text.strip()
                break

            # استدعاء أداة
            if part and hasattr(part, "function_call") and part.function_call:
                fc       = part.function_call
                tool_name = fc.name
                args      = dict(fc.args) if fc.args else {}

                start  = time.time()
                result = self._execute_tool(tool_name, args)
                elapsed = round(time.time() - start, 3)

                tc = ToolCall(
                    tool_name = tool_name,
                    arguments = args,
                    result    = result if isinstance(result, str) else str(result),
                    elapsed_s = elapsed,
                )
                response.tool_calls.append(tc)
                response.used_tools.append(tool_name)
                logger.debug(f"  🔧 {tool_name}({args}) → {result}")

                # أضف نتيجة الأداة للمحادثة
                messages.append({
                    "role": "tool",
                    "content": json.dumps({"tool": tool_name, "result": result}),
                })
            else:
                break

        if not response.final_text:
            response.final_text = "لم أتمكن من توليد إجابة"

        return response

    # ════════════════════════════════════
    # تنفيذ الأدوات
    # ════════════════════════════════════

    def _execute_tool(self, name: str, args: dict) -> str:
        handler = self._tool_handlers.get(name)
        if not handler:
            return f"أداة '{name}' غير معروفة"
        try:
            return handler(**args)
        except Exception as e:
            return f"خطأ في {name}: {e}"

    def _tool_calculator(self, expression: str) -> str:
        """حاسبة آمنة"""
        try:
            # فقط الأرقام والعمليات الأساسية
            import ast
            safe_expr = expression.replace("×","*").replace("÷","/").replace("،","")
            tree = ast.parse(safe_expr, mode="eval")
            # تحقق أن الـ AST آمن
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp,
                                          ast.Num, ast.Constant, ast.Add, ast.Sub,
                                          ast.Mult, ast.Div, ast.Pow, ast.Mod)):
                    return "تعبير غير آمن"
            result = eval(compile(tree, "<calc>", "eval"))
            return f"{result:,.2f}"
        except Exception as e:
            return f"خطأ في الحساب: {e}"

    def _tool_currency(self, amount: float, from_currency: str, to_currency: str) -> str:
        """تحويل عملات (أسعار تقريبية)"""
        rates = {"SAR": 1.0, "USD": 3.75, "EUR": 4.05, "AED": 1.02, "GBP": 4.75}
        f = from_currency.upper()
        t = to_currency.upper()
        if f not in rates or t not in rates:
            return f"عملة غير معروفة: {f} أو {t}"
        result = amount * rates[t] / rates[f]
        return f"{amount:,.2f} {f} = {result:,.2f} {t} (سعر تقريبي)"

    def _tool_date_diff(self, date1: str, date2: str) -> str:
        """الفرق بين تاريخين"""
        from datetime import datetime
        formats = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]
        d1 = d2 = None
        for fmt in formats:
            try:
                d1 = d1 or datetime.strptime(date1, fmt)
                d2 = d2 or datetime.strptime(date2, fmt)
            except ValueError:
                pass
        if not d1 or not d2:
            return f"تعذّر تحليل التواريخ: {date1}, {date2}"
        diff = abs((d2 - d1).days)
        return f"الفرق: {diff} يوم ({diff//30} شهر تقريباً)"

    def _tool_doc_lookup(self, filename_hint: str) -> str:
        """البحث عن ملف بالاسم"""
        from pathlib import Path
        upload_dir = Path(os.getenv("UPLOAD_DIR", "All_Invoices_Files"))
        if not upload_dir.exists():
            return "مجلد المستندات غير موجود"
        files = list(upload_dir.iterdir())
        hint  = filename_hint.lower()
        matches = [f.name for f in files if hint in f.name.lower()]
        if matches:
            return f"وُجد: {', '.join(matches[:5])}"
        return f"لا يوجد ملف يحتوي '{filename_hint}'"

    # ════════════════════════════════════
    # تعريف الأدوات لـ Gemini
    # ════════════════════════════════════

    def _define_tools(self):
        """تعريف الأدوات بصيغة Gemini Function Calling"""
        try:
            return [
                types.Tool(function_declarations=[
                    types.FunctionDeclaration(
                        name        = "calculator",
                        description = "إجراء عمليات حسابية رياضية",
                        parameters  = types.Schema(
                            type       = "OBJECT",
                            properties = {"expression": types.Schema(type="STRING", description="التعبير الرياضي")},
                            required   = ["expression"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name        = "currency_convert",
                        description = "تحويل مبالغ بين العملات",
                        parameters  = types.Schema(
                            type       = "OBJECT",
                            properties = {
                                "amount"       : types.Schema(type="NUMBER"),
                                "from_currency": types.Schema(type="STRING"),
                                "to_currency"  : types.Schema(type="STRING"),
                            },
                            required   = ["amount", "from_currency", "to_currency"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name        = "date_diff",
                        description = "حساب الفرق بين تاريخين",
                        parameters  = types.Schema(
                            type       = "OBJECT",
                            properties = {
                                "date1": types.Schema(type="STRING"),
                                "date2": types.Schema(type="STRING"),
                            },
                            required   = ["date1", "date2"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name        = "doc_lookup",
                        description = "البحث عن مستند بالاسم",
                        parameters  = types.Schema(
                            type       = "OBJECT",
                            properties = {"filename_hint": types.Schema(type="STRING")},
                            required   = ["filename_hint"],
                        ),
                    ),
                ])
            ]
        except Exception as e:
            logger.warning(f"⚠️ MCP tool definition: {e}")
            return []