# ============================================================
# src/security/strategist.py
# المنسق الاستراتيجي للأمان — يُنسّق بين Gatekeeper و Auditor
#
# هو "قائد الأمان" الذي يقرر:
#   • مستوى الفحص المناسب لكل طلب
#   • متى يُفعّل الـ Auditor الكامل
#   • كيف يتعامل مع الهجمات المتكررة
#   • تسجيل الحوادث الأمنية
# ============================================================

import os
import time
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from loguru import logger

from src.security.gatekeeper import Gatekeeper
from src.security.auditor    import Auditor, AuditResult


@dataclass
class SecurityDecision:
    allowed      : bool
    query        : str   = ""         # الاستعلام بعد التنظيف (إن توفر)
    original_query: str  = ""         # الاستعلام الأصلي (إن توفر)
    risk_level   : str   = "low"      # low | medium | high | critical
    check_level  : str   = "standard" # minimal | standard | strict
    reason       : str   = ""
    warnings     : list[str] = field(default_factory=list)
    audit_result : AuditResult | None = None


class Strategist:
    """
    المنسق الاستراتيجي — يُشغّل طبقات الأمان بذكاء.

    لا يُشغّل Auditor الكامل على كل طلب (تكلفة عالية).
    بدلاً من ذلك، يُقيّم مستوى الخطر ويختار الفحص المناسب.

    الاستخدام:
        strategist = Strategist()

        # قبل المعالجة
        pre = strategist.pre_check(query, user_id)
        if not pre.allowed: return pre.reason

        # بعد توليد الإجابة
        post = strategist.post_check(query, answer, context, user_id)
        final_answer = post.audit_result.redacted_answer or answer
    """

    # عدد الطلبات العالية الخطورة التي تُفعّل Lockdown
    LOCKDOWN_THRESHOLD = 5
    LOCKDOWN_WINDOW    = 300   # 5 دقائق

    def __init__(self):
        self.gatekeeper = Gatekeeper(use_llm_check=False)
        self.auditor    = Auditor()

        # تتبع سلوك المستخدمين
        self._risk_history  : dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self._incident_log  : list[dict]       = []
        self._locked_users  : dict[str, float] = {}   # user_id → unlock_time

        logger.success("✅ Strategist جاهز | Gatekeeper + Auditor")

    # ════════════════════════════════════
    # فحص ما قبل المعالجة
    # ════════════════════════════════════

    def pre_check(self, query: str, user_id: str = "default") -> SecurityDecision:
        """
        فحص الطلب قبل المعالجة.

        Returns:
            SecurityDecision
        """
        # ── 1. فحص Lockdown ──
        if self._is_locked(user_id):
            return SecurityDecision(
                allowed    = False,
                query      = query,
                original_query = query,
                risk_level = "critical",
                reason     = "المستخدم محظور مؤقتاً بسبب نشاط مشبوه متكرر",
            )

        # ── 2. Gatekeeper ──
        gate = self.gatekeeper.check(query, user_id=user_id)
        if not gate.allowed:
            self._record_incident(user_id, query, gate.risk_level, gate.reason)
            self._check_lockdown(user_id)
            return SecurityDecision(
                allowed    = False,
                query      = gate.query,
                original_query = gate.original_query,
                risk_level = gate.risk_level,
                reason     = gate.reason,
                warnings   = gate.warnings,
            )

        # ── 3. تحديد مستوى الفحص ──
        history     = list(self._risk_history[user_id])
        recent_high = sum(1 for r in history[-10:] if r in ("high","medium"))
        check_level = "strict" if recent_high >= 3 else "standard"

        # تسجيل مستوى الخطر
        self._risk_history[user_id].append(gate.risk_level)

        return SecurityDecision(
            allowed     = True,
            query       = gate.query,
            original_query = gate.original_query,
            risk_level  = gate.risk_level,
            check_level = check_level,
            warnings    = gate.warnings,
            reason      = gate.reason or "مقبول",
        )

    # ════════════════════════════════════
    # فحص ما بعد التوليد
    # ════════════════════════════════════

    def post_check(
        self,
        query    : str,
        answer   : str,
        context  : str,
        user_id  : str  = "default",
        force    : bool = False,
    ) -> SecurityDecision:
        """
        فحص الإجابة بعد التوليد.

        Args:
            force: تشغيل Auditor حتى لو مستوى الخطر منخفض
        """
        history    = list(self._risk_history.get(user_id, []))
        risk_level = history[-1] if history else "low"

        # تشغيل Auditor فقط عند الحاجة (توفير API calls)
        should_audit = (
            force or
            risk_level in ("medium", "high") or
            self._answer_looks_suspicious(answer)
        )

        if should_audit:
            audit = self.auditor.audit(query, answer, context)
            logger.debug(f"🔍 Auditor: passed={audit.passed} | issues={audit.issues}")
            if not audit.passed:
                self._record_incident(user_id, query, "medium", f"Auditor: {audit.issues}")

            return SecurityDecision(
                allowed      = True,
                risk_level   = risk_level,
                check_level  = "strict",
                audit_result = audit,
            )

        return SecurityDecision(
            allowed     = True,
            risk_level  = risk_level,
            check_level = "minimal",
        )

    # ════════════════════════════════════
    # تقرير الأمان
    # ════════════════════════════════════

    def security_report(self) -> dict:
        """تقرير حوادث الأمان"""
        risk_counts = defaultdict(int)
        for incident in self._incident_log:
            risk_counts[incident.get("risk","low")] += 1

        return {
            "total_incidents"  : len(self._incident_log),
            "by_risk"          : dict(risk_counts),
            "locked_users"     : list(self._locked_users.keys()),
            "recent_incidents" : self._incident_log[-5:],
        }

    # ════════════════════════════════════
    # Helpers
    # ════════════════════════════════════

    def _answer_looks_suspicious(self, answer: str) -> bool:
        """فحص سريع للإجابة"""
        suspicious = [
            "ignore", "bypass", "jailbreak",
            "تجاهل", "تجاوز", "أنت الآن",
        ]
        low = answer.lower()
        return any(s in low for s in suspicious)

    def _record_incident(self, user_id: str, query: str, risk: str, reason: str):
        self._incident_log.append({
            "user_id"  : user_id,
            "query"    : query[:100],
            "risk"     : risk,
            "reason"   : reason,
            "timestamp": time.time(),
        })
        if len(self._incident_log) > 1000:
            self._incident_log = self._incident_log[-500:]

    def _check_lockdown(self, user_id: str):
        """تقييم ما إذا كان يجب حظر المستخدم"""
        history = list(self._risk_history[user_id])
        recent_blocks = sum(1 for r in history[-10:] if r in ("high","critical"))
        if recent_blocks >= self.LOCKDOWN_THRESHOLD:
            unlock_at = time.time() + self.LOCKDOWN_WINDOW
            self._locked_users[user_id] = unlock_at
            logger.warning(f"🔒 Lockdown: {user_id} | {self.LOCKDOWN_WINDOW}s")

    def _is_locked(self, user_id: str) -> bool:
        if user_id not in self._locked_users:
            return False
        if time.time() > self._locked_users[user_id]:
            del self._locked_users[user_id]
            return False
        return True