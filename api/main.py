# ============================================================
# api/main.py  — FastAPI Backend + Upload  (إصدار مُصحَّح)
# ============================================================

import sys
sys.path.append(".")
from dotenv import load_dotenv
load_dotenv()

import os, re, time, uuid, asyncio, subprocess
from pathlib import Path
from typing  import Optional, List
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru  import logger

ALLOWED_EXT = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".md", ".txt"}
MAX_MB      = int(os.getenv("MAX_UPLOAD_MB", "20"))
UPLOAD_DIR  = Path(os.getenv("UPLOAD_DIR", "All_Invoices_Files"))

# ── App ──
app = FastAPI(title="Agentic RAG API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_pipeline  : Optional[object] = None
_gatekeeper: Optional[object] = None

def get_pipeline():
    global _pipeline
    if not _pipeline:
        from src.agents.langgraph_pipeline import RAGPipeline
        _pipeline = RAGPipeline()
    return _pipeline

def get_gatekeeper():
    global _gatekeeper
    if not _gatekeeper:
        from src.security.gatekeeper import Gatekeeper
        _gatekeeper = Gatekeeper(use_llm_check=False)
    return _gatekeeper


# ── Models ──
class QueryRequest(BaseModel):
    query  : str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(default="web_user")

class QueryResponse(BaseModel):
    request_id: str; query: str; answer: str
    sources: list[str]; follow_up: list[str]
    confidence: float; elapsed_s: float
    blocked: bool = False; risk_level: str = "low"
    warnings: list[str] = []

class StatusResponse(BaseModel):
    status: str; total_chunks: int; total_sources: int
    model: str; version: str

class UploadResult(BaseModel):
    filename: str; size_kb: float; status: str
    chunks: int = 0; message: str = ""

class UploadResponse(BaseModel):
    uploaded: int; failed: int
    results: list[UploadResult]; elapsed_s: float


# ════════════════════════════════════
# Health
# ════════════════════════════════════

@app.get("/", tags=["Health"])
async def root(): return {"status": "ok", "version": "1.0.0"}

@app.get("/health", tags=["Health"])
async def health(): return {"status": "healthy", "ts": time.time()}


# ════════════════════════════════════
# Status
# ════════════════════════════════════

@app.get("/status", response_model=StatusResponse, tags=["Info"])
async def get_status():
    try:
        from src.database.db_setup import check_database_status
        db = check_database_status()
    except Exception:
        db = {"total_chunks": 0, "total_sources": 0, "status": "unknown"}
    return StatusResponse(
        status        = db.get("status", "unknown"),
        total_chunks  = db.get("total_chunks", 0),
        total_sources = db.get("total_sources", 0),
        model         = os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        version       = "1.0.0",
    )


# ════════════════════════════════════
# Query  (مُصحَّح ✅)
# ════════════════════════════════════

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(req: QueryRequest):
    rid   = str(uuid.uuid4())[:8]
    start = time.time()

    gate = get_gatekeeper().check(req.query, user_id=req.user_id)
    if not gate.allowed:
        return QueryResponse(
            request_id=rid, query=req.query,
            answer=f"عذراً، لا يمكنني معالجة هذا الطلب. ({gate.reason})",
            sources=[], follow_up=[], confidence=0.0,
            elapsed_s=round(time.time()-start, 2),
            blocked=True, risk_level=gate.risk_level, warnings=gate.warnings,
        )
    try:
        result = get_pipeline().run(gate.query, user_id=req.user_id)
        elapsed = round(time.time()-start, 2)

        # ── استخراج الإجابة بأمان ──
        if isinstance(result, dict):
            answer = result.get("final_answer", "")
        else:
            # GraphState أو أي كائن آخر
            answer = getattr(result, "final_answer", "") or result.get("final_answer", "") if hasattr(result, "get") else ""

        if not answer:
            answer = "لم أتمكن من إيجاد إجابة في المستندات المتاحة."

        return QueryResponse(
            request_id=rid, query=req.query,
            answer=_clean(answer),
            sources=_sources(result if isinstance(result, dict) else dict(result)),
            follow_up=_follow_up(answer),
            confidence=result.get("confidence", 0.8) if hasattr(result, "get") else 0.8,
            elapsed_s=elapsed, warnings=gate.warnings,
        )
    except AttributeError as e:
        # ── إصلاح: GraphState.create() غير موجود ──
        if "create" in str(e):
            logger.error(f"❌ state.py قديم: {e}")
            raise HTTPException(
                status_code=500,
                detail=(
                    "⚠️ ملف state.py المحلي قديم ولا يحتوي دالة create().\n"
                    "انسخ النسخة المُحدَّثة: "
                    "src/agents/state.py من مجلد الـ outputs."
                )
            )
        logger.error(f"❌ {e}")
        raise HTTPException(500, str(e))
    except Exception as e:
        logger.error(f"❌ {e}")
        raise HTTPException(500, str(e))





# ════════════════════════════════════
# Upload  ★ (مُصحَّح ✅)
# ════════════════════════════════════

@app.post("/upload", response_model=UploadResponse, tags=["Upload"])
async def upload_files(
    files  : List[UploadFile] = File(...),
    ingest : bool             = Form(default=True),
    user_id: str              = Form(default="web_user"),
    bg_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    ✅ رفع ملفات بسرعة مع معالجة متوازية
    - حفظ الملفات: متوازي (سريع جداً)
    - الإدخال: في الخلفية (لا تنتظري)
    """
    start = time.time()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    results: list[UploadResult] = []

    # ── خطوة 1: حفظ الملفات بسرعة (بدون انتظار ingest) ──
    files_to_ingest = []  # قائمة ملفات للإدخال في الخلفية

    for f in files:
        raw_name = f.filename or "unknown"
        try:
            fname = raw_name.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            fname = raw_name
        fname  = Path(fname).name
        suffix = Path(fname).suffix.lower()

        if suffix not in ALLOWED_EXT:
            results.append(UploadResult(
                filename=fname, size_kb=0, status="error",
                message=f"امتداد غير مسموح: {suffix}",
            ))
            continue

        content = await f.read()
        size_kb = round(len(content) / 1024, 1)

        if size_kb > MAX_MB * 1024:
            results.append(UploadResult(
                filename=fname, size_kb=size_kb, status="error",
                message=f"الحجم كبير جداً ({size_kb:.0f} KB) | الحد: {MAX_MB} MB",
            ))
            continue

        dest = UPLOAD_DIR / fname
        try:
            dest.write_bytes(content)
            logger.info(f"💾 حُفظ: {fname} ({size_kb} KB)")
            
            # أضف للقائمة لإدخالها لاحقاً في الخلفية
            if ingest and suffix not in {".txt"}:
                files_to_ingest.append((dest, fname))
                results.append(UploadResult(
                    filename=fname, size_kb=size_kb,
                    status="pending",
                    message="💾 تم الحفظ | 🔄 الإدخال قيد التقدم...",
                ))
            else:
                results.append(UploadResult(
                    filename=fname, size_kb=size_kb, status="saved",
                    message="تم الحفظ (بدون إدخال)",
                ))
        except Exception as e:
            results.append(UploadResult(filename=fname, size_kb=size_kb, status="error", message=str(e)))
            continue

    # ── خطوة 2: إضف مهام الإدخال للخلفية (في نفس الوقت) ──
    if files_to_ingest:
        bg_tasks.add_task(_ingest_batch_background, files_to_ingest)
        logger.info(f"🚀 تم إرسال {len(files_to_ingest)} ملف للإدخال في الخلفية")

    ok   = sum(1 for r in results if r.status in ("saved", "pending", "ingested"))
    fail = len(results) - ok
    elapsed = round(time.time() - start, 2)
    
    logger.info(f"📤 رفع سريع | ✅{ok} ❌{fail} | ⏱️ {elapsed}s")
    return UploadResponse(uploaded=ok, failed=fail, results=results, elapsed_s=elapsed)


@app.get("/files", tags=["Upload"])
async def list_files():
    """قائمة الملفات مع حالة الإدخال"""
    if not UPLOAD_DIR.exists():
        return {"files": [], "total": 0}
    files = []
    for f in sorted(UPLOAD_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in ALLOWED_EXT:
            files.append({
                "name": f.name, 
                "size_kb": round(f.stat().st_size / 1024, 1),
                "extension": f.suffix.lower(),
                "uploaded_at": time.ctime(f.stat().st_mtime),
            })
    return {"files": files, "total": len(files)}


@app.delete("/files/{filename}", tags=["Upload"])
async def delete_file(filename: str):
    target = UPLOAD_DIR / Path(filename).name
    if not target.exists():
        raise HTTPException(404, "الملف غير موجود")
    target.unlink()
    return {"deleted": target.name}


# ════════════════════════════════════
# Ingest في الخلفية (متوازي) ⚡
# معالجة عدة ملفات في نفس الوقت
# ════════════════════════════════════

def _ingest_batch_background(files_to_ingest: list[tuple[Path, str]]) -> None:
    """
    معالجة متوازية للملفات في الخلفية (في thread منفصل).
    يحسّن الأداء عند رفع عدة ملفات.
    """
    logger.info(f"🔄 بدء إدخال {len(files_to_ingest)} ملف في الخلفية...")
    
    # استخدم ThreadPoolExecutor لمعالجة متوازية (حتى 3 ملفات في نفس الوقت)
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(_ingest_single_sync, dest, fname): fname 
            for dest, fname in files_to_ingest
        }
        
        for future in futures:
            fname = futures[future]
            try:
                chunks_count = future.result(timeout=600)  # 10 دقائق لكل ملف
                logger.success(f"✅ أكتمل إدخال: {fname} ({chunks_count} قطعة)")
            except Exception as e:
                logger.error(f"❌ فشل إدخال {fname}: {e}")


def _ingest_single_sync(filepath: Path, filename: str) -> int:
    """
    إدخال ملف واحد بشكل متزامن (للاستخدام مع ThreadPoolExecutor).
    """
    try:
        # استدعاء subprocess للإدخال (تجنب مشاكل import)
        result = subprocess.run(
            [sys.executable, "scripts/ingest_docs.py", "--file", str(filepath)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=600,
            cwd=str(Path.cwd()),
        )
        
        if result.returncode == 0:
            n = _parse_chunks_count(result.stdout)
            return max(n, 1)
        else:
            logger.warning(f"⚠️ subprocess فشل: {filename} | rc={result.returncode}")
            return 0
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ timeout: {filename}")
        return 0
    except Exception as e:
        logger.error(f"❌ ingest error: {filename}: {e}")
        return 0


# ════════════════════════════════════
# Ingest عبر subprocess  ✅
# يتجنب مشاكل import بالكامل
# ════════════════════════════════════

async def _ingest_via_subprocess(filepath: Path) -> int:
    """
    يستدعي ingest_docs.py --file <path> كـ subprocess.
    يلتقط stdout/stderr ويُسجّلها للتشخيص.
    """
    script = Path("scripts") / "ingest_docs.py"
    if not script.exists():
        logger.warning("⚠️ scripts/ingest_docs.py غير موجود — fallback مباشر")
        return _ingest_direct(filepath)

    abs_path = filepath.resolve()
    logger.info(f"⚙️ subprocess ingest: {filepath.name} | {abs_path}")

    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                [sys.executable, str(script), "--file", str(abs_path)],
                capture_output=True, text=True,
                encoding='utf-8',  # ✅ حدد الترميز صراحة
                errors='replace',  # ✅ استبدل الأحرف التي لا يمكن فك تشفيرها
                timeout=1800,  # 30 دقيقة للملفات الكبيرة
                cwd=str(Path.cwd()),
                env={**os.environ},
            )
        )

        # سجّل آخر 15 سطر من stdout
        if result.stdout.strip():
            for line in result.stdout.strip().split("\n")[-15:]:
                if line.strip():
                    logger.debug(f"  [ingest] {line}")

        # سجّل stderr دائماً إذا وُجد
        if result.stderr.strip():
            stderr_lines = [l for l in result.stderr.strip().split("\n") if l.strip()]
            for line in stderr_lines[-8:]:
                logger.warning(f"  [ingest.err] {line[:200]}")

        if result.returncode == 0:
            n = _parse_chunks_count(result.stdout)
            logger.success(f"✅ ingest نجح: {filepath.name} → {n} قطعة")
            return max(n, 1)
        else:
            logger.error(f"❌ ingest subprocess فشل | returncode={result.returncode}")
            logger.info("🔄 تجربة ingest مباشر كـ fallback...")
            return _ingest_direct(filepath)

    except subprocess.TimeoutExpired:
        logger.error(f"⏰ ingest timeout (5 دقائق): {filepath.name}")
        return 0
    except Exception as e:
        logger.error(f"❌ subprocess خطأ غير متوقع: {e}")
        return _ingest_direct(filepath)


def _ingest_direct(filepath: Path) -> int:
    """
    محاولة الإدخال المباشر بدون subprocess.
    يجرب أسماء كلاسات متعددة لتجنب مشكلة الاسم.
    """
    try:
        import importlib, inspect

        # ── تحميل loader.py وإيجاد الكلاس الصحيح ──
        loader_mod = importlib.import_module("src.data_sources.loader")
        loader_cls = None
        for name, obj in inspect.getmembers(loader_mod, inspect.isclass):
            # أي كلاس يحتوي load_file أو load_documents
            if hasattr(obj, "load_file") or hasattr(obj, "load_documents"):
                loader_cls = obj
                break

        if loader_cls is None:
            logger.error("❌ لم يُعثر على كلاس loader في src.data_sources.loader")
            return 0

        loader = loader_cls()
        load_fn = getattr(loader, "load_file", None) or getattr(loader, "load_documents", None)
        docs = load_fn(str(filepath))

        # ── تحميل parser ──
        try:
            from src.data_processing.restructuring.document_parser import DocumentParser
            parsed = DocumentParser().parse(docs)
        except Exception:
            parsed = docs

        # ── تحميل chunker ──
        chunks = []
        try:
            from src.data_processing.chunking.boundary_detector import BoundaryDetector
            chunks = BoundaryDetector().chunk(parsed)
        except Exception:
            chunks = parsed if isinstance(parsed, list) else []

        if not chunks:
            return 0

        # ── حفظ في DB ──
        try:
            from src.database.relational_db import RelationalDB
            RelationalDB().insert_chunks(chunks)
        except Exception as e:
            logger.warning(f"⚠️ relational_db: {e}")

        try:
            from src.database.vector_store import VectorStore
            VectorStore().add_chunks(chunks)
        except Exception as e:
            logger.warning(f"⚠️ vector_store: {e}")

        logger.success(f"✅ direct ingest: {filepath.name} → {len(chunks)} قطعة")
        return len(chunks)

    except Exception as e:
        logger.error(f"❌ direct ingest فشل: {e}")
        return 0


def _parse_chunks_count(stdout: str) -> int:
    """استخراج عدد القطع من مخرجات ingest_docs.py"""
    for line in reversed(stdout.split("\n")):
        m = re.search(r"CHUNKS:(\d+)", line)
        if m: return int(m.group(1))
    for line in reversed(stdout.split("\n")):
        m = re.search(r"(\d+)\s*(قطعة|chunk|chunks)", line, re.IGNORECASE)
        if m: return int(m.group(1))
    return 0


# ════════════════════════════════════
# Helpers
# ════════════════════════════════════

def _sources(result: dict) -> list[str]:
    answer = result.get("final_answer", "")
    m = re.search(r"\*\*المصادر:\*\*\s*(.+)", answer)
    if m: return [s.strip() for s in m.group(1).split("|") if s.strip()][:5]
    seen, out = set(), []
    for c in result.get("retrieved_chunks", []):
        src = c.get("source", "") if isinstance(c, dict) else getattr(c, "source_file", "") or getattr(c, "source", "")
        if src and src not in seen: out.append(src); seen.add(src)
    return out[:5]

def _follow_up(answer: str) -> list[str]:
    qs, cap = [], False
    for line in answer.split("\n"):
        if "أسئلة مقترحة" in line or "أسئلة المتابعة" in line: cap = True; continue
        if cap and line.strip().startswith("•"):
            q = line.strip().lstrip("•").strip()
            if q: qs.append(q)
    return qs[:3]

def _clean(answer: str) -> str:
    answer = re.sub(r"\n📎\s*\*\*المصادر:\*\*.*",       "", answer, flags=re.DOTALL)
    answer = re.sub(r"\n💡\s*\*\*أسئلة مقترحة:\*\*.*", "", answer, flags=re.DOTALL)
    return answer.strip()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)