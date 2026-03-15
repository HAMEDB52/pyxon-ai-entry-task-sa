// ============================================================
// frontend/server.js  —  Express + Upload Proxy
// ============================================================
const express  = require("express");
const path     = require("path");
const axios    = require("axios");
const multer   = require("multer");
const FormData = require("form-data");

const app     = express();
const PORT    = process.env.PORT    || 3000;
const API_URL = process.env.API_URL || "http://127.0.0.1:8000";

// ── Multer: رفع في الذاكرة (ننقله مباشرة لـ FastAPI) ──
const upload = multer({
  storage : multer.memoryStorage(),
  limits  : { fileSize: 25 * 1024 * 1024 },   // 25 MB
});

app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// ══════════════════════════════════════
// PROXY: Query
// ══════════════════════════════════════
app.post("/api/query", async (req, res) => {
  try {
    const r = await axios.post(`${API_URL}/query`, req.body, { timeout: 300_000 });
    res.json(r.data);
  } catch (err) {
    res.status(err.response?.status || 500).json({ error: err.response?.data?.detail || err.message });
  }
});

// ══════════════════════════════════════
// PROXY: Status & Suggestions
// ══════════════════════════════════════
app.get("/api/status", async (req, res) => {
  try {
    const r = await axios.get(`${API_URL}/status`, { timeout: 10_000 });
    res.json(r.data);
  } catch { res.status(500).json({ error: "API غير متاح" }); }
});

app.get("/api/suggestions", async (req, res) => {
  try {
    const r = await axios.get(`${API_URL}/suggestions`, { timeout: 5_000 });
    res.json(r.data);
  } catch { res.json({ suggestions: [] }); }
});

// ══════════════════════════════════════
// PROXY: Upload  ★
// ══════════════════════════════════════
app.post("/api/upload", upload.array("files", 20), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0)
      return res.status(400).json({ error: "لم يتم إرسال أي ملفات" });

    // بناء FormData لإعادة إرساله لـ FastAPI
    const form = new FormData();

    req.files.forEach(f => {
      // إصلاح encoding الاسم العربي — Buffer.from يضمن UTF-8
      const safeFilename = Buffer.from(f.originalname, "latin1").toString("utf8");
      form.append("files", f.buffer, {
        filename    : safeFilename,
        contentType : f.mimetype,
        knownLength : f.size,
      });
    });

    const ingest  = req.body.ingest  !== "false";
    const user_id = req.body.user_id || "web_user";
    form.append("ingest",   String(ingest));
    form.append("user_id",  user_id);

    const r = await axios.post(`${API_URL}/upload`, form, {
      headers : { ...form.getHeaders() },
      timeout : 300_000,   // 5 دقائق لإدخال كبير
      maxContentLength: Infinity,
      maxBodyLength   : Infinity,
    });
    res.json(r.data);
  } catch (err) {
    console.error("Upload error:", err.message);
    res.status(err.response?.status || 500)
       .json({ error: err.response?.data?.detail || err.message });
  }
});

// ══════════════════════════════════════
// PROXY: List Files
// ══════════════════════════════════════
app.get("/api/files", async (req, res) => {
  try {
    const r = await axios.get(`${API_URL}/files`, { timeout: 10_000 });
    res.json(r.data);
  } catch { res.json({ files: [], total: 0 }); }
});

// ══════════════════════════════════════
// PROXY: Delete File
// ══════════════════════════════════════
app.delete("/api/files/:filename", async (req, res) => {
  try {
    const r = await axios.delete(`${API_URL}/files/${encodeURIComponent(req.params.filename)}`, { timeout: 10_000 });
    res.json(r.data);
  } catch (err) {
    res.status(err.response?.status || 500).json({ error: err.message });
  }
});

// ══════════════════════════════════════
// SPA Fallback
// ══════════════════════════════════════
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.listen(PORT, () => {
  console.log(`\n🚀 Frontend : http://localhost:${PORT}`);
  console.log(`🔗 FastAPI  : ${API_URL}\n`);
});