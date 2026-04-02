/* ═══════════════════════════════════════════════════
   LYNCS AI — Frontend Application
   ═══════════════════════════════════════════════════ */

const API = ""; // Same origin as Express server

// ── State ──
const state = {
    loading: false,
    history: JSON.parse(localStorage.getItem("lynscs_history") || "[]"),
    userId: localStorage.getItem("lynscs_user_id") || `user_${Date.now()}`,
    theme: localStorage.getItem("lynscs_theme") || "light",
};
localStorage.setItem("lynscs_user_id", state.userId);

// ── Theme Management ──
function initTheme() {
    if (state.theme === "dark") {
        document.documentElement.setAttribute("data-theme", "dark");
    }
    updateThemeIcon();
}

function toggleTheme() {
    state.theme = state.theme === "light" ? "dark" : "light";
    localStorage.setItem("lynscs_theme", state.theme);
    
    if (state.theme === "dark") {
        document.documentElement.setAttribute("data-theme", "dark");
    } else {
        document.documentElement.removeAttribute("data-theme");
    }
    updateThemeIcon();
}

function updateThemeIcon() {
    const themeToggle = $("themeToggle");
    if (themeToggle) {
        themeToggle.setAttribute("title", state.theme === "light" ? "الوضع الداكن" : "الوضع الفاتح");
    }
}

// ── DOM refs ──
const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const inputEl = $("queryInput");
const sendBtn = $("sendBtn");
const welcomeScreen = $("welcomeScreen");
const suggestionsEl = $("suggestions");
const toastEl = $("toast");

/* ═══════════════════════
   INIT
═══════════════════════ */
(async function init() {
    initTheme(); // Initialize theme
    await loadSuggestions();
    await loadStatus();
    renderHistory();

    // Theme toggle
    $("themeToggle")?.addEventListener("click", toggleTheme);

    // Nav switching
    document.querySelectorAll(".nav-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const view = btn.dataset.view;
            switchView(view);
            document.querySelectorAll(".nav-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
        });
    });

    // Input events
    inputEl.addEventListener("input", onInput);
    inputEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
        }
    });

    sendBtn.addEventListener("click", submit);
    $("clearHistory")?.addEventListener("click", clearHistory);
    $("refreshStatus")?.addEventListener("click", loadStatus);

    // Initialize upload module
    initUpload();
})();

/* ═══════════════════════
   VIEW SWITCHING
═══════════════════════ */
function switchView(name) {
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    $(`view-${name}`)?.classList.add("active");
    if (name === "history") renderHistory();
    if (name === "settings") loadStatus();
}

/* ═══════════════════════
   SUGGESTIONS
═══════════════════════ */
async function loadSuggestions() {
    try {
        const res = await fetch(`${API}/api/suggestions`);
        const data = await res.json();
        suggestionsEl.innerHTML = "";
        (data.suggestions || []).forEach((q) => {
            const chip = document.createElement("button");
            chip.className = "suggestion-chip";
            chip.textContent = q;
            chip.onclick = () => {
                inputEl.value = q;
                onInput();
                submit();
            };
            suggestionsEl.appendChild(chip);
        });
    } catch (e) {
        console.error("Failed to load suggestions:", e);
        // Default suggestions
        const defaults = [
            "ما هو إجمالي الفاتورة؟",
            "من هو العميل في هذه الفاتورة؟",
            "هل تم استلام الدفعة؟",
            "قارن بين الفواتير",
        ];
        suggestionsEl.innerHTML = "";
        defaults.forEach((q) => {
            const chip = document.createElement("button");
            chip.className = "suggestion-chip";
            chip.textContent = q;
            chip.onclick = () => {
                inputEl.value = q;
                onInput();
                submit();
            };
            suggestionsEl.appendChild(chip);
        });
    }
}

/* ═══════════════════════
   STATUS
═══════════════════════ */
async function loadStatus() {
    try {
        const res = await fetch(`${API}/api/status`);
        const data = await res.json();

        const statChunks = $("stat-chunks");
        const statSources = $("stat-sources");
        const statModel = $("stat-model");
        const statStatus = $("stat-status");
        const infoModel = $("info-model");
        const infoDb = $("info-db");

        if (statChunks) statChunks.textContent = data.total_chunks ?? "—";
        if (statSources) statSources.textContent = data.total_sources ?? "—";
        if (statModel) statModel.textContent = (data.model || "—").replace("gemini-", "");
        if (statStatus) statStatus.textContent = data.status ?? "—";
        if (infoModel) infoModel.textContent = (data.model || "Gemini").replace("gemini-", "");
        if (infoDb) infoDb.textContent = data.total_chunks > 0 ? "متصلة" : "غير متصلة";

        // Update sidebar status
        const statusDot = document.querySelector(".status-dot");
        const statusText = document.querySelector(".status-text");
        const isOk = data.total_chunks > 0;

        if (statusDot) {
            statusDot.style.background = isOk ? "var(--success)" : "var(--warning)";
        }
        if (statusText) {
            statusText.textContent = isOk ? `${data.total_chunks} قطعة` : "قاعدة البيانات فارغة";
        }
    } catch (e) {
        console.error("Failed to load status:", e);
        const statusDot = document.querySelector(".status-dot");
        const statusText = document.querySelector(".status-text");
        if (statusDot) statusDot.style.background = "var(--error)";
        if (statusText) statusText.textContent = "API غير متاح";
    }
}

/* ═══════════════════════
   INPUT
═══════════════════════ */
function onInput() {
    const val = inputEl.value.trim();
    sendBtn.disabled = !val || state.loading;

    // Auto-resize
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 150) + "px";
}

/* ═══════════════════════
   SUBMIT
═══════════════════════ */
async function submit() {
    const query = inputEl.value.trim();
    if (!query || state.loading) return;

    state.loading = true;
    sendBtn.disabled = true;
    inputEl.value = "";
    inputEl.style.height = "auto";

    // Hide welcome screen
    if (welcomeScreen) welcomeScreen.style.display = "none";

    // User message
    appendMessage("user", query);

    // Typing indicator
    const typingId = showTyping();

    try {
        const res = await fetch(`${API}/api/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, user_id: state.userId }),
        });

        removeTyping(typingId);

        if (!res.ok) {
            const err = await res.json();
            appendMessage("assistant", `❌ خطأ: ${err.error || "فشل الاتصال"}`, { error: true });
            return;
        }

        const data = await res.json();

        appendMessage("assistant", data.answer, {
            sources: data.sources,
            elapsed: data.elapsed_s,
            blocked: data.blocked,
            warnings: data.warnings,
            follow_up: data.follow_up,
            confidence: data.confidence,
        });

        // Save history
        saveHistory(query, data.answer);
    } catch (e) {
        removeTyping(typingId);
        appendMessage("assistant", `❌ لا يمكن الاتصال بالخادم. تأكد من تشغيل الـ API.`, { error: true });
    } finally {
        state.loading = false;
        sendBtn.disabled = false;
        onInput();
        scrollToBottom();
    }
}

/* ═══════════════════════
   MESSAGES
═══════════════════════ */
function appendMessage(role, text, meta = {}) {
    const wrapper = document.createElement("div");
    wrapper.className = `message message--${role === "user" ? "user" : "assistant"}`;

    // Header
    const header = document.createElement("div");
    header.className = "message-header";
    header.textContent = role === "user" ? "أنت" : "LYNCS AI";
    wrapper.appendChild(header);

    // Content
    const content = document.createElement("div");
    content.className = "message-content";
    content.innerHTML = formatText(text);
    wrapper.appendChild(content);

    // Meta
    if (role === "assistant") {
        const metaRow = document.createElement("div");
        metaRow.className = "message-meta";

        if (meta.elapsed) {
            const t = document.createElement("span");
            t.className = "meta-tag";
            t.textContent = `⏱ ${meta.elapsed}s`;
            metaRow.appendChild(t);
        }

        if (meta.blocked) {
            const b = document.createElement("span");
            b.className = "meta-tag";
            b.textContent = "🚫 محجوب";
            metaRow.appendChild(b);
        }

        (meta.sources || []).forEach((src) => {
            const s = document.createElement("span");
            s.className = "meta-tag meta-tag--source";
            s.textContent = `📎 ${src}`;
            metaRow.appendChild(s);
        });

        if (metaRow.children.length) wrapper.appendChild(metaRow);

        // Follow-up
        if (meta.follow_up?.length) {
            const fu = document.createElement("div");
            fu.className = "follow-up";
            const lbl = document.createElement("div");
            lbl.className = "follow-up-label";
            lbl.textContent = "أسئلة مقترحة:";
            fu.appendChild(lbl);

            const chips = document.createElement("div");
            chips.className = "follow-up-chips";
            meta.follow_up.forEach((q) => {
                const chip = document.createElement("button");
                chip.className = "follow-chip";
                chip.textContent = q;
                chip.onclick = () => {
                    inputEl.value = q;
                    onInput();
                    submit();
                };
                chips.appendChild(chip);
            });
            fu.appendChild(chips);
            wrapper.appendChild(fu);
        }
    }

    messagesEl.appendChild(wrapper);
    scrollToBottom();
}

function formatText(text) {
    return text
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>")
        .replace(/`(.*?)`/g, `<code>$1</code>`)
        .replace(/\n/g, "<br>");
}

/* ═══════════════════════
   TYPING INDICATOR
═══════════════════════ */
let _typingCounter = 0;
function showTyping() {
    const id = ++_typingCounter;
    const el = document.createElement("div");
    el.className = "typing-indicator";
    el.dataset.typingId = id;
    el.innerHTML = `<span></span><span></span><span></span>`;
    messagesEl.appendChild(el);
    scrollToBottom();
    return id;
}
function removeTyping(id) {
    const el = messagesEl.querySelector(`[data-typing-id="${id}"]`);
    if (el) el.remove();
}

/* ═══════════════════════
   HISTORY
═══════════════════════ */
function saveHistory(query, answer) {
    state.history.unshift({ query, answer, time: new Date().toLocaleString("ar-SA") });
    state.history = state.history.slice(0, 50);
    localStorage.setItem("lynscs_history", JSON.stringify(state.history));
}

function renderHistory() {
    const list = $("historyList");
    if (!state.history.length) {
        list.innerHTML = `<p class="empty-state">لا توجد محادثات سابقة</p>`;
        return;
    }
    list.innerHTML = "";
    state.history.forEach((item) => {
        const el = document.createElement("div");
        el.className = "history-item";
        el.innerHTML = `
      <div class="history-item-query">${escHtml(item.query)}</div>
      <div class="history-item-answer">${escHtml(item.answer)}</div>
      <div class="history-item-time">${item.time}</div>
    `;
        el.onclick = () => {
            switchView("chat");
            document.querySelectorAll(".nav-btn").forEach((b) => b.classList.remove("active"));
            document.querySelector('[data-view="chat"]')?.classList.add("active");
            inputEl.value = item.query;
            onInput();
        };
        list.appendChild(el);
    });
}

function clearHistory() {
    if (!confirm("هل أنت متأكد من مسح كل السجل؟")) return;
    state.history = [];
    localStorage.removeItem("lynscs_history");
    renderHistory();
    showToast("تم مسح السجل");
}

/* ═══════════════════════
   HELPERS
═══════════════════════ */
function scrollToBottom() {
    requestAnimationFrame(() => {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    });
}

function showToast(msg, duration = 2500) {
    toastEl.textContent = msg;
    toastEl.classList.add("show");
    setTimeout(() => toastEl.classList.remove("show"), duration);
}

function escHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/* ══════════════════════════════════════════════════
   UPLOAD MODULE
══════════════════════════════════════════════════ */

// ── State ──
const uploadState = {
    queue: [], // { file, id }
    uploading: false,
};

// ── DOM refs ──
let dropZone,
    fileInput,
    selectFilesBtn,
    uploadQueue,
    queueList,
    queueTitle,
    startUploadBtn,
    clearQueueBtn,
    uploadResults,
    resultsList,
    filesList,
    ingestToggle,
    refreshFiles;

// ── Extension → emoji icon ──
const EXT_ICON = {
    ".pdf": "📄",
    ".docx": "📝",
    ".doc": "📝",
    ".png": "🖼️",
    ".jpg": "🖼️",
    ".jpeg": "🖼️",
    ".md": "📋",
    ".txt": "📃",
};
const extIcon = (name) => EXT_ICON[name.slice(name.lastIndexOf(".")).toLowerCase()] || "📎";

// ── Allowed extensions ──
const ALLOWED = new Set([".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg", ".md", ".txt"]);
const isAllowed = (f) => ALLOWED.has(f.name.slice(f.name.lastIndexOf(".")).toLowerCase());

/* ─────────────────────────────────
   Init upload view
───────────────────────────────── */
function initUpload() {
    dropZone = $("dropZone");
    fileInput = $("fileInput");
    selectFilesBtn = $("selectFilesBtn");
    uploadQueue = $("uploadQueue");
    queueList = $("queueList");
    queueTitle = $("queueTitle");
    startUploadBtn = $("startUploadBtn");
    clearQueueBtn = $("clearQueueBtn");
    uploadResults = $("uploadResults");
    resultsList = $("resultsList");
    filesList = $("filesList");
    ingestToggle = $("ingestToggle");
    refreshFiles = $("refreshFiles");

    if (!dropZone) return;

    // Open file picker on zone click
    dropZone.addEventListener("click", (e) => {
        if (e.target !== selectFilesBtn) fileInput.click();
    });
    selectFilesBtn?.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    fileInput.addEventListener("change", () => addFiles([...fileInput.files]));

    // Drag-and-drop
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });
    dropZone.addEventListener("dragleave", (e) => {
        if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove("drag-over");
    });
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        addFiles([...e.dataTransfer.files]);
    });

    // Queue buttons
    startUploadBtn?.addEventListener("click", startUpload);
    clearQueueBtn?.addEventListener("click", clearQueue);
    refreshFiles?.addEventListener("click", loadExistingFiles);
}

/* ─────────────────────────────────
   Add files to queue
───────────────────────────────── */
function addFiles(files) {
    const validFiles = files.filter((f) => {
        if (!isAllowed(f)) {
            showToast(`⛔ ${f.name} — امتداد غير مسموح`);
            return false;
        }
        if (f.size > 20 * 1024 * 1024) {
            showToast(`⛔ ${f.name} — الحجم أكبر من 20MB`);
            return false;
        }
        // Prevent duplicates
        if (uploadState.queue.some((q) => q.file.name === f.name && q.file.size === f.size)) return false;
        return true;
    });

    if (!validFiles.length) return;

    validFiles.forEach((f) => {
        uploadState.queue.push({
            file: f,
            id: `q_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        });
    });

    renderQueue();
    fileInput.value = ""; // Reset so same file can be picked again
}

/* ─────────────────────────────────
   Render queue
───────────────────────────────── */
function renderQueue() {
    if (!uploadState.queue.length) {
        uploadQueue.style.display = "none";
        return;
    }

    uploadQueue.style.display = "block";
    queueTitle.textContent = `${uploadState.queue.length} ملف${uploadState.queue.length > 1 ? " محددة" : " محدد"}`;
    queueList.innerHTML = "";

    uploadState.queue.forEach(({ file, id }) => {
        const row = document.createElement("div");
        row.className = "queue-item";
        row.dataset.qid = id;

        const sizeText =
            file.size > 1024 * 1024
                ? `${(file.size / 1024 / 1024).toFixed(1)} MB`
                : `${Math.round(file.size / 1024)} KB`;

        row.innerHTML = `
      <span class="queue-icon">${extIcon(file.name)}</span>
      <span class="queue-item-name">${escHtml(file.name)}</span>
      <span class="queue-item-size">${sizeText}</span>
      <button class="queue-item-remove" data-qid="${id}" title="حذف">×</button>
    `;
        queueList.appendChild(row);
    });

    // Remove listeners
    queueList.querySelectorAll(".queue-item-remove").forEach((btn) => {
        btn.addEventListener("click", (e) => {
            const qid = e.currentTarget.dataset.qid;
            uploadState.queue = uploadState.queue.filter((q) => q.id !== qid);
            renderQueue();
        });
    });
}

/* ─────────────────────────────────
   Start Upload
───────────────────────────────── */
async function startUpload() {
    if (!uploadState.queue.length || uploadState.uploading) return;

    uploadState.uploading = true;
    startUploadBtn.disabled = true;
    startUploadBtn.textContent = "⏳ جارٍ الرفع…";
    uploadResults.style.display = "none";
    resultsList.innerHTML = "";

    const formData = new FormData();
    uploadState.queue.forEach(({ file }) => formData.append("files", file));
    formData.append("ingest", ingestToggle.checked ? "true" : "false");
    formData.append("user_id", state.userId);

    // Animate progress
    const progressInterval = setInterval(() => {
        const bars = document.querySelectorAll(".progress-bar");
        bars.forEach((bar) => {
            const w = parseFloat(bar.style.width || "0");
            if (w < 85) bar.style.width = (w + Math.random() * 6) + "%";
        });
    }, 300);

    try {
        const res = await fetch(`${API}/api/upload`, { method: "POST", body: formData });
        const data = await res.json();
        clearInterval(progressInterval);

        if (res.ok) {
            await new Promise((r) => setTimeout(r, 400));
            renderResults(data);
            uploadState.queue = [];
            renderQueue();
            showToast(`✅ ${data.uploaded} ملف رُفع بنجاح`);
            await loadExistingFiles();
        } else {
            showToast(`❌ ${data.error || "فشل الرفع"}`);
        }
    } catch (err) {
        clearInterval(progressInterval);
        showToast("❌ لا يمكن الاتصال بالخادم");
        console.error(err);
    } finally {
        uploadState.uploading = false;
        startUploadBtn.disabled = false;
        startUploadBtn.textContent = "رفع";
    }
}

/* ─────────────────────────────────
   Render Results
───────────────────────────────── */
function renderResults(data) {
    uploadResults.style.display = "block";
    resultsList.innerHTML = "";

    // Summary
    const summary = document.createElement("div");
    summary.style.cssText = "display:flex;gap:12px;margin-bottom:10px;font-size:0.8rem";
    summary.innerHTML = `
    <span style="color:var(--success)">✅ نجح: ${data.uploaded}</span>
    ${data.failed ? `<span style="color:var(--error)">❌ فشل: ${data.failed}</span>` : ""}
    <span style="color:var(--text-tertiary)">⏱ ${data.elapsed_s}s</span>
  `;
    resultsList.appendChild(summary);

    (data.results || []).forEach((r) => {
        const el = document.createElement("div");
        el.className = `result-item result-item--${r.status}`;
        const itemTaskId = `task_${Math.random().toString(36).slice(2, 7)}`;
        el.id = itemTaskId;

        const icon =
            r.status === "ingested"
                ? "✅"
                : r.status === "saved"
                ? "💾"
                : r.status === "processing"
                ? "⏳"
                : "❌";

        el.innerHTML = `
      <span class="result-item-icon">${icon}</span>
      <span class="result-item-name">${escHtml(r.filename)}</span>
      <span class="result-item-msg">${escHtml(r.message)}</span>
      ${r.chunks ? `<span class="result-item-chunks">${r.chunks} قطعة</span>` : ""}
    `;
        resultsList.appendChild(el);

        if (r.status === "processing") {
            pollIngestStatus(r.filename, itemTaskId);
        }
    });
}

/* ─────────────────────────────────
   Poll Ingest Status
───────────────────────────────── */
async function pollIngestStatus(filename, elementId) {
    const card = document.getElementById(elementId);
    if (!card) return;

    let polls = 0;
    const maxPolls = 60;

    const interval = setInterval(async () => {
        polls++;

        try {
            const res = await fetch(`${API}/api/ingest-status/${encodeURIComponent(filename)}`);
            const data = await res.json();

            if (data.status === "completed") {
                clearInterval(interval);
                card.className = "result-item result-item--ingested";
                card.querySelector(".result-item-icon").textContent = "✅";
                card.querySelector(".result-item-msg").textContent = "تمت المعالجة بنجاح!";

                if (data.chunks) {
                    let chunkSpan = card.querySelector(".result-item-chunks");
                    if (!chunkSpan) {
                        chunkSpan = document.createElement("span");
                        chunkSpan.className = "result-item-chunks";
                        card.appendChild(chunkSpan);
                    }
                    chunkSpan.textContent = `${data.chunks} قطعة`;
                }

                showToast(`✨ اكتملت معالجة: ${filename}`);
                await loadStatus();
                await loadExistingFiles();
            }

            if (data.status === "error" || polls >= maxPolls) {
                clearInterval(interval);
                card.className = "result-item result-item--error";
                card.querySelector(".result-item-icon").textContent = "❌";
                card.querySelector(".result-item-msg").textContent = data.message || "انتهت مهلة المعالجة";
            }
        } catch (e) {
            console.error("Polling error:", e);
        }
    }, 2000);
}

/* ─────────────────────────────────
   Clear Queue
───────────────────────────────── */
function clearQueue() {
    uploadState.queue = [];
    renderQueue();
    uploadResults.style.display = "none";
}

/* ─────────────────────────────────
   Load Existing Files
───────────────────────────────── */
async function loadExistingFiles() {
    if (!filesList) return;

    filesList.innerHTML = `<p class="empty-state">⏳ جارٍ التحميل…</p>`;
    try {
        const res = await fetch(`${API}/api/files`);
        const data = await res.json();

        if (!data.files || data.files.length === 0) {
            filesList.innerHTML = `<p class="empty-state">لا توجد ملفات بعد — ارفع أول مستند</p>`;
            return;
        }

        filesList.innerHTML = "";
        data.files.forEach((f) => {
            const el = document.createElement("div");
            el.className = "file-item";
            el.innerHTML = `
        <span class="file-icon">${extIcon(f.name)}</span>
        <span class="file-name">${escHtml(f.name)}</span>
        <span class="file-size">${f.size_kb} KB</span>
        <button class="file-delete" data-name="${escHtml(f.name)}" title="حذف">🗑</button>
      `;
            filesList.appendChild(el);
        });

        // Delete handlers
        filesList.querySelectorAll(".file-delete").forEach((btn) => {
            btn.addEventListener("click", async () => {
                const name = btn.dataset.name;
                if (!confirm(`هل تريد حذف "${name}"؟`)) return;
                await deleteFile(name);
            });
        });
    } catch (e) {
        console.error("Failed to load files:", e);
        filesList.innerHTML = `<p class="empty-state">⚠️ تعذّر تحميل قائمة الملفات</p>`;
    }
}

/* ─────────────────────────────────
   Delete File
───────────────────────────────── */
async function deleteFile(filename) {
    try {
        const res = await fetch(`${API}/api/files/${encodeURIComponent(filename)}`, { method: "DELETE" });
        if (res.ok) {
            showToast(`🗑 تم حذف ${filename}`);
            await loadExistingFiles();
        } else {
            showToast("❌ فشل الحذف");
        }
    } catch (e) {
        showToast("❌ خطأ في الاتصال");
        console.error(e);
    }
}

// Hook into nav switching to load files on view open
const _origSwitchView = switchView;
window.switchView = function (name) {
    _origSwitchView(name);
    if (name === "upload") loadExistingFiles();
};
