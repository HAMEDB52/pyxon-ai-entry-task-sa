# 🌐 LYNCS AI — Web Interface

## الهيكل
```
api/
  main.py           ← FastAPI (port 8000)
  requirements.txt

frontend/
  server.js         ← Express Node.js (port 3000)
  package.json
  public/
    index.html      ← LYNCS AI Modern UI
    style.css       ← Clean, minimal design
    app.js          ← Frontend logic
```

## ✨ المميزات الجديدة

### التصميم الجديد
- **واجهة عصرية نظيفة** - تصميم Minimalist مشابه للصور المرفوعة
- **Dark Mode تلقائي** - يدعم الوضع الداكن تلقائياً
- **خطوط عربية محسّنة** - IBM Plex Sans Arabic
- **حركات سلسة** - Animations احترافية

### الهوية الجديدة
- **LYNCS AI** - مساعد ذكي متعدد الأغراض
- **RAG متطور** - بحث واسترجاع معزّز
- **استشهاد بالمصادر** - كل معلومة موثقة
- **لغة مرنة** - عربي/إنجليزي حسب المستخدم

## تشغيل الـ Backend (FastAPI)

```bash
# تثبيت المتطلبات
pip install -r requirements.txt

# تشغيل
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

الوصول: http://localhost:8000/docs

## تشغيل الـ Frontend (Node.js)

```bash
cd frontend

# تثبيت الحزم
npm install

# تشغيل
npm start

# أو للتطوير مع auto-reload
npm run dev
```

الوصول: http://localhost:3000

## متغيرات البيئة

```bash
# .env
PORT=3000                    # منفذ الـ Frontend
API_URL=http://localhost:8000 # عنوان الـ FastAPI

# Google Gemini
GOOGLE_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash

# Database
DATABASE_URL=postgresql://...
```

## الاختصارات

| الإجراء | الاختصار |
|---------|----------|
| إرسال | Enter |
| سطر جديد | Shift+Enter |
| مسح المحادثة | زر 🗑 في الأعلى |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | حالة النظام |
| GET | `/api/suggestions` | اقتراحات |
| POST | `/api/query` | إرسال سؤال |
| POST | `/api/upload` | رفع ملفات |
| GET | `/api/files` | قائمة الملفات |
| DELETE | `/api/files/:name` | حذف ملف |

## استكشاف الأخطاء

### الـ Frontend لا يتصل
```bash
# تأكد من تشغيل الـ Backend
python -m uvicorn api.main:app --port 8000

# تحقق من API_URL في server.js
```

### رفع الملفات يفشل
```bash
# تحقق من الصلاحيات
chmod 755 All_Invoices_Files/

# تأكد من وجود المسار
mkdir -p All_Invoices_Files
```

### قاعدة البيانات فارغة
```bash
# شغّل ingest_docs.py
python scripts/ingest_docs.py --all
```

---

**LYNCS AI** — مساعدك الذكي المدعوم بتقنية RAG 🚀
