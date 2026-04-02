# 🎯 Pyxon AI Entry Task - الدليل الشامل

**دليل متكامل لتنفيذ وتسليم مهمة Pyxon AI Junior Engineer**

---

## 📋 جدول المحتويات

1. [نظرة عامة](#-نظرة-عامة)
2. [المتطلبات](#-المتطلبات)
3. [التثبيت والتشغيل](#-التثبيت-والتشغيل)
4. [المكونات التقنية](#-المكونات-التقنية)
5. [اختبار Benchmark](#-اختبار-benchmark)
6. [خطوات التسليم](#-خطوات-التسليم)
7. [النشر](#-النشر)
8. [الأسئلة الشائعة](#-الأسئلة-الشائعة)

---

## ✨ نظرة عامة

هذا المشروع ينفذ جميع متطلبات **Pyxon AI Entry Task**:

✅ **Document Parser** - معالجة PDF, DOC/DOCX, TXT  
✅ **Intelligent Chunking** - تجزئة ذكية Fixed/Dynamic  
✅ **Vector DB** - تخزين متجهي مع pgvector  
✅ **SQL DB** - تخزين علائقي مع PostgreSQL  
✅ **Arabic Support** - دعم كامل للعربية مع التشكيل  
✅ **Benchmark Suite** - 12 اختبار شامل  
✅ **RAG Integration** - جاهز للتكامل مع RAG  
✅ **Live Demo** - قابل للنشر المباشر  

---

## 📋 المتطلبات

### الأساسية
- Python 3.10+
- Node.js 18+ (للواجهة)
- PostgreSQL 14+ مع pgvector
- Google Gemini API Key (مجاني)

### الحصول على المفاتيح
1. **Google Gemini API:** https://aistudio.google.com/app/apikey
2. **Neon Database:** https://neon.tech (PostgreSQL مجاني)

---

## ⚙️ التثبيت والتشغيل

### 1. تثبيت المكتبات

```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### 2. إعداد البيئة

```bash
# انسخ ملف البيئة
cp .env.example .env
```

**عدّل ملف `.env`:**
```bash
# Google Gemini (مجاني)
GOOGLE_API_KEY=AIzaSy...your_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

# Neon Database
DATABASE_URL=postgresql://user:pass@host.neon.tech/neondb?sslmode=require
```

### 3. إعداد قاعدة البيانات

```bash
# تفعيل pgvector
psql "$DATABASE_URL" -c "CREATE EXTENSION IF NOT EXISTS vector;"

# إنشاء الجداول
python src/database/db_setup.py
```

### 4. إدخال مستندات

```bash
# ضع ملفات PDF/DOCX/TXT في المجلد
mkdir -p All_Invoices_Files
cp your_files.pdf All_Invoices_Files/

# تشغيل الإدخال
python scripts/ingest_docs.py
```

### 5. تشغيل النظام

**Terminal 1 - Backend:**
```bash
python -m uvicorn api.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

**الوصول:**
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

---

## 🧠 المكونات التقنية

### 1. Intelligent Chunking Strategy Selector

**الملف:** `src/data_processing/chunking/strategy_selector.py`

**الوظيفة:** يحلل المستند ويختار أفضل استراتيجية chunking تلقائياً

**الاستراتيجيات:**
- **Fixed:** للمستندات المنتظمة (فواتير، نماذج)
- **Dynamic:** للمستندات غير المنتظمة (تقارير، مقالات)
- **Semantic:** للمستندات المهيكلة (كتب، أدلة)
- **Hybrid:** للمستندات المعقدة

**التحليل:**
```python
from src.data_processing.chunking.strategy_selector import analyze_document

analysis = analyze_document(text, "document.pdf")
print(f"Strategy: {analysis.recommended_strategy.value}")
print(f"Confidence: {analysis.confidence:.2f}")
print(f"Language: {analysis.language}")
print(f"Has Diacritics: {analysis.has_diacritics}")
```

**معايير الاختيار:**
- كثافة العناوين
- عدد الجداول والقوائم
- طول المستند
- تنوع المفردات
- اللغة
- وجود التشكيل

---

### 2. Arabic Diacritics Support

**الملف:** `src/data_processing/arabic_lemmatizer.py`

**الميزات:**
- كشف التشكيل (Diacritics Detection)
- الحفاظ على التشكيل (Preservation)
- إزالة التشكيل الاختيارية (Optional Removal)
- مقارنة مع تجاهل التشكيل (Agnostic Comparison)

**التشكيل المدعوم:**
```
َ  ُ  ِ  ّ  ْ  ً  ٌ  ٍ  ٓ  ٔ  ٕ
```

**الاستخدام:**
```python
from src.data_processing.arabic_lemmatizer import get_lemmatizer

lemmatizer = get_lemmatizer()

# 1. مقارنة مع تجاهل التشكيل
match = lemmatizer.compare_with_diacritics("الْحَمْدُ", "الحمد")
# النتيجة: True

# 2. الحفاظ على التشكيل
preserved = lemmatizer.preserve_diacritics("بِسْمِ اللَّهِ الرَّحْمَٰنِ")

# 3. إزالة التشكيل
without_tashkeel = lemmatizer.normalize_with_diacritics(
    "الْحَمْدُ لِلَّهِ", 
    remove_tashkeel=True
)

# 4. استخراج نمط التشكيل
pattern = lemmatizer.extract_diacritics_pattern("الْحَمْدُ")
```

---

### 3. Benchmark Suite

**الملف:** `scripts/pyxon_task_benchmark.py`

**الفئات:**
1. **Strategy Selection** (اختباران) - دقة اختيار الاستراتيجية
2. **Arabic Support** (اختباران) - دعم اللغة العربية
3. **Diacritics Support** (4 اختبارات) - دعم التشكيل
4. **Chunking Quality** (اختبار) - جودة التجزئة
5. **Performance** (اختباران) - الأداء والسرعة
6. **Multi-format** (اختبار) - دعم الصيغ المتعددة

**التشغيل:**
```bash
python scripts/pyxon_task_benchmark.py
```

**النتائج المتوقعة:**
```
============================================================
📊 تقرير اختبارات Pyxon AI
============================================================

✅ النتائج العامة:
  إجمالي الاختبارات: 12
  الناجحة: 11
  الفاشلة: 1
  معدل النجاح: 91.7%
  متوسط الدرجة: 0.95
  متوسط الوقت: 3.98ms

📈 النتائج حسب الفئة:
  strategy_selection: 0.95 (2 tests)
  arabic_support: 0.75 (2 tests)
  diacritics_support: 1.00 (4 tests)
  chunking_quality: 1.00 (1 tests)
  performance: 1.00 (2 tests)
  multi_format: 1.00 (1 tests)
```

---

## 📤 خطوات التسليم

### الخطوة 1: Fork Repository

```
1. اذهب إلى: https://github.com/pyxon-ai/pyxon-ai-entry-task-sa
2. اضغط "Fork" في الزاوية العلوية اليمنى
3. سيتم إنشاء نسخة في حسابك GitHub
```

### الخطوة 2: تحديث معلومات الاتصال

**افتح هذا الملف وحدّث المعلومات:**

```markdown
# في نهاية هذا الملف، حدّث:

**Name:** Hamed Alruwaili
**Email:** your.email@gmail.com
**Phone:** +966 XX XXX XXXX
**GitHub:** @your_username
**Demo Link:** [سيتم إضافته بعد النشر]
```

### الخطوة 3: Commit & Push

```bash
# إضافة جميع الملفات
git add .

# Commit
git commit -m "Complete Pyxon AI Entry Task

- Intelligent Chunking Strategy Selector
- Enhanced Arabic Diacritics Support  
- Comprehensive Benchmark Suite (91.7% success)
- Full documentation"

# Push
git push origin main
```

### الخطوة 4: إنشاء Pull Request

```
1. اذهب إلى: https://github.com/pyxon-ai/pyxon-ai-entry-task-sa
2. اضغط "Pull requests"
3. اضغط "New pull request"
4. اختر:
   - base: pyxon-ai/pyxon-ai-entry-task-sa:main
   - compare: YOUR_USERNAME/Agentic_RAG:main
5. اضغط "Create pull request"
```

### الخطوة 5: ملء Pull Request

**انسخ هذا المحتوى في وصف PR:**

```markdown
## 📋 Contact Information

**Name:** Hamed Alruwaili
**Email:** your.email@gmail.com
**Phone:** +966 XX XXX XXXX
**GitHub:** @your_username
**Demo Link:** [URL بعد النشر]

## ✅ Requirements Checklist

- [x] Document Parser - PDF, DOC/DOCX, TXT
- [x] Intelligent Chunking - Fixed & Dynamic strategies
- [x] Vector DB Storage - PostgreSQL + pgvector
- [x] SQL DB Storage - PostgreSQL relational
- [x] Arabic Support - Full Unicode with diacritics
- [x] Benchmark Suite - 12 tests, 91.7% success
- [x] RAG Integration Ready
- [x] Documentation - Complete

## 📊 Benchmark Results

- Total Tests: 12
- Success Rate: 91.7%
- Average Score: 0.95/1.0
- Average Time: 3.98ms

## 🔗 Demo

[رابط الـDemo بعد النشر]
```

### الخطوة 6: التأكيد

**بعد إنشاء PR:**
1. ستصلك رسالة تأكيد على البريد الإلكتروني
2. **رد على الرسالة** لتأكيد الاستلام
3. أكد توفرک للمقابلة

---

## 🚀 النشر

### الخيار 1: Render (الأسهل - موصى به)

**المميزات:**
- مجاني للخطة الأساسية
- دعم PostgreSQL مدمج
- نشر تلقائي من GitHub
- شهادة SSL مجانية

**الخطوات:**

#### 1. إعداد قاعدة البيانات

```
1. سجل في https://neon.tech
2. Create Project
3. انسخ Connection String
4. فعّل pgvector:
   psql "$DATABASE_URL" -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

#### 2. النشر على Render

```
1. سجل في https://render.com
2. New → Web Service
3. Connect GitHub repository
4. اختر fork الخاص بك
```

**الإعدادات:**
```
Name: pyxon-ai-demo
Region: Bahrain
Branch: main
Runtime: Python 3
Build Command: pip install -r requirements.txt
Start Command: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

**Environment Variables:**
```
GOOGLE_API_KEY=your_key_here
DATABASE_URL=postgresql://...
PORT=10000
```

#### 3. الانتظار

```
انتظر 5-10 دقائق
ستحصل على URL مثل:
https://pyxon-ai-demo.onrender.com
```

---

### الخيار 2: Railway

**المميزات:**
- $5 رصيد مجاني شهرياً
- نشر أسهل
- PostgreSQL مدمج

**الخطوات:**

```
1. سجل في https://railway.app
2. New Project → Deploy from GitHub
3. اختر repository
4. أضف PostgreSQL من Marketplace
5. أضف المتغيرات البيئية
6. Deploy
```

---

### الخيار 3: Vercel (للواجهة فقط)

**ملاحظة:** Vercel لا يدعم FastAPI مباشرة

```
Backend: Render/Railway
Frontend: Vercel
```

**النشر:**

```bash
cd frontend
npm install -g vercel
vercel --prod
```

**تعديل API URL:**

```javascript
// في frontend/public/app.js
const API_URL = 'https://your-backend.onrender.com';
```

---

## 🧪 اختبار النشر

### Health Check

```bash
curl https://your-demo.onrender.com/health
```

**الاستجابة المتوقعة:**
```json
{
  "status": "healthy",
  "database": "connected",
  "vector_store": "ready"
}
```

### اختبار رفع مستند

```bash
curl -X POST https://your-demo.onrender.com/upload \
  -F "files=@test.pdf" \
  -F "ingest=true"
```

### اختبار استعلام

```bash
curl -X POST https://your-demo.onrender.com/query \
  -H "Content-Type: application/json" \
  -d '{"query": "ما هو الذكاء الاصطناعي؟"}'
```

---

## ❓ الأسئلة الشائعة

### 1. ما هو الموعد النهائي؟

**Saturday, April 4th, 13:00 Saudi time**

### 2. هل النشر إلزامي؟

نشر Demo مباشر **مطلوب** حسب المتطلبات.

### 3. كيف أحصل على Google Gemini API Key؟

1. اذهب إلى: https://aistudio.google.com/app/apikey
2. سجل الدخول بحساب Google
3. اضغط "Get API Key"
4. انسخ المفتاح وضعه في `.env`

### 4. ما هي قاعدة البيانات المطلوبة؟

PostgreSQL 14+ مع ملحق pgvector للبحث المتجهي.

يمكنك استخدام:
- **Neon.tech** (مجاني - موصى به)
- **Railway** (مجاني مع رصيد $5)
- **محلي** (للاختبار فقط)

### 5. كيف أختبر النظام محلياً؟

```bash
# 1. تثبيت المكتبات
pip install -r requirements.txt

# 2. إعداد .env
cp .env.example .env
# عدّل وأضف GOOGLE_API_KEY و DATABASE_URL

# 3. تشغيل Benchmark
python scripts/pyxon_task_benchmark.py

# 4. تشغيل Backend
python -m uvicorn api.main:app --reload

# 5. تشغيل Frontend
cd frontend && npm start
```

### 6. ما هي نتائج Benchmark المتوقعة؟

```
Total Tests: 12
Success Rate: 91.7%
Average Score: 0.95/1.0
Average Time: 3.98ms
```

### 7. ماذا أفعل إذا فشل Benchmark؟

تحقق من:
1. `GOOGLE_API_KEY` في `.env` صحيح
2. `DATABASE_URL` صحيح و pgvector مفعّل
3. جميع المكتبات مثبتة: `pip install -r requirements.txt`

### 8. كيف أضيف معلومات الاتصال؟

في نهاية هذا الملف، حدّث:

```markdown
## 📋 Contact Information

**Name:** Hamed Alruwaili
**Email:** your.email@gmail.com
**Phone:** +966 XX XXX XXXX
**GitHub:** @your_username
```

### 9. ماذا أكتب في Pull Request؟

استخدم القالب في قسم **"الخطوة 5: ملء Pull Request"** أعلاه.

### 10. متى أتوقع ردًا؟

- **خلال 24 ساعة:** رسالة تأكيد
- **خلال أسبوع:** مراجعة تقنية
- **إذا تم اختيارك:** مقابلة

---

## 📊 هيكل المشروع

```
Agentic_RAG/
│
├── src/data_processing/
│   ├── chunking/
│   │   ├── strategy_selector.py      ⭐ اختيار استراتيجية ذكي
│   │   └── boundary_detector.py      # تجزئة
│   ├── arabic_lemmatizer.py          ⭐ دعم التشكيل العربي
│   └── ...
│
├── scripts/
│   ├── pyxon_task_benchmark.py       ⭐ مجموعة اختبارات
│   └── ingest_docs.py                # إدخال مستندات
│
├── api/
│   └── main.py                       # FastAPI Backend
│
├── frontend/
│   ├── server.js                     # Node.js Server
│   └── public/                       # واجهة ويب
│
├── .env                              # متغيرات البيئة
├── requirements.txt                  # مكتبات Python
└── README.md                         # توثيق
```

---

## 📋 Contact Information

**حدّث المعلومات أدناه قبل التسليم:**

```
Name: Hamed Alruwaili
Email: [YOUR_EMAIL_HERE]
Phone: [YOUR_PHONE_HERE]
GitHub: [@YOUR_USERNAME_HERE]
Location: Saudi Arabia
```

---

## ✅ قائمة التحقق النهائية

قبل التسليم، تأكد من:

- [ ] Benchmark شغّل بنجاح (91.7%+)
- [ ] معلومات الاتصال مُحدّثة
- [ ] Fork repository على GitHub
- [ ] Commit & Push جميع الملفات
- [ ] Pull Request مُنشأ
- [ ] Demo URL مُضاف (إذا تم النشر)
- [ ] الرد على رسالة التأكيد

---

## 🎉 بالتوفيق!

**تمنياتنا لك بالتوفيق في مقابلة Pyxon AI!**

**الحالة:** جاهز للتسليم ✅

---

*آخر تحديث: أبريل 2024*
