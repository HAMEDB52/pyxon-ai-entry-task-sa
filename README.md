# 🚀 LYNCS AI — Agentic RAG System

نظام ذكاء اصطناعي متقدم للمحادثة والاسترجاع المعزّز (RAG) مع وكلاء متعددين

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ المميزات

- **🤖 وكلاء متعددون**: 4 وكلاء متخصصون (بحث، تحقق، تصحيح، إجابة)
- **🔍 بحث متقدم**: Hybrid Search + Reranker + Query Expansion
- **📄 معالجة مستندات**: PDF, DOCX, PNG, JPG مع OCR للعربية
- **💬 واجهة حديثة**: UI نظيف مع Dark/Light Mode
- **🔒 أمان**: Gatekeeper للحماية من Prompt Injection
- **📊 قاعدة بيانات**: PostgreSQL + pgvector للبحث المتجهي

## 🏗️ البنية

```
Agentic_RAG/
├── api/                  # FastAPI Backend
│   └── main.py
├── frontend/             # Node.js Frontend
│   ├── server.js
│   └── public/
│       ├── index.html
│       ├── style.css
│       └── app.js
├── src/
│   ├── agents/           # الوكلاء
│   │   ├── agent1_research.py
│   │   ├── agent2_verification.py
│   │   ├── agent3_correction.py
│   │   ├── agent4_answer.py
│   │   ├── langgraph_pipeline.py
│   │   └── state.py
│   ├── database/         # قاعدة البيانات
│   ├── data_processing/  # معالجة النصوص
│   ├── reasoning_engine/ # المنطق
│   └── security/         # الأمان
├── scripts/
│   └── ingest_docs.py    # إدخال المستندات
└── All_Invoices_Files/   # المستندات
```

## 🚀 البدء السريع

### 1. المتطلبات

- Python 3.10+
- Node.js 18+
- PostgreSQL (Neon.tech أو محلي)

### 2. التثبيت

```bash
# استنساخ المشروع
git clone https://github.com/YOUR_USERNAME/Agentic_RAG.git
cd Agentic_RAG

# تثبيت Python dependencies
pip install -r requirements.txt

# تثبيت Node.js dependencies
cd frontend
npm install
cd ..
```

### 3. الإعداد

```bash
# انسخ ملف البيئة
cp .env.example .env

# عدّل .env وأضف مفاتيحك
# - GOOGLE_API_KEY
# - DATABASE_URL
```

### 4. إدخال المستندات

```bash
# ضع ملفات PDF في All_Invoices_Files/
python scripts/ingest_docs.py
```

### 5. التشغيل

```bash
# Terminal 1: Backend (FastAPI)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend (Node.js)
cd frontend
npm start
```

الوصول:
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

## 📖 الاستخدام

### من الواجهة web

1. افتح http://localhost:3000
2. اسأل عن أي معلومات في المستندات
3. رفع ملفات جديدة من تبويب "رفع ملفات"

### من API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "ما إجمالي الفاتورة؟"}'
```

## ⚙️ التكوين

### تحسين الأداء

في `scripts/ingest_docs.py`:

```python
ENABLE_ARABIC_ENHANCER = True   # تحسين النص العربي (بطيء)
ENABLE_SUMMARY_GEN     = True   # توليد الملخصات
ENABLE_KEYWORD_EXT     = True   # استخراج الكلمات
```

### Dark/Light Mode

اضغط على زر 🌙/☀️ في sidebar للتبديل بين الوضعين.

## 🧪 الاختبار

```bash
# تشغيل الاستعلامات
python scripts/run_query.py "ما إجمالي الفاتورة؟"

# فحص قاعدة البيانات
python scripts/debug_db.py
```

## 📊 الإحصائيات

| الميزة | القيمة |
|--------|--------|
| وكلاء | 4 |
| مستندات مدعومة | PDF, DOCX, PNG, JPG |
| نماذج LLM | Gemini 2.5 Flash |
| Embeddings | Gemini 3072 أبعاد |
| قاعدة بيانات | PostgreSQL + pgvector |

## 🔧 استكشاف الأخطاء

### "لا توجد مستندات"

```bash
# تأكد من إدخال المستندات
python scripts/ingest_docs.py

# تحقق من قاعدة البيانات
python scripts/debug_db.py
```

### "API غير متاح"

```bash
# تأكد من تشغيل Backend
python -m uvicorn api.main:app --port 8000

# تحقق من API_URL في frontend/server.js
```

### وقت معالجة طويل

```python
# عطّل Arabic Enhancer في scripts/ingest_docs.py
ENABLE_ARABIC_ENHANCER = False
```

## 📝 الترخيص

MIT License — راجع [LICENSE](LICENSE) للتفاصيل.

## 🤝 المساهمة

المساهمات مرحب بها! يرجى:

1. Fork المشروع
2. إنشاء فرع (`git checkout -b feature/AmazingFeature`)
3. Commit التغييرات (`git commit -m 'Add AmazingFeature'`)
4. Push للفرع (`git push origin feature/AmazingFeature`)
5. افتح Pull Request

## 📬 التواصل

- **GitHub Issues**: للأسئلة التقنية والمشاكل
- **Email**: your.email@example.com

---

**صنع بـ ❤️ بواسطة LYNCS AI Team**
