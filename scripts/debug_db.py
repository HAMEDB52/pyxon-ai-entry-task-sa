# ============================================================
# scripts/debug_db.py
# فحص البيانات المخزنة في قاعدة البيانات
# ============================================================

import sys
sys.path.append(".")

from dotenv import load_dotenv
load_dotenv()

from src.database.relational_db import RelationalDB
from src.database.db_setup import get_connection

# ════════════════════════════════════════
# فحص القطع المخزنة
# ════════════════════════════════════════

db = RelationalDB()

# 1. إحصائيات عامة
print("\n" + "="*60)
print("📊 إحصائيات قاعدة البيانات")
print("="*60)
stats = db.get_statistics()
for k, v in stats.items():
    if k != "top_sources":
        print(f"  {k}: {v}")

# 2. جميع المصادر
print("\n" + "="*60)
print("📁 المصادر المُدخلة")
print("="*60)
sources = db.get_all_sources()
for src in sources:
    print(f"  [{src.id}] {src.file_name} → {src.num_chunks} قطعة")

# 3. فحص قطع "رخصة البناء"
print("\n" + "="*60)
print("🔍 فحص قطع 'رخصة البناء'")
print("="*60)

conn = get_connection()
cur = conn.cursor()

try:
    # جلب جميع قطع رخصة البناء
    cur.execute("""
        SELECT id, chunk_id, content, chunk_type, source_file, 
               page_number, parent_heading, summary, keywords
        FROM doc_chunks
        WHERE source_file LIKE '%رخصة%'
        ORDER BY id
    """)
    
    rows = cur.fetchall()
    print(f"\n✅ وُجد {len(rows)} قطعة\n")
    
    for i, row in enumerate(rows, 1):
        chunk_id = row[1]
        content = row[2][:500].replace('\n', ' ')
        chunk_type = row[3]
        page = row[5]
        heading = row[6] or "بدون عنوان"
        
        print(f"{'='*60}")
        print(f"قطعة #{i} | ID: {chunk_id}")
        print(f"النوع: {chunk_type} | الصفحة: {page}")
        print(f"العنوان الأب: {heading}")
        print(f"{'-'*60}")
        print(f"المحتوى:\n{content}...")
        print()
        
finally:
    cur.close()
    conn.close()

# 4. بحث نصي مباشر
print("\n" + "="*60)
print("🔎 بحث نصي عن 'إبراهيم' في قاعدة البيانات")
print("="*60)

conn2 = get_connection()
cur = conn2.cursor()
try:
    cur.execute("""
        SELECT chunk_id, source_file, content
        FROM doc_chunks
        WHERE content ILIKE '%إبراهيم%'
        LIMIT 3
    """)
    rows = cur.fetchall()
    
    if rows:
        for r in rows:
            print(f"\n📍 {r[1]} | chunk: {r[0]}")
            print(f"   المحتوى: {r[2][:300]}...")
    else:
        print("  ❌ لم يُعثر على 'إبراهيم' في أي قطعة!")
        
finally:
    cur.close()
    conn2.close()

print("\n✅ اكتمل الفحص")
