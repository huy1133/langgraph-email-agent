import sqlite3
import sys
import io
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding='utf-8')

db_path = Path(__file__).resolve().parent / "data" / "customers.db"
if not db_path.exists():
    print("Database SQL chưa được tạo (chưa chạy Ingest).")
    sys.exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

def print_table(cursor, table_name, title):
    try:
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        print(f"\n[ BẢNG {title.upper()} ] - Tổng số: {len(rows)}")
        print("-" * 40)
        print(f"{'ID':<5} | {'TÊN'}")
        print("-" * 40)
        for r in rows:
            print(f"{r[0]:<5} | {str(r[1])}")
        print("-" * 40)
    except sqlite3.OperationalError:
        print(f"Bảng {table_name} không tồn tại!")

print("=== BÁO CÁO THỰC THỂ SQLITE CHUẨN HÓA (NORMALIZED) ===")
print_table(cursor, "customers", "Khách hàng")
print_table(cursor, "companies", "Công ty")
print_table(cursor, "industries", "Ngành nghề")

conn.close()
