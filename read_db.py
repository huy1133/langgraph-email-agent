from pathlib import Path
import chromadb
import sys
import io

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding='utf-8')

# Khởi tạo đường dẫn tới chroma_data
db_path = Path(__file__).resolve().parent / "chroma_data"

if not db_path.exists():
    print("Database chưa được tạo (thư mục chroma_data không tồn tại).")
    sys.exit(1)

# Trỏ Client vào thư mục
client = chromadb.PersistentClient(path=str(db_path))

try:
    # Lấy collection gốc mà ta đã cấu hình là 'email_chunks'
    collection = client.get_collection(name="email_chunks")
except Exception as e:
    print("Không tìm thấy collection 'email_chunks'. Bạn đã chạy ingest thành công chưa?")
    sys.exit(1)

# Lấy 5 bản ghi đầu tiên trong DB để xem
results = collection.peek(limit=5)
total_count = collection.count()

print(f"Tổng số chunks (đoạn text) đang có trong DB: {total_count}")
print("-" * 50)

# In thử dữ liệu
if total_count > 0:
    print("Hiển thị 5 bản ghi đầu tiên:")
    for i in range(len(results['ids'])):
        print(f"[{i+1}] ID: {results['ids'][i]}")
        print(f" - Metadata: {results['metadatas'][i]}")
        print(f" - Nội dung: {results['documents'][i][:200]}...") # Chỉ in 200 ký tự đầu tiên
        print("-" * 50)
else:
    print("DB hiện đang rỗng!")
