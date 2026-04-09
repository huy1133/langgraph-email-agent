# Hướng Dẫn Chuẩn Bị & Quản Lý Dữ Liệu Email RAG

Tài liệu này mô tả toàn bộ kiến trúc xử lý từ dữ liệu thô (raw email) thành dữ liệu có cấu trúc tại cơ sở dữ liệu SQLite và Cơ Sở Dữ Liệu Vector (ChromaDB). Thiết kế hệ thống được tách rẽ (**Normalization**) để phục vụ xây dựng UI tốt nhất.

---

## 1. Định dạng Dữ Liệu Khởi Tạo (Raw Data)
Hệ thống không giới hạn định dạng chữ đầu vào. Người dùng có thể copy nguyên văn 1 email, một đoạn chat nội bộ hay ghi chú chuyển tiếp. 

Bạn chỉ cần gom chúng lại vào một file dạng `.txt` (ví dụ: `data/emails_10_companies_10_mails.txt`) và phân cách từng email/thực thể bằng dải ngăn cách:
`==========`

**Ví dụ:**
```txt
==========
Từ: Nguyen Van A <a@alphatech.vn>
Công ty Alpha Tech bảo là bên anh sửa lại hợp đồng đi nhé.
==========
Ghi chú ngày hôm nay: 
Khách B bên bệnh viên Chợ Rẫy (Y Tế) yêu cầu check lại bill thanh toán.
==========
```

---

## 2. Kiến Trúc Luồng Code (Data Flow)

Khi tiến trình bắt đầu đọc dòng văn bản trên, nó sẽ kích hoạt khối AI (LLM) để rẽ nhánh dữ liệu ra 2 mảng lưu trữ tách biệt theo Chuẩn hóa bậc 3 (3NF).

### Bước 1: Trích xuất thực thể Master (SQL)
LLM Model sẽ đọc từng email và móc ra 3 thông số bất phân ly: **Tên khách hàng, Tăng công ty, Tên ngành**.
Hệ thống kết nối đến file `data/customers.db` và **TỰ ĐỘNG GỘP TRÙNG LẶP** (INSERT OR IGNORE) rồi rót vào 3 bảng riêng biệt:
- Bảng `customers` (ID, name)
- Bảng `companies` (ID, name)
- Bảng `industries` (ID, name)

*Nhờ việc tách 3 bảng này, UI Dropdown Menu của ứng dụng đọc từng bảng sẽ vô cùng trong sạch, gọn gàng và không lặp nhau.*

### Bước 2: Lưu lõi Vector Khóa Ngoại (ChromaDB)
Toàn bộ phần lõi nguyên bản chữ (Body) của email sẽ được đem đi băm nhỏ (Chunk) và gắn Vector bởi `LLM_EMBEDDING_MODEL`. Khối Vector này lưu cục bộ tại `chroma_data`.

Đặc biệt, bên trong Vector thẻ Metadata sẽ được cấy 3 chiếc Khóa Ngoại (Foreign Keys) tương ứng:
```json
{
  "customer_id": 1,
  "company_id": 5,
  "industry_id": 3
}
```
**Lợi ích**: Khi UI muốn tìm kiếm dữ liệu RAG cho Ngành Y Tế (ID 3), Vector Query cực kỳ nhanh, mượt mà và giới hạn đích ngắm chuẩn xác 100%.

---

## 3. Cách Dùng Lệnh Chạy (How to Run)

### A. Thiết lập API Key
Trong file `.env`:
```env
LLM_API_KEY=YOUR_OPENAI_OR_PROXY_KEY
LLM_CHAT_MODEL=gpt-4o-mini
LLM_EMBEDDING_MODEL=text-embedding-3-small
# BASE_URL=https://... (Comment đi nếu xài API gốc OpenAI)
```

### B. Chạy tiến trình đưa vào Data (Ingest)
Truy nhập Terminal tại thư mục `langgraph-email-agent`, gõ lệnh:
```bash
python -m src.ingest_from_text data/emails_10_companies_10_mails.txt
```
*Ghi chú: Lệnh có cơ chế Auto-catch lỗi 503 proxy nên sẽ không bị đứt gãy nếu mạng gián đoạn.*

### C. Xem báo cáo kết quả Data đã bóc tách (Review)
Sau khi tạo xong, bạn có thể kiểm chứng kho Data Thực Thể (Entities) của mình có đẹp và chuẩn hóa hay không qua lệnh xem Report:
```bash
python read_sqlite.py
```
Màn hình CMD sẽ in danh sách liệt kê phân mục rõ ràng từ SQLite.
