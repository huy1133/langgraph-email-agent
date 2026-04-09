# Hướng Dẫn Sử Dụng Web Portal Kéo Thả Dữ Liệu RAG

Web Portal này được sinh ra để giúp bất kì ai trong team (không cần biết code hay Python Terminal) cũng có thể trực tiếp đưa dữ liệu thô vào hệ thống Máy trí tuệ nhân tạo (RAG/Vector DB).

---

## 🚀 1. Cách Gọi Giao Diện (Chỉ cần làm 1 lần)

Đảm bảo bạn đã tải đủ các thư viện trong `requirements.txt`.
Mở cửa sổ Terminal/Command Prompt ngay tại thư mục dự án này, gõ lệnh:

```bash
python -m src.web_app
```

*Hoặc nếu muốn hệ thống có chế độ Tự reload khi cập nhật Code:*
```bash
uvicorn src.web_app:app --reload
```

---

## 🖥 2. Cách Nhập Liệu Chuyên Nghiệp

Khi server chạy thành công, hãy mở Trình duyệt Chrome / Safari lên đường dẫn:
👉 **http://localhost:8000**

- Màn hình đen (Dark mode Glassmorphism) sẽ hiện lên. 
- Bạn chỉ việc Copy bất kỳ Text gì (Đoạn mã, Chat Log, Email Forward) rồi **Paste thẳng vào ô Nhập Văn bản bự ở giữa**. 
- Nếu bạn dán một lúc 10 email khác nhau, nhớ dùng thanh ngăn cách là 10 dấu bằng: `==========` đặt giữa mỗi email.
- Bấm **"Bắt đầu Trích xuất & Nhúng DB"** và xem phép thuật xảy ra ở khung Console mô phỏng bên dưới.

---

## 🛠 3. Cơ Chế Hoạt Động Ngầm (Cần Biết Cho Dev)

Ngay khi bạn bấm nút trên UI, một luồng (Requests) sẽ chạy ngầm truyền tải toàn bộ số chữ kia vào REST API (`POST /api/ingest`):
1. **Chia khối (Splitting)**: Đoạn Text lớn bị cắt dựa trên thẻ `==========`.
2. **AI Bóc Tách (LLM)**: File backend sẽ tự động giao tiếp với `LLM_CHAT_MODEL` được bọc bên file `.env` để trích ra 3 thẻ: Nhóm Khách Hàng, Tên Công Ty, Phân Loại Ngành.
3. **Cập Nhật Data Master**: Python tự động "Nhét" (Insert) 3 thẻ trên vào `data/customers.db`. Quá trình này được kích hoạt khoá `UNIQUE`, tức là không bao giờ bị đẻ trùng tên khi bạn lỡ tay ấn 2 lần. Phân rã theo Chuẩn 3 (3NF).
4. **Nhúng Semantic (ChromaDB)**: 3 ID duy nhất sẽ được gắn thành mã số trên lưng của Vector chữ. Bọc lại và nằm ngoan ngoãn bên trong Folder `./chroma_data`.
5. **Hiển thị tiến trình**: Ngay khi hoàn tất mỗi bước này, Web UI trên trình duyệt của bạn sẽ hiện Log xanh lè báo thành công.

---

## 📦 4. Phục hồi và Cài đặt Sang máy mới
Nếu đem dự án này qua máy mới, chỉ lôi Terminal ra cài:
```bash
pip install -r requirements.txt
```
Là UI chạy lại ầm ầm! Hưởng thụ thôi!
