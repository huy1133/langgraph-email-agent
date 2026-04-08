# 🚀 Lộ trình Phát triển Hệ thống AI Email Automation (Development Roadmap)

Để xây dựng một hệ thống hoạt động ổn định, thông minh và có khả năng tự sửa lỗi, quy trình phát triển được chia làm 4 giai đoạn cốt lõi.

---

## 📂 Giai đoạn 1: Xử lý Dữ liệu & Xây dựng Vector DB (Nền tảng RAG)
Đây là bước quan trọng nhất quyết định "trí nhớ" và độ chính xác của hệ thống.

1.  **Thu thập & Làm sạch (Data Cleaning):**
    *   Trích xuất lịch sử email từ CRM/Outlook/Gmail.
    *   Loại bỏ HTML tags, chữ ký (signatures), các thông tin rác hoặc email tự động.
2.  **Phân mảnh (Chunking):**
    *   Chia email thành các đoạn nhỏ có ý nghĩa.
    *   Chiến lược: Chia theo mốc thời gian hoặc theo từng chủ đề thảo luận (thread).
3.  **Embedding & Lưu trữ:**
    *   Sử dụng mô hình Embedding (ví dụ: `text-embedding-3-small` của OpenAI).
    *   Lưu trữ vào **Vector Database** (Pinecone, Milvus, Qdrant hoặc ChromaDB).
    *   **Metadata:** Thiết kế chuẩn các trường dữ liệu như `client_id`, `date`, `sender_role` để truy xuất chính xác.

---

## 🧠 Giai đoạn 2: Thiết kế LangGraph Agent (Bộ não hệ thống)
Sử dụng kiến trúc State Graph để thay thế cho các prompt dài đơn lẻ, giúp kiểm soát luồng suy nghĩ của AI.

### 1. Định nghĩa Trạng thái (State)
Khởi tạo một `TypedDict` để lưu trữ và truyền dữ liệu qua các Node:
*   `User_Inputs`: Tên, ngành nghề, mục tiêu của email.
*   `Email_History`: Dữ liệu ngữ cảnh lấy từ Vector DB.
*   `Current_Draft`: Bản thảo hiện tại.
*   `Feedback`: Các lỗi hoặc yêu cầu chỉnh sửa từ Node kiểm duyệt.

### 2. Xây dựng các Nodes (Nút xử lý)
*   **Retrieve_Node:** Query vào Vector DB để rút trích top 3-5 email liên quan nhất của khách hàng mục tiêu.
*   **Draft_Node:** Kết hợp dữ liệu từ Form UI và Context lịch sử để viết bản nháp đầu tiên.
*   **Evaluate_Node (The Critic):** Sử dụng một LLM khác hoặc Code Logic để kiểm tra:
    *   Độ dài (100-150 từ)?
    *   Có Call-to-Action (CTA) chưa?
    *   Có bị dính từ ngữ sáo rỗng (clichés) không?
*   **Rewrite_Node:** Nếu `Evaluate_Node` trả về lỗi, node này sẽ viết lại dựa trên feedback cụ thể.

### 3. Định tuyến (Conditional Edges)
*   **Nếu đạt chuẩn:** Xuất kết quả cuối cùng ra UI.
*   **Nếu chưa đạt:** Quay lại `Rewrite_Node` để sửa đổi.

---

## 💻 Giai đoạn 3: Phát triển Backend API & Tích hợp UI

1.  **Backend API:**
    *   Bọc luồng LangGraph vào một API Endpoint (Sử dụng FastAPI hoặc Flask).
    *   Nhận dữ liệu JSON từ Frontend và trả về kết quả từ Agent.
2.  **Frontend UI:**
    *   Thiết kế Form nhập liệu (Tên, Công ty, Ngành, Mục tiêu).
    *   **Human-in-the-loop:** Giao diện cho phép người dùng đọc bản nháp AI, chỉnh sửa thủ công trước khi bấm "Gửi".
3.  **Streaming (Cải thiện UX):**
    *   Implement tính năng streaming trạng thái để người dùng theo dõi tiến độ: *"Đang tìm lịch sử..."* → *"Đang viết nháp..."* → *"Đang duyệt lại..."*

---

## 🧪 Giai đoạn 4: Đánh giá & Tối ưu hóa (Testing)

1.  **Kiểm tra chất lượng (Prompt Engineering):**
    *   Chuẩn bị bộ 50 test cases thực tế.
    *   Tinh chỉnh Prompt cho `Evaluate_Node` để đảm bảo tiêu chuẩn khắt khe nhất.
2.  **Kiểm tra hiệu suất RAG:**
    *   Đảm bảo cơ chế lọc Metadata hoạt động tốt.
    *   **Tuyệt đối tránh nhầm lẫn:** Đảm bảo Agent không lấy nhầm lịch sử của khách hàng A để viết email cho khách hàng B.
3.  **Tối ưu chi phí:**
    *   Sử dụng các model rẻ hơn (như GPT-4o-mini) cho các node kiểm duyệt đơn giản để tiết kiệm tài nguyên.

---
