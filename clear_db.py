import sqlite3
import os
import sys
from pathlib import Path

def clear_data():
    # Lấy thư mục gốc của dự án (nơi chứa file clear_db.py)
    project_root = Path(__file__).resolve().parent
    db_path = project_root / 'data' / 'customers.db'
    chroma_path = project_root / 'chroma_data'
    
    print("\n=== ĐANG TIẾN HÀNH DỌN DẸP TOÀN BỘ DỮ LIỆU ===")
    
    # Dọn dẹp SQLite
    if db_path.exists():
        try:
            con = sqlite3.connect(str(db_path))
            c = con.cursor()
            
            # Xóa sạch dữ liệu trong bảng contacts
            c.execute('DELETE FROM contacts')
            
            # Reset lại chỉ số AUTOINCREMENT (ID về lại từ đầu)
            try:
                c.execute("UPDATE sqlite_sequence SET seq = 0 WHERE name = 'contacts'")
            except sqlite3.OperationalError:
                pass # Bỏ qua nếu bảng sqlite_sequence chưa được tạo
            
            con.commit()
            con.close()
            print(f"[OK] SQLite: Đã xóa sạch toàn bộ danh bạ khách hàng. Bộ đếm ID đã reset.")
        except Exception as e:
            print(f"[LỖI] Xóa SQLite thất bại: {e}")
    else:
        print("[INFO] SQLite: Không có file dữ liệu (trống).")
        
    # Dọn dẹp ChromaDB
    if chroma_path.exists():
        try:
            import chromadb
            import logging
            # Giảm log của chromadb khi chạy
            logging.getLogger("chromadb").setLevel(logging.ERROR)
            
            client = chromadb.PersistentClient(path=str(chroma_path))
            try:
                client.delete_collection('email_chunks')
                print(f"[OK] ChromaDB: Đã xóa toàn bộ kho Vector cũ.")
            except Exception:
                print("[OK] ChromaDB: Hiện đang trống, không cần xóa.")
        except ImportError:
            print("[LỖI] Chưa cài đặt thư viện 'chromadb'.")
        except Exception as e:
            print(f"[LỖI] Xóa ChromaDB thất bại: {e}")
    else:
        print("[INFO] ChromaDB: Không có thư mục Vector dữ liệu (trống).")
        
    print("=== DỌN DẸP HOÀN TẤT. SẴN SÀNG CHO PHIÊN TEST MỚI ===\n")

if __name__ == "__main__":
    clear_data()
