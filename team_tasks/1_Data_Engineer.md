# 🧑‍💻 Thành viên 1: Chuyên gia Dữ liệu (Synthetic Data Generation)

## 📌 Tổng quan mục tiêu
Tạo ra một **Golden Dataset** (Tập dữ liệu Vàng) gồm ít nhất 50 test cases. Tập dữ liệu này sẽ được dùng làm thước đo chuẩn để chấm điểm hệ thống RAG tiếp theo.

## 📁 File phần quyền xử lý chính
- `data/synthetic_gen.py`

## 🛠️ Trình tự các bước cần làm
1. **Khởi tạo kết nối AI:** Trong hàm `generate_qa_from_text`, hãy dùng một thư viện LLM (như OpenAI, Anthropic, Gemini, hoặc HuggingFace) để tự động sinh câu hỏi.
2. **Thiết kế Prompt sinh dữ liệu:** Gợi ý LLM đọc `raw_text` và sinh ra cấu trúc chứa:
   - `question`: Câu hỏi từ người dùng.
   - `expected_answer`: Câu trả lời lý tưởng (Ground Truth).
   - `context`: Đoạn văn bản chứa câu trả lời.
   - `expected_ids`: ID của tài liệu/chunk (Dùng cho đánh giá Hit Rate sau này).
3. **Thêm dữ liệu nhiễu/Red Teaming:** Đảm bảo prompt yêu cầu model tạo ra ít nhất một số câu hỏi khó, đánh đố, hoặc hỏi về thứ không có trong text để kiểm tra hallucination.
4. **Kiểm thử script:** Chạy thử `python data/synthetic_gen.py` xem file xuất ra có chuẩn định dạng cấu trúc JSONL không.

## ✅ Tiêu chí hoàn thành (Checklist)
- [x] Chạy lệnh `python data/synthetic_gen.py` không bị lỗi.
- [x] Hàm sinh ra thành công file `data/golden_set.jsonl`.
- [x] File `golden_set.jsonl` có đủ ít nhất **50 dòng** (test cases).
- [x] Mỗi JSON lines có đủ các key: `question`, `expected_answer`, `context`, `expected_ids` (hoặc cấu trúc tương đương).
- [x] Có ít nhất 5 câu hỏi mang tính chất "Red Teaming" (lừa/đánh đố).

### Ghi nhận kiểm thử nhanh
- Kết quả xác minh tự động sau khi chạy script: `rows=50 valid=True red=7`.

## 💡 Nơi nhận kết quả
Sau khi xong, các bạn ở nhóm khác (đặc biệt Thành viên 4 và 5) sẽ dùng trực tiếp file `data/golden_set.jsonl` mà bạn tạo ra để chạy Test Benchmark.
