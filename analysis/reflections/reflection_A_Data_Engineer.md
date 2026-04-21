# Báo cáo Cá nhân - Thành viên 1: Data Engineer

- **Họ và tên:** Lương Thanh Hậu
- **MSSV:** 2A202600115
- **Vai trò:** Data Engineer (Retrieval & Synthetic Data Generation)

---

## 🛠️ Đóng góp kỹ thuật (Engineering Contribution)
*Mô tả các file bạn đã viết, các tính năng bạn đã cài đặt trong `data/synthetic_gen.py`.*
- **File chính đã triển khai/chỉnh sửa:** `data/synthetic_gen.py`, `data/golden_set.jsonl`, `.env.example`, `team_tasks/1_Data_Engineer.md`.
- **Xây dựng pipeline sinh dữ liệu tổng hợp bằng LLM:**
  - Kết nối OpenAI qua biến môi trường `OPENAI_API_KEY` và kiểm tra key đầu vào để tránh lỗi ngầm.
  - Thiết kế cơ chế sinh dữ liệu theo batch (mặc định 10 mẫu/lần) thay cho sinh một lần lớn để giảm rủi ro vượt giới hạn token.
- **Gia cố chất lượng và tính đúng chuẩn của dữ liệu đầu ra:**
  - Validate từng JSON line theo schema bắt buộc: `question`, `expected_answer`, `context`, `expected_ids`.
  - Bổ sung trường `is_red_teaming` để theo dõi tỉ lệ mẫu nhiễu/đối kháng.
  - Thêm retry khi model trả JSON lỗi hoặc không đạt số lượng/chất lượng batch kỳ vọng.
- **Đảm bảo tính nhất quán benchmark khi chạy lặp:**
  - Đổi cơ chế ghi file từ append sang overwrite để tránh chồng dữ liệu cũ mới.
  - Mỗi lần chạy tạo lại dataset sạch, ổn định cho các bước đánh giá retrieval và judge.
- **Kết quả định lượng sau khi chạy script:**
  - Tạo thành công `data/golden_set.jsonl` với `rows=50`, `valid=True`, `red=7`.
  - Đối chiếu checklist Data Engineer: hoàn thành đủ các yêu cầu bắt buộc trong `team_tasks/1_Data_Engineer.md`.

---

## 🧠 Chiều sâu chuyên môn (Technical Depth)
*Giải thích ngắn gọn cách bạn thiết kế Prompt để sinh dữ liệu đa dạng và cách bạn triển khai bộ dữ liệu "Red Teaming".*
- **Thiết kế prompt sinh dữ liệu đa dạng:**
  - Prompt được ràng buộc output ở dạng JSON có schema cố định để giảm sai lệch định dạng.
  - Dữ liệu được sinh theo nhiều đợt batch nhỏ để tăng khả năng kiểm soát chất lượng từng cụm câu hỏi.
  - Dùng vòng lặp sinh - kiểm tra - bù mẫu thiếu để đảm bảo đủ số lượng mục tiêu mà vẫn giữ chuẩn dữ liệu.
- **Triển khai Red Teaming trong golden set:**
  - Chủ động gán nhãn `is_red_teaming` để tách rõ nhóm câu hỏi gây nhiễu/đối kháng.
  - Đảm bảo số mẫu red team đạt ngưỡng tối thiểu theo checklist (>=5), thực tế đạt 7 mẫu.
  - Mục tiêu của tập red team là kiểm tra độ bền retrieval/judge trước các truy vấn dễ gây nhầm lẫn hoặc ngoài ngữ cảnh.

---

## 🚀 Giải quyết vấn đề (Problem Solving)
*Mô tả một vấn đề bạn gặp phải (ví dụ: lỗi JSON, Rate Limit API) và cách bạn xử lý nó.*
- **Vấn đề chính gặp phải:**
  - Khi yêu cầu model sinh số lượng lớn trong một lần gọi, kết quả dễ bị lỗi JSON hoặc bị cắt do giới hạn token.
  - Dữ liệu benchmark bị trùng khi script dùng append mode và được chạy nhiều lần.
- **Cách xử lý:**
  - Chuyển sang chiến lược sinh theo batch nhỏ kết hợp parse/validate chặt cho từng mẫu.
  - Thêm cơ chế retry có kiểm soát khi output không hợp lệ.
  - Đổi sang overwrite file đầu ra để loại bỏ hoàn toàn hiện tượng chồng lặp dữ liệu giữa các lần chạy.
- **Tác động sau xử lý:**
  - Script chạy ổn định hơn, giảm lỗi đầu ra.
  - Dataset tạo mới nhất quán, đáng tin cậy cho toàn bộ pipeline đánh giá.

---

## ✍️ Tự đánh giá (Self-Reflection)
*Bạn học được gì qua bài lab này? Bạn tâm đắc nhất điều gì ở hệ thống mà bạn xây dựng?*
- Qua bài lab này, tôi học được cách biến một bài toán "sinh dữ liệu bằng LLM" thành một pipeline kỹ thuật có kiểm soát chất lượng: có ràng buộc schema, validate tự động, retry, và tiêu chí định lượng rõ ràng.
- Điều tôi tâm đắc nhất là cách thiết kế tập dữ liệu phục vụ trực tiếp cho benchmarking: không chỉ đủ số lượng, mà còn có thành phần Red Teaming để đo độ bền của hệ thống trong tình huống khó.
- Tôi cũng rút ra rằng độ ổn định của benchmark phụ thuộc mạnh vào tính nhất quán dữ liệu đầu vào; vì vậy các quyết định như sinh theo batch và overwrite file là rất quan trọng.
- Nếu làm lại, tôi sẽ ưu tiên bổ sung bước khử trùng câu hỏi và QA report tự động để nâng chất lượng dữ liệu trước khi bàn giao cho pipeline đánh giá.
