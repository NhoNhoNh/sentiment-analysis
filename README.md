# Aspect-Based Sentiment Analysis (ABSA) - Phân Tích Cảm Xúc Theo Khía Cạnh

Dự án này triển khai một hệ thống Phân Tích Cảm Xúc Theo Khía Cạnh (ABSA) hoàn chỉnh, cho phép tự động tách câu, trích xuất các đối tượng/khía cạnh (Aspects) được nhắc đến trong văn bản và phân loại cảm xúc (Positive, Negative, Neutral) cho từng khía cạnh đó. Hệ thống cung cấp giao diện trực quan thân thiện thông qua Gradio.

---

## 📊 Nguồn Dữ Liệu (Data Source)

Dữ liệu huấn luyện và đánh giá được sử dụng trong dự án bắt nguồn từ tập dữ liệu chuẩn **SemEval-2014 Task 4 (Restaurant Reviews)**. 

- **Đặc điểm tập dữ liệu:** Tập dữ liệu chứa các đánh giá thực tế của khách hàng về các nhà hàng, trong đó mỗi câu đã được dán nhãn chi tiết về các khía cạnh (ví dụ: `food`, `service`, `price`, `ambience`) kèm theo cực cảm xúc tương ứng (Tích cực, Tiêu cực hoặc Trung tính).
- **Cấu trúc dữ liệu:** Dữ liệu được lưu trữ dưới định dạng XML (`train.xml`, `val.xml`, `test.xml`) trong thư mục `train/dataset/`, đồng thời được tích hợp và xử lý thông qua thư viện `datasets` của Hugging Face (`tomaarsen/setfit-absa-semeval-restaurants`).

---

## 🏗️ Kiến Trúc Hệ Thống (System Architecture)

Hệ thống ABSA được thiết kế theo quy trình 2 giai đoạn (Two-stage Pipeline) sử dụng các mô hình học sâu dựa trên kiến trúc BERT (`bert-base-uncased`):

1. **Giai đoạn 1: Trích xuất Khía cạnh (Aspect Extractor - Token Classification)**
   - **Nhiệm vụ:** Xác định các từ/cụm từ đóng vai trò là khía cạnh được đánh giá trong câu.
   - **Phương pháp:** Bài toán gán nhãn chuỗi (Sequence Labeling) theo chuẩn BIO (`B-ASP`, `I-ASP`, `O`).
   - **Mô hình đầu ra:** `train/saved_models/aspect_extractor`.

2. **Giai đoạn 2: Phân loại Cảm xúc (Sentiment Classifier - Sequence Classification)**
   - **Nhiệm vụ:** Xác định cảm xúc dành cho khía cạnh vừa được trích xuất.
   - **Phương pháp:** Mô hình nhận đầu vào là cặp câu (Sentence, Aspect) và dự đoán nhãn cảm xúc thuộc 3 lớp: `Positive`, `Negative`, `Neutral`.
   - **Mô hình đầu ra:** `train/saved_models/sentiment_classifier`.

---

## 📂 Cấu Trúc Thư Mục (Project Structure)

```text
├── src/
│   └── model_inference.py    # Chứa lớp ABSAPipeline xử lý toàn bộ quy trình dự đoán
├── train/
│   ├── dataset/              # Dữ liệu SemEval-2014 Task 4 (train, val, test dạng XML)
│   ├── saved_models/         # Thư mục chứa các mô hình đã được huấn luyện hoàn chỉnh
│   └── train_model.ipynb     # Notebook tiền xử lý dữ liệu, huấn luyện và đánh giá mô hình
├── demo.py                   # Ứng dụng giao diện web Gradio
├── .gitignore                # Cấu hình bỏ qua các file môi trường, cache và mô hình lớn
└── README.md                 # Tài liệu hướng dẫn dự án
```

---

## 🚀 Hướng Dẫn Sử Dụng (Getting Started)

### 1. Cài đặt Môi trường
Đảm bảo bạn đã cài đặt Python (khuyến nghị 3.10+) và cài đặt các thư viện cần thiết:

```bash
pip install torch numpy pandas transformers datasets evaluate seqeval gradio nltk
```

### 2. Huấn luyện Mô hình (Tùy chọn)
Nếu bạn muốn tự huấn luyện lại mô hình từ đầu, hãy mở và chạy các cell trong file notebook:
`train/train_model.ipynb`
Quá trình này sẽ tải dữ liệu, tiền xử lý, huấn luyện hai mô hình BERT và lưu kết quả vào thư mục `train/saved_models/`.

### 3. Khởi chạy Giao diện Web (Gradio Demo)
Để trải nghiệm trực tiếp mô hình qua giao diện web trực quan, hãy chạy lệnh sau tại thư mục gốc:

```bash
python demo.py
```
Hệ thống sẽ tự động khởi chạy một máy chủ cục bộ và mở giao diện trên trình duyệt của bạn (thông thường tại `http://127.0.0.1:7860`). Bạn có thể nhập các đoạn đánh giá nhà hàng bất kỳ để xem mô hình phân tích chi tiết.
