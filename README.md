# Báo Cáo Kỹ Thuật: Hệ Thống Phân Tích Cảm Xúc Theo Khía Cạnh (Aspect-Based Sentiment Analysis - ABSA)

---

## 📑 Mục Lục
1. [Giới Thiệu Tổng Quan](#1-giới-thiệu-tổng-quan)
2. [Video Demo Hệ Thống](#2-video-demo-hệ-thống)
3. [Nguồn Dữ Liệu & Thống Kê](#3-nguồn-dữ-liệu--thống-kê)
4. [Kiến Trúc Hệ Thống (Two-Stage Pipeline)](#4-kiến-trúc-hệ-thống-two-stage-pipeline)
5. [Quá Trình Huấn Luyện & Cấu Hình Tham Số](#5-quá-trình-huấn-luyện--cấu-hình-tham-số)
6. [Kết Quả Đánh Giá & Chỉ Số Độ Đo (Evaluation Metrics)](#6-kết-quả-đánh-giá--chỉ-số-độ-đo-evaluation-metrics)
7. [Cấu Trúc Thư Mục Dự Án](#7-cấu-trúc-thư-mục-dự-án)
8. [Hướng Dẫn Cài Đặt & Sử Dụng](#8-hướng-dẫn-cài-đặt--sử-dụng)

---

## 1. Giới Thiệu Tổng Quan

**Aspect-Based Sentiment Analysis (ABSA)** là một bài toán xử lý ngôn ngữ tự nhiên (NLP) nâng cao, vượt trội hơn so với phân tích cảm xúc truyền thống (chỉ phân loại toàn bộ văn bản thành Tích cực/Tiêu cực). ABSA đi sâu vào việc nhận diện chính xác **từng khía cạnh/đối tượng (Aspect)** xuất hiện trong câu và phân tích **cảm xúc cụ thể (Sentiment)** dành cho từng khía cạnh đó.

**Ví dụ thực tế:** 
> *"The screen is amazing but battery life is terrible."*
- Hệ thống truyền thống: Có thể phân loại nhầm thành Trung tính (Neutral).
- Hệ thống ABSA:
  - Khía cạnh 1: `screen` $\rightarrow$ Cảm xúc: `Positive`
  - Khía cạnh 2: `battery life` $\rightarrow$ Cảm xúc: `Negative`

Dự án này cung cấp một giải pháp toàn diện từ khâu tiền xử lý dữ liệu, huấn luyện mô hình học sâu với kiến trúc BERT, cho đến việc đóng gói thành một Pipeline hoàn chỉnh tích hợp giao diện người dùng trực quan trên nền tảng Gradio.

---

## 2. Video Demo Hệ Thống

Bấm vào hình ảnh dưới đây để xem video demo chi tiết về cách hệ thống hoạt động thực tế trên giao diện web Gradio:

[![Demo ABSA System](https://img.youtube.com/vi/4iauET7w4Fo/0.jpg)](https://www.youtube.com/watch?v=4iauET7w4Fo)

---

## 3. Nguồn Dữ Liệu & Thống Kê

Dữ liệu huấn luyện và kiểm thử được trích xuất từ tập dữ liệu chuẩn mực trong nghiên cứu NLP: **SemEval-2014 Task 4 (Subtask 1 - Restaurant Reviews)**.

### 📊 Đặc điểm & Phân bổ dữ liệu
- **Lĩnh vực (Domain):** Đánh giá dịch vụ nhà hàng (Ẩm thực, Phục vụ, Không gian, Giá cả,...).
- **Định dạng gốc:** XML (`train.xml`, `val.xml`, `test.xml`) chứa các thẻ cấu trúc ghi nhận chính xác vị trí bắt đầu/kết thúc (span offsets) của từ khóa khía cạnh và cực cảm xúc.
- **Tiền xử lý & Tích hợp:** Dữ liệu được tải và đồng bộ hóa thông qua thư viện `datasets` của Hugging Face (`tomaarsen/setfit-absa-semeval-restaurants`).

### 📈 Thống kê chi tiết các tập dữ liệu:
1. **Tập huấn luyện Aspect Extractor:**
   - Số lượng câu Train: **2,019 câu**
   - Số lượng câu Test: **606 câu**
2. **Tập huấn luyện Sentiment Classifier:**
   - Tổng số lượng mẫu Train gốc: **3,693 mẫu** (được chia thành Train: **3,139 mẫu** / Validation: **554 mẫu** theo tỷ lệ 85:15).

---

## 4. Kiến Trúc Hệ Thống (Two-Stage Pipeline)

Hệ thống được thiết kế theo quy trình 2 giai đoạn nối tiếp (Two-stage Pipeline), tận dụng sức mạnh biểu diễn ngôn ngữ vượt trội của mô hình **BERT (`bert-base-uncased`)**:

```
[Văn bản đầu vào] 
       │
       ▼ (NLTK Sentence Tokenizer)
[Các câu đơn lẻ]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1: ASPECT EXTRACTOR (Token Classification)   │
│ Nhận diện các thực thể/khía cạnh theo chuẩn BIO (B/I/O)│
└──────────────────────────┬─────────────────────────────┘
                           │ Danh sách các Aspects (ví dụ: "food", "service")
                           ▼
┌────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 2: SENTIMENT CLASSIFIER (Sequence Class.)    │
│ Đánh giá cặp (Câu, Khía Cạnh) -> Positive/Negative/Neut│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
[Bảng Kết Quả Tổng Hợp: Câu | Khía Cạnh | Cảm Xúc]
```

### 🔍 Giai đoạn 1: Aspect Extractor (Trích xuất Khía cạnh)
- **Kiến trúc:** `BertForTokenClassification` (Mô hình gán nhãn chuỗi cấp độ từ).
- **Quy tắc dán nhãn (BIO Scheme):**
  - `B-ASP` (Begin): Token mở đầu của một khía cạnh.
  - `I-ASP` (Inside): Token tiếp theo thuộc cùng một khía cạnh.
  - `O` (Outside): Các từ ngữ thông thường không phải khía cạnh.

### 🔍 Giai đoạn 2: Sentiment Classifier (Phân loại Cảm xúc)
- **Kiến trúc:** `BertForSequenceClassification` (Mô hình phân loại chuỗi).
- **Cơ chế hoạt động:** Nhận đầu vào là một chuỗi kết hợp giữa câu gốc và khía cạnh cụ thể `[CLS] Câu gốc [SEP] Khía cạnh [SEP]`. Mô hình học cách tập trung sự chú ý (attention) vào khía cạnh mục tiêu để đưa ra quyết định phân loại vào 3 lớp nhãn: `Positive`, `Negative`, `Neutral`.

---

## 5. Quá Trình Huấn Luyện & Cấu Hình Tham Số

Cả hai mô hình đều được tinh chỉnh (fine-tuning) trên GPU với bộ tham số tối ưu hóa nhằm đạt độ chính xác cao và tránh hiện tượng quá khớp (overfitting).

### ⚙️ Bảng Cấu Hình Siêu Tham Số (Hyperparameters)

| Tham số | Aspect Extractor | Sentiment Classifier | Giải thích / Ghi chú |
| :--- | :---: | :---: | :--- |
| **Pre-trained Model** | `bert-base-uncased` | `bert-base-uncased` | Mô hình ngôn ngữ cơ sở 110M tham số |
| **Số Epochs** | 5 | 5 | Số chu kỳ duyệt qua toàn bộ dữ liệu |
| **Learning Rate** | $2 \times 10^{-5}$ | $2 \times 10^{-5}$ | Tốc độ học an toàn cho fine-tuning BERT |
| **Batch Size (Train/Eval)**| 8 / 8 | 16 / 16 | Kích thước lô huấn luyện |
| **Weight Decay** | 0.01 | 0.01 | Chuẩn hóa L2 giúp giảm overfitting |
| **Warmup Ratio** | 0.1 | 0.1 | Tăng dần learning rate ở 10% số bước đầu |
| **Save/Eval Strategy** | Epoch | Epoch | Đánh giá và lưu mô hình sau mỗi epoch |
| **Best Model Metric** | `F1-Score` | `Accuracy` | Tiêu chí chọn mô hình tốt nhất |

---

## 6. Kết Quả Đánh Giá & Chỉ Số Độ Đo (Evaluation Metrics)

Sau quá trình huấn luyện, các mô hình được đánh giá nghiêm ngặt trên tập kiểm thử/kiểm chứng độc lập (Test/Validation set). Dưới đây là bảng tổng hợp các chỉ số độ đo chi tiết qua từng Epoch và kết quả của mô hình tốt nhất.

### 🏆 6.1. Kết Quả Mô Hình Aspect Extractor (Đánh giá trên tập Test 606 câu)

Mô hình đạt hiệu năng cao nhất tại **Epoch 5** với F1-Score vượt trội **87.75%**:

| Epoch | Training Loss | Validation Loss | Precision | Recall | **F1-Score** | Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.1263 | 0.1072 | 79.31% | 86.60% | 82.80% | 96.18% |
| 2 | 0.0852 | 0.0988 | 83.05% | 88.35% | 85.62% | 96.90% |
| 3 | 0.0358 | 0.1135 | 85.82% | 89.58% | 87.66% | 97.17% |
| 4 | 0.0174 | 0.1276 | 84.93% | 89.84% | 87.32% | 97.09% |
| **5 (Best)** | **0.0159** | **0.1387** | **85.91%** | **89.67%** | **87.75%** | **97.09%** |

- **Nhận xét:** Chỉ số Recall đạt xấp xỉ 90%, cho thấy mô hình có khả năng phát hiện hầu như toàn bộ các khía cạnh xuất hiện trong văn bản mà ít bị bỏ sót.

### 🏆 6.2. Kết Quả Mô Hình Sentiment Classifier (Đánh giá trên tập Validation 554 câu)

Mô hình đạt độ chính xác tổng thể cao nhất tại **Epoch 5** với Accuracy **79.06%**:

| Epoch | Training Loss | Validation Loss | **Accuracy** | Macro F1 |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.6723 | 0.6316 | 74.19% | 65.07% |
| 2 | 0.5292 | 0.5934 | 77.98% | 69.24% |
| 3 | 0.3469 | 0.6061 | 77.80% | 72.04% |
| 4 | 0.1867 | 0.6534 | 78.88% | **73.67%** |
| **5 (Best)** | **0.1413** | **0.6717** | **79.06%** | 73.40% |

- **Nhận xét:** Mô hình phân loại cảm xúc hoạt động ổn định và cân bằng giữa các lớp (Macro F1 duy trì trên 73%). Việc tách riêng biệt giai đoạn trích xuất và phân loại giúp giảm thiểu nhiễu và tối ưu hóa hiệu suất cho từng tác vụ chuyên biệt.

---

## 7. Cấu Trúc Thư Mục Dự Án

```text
├── src/
│   └── model_inference.py    # Lớp ABSAPipeline: Tích hợp bộ tách câu NLTK và xử lý dự đoán end-to-end
├── train/
│   ├── dataset/              # Tập dữ liệu gốc SemEval-2014 Task 4 (train.xml, val.xml, test.xml)
│   ├── saved_models/         # Thư mục lưu trữ 2 mô hình BERT đã huấn luyện hoàn chỉnh
│   │   ├── aspect_extractor/ # Trọng số mô hình Extractor (model.safetensors, config, tokenizer)
│   │   └── sentiment_classifier/ # Trọng số mô hình Classifier (model.safetensors, config, tokenizer)
│   └── train_model.ipynb     # Notebook chi tiết toàn bộ quy trình tải dữ liệu, huấn luyện và đánh giá
├── demo.py                   # Ứng dụng web trực quan xây dựng bằng Gradio
├── .gitignore                # Cấu hình bỏ qua môi trường ảo, file trọng số lớn và cache
└── README.md                 # Báo cáo kỹ thuật và tài liệu dự án
```

---

## 8. Hướng Dẫn Cài Đặt & Sử Dụng

### 🛠️ 8.1. Cài đặt Môi trường
Dự án yêu cầu Python 3.10 trở lên. Khuyến nghị sử dụng môi trường ảo (`venv` hoặc `conda`). Cài đặt các thư viện phụ thuộc bằng lệnh:

```bash
pip install torch numpy pandas transformers datasets evaluate seqeval gradio nltk
```

### 🧠 8.2. Huấn luyện lại Mô hình (Tùy chọn)
Nếu bạn muốn tự kiểm chứng hoặc huấn luyện lại mô hình trên tập dữ liệu mới, hãy mở và chạy lần lượt các cell trong file Jupyter Notebook:
`train/train_model.ipynb`
Hệ thống sẽ tự động tải dữ liệu, dán nhãn, khởi chạy `Trainer` của Hugging Face và xuất các mô hình tốt nhất vào thư mục `train/saved_models/`.

### 💻 8.3. Khởi chạy Giao diện Demo (Gradio Web UI)
Để chạy thử nghiệm trực tiếp với các câu đánh giá thực tế, thực thi lệnh sau tại thư mục gốc của dự án:

```bash
python demo.py
```

Hệ thống sẽ khởi tạo một máy chủ cục bộ và mở giao diện web tại địa chỉ mặc định `http://127.0.0.1:7860`. Tại đây, bạn có thể nhập bất kỳ đoạn văn bản đánh giá nhà hàng nào, hệ thống sẽ tự động tách câu, trích xuất các khía cạnh và hiển thị cực cảm xúc tương ứng trong bảng kết quả.
