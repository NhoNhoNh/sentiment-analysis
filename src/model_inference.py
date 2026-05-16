import torch
import numpy as np
import nltk
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification

# Tải bộ tách câu của NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ABSAPipeline:
    def __init__(self, ext_model_path, cls_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(ext_model_path)
        self.extractor = AutoModelForTokenClassification.from_pretrained(ext_model_path).to(self.device)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(cls_model_path).to(self.device)
        
        self.sentiment_labels = ["Negative", "Neutral", "Positive"]

    def predict(self, document_text):
        if not document_text.strip():
            return []

        self.extractor.eval()
        self.classifier.eval()
        
        sentences = nltk.sent_tokenize(document_text)
        all_results = []

        with torch.inference_mode():
            for sentence in sentences:
                if len(sentence.strip()) < 2: continue
                
                inputs = self.tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=128)
                offsets = inputs.pop("offset_mapping")[0].tolist()
                
                outputs = self.extractor(**inputs.to(self.device))
                predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
                
                aspect_spans = []
                current_start = -1
                current_end = -1
                
                for idx, pred in enumerate(predictions):
                    # Bỏ qua các token đặc biệt (như [CLS], [SEP]) có offset là (0,0)
                    if offsets[idx] == (0, 0):
                        continue
                        
                    if pred == 1:  # B-ASP (Bắt đầu 1 khía cạnh mới)
                        if current_start != -1:
                            aspect_spans.append(sentence[current_start:current_end])
                        current_start = offsets[idx][0]
                        current_end = offsets[idx][1]
                    elif pred == 2:  # I-ASP (Phần tiếp theo của khía cạnh)
                        if current_start != -1:
                            current_end = offsets[idx][1]
                    elif pred == 0:  # O (Từ bình thường)
                        if current_start != -1:
                            aspect_spans.append(sentence[current_start:current_end])
                            current_start = -1
                            current_end = -1

                # Chốt lại từ khóa cuối cùng nếu câu kết thúc ngay bằng Aspect
                if current_start != -1:
                    aspect_spans.append(sentence[current_start:current_end])

                # Bộ lọc an toàn: Loại bỏ các chuỗi rỗng hoặc quá ngắn
                aspect_spans = list(set([span.strip() for span in aspect_spans if len(span.strip()) > 1]))

                # 2. Đưa qua Sentiment Classifier
                for aspect in aspect_spans:
                    inputs_cls = self.tokenizer(sentence, aspect, return_tensors="pt", truncation=True, max_length=128).to(self.device)
                    outputs_cls = self.classifier(**inputs_cls)
                    pred_cls = torch.argmax(outputs_cls.logits, dim=1).item()
                    
                    all_results.append([sentence, aspect, self.sentiment_labels[pred_cls]]) # type: ignore

        return all_results