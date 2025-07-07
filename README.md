# 🔍 Invoice Information Extraction with GCN

Dự án này sử dụng **Graph Convolutional Networks (GCN)** để trích xuất thông tin từ ảnh hóa đơn sau khi được OCR.

---


### 1. Cài đặt thư viện


```bash
pip install -r requirements.txt
```

---

### 2. Huấn luyện mô hình

#### Huấn luyện mặc định:
```bash
python main.py --mode train
```

#### Huấn luyện với số epoch tùy chọn:
```bash
python main.py --mode train --epochs 3000
```

---

### 3. Thực nghiệm dự đoán

Sau khi huấn luyện xong, có thể thực hiện dự đoán trên ảnh hóa đơn bằng ID ảnh:

```bash
python main.py --mode predict --input <ID_CỦA_ẢNH>
```

Ví dụ:
```bash
python main.py --mode predict --input 1
```

---

## 📁 Cấu trúc thư mục (gợi ý)

```
.
├── main.py
├── grapher.py
├── model/
│   └── gcn_model.pt
├── data/
│   ├── ocr_texts/
│   ├── graphs/
│   └── annotations/
├── requirements.txt
└── README.md
```

