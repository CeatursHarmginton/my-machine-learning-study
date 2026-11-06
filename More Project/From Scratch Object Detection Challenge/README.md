# 🎯 From Scratch: Object Detection Challenge (Kaggle Competition)

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/from-scratch-object-detection-challenge)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/ceatur/from-scratch-object-detection-challenge-dataset)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-From%20Scratch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Computer Vision](https://img.shields.io/badge/Task-Object%20Detection-blue)](https://pytorch.org/)

Dự án tự xây dựng toàn bộ pipeline phát hiện và định vị đối tượng đa lớp (**Object Detection from Scratch**) từ con số 0 trên nền tảng PyTorch, thuộc cuộc thi **[From Scratch: Object Detection Challenge trên Kaggle](https://www.kaggle.com/competitions/from-scratch-object-detection-challenge/)**.

Tập dữ liệu hoàn chỉnh đã được đóng gói và sẵn sàng tải về tại: **[Hugging Face Dataset - ceatur/from-scratch-object-detection-challenge-dataset](https://huggingface.co/datasets/ceatur/from-scratch-object-detection-challenge-dataset)**.

---

## 📌 Mục lục

- [Tổng quan dự án & Thử thách "From Scratch"](#-tổng-quan-dự-án--thử-thách-from-scratch)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Đặc điểm tập dữ liệu & 5 Lớp đối tượng](#-đặc-điểm-tập-dữ-liệu--5-lớp-đối-tượng)
- [Kiến trúc & Pipeline Tự Cài Đặt (From Scratch)](#-kiến-trúc--pipeline-tự-cài-đặt-from-scratch)
- [Thước đo đánh giá (Evaluation Metric: mAP@0.5)](#-thước-đo-đánh-giá-evaluation-metric-map05)
- [Công cụ hỗ trợ (Tools & Utilities)](#-công-cụ-hỗ-trợ-tools--utilities)
- [Hướng dẫn cài đặt & Thực thi](#-hướng-dẫn-cài-đặt--thực-thi)
- [Định dạng Submission](#-định-dạng-submission)
- [Chiến lược tối ưu nâng cao](#-chiến-lược-tối-ưu-nâng-cao)
- [Tài liệu tham khảo](#-tài-liệu-tham-khảo)

---

## 🎯 Tổng quan dự án & Thử thách "From Scratch"

Trong khi các thư viện thị giác máy tính hiện đại cung cấp sẵn các detector dựng sẵn (off-the-shelf) như YOLO (v5/v8), Faster R-CNN hay RetinaNet, cuộc thi **From Scratch: Object Detection Challenge** đặt ra bài toán làm chủ bản chất của thị giác máy tính bằng cách yêu cầu lập trình viên **tự thiết kế và hiện thực hóa toàn bộ luồng huấn luyện, hàm mất mát và thuật toán suy luận**:

### 🚫 Quy tắc & Ràng buộc cốt lõi:
1. **Không sử dụng Off-the-Shelf Detectors**: Nghiêm cấm sử dụng các framework phát hiện đối tượng dựng sẵn (như `ultralytics`, `torchvision.models.detection.*`, `detectron2`, `mmdetection`).
2. **Tự cài đặt toàn bộ Head & Loss**: Được phép sử dụng pre-trained backbone cơ bản (như ResNet, VGG, MobileNet) để trích xuất đặc trưng (feature extractor), nhưng toàn bộ Detection Head, Anchor/Grid generator, Target Matching, Loss Functions và thuật toán NMS phải được tự lập trình.
3. **5 Lớp đối tượng mục tiêu**: Nhận diện và định vị chính xác: `person` (người), `car` (xe hơi), `chair` (ghế), `dog` (chó), và `cat` (mèo).

---

## 📁 Cấu trúc thư mục

```text
More Project/
└── From Scratch Object Detection Challenge/
    ├── Data/
    │   ├── annotations/
    │   │   ├── train.json                    # Nhãn huấn luyện (images, annotations, bbox)
    │   │   ├── val.json                      # Nhãn validation
    │   │   ├── oracle_train_predictions.json # Oracle predictions tham khảo cho tập train
    │   │   └── oracle_val_predictions.json   # Oracle predictions tham khảo cho tập val
    │   ├── train/
    │   │   └── images/                       # Thư mục ảnh huấn luyện
    │   ├── val/
    │   │   └── images/                       # Thư mục ảnh validation
    │   ├── test/
    │   │   └── images/                       # Thư mục ảnh kiểm thử (12,474 ảnh test)
    │   ├── tools/
    │   │   ├── convert.py                    # Chuyển đổi JSON predictions sang CSV nộp bài
    │   │   └── evaluate_predictions.py       # Đánh giá mAP@0.5 trên tập validation
    │   └── classes.json                      # Danh sách 5 lớp đối tượng
    ├── Notebooks/
    │   └── solve.ipynb                       # Jupyter Notebook phát triển mô hình & inference
    ├── Submissions/                          # Chứa file kết quả nộp bài Kaggle (.csv)
    └── README.md                             # Tài liệu giới thiệu dự án
```

---

## 📊 Đặc điểm tập dữ liệu & 5 Lớp đối tượng

### 1. Định dạng toạ độ Bounding Box
Tọa độ bounding box trong `train.json` và `val.json` được biểu diễn theo chuẩn pixel thực tế:
$$\text{bbox} = [x_{\min}, y_{\min}, x_{\max}, y_{\max}]$$

- $x_{\min}, y_{\min}$: Toạ độ góc trên bên trái của vật thể ($0 \le x_{\min} < x_{\max} \le W$, $0 \le y_{\min} < y_{\max} \le H$).
- $x_{\max}, y_{\max}$: Toạ độ góc dưới bên phải của vật thể.

### 2. Danh sách 5 lớp & Phân bố dữ liệu

| STT | Lớp (Class) | Ý nghĩa | Số lượng nhãn Train | Số lượng nhãn Val | Tỷ lệ |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | `person` | Người | 5,829 | 1,074 | ~54.8% |
| 2 | `chair` | Ghế | 1,613 | 282 | ~15.1% |
| 3 | `car` | Xe hơi | 1,339 | 283 | ~12.6% |
| 4 | `dog` | Chó | 1,028 | 206 | ~9.7% |
| 5 | `cat` | Mèo | 833 | 176 | ~7.8% |
| **Tổng** | **5 classes** | | **10,642 boxes** | **2,021 boxes** | **100%** |

| Thuộc tính | Chi tiết |
| :--- | :--- |
| **Số lượng ảnh Train** | 5,064 ảnh (7,500 image entries trong train.json) |
| **Số lượng ảnh Validation** | 1,271 ảnh (1,500 image entries trong val.json) |
| **Số lượng ảnh Test** | 12,474 ảnh test |
| **Kho lưu trữ Dataset** | [Hugging Face: ceatur/from-scratch-object-detection-challenge-dataset](https://huggingface.co/datasets/ceatur/from-scratch-object-detection-challenge-dataset) |

---

## 🔬 Kiến trúc & Pipeline Tự Cài Đặt (From Scratch)

Để giải quyết bài toán phát hiện đối tượng mà không sử dụng detector đóng gói sẵn, pipeline bao gồm 4 thành phần tự xây dựng:

```text
┌─────────────────┐     ┌─────────────────────┐     ┌────────────────────────┐
│  Input Image    │ ──► │  Backbone (ResNet)  │ ──► │  Feature Maps (C3,C4)  │
└─────────────────┘     └─────────────────────┘     └───────────┬────────────┘
                                                                │
                                                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    Custom Detection Head (Anchor/Grid)                     │
│  - Bounding Box Regressor: (dx, dy, dw, dh) hoặc (xmin, ymin, xmax, ymax)  │
│  - Objectness Score: p(Object)                                             │
│  - Class Probabilities: (person, car, dog, cat, chair)                     │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
   ┌───────────────────────────┐             ┌───────────────────────────┐
   │       TRAINING MODE       │             │      INFERENCE MODE       │
   │  - IoU Target Matching    │             │  - Box Decoding           │
   │  - Localization Loss      │             │  - Confidence Filter      │
   │  - Objectness / Class Loss│             │  - Custom Class-wise NMS  │
   └───────────────────────────┘             └───────────────────────────┘
```

### 1. Custom Dataset & Data Augmentation
- Tự chuẩn hóa kích thước ảnh về kích thước cố định (ví dụ $448 \times 448$ hoặc $512 \times 512$).
- Đồng bộ biến đổi tọa độ Bounding Box khi ảnh bị Resize, Crop hoặc Flip:
  - **Random Horizontal Flip**: Đổi toạ độ $x'_{\min} = W - x_{\max}$, $x'_{\max} = W - x_{\min}$.
  - **Color Jitter & Normalization**: Thay đổi độ sáng, tương phản và chuẩn hóa ImageNet (`mean`, `std`).

### 2. Multi-Task Custom Loss Function
Tổng loss là sự kết hợp có trọng số của 3 thành phần:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{loc}} + \lambda_{\text{obj}} \mathcal{L}_{\text{obj}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}$$

- **$\mathcal{L}_{\text{loc}}$ (Localization Loss)**: Smooth L1 Loss hoặc GIoU / CIoU Loss tính trên các anchor/cell chứa ground truth.
- **$\mathcal{L}_{\text{obj}}$ (Objectness Loss)**: Binary Cross-Entropy (BCE) hoặc Focal Loss nhằm phân biệt cell chứa vật thể và nền (background).
- **$\mathcal{L}_{\text{cls}}$ (Classification Loss)**: Cross-Entropy Loss hoặc Focal Loss giữa 5 lớp đối tượng.

### 3. Tự cài đặt Non-Maximum Suppression (NMS)
Sau khi mạng dự đoán hàng nghìn box ứng viên, thuật toán NMS thủ công sẽ:
1. Lọc bỏ các box có `confidence < threshold` (ví dụ: $< 0.1$).
2. Sắp xếp các box còn lại theo độ tin cậy giảm dần.
3. Với mỗi box có confidence cao nhất, loại bỏ tất cả các box cùng lớp có $\text{IoU} > \text{nms\_threshold}$ (ví dụ: $> 0.5$).

---

## 📐 Thước đo đánh giá (Evaluation Metric: mAP@0.5)

Chất lượng mô hình được đánh giá thông qua chỉ số **mean Average Precision tại ngưỡng IoU 0.50 (mAP@0.5)**:

### 1. Intersection over Union (IoU)
$$\text{IoU}(B_p, B_{gt}) = \frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p \cup B_{gt})}$$

- Dự đoán được coi là **True Positive (TP)** nếu $\text{IoU}(B_p, B_{gt}) \ge 0.5$ và chưa có dự đoán nào match với $B_{gt}$ trước đó.
- Nếu không đạt ngưỡng hoặc ground truth đã match, dự đoán là **False Positive (FP)**.
- Ground truth không có dự đoán nào match là **False Negative (FN)**.

### 2. Precision, Recall & mAP
Đường cong Precision-Recall được nội suy tại 101 điểm hoặc theo diện tích dưới đường cong:
$$\text{AP}_c = \sum_{k} (R_k - R_{k-1}) \cdot P_{\text{interp}}(R_k)$$
$$\text{mAP@0.5} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c \quad (C = 5)$$

---

## 🛠️ Công cụ hỗ trợ (Tools & Utilities)

Dự án đi kèm các script sẵn trong thư mục `Data/tools/`:

### 1. Đánh giá cục bộ với `evaluate_predictions.py`
Kiểm tra độ chính xác mAP@0.5 của file dự đoán JSON trên tập validation hoặc oracle:
```bash
python Data/tools/evaluate_predictions.py \
    --ground_truth Data/annotations/val.json \
    --predictions Data/annotations/oracle_val_predictions.json \
    --iou_threshold 0.5
```

### 2. Chuyển đổi JSON sang CSV nộp bài với `convert.py`
Chuyển đổi file dự đoán JSON của tập test thành định dạng CSV chuẩn Kaggle:
```bash
python Data/tools/convert.py \
    --input predictions_test.json \
    --output Submissions/submission.csv \
    --classes Data/classes.json
```

---

## 🚀 Hướng dẫn cài đặt & Thực thi

### 1. Cài đặt môi trường

```bash
# Clone repository
git clone https://github.com/CeatursHarmginton/my-machine-learning-study.git
cd "my-machine-learning-study/More Project/From Scratch Object Detection Challenge"

# Cài đặt các thư viện phụ thuộc
pip install torch torchvision numpy pandas matplotlib pillow opencv-python tqdm
```

### 2. Tải & Chuẩn bị dữ liệu từ Hugging Face

Toàn bộ dataset đã được lưu trữ và tối ưu trên [Hugging Face](https://huggingface.co/datasets/ceatur/from-scratch-object-detection-challenge-dataset):

```bash
# Tải tệp dữ liệu nén (.zip)
wget -q -O data.zip "https://huggingface.co/datasets/ceatur/from-scratch-object-detection-challenge-dataset/resolve/main/from-scratch-object-detection-challenge-dataset.zip?download=true"

# Giải nén vào thư mục dự án
unzip -q data.zip -d ./
```

### 3. Huấn luyện & Chạy Notebook
Khởi chạy Jupyter Notebook:
```bash
jupyter notebook Notebooks/solve.ipynb
```

---

## 📄 Định dạng Submission

File nộp bài `submission.csv` bao gồm 2 cột `image_id` và `bounding_boxes` (chuỗi JSON chứa danh sách các bounding boxes dự đoán):

```csv
image_id,bounding_boxes
img_0003a558b10b.jpg,"[{""x_min"": 112.5, ""y_min"": 45.0, ""x_max"": 320.0, ""y_max"": 410.0, ""class"": ""person"", ""confidence"": 0.89}]"
img_0006e1b46bdd.jpg,"[{""x_min"": 50.0, ""y_min"": 120.0, ""x_max"": 480.0, ""y_max"": 350.0, ""class"": ""car"", ""confidence"": 0.94}, {""x_min"": 10.0, ""y_min"": 200.0, ""x_max"": 90.0, ""y_max"": 310.0, ""class"": ""chair"", ""confidence"": 0.72}]"
...
```

---

## 💡 Chiến lược tối ưu nâng cao

1. **Grid Resolution / Multi-Scale Feature Maps (FPN)**: Kết hợp feature map từ nhiều tầng (ví dụ: $14 \times 14$, $28 \times 28$) để vừa bắt được vật thể lớn (`car`, `person`) vừa nhận diện tốt vật thể nhỏ (`cat`, `chair`).
2. **K-Means Anchor Clustering**: Nếu sử dụng phương pháp Anchor-based, chạy K-Means trên tập nhãn Train để tìm tỷ lệ aspect ratio và kích thước anchor tối ưu cho 5 lớp.
3. **Hard Negative Mining / Focal Loss**: Tỷ lệ cell chứa background vượt trội so với foreground; sử dụng Focal Loss ($\gamma = 2.0$) giúp cân bằng gradient và tránh mô hình bị áp đảo bởi background.
4. **Data Augmentation đa dạng**: Random Scaling, Color Jitter, Mosaic / CutMix giúp tăng tính bất biến với tỷ lệ và ánh sáng.

---

## 📚 Tài liệu tham khảo

- [Kaggle From Scratch: Object Detection Challenge](https://www.kaggle.com/competitions/from-scratch-object-detection-challenge/)
- [Hugging Face Dataset - ceatur/from-scratch-object-detection-challenge-dataset](https://huggingface.co/datasets/ceatur/from-scratch-object-detection-challenge-dataset)
- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You only look once: Unified, real-time object detection*. In CVPR.
- Lin, T. Y. et al. (2017). *Focal Loss for Dense Object Detection*. In ICCV.
- Bodla, N. et al. (2017). *Soft-NMS – Improving Object Detection With One Line of Code*. In ICCV.

