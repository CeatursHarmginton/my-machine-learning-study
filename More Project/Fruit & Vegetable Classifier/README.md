# 🍎 Fruit & Vegetable Classifier (Kaggle Competition)

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/utn-ia-2026-fruit-vegetable-classifier)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/ceatur/fruit-vegetable-classifier-dataset)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Image%20Classification-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/Task-Multi--Class%20Classification-blue)](https://pytorch.org/)

Dự án phân loại hình ảnh trái cây và rau củ phổ biến sử dụng kỹ thuật Deep Learning và Transfer Learning trên nền tảng PyTorch, thuộc cuộc thi **[UTN-IA 2026: Fruit & Vegetable Classifier trên Kaggle](https://www.kaggle.com/competitions/utn-ia-2026-fruit-vegetable-classifier/overview)**.

Tập dữ liệu hoàn chỉnh đã được đóng gói và sẵn sàng tải về tại: **[Hugging Face Dataset - ceatur/fruit-vegetable-classifier-dataset](https://huggingface.co/datasets/ceatur/fruit-vegetable-classifier-dataset)**.

---

## 📌 Mục lục

- [Tổng quan dự án](#-tổng-quan-dự-án)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Danh sách 5 loại nông sản & Thống kê dữ liệu](#-danh-sách-5-loại-nông-sản--thống-kê-dữ-liệu)
- [Kiến trúc & Pipeline Huấn luyện](#-kiến-trúc--pipeline-huấn-luyện)
- [Trực quan hóa mô hình (Grad-CAM Explainability)](#-trực-quan-hóa-mô-hình-grad-cam-explainability)
- [Thước đo đánh giá (Evaluation Metric)](#-thước-đo-đánh-giá-evaluation-metric)
- [Hướng dẫn cài đặt & Thực thi](#-hướng-dẫn-cài-đặt--thực-thi)
- [Định dạng Submission](#-định-dạng-submission)
- [Chiến lược tối ưu cho tập dữ liệu nhỏ](#-chiến-lược-tối-ưu-cho-tập-dữ-liệu-nhỏ)
- [Tài liệu tham khảo](#-tài-liệu-tham-khảo)

---

## 🎯 Tổng quan dự án

Phân loại tự động các loại rau củ và trái cây qua hình ảnh là bài toán ứng dụng thiết thực trong:
1. **Thanh toán tự động (Smart Retail & Checkout)**: Nhận diện nhanh loại nông sản tại quầy cân siêu thị không người bán.
2. **Quản lý chuỗi cung ứng & Kho bãi (Inventory Automation)**: Phân loại và kiểm kê số lượng sản phẩm nhập/xuất kho tự động.
3. **Ứng dụng theo dõi dinh dưỡng (Health & Calorie Tracker)**: Tự động phát hiện thực phẩm và ước tính khẩu phần ăn hàng ngày.

**Thách thức chính**:
- Tập dữ liệu có kích thước vừa và nhỏ (325 ảnh huấn luyện chia cho 5 lớp), đòi hỏi kỹ thuật Transfer Learning và Data Augmentation mạnh để chống hiện tượng quá khớp (overfitting).
- Biến động về góc chụp, tỷ lệ khung hình, ánh sáng, độ tươi và cách sắp đặt các quả (riêng lẻ hoặc theo chùm như nho).

---

## 📁 Cấu trúc thư mục

```text
More Project/
└── Fruit & Vegetable Classifier/
    ├── Data/
    │   ├── train/
    │   │   ├── apple/                    # 53 ảnh quả táo
    │   │   ├── banana/                   # 60 ảnh quả chuối
    │   │   ├── grapes/                   # 78 ảnh chùm nho
    │   │   ├── potato/                   # 62 ảnh củ khoai tây
    │   │   └── tomato/                   # 72 ảnh quả cà chua
    │   ├── test/                         # 89 ảnh kiểm thử (.jpg, img_00000 -> img_00088)
    │   └── sample_submission.csv         # File mẫu định dạng nộp bài (image_id, label)
    ├── Notebooks/
    │   └── solve.ipynb                   # Jupyter Notebook huấn luyện mô hình & inference
    ├── Submissions/                      # File kết quả nộp bài (.csv)
    └── README.md                         # Tài liệu hướng dẫn chi tiết dự án
```

---

## 🍇 Danh sách 5 loại nông sản & Thống kê dữ liệu

Tập dữ liệu bao gồm 5 lớp nông sản thông dụng:

| STT | Lớp (Class Name) | Tiếng Việt | Số lượng ảnh Train | Tỷ lệ | Đặc trưng nhận diện chính |
| :---: | :--- | :--- | :---: | :---: | :--- |
| 1 | `apple` | Quả táo | 53 | ~16.3% | Dáng tròn, cuống lõm, vỏ đỏ/xanh bóng |
| 2 | `banana` | Quả chuối | 60 | ~18.5% | Dáng thuôn dài uốn cong, vỏ vàng/xanh, thường mọc theo nải |
| 3 | `grapes` | Chùm nho | 78 | ~24.0% | Kết chùm hạt nhỏ hình cầu/bầu dục, vỏ tím/xanh/đỏ mọng |
| 4 | `potato` | Củ khoai tây | 62 | ~19.1% | Dáng bầu dục gồ ghề, vỏ nâu đất nhám |
| 5 | `tomato` | Quả cà chua | 72 | ~22.1% | Dáng tròn đỏ mọng nước, núm cuống lá xanh xòe |
| **Tổng** | **5 classes** | | **325 ảnh** | **100%** | |

| Thuộc tính | Chi tiết |
| :--- | :--- |
| **Tổng số ảnh Train** | 325 ảnh (5 thư mục lớp theo cấu trúc ImageFolder) |
| **Tổng số ảnh Test** | 89 ảnh (`img_00000.jpg` $\rightarrow$ `img_00088.jpg`) |
| **Cấu trúc lưu trữ** | Thư mục phân lớp trực quan, sẵn sàng nạp qua `torchvision.datasets.ImageFolder` |
| **Kho lưu trữ Dataset** | [Hugging Face: ceatur/fruit-vegetable-classifier-dataset](https://huggingface.co/datasets/ceatur/fruit-vegetable-classifier-dataset) |

---

## 🔬 Kiến trúc & Pipeline Huấn luyện

```text
┌─────────────────┐     ┌────────────────────────────┐     ┌───────────────────────┐
│   Input Image   │ ──► │     Data Augmentation      │ ──► │  Pretrained Backbone  │
│  (RGB 224x224)  │     │ (Flip, Affine, ColorJitter)│     │  (ResNet-18 / 34 / 50) │
└─────────────────┘     └────────────────────────────┘     └──────────┬────────────┘
                                                                      │
                                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Classification Head                              │
│   Linear(num_features, 256) ──► ReLU ──► Dropout(0.3) ──► Linear(256, 5)   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
   ┌───────────────────────────┐             ┌───────────────────────────┐
   │       TRAINING PHASE      │             │      INFERENCE PHASE      │
   │  - Cross-Entropy + Smooth │             │  - Softmax Probabilities  │
   │  - AdamW / CosineAnnealing│             │  - Argmax Class Name      │
   │  - Stratified 5-Fold CV   │             │  - Export Submission CSV  │
   └───────────────────────────┘             └───────────────────────────┘
```

### 1. Tiền xử lý & Augmentation
Đối với dataset kích thước nhỏ (~325 mẫu), Augmentation đóng vai trò quyết định:
- **Spatial Transforms**: `RandomResizedCrop(224, scale=(0.7, 1.0))`, `RandomHorizontalFlip(p=0.5)`, `RandomRotation(degrees=20)`.
- **Color & Lighting**: `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)`.
- **Normalization**: Chuẩn hóa theo thống kê ImageNet:
  $$\text{mean} = [0.485, 0.456, 0.406], \quad \text{std} = [0.229, 0.224, 0.225]$$

### 2. Mô hình Backbone & Transfer Learning
- Sử dụng pre-trained **ResNet-18** / **ResNet-50** hoặc **EfficientNet-B0** đã được huấn luyện trước trên 1.4 triệu ảnh ImageNet-1k.
- Đóng băng (Freeze) các tầng đặc trưng ban đầu và chỉ tinh chỉnh (Fine-tune) các block tích chập sâu cùng Classifier Head.

### 3. Hyperparameters & Chiến lược hội tụ
- **Optimizer**: AdamW với Learning rate $= 1 \times 10^{-4}$, Weight Decay $= 10^{-2}$.
- **Loss Function**: `nn.CrossEntropyLoss(label_smoothing=0.1)` giúp giảm thiểu overconfidence của mạng trên tập dữ liệu nhỏ.
- **LR Scheduler**: `CosineAnnealingLR` hoặc `ReduceLROnPlateau`.
- **Batch Size**: 16 hoặc 32.
- **Epochs**: 25 - 40 epochs.

---

## 🔍 Trực quan hóa mô hình (Grad-CAM Explainability)

Dự án tích hợp thuật toán **Grad-CAM (Gradient-weighted Class Activation Mapping)** để kiểm tra vùng thị giác mà mạng AI tập trung trước khi đưa ra kết luận:
- Giúp xác nhận mô hình đang thực sự nhìn vào vỏ trái cây, cuống lá, màu sắc và hình dạng thay vì các chi tiết nền (bàn gỗ, rổ đựng).

```text
Ảnh đầu vào ──► Convolutional Layers ──► Gradients Activation ──► Heatmap Overlay
```

---

## 📐 Thước đo đánh giá (Evaluation Metric)

Cuộc thi đánh giá độ chính xác phân loại bằng **Accuracy (Độ chính xác tổng thể)** hoặc **Multi-Class Log Loss / Macro F1-Score**:

$$\text{Accuracy} = \frac{\sum_{i=1}^{N} \mathbb{I}(\hat{y}_i = y_i)}{N}$$

$$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c} \quad (C = 5)$$

---

## 🛠️ Hướng dẫn cài đặt & Thực thi

### 1. Cài đặt môi trường

```bash
# Clone repository
git clone https://github.com/CeatursHarmginton/my-machine-learning-study.git
cd "my-machine-learning-study/More Project/Fruit & Vegetable Classifier"

# Cài đặt thư viện cần thiết
pip install torch torchvision pandas numpy matplotlib pillow opencv-python tqdm
```

### 2. Tải & Chuẩn bị dữ liệu từ Hugging Face

Bạn có thể tải nhanh toàn bộ dataset đã đóng gói sẵn từ [Hugging Face](https://huggingface.co/datasets/ceatur/fruit-vegetable-classifier-dataset):

```bash
# Tải tệp zip dataset trực tiếp từ Hugging Face
wget -q -O data.zip "https://huggingface.co/datasets/ceatur/fruit-vegetable-classifier-dataset/resolve/main/fruit-vegetable-classifier-dataset.zip?download=true"

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

File kết quả nộp bài `submission.csv` bao gồm 2 cột: `image_id` (mã định danh không kèm đuôi file `.jpg`) và `label` (tên lớp dự đoán):

```csv
image_id,label
img_00000,apple
img_00001,banana
img_00002,grapes
img_00003,potato
img_00004,tomato
...
img_00088,banana
```

---

## 💡 Chiến lược tối ưu cho tập dữ liệu nhỏ

1. **Stratified 5-Fold Cross Validation**: Chia tập 325 ảnh thành 5 fold đều cho mỗi lớp, huấn luyện 5 mô hình độc lập và trung bình xác suất (Soft Voting Ensemble) cho tập test.
2. **Two-Stage Fine-Tuning**: 
   - *Giai đoạn 1*: Freeze backbone, chỉ train classifier head với learning rate cao ($10^{-3}$) trong 10 epochs.
   - *Giai đoạn 2*: Unfreeze toàn bộ backbone, train toàn bộ mạng với learning rate nhỏ ($10^{-5}$) trong 20 epochs.
3. **Test-Time Augmentation (TTA)**: Tại bước inference trên 89 ảnh test, dự đoán qua nhiều biến thể (ảnh gốc, lật ngang, thay đổi tỷ lệ nhẹ) rồi lấy trung bình xác suất.
4. **MixUp & CutMix**: Kết hợp ngẫu nhiên 2 ảnh để tạo mẫu dữ liệu mới, tăng tính trơn của không gian quyết định.

---

## 📚 Tài liệu tham khảo

- [UTN-IA 2026: Fruit & Vegetable Classifier Overview (Kaggle)](https://www.kaggle.com/competitions/utn-ia-2026-fruit-vegetable-classifier/overview)
- [Hugging Face Fruit & Vegetable Dataset](https://huggingface.co/datasets/ceatur/fruit-vegetable-classifier-dataset)
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. In CVPR.
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization*. In ICCV.
- Zhang, H. et al. (2018). *mixup: Beyond Empirical Risk Minimization*. In ICLR.

