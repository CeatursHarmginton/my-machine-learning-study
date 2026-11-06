# 🍜 Street Food Image Classification (Kaggle Competition)

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/street-food-image-classification)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/ceatur/street-food-image-classification-dataset)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

Dự án phân loại hình ảnh các món ăn đường phố phổ biến trên thế giới sử dụng Deep Learning và Transfer Learning, thuộc cuộc thi **[Street Food Image Classification trên Kaggle](https://www.kaggle.com/competitions/street-food-image-classification)**.

Tập dữ liệu hoàn chỉnh đã được lưu trữ và sẵn sàng tải về tại: **[Hugging Face Dataset - ceatur/street-food-image-classification-dataset](https://huggingface.co/datasets/ceatur/street-food-image-classification-dataset)**.

---

## 📌 Mục lục

- [Tổng quan dự án](#-tổng-quan-dự-án)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Danh sách 10 loại món ăn](#-danh-sách-10-loại-món-ăn)
- [Kiến trúc & Pipeline Huấn luyện](#-kiến-trúc--pipeline-huấn-luyện)
- [Trực quan hóa mô hình (Grad-CAM Explainability)](#-trực-quan-hóa-mô-hình-grad-cam-explainability)
- [Kết quả thực nghiệm](#-kết-quả-thực-nghiệm)
- [Hướng dẫn cài đặt & Thực thi](#-hướng-dẫn-cài-đặt--thực-thi)
- [Định dạng Submission](#-định-dạng-submission)
- [Tài liệu tham khảo](#-tài-liệu-tham-khảo)

---

## 🎯 Tổng quan dự án

Nhận diện món ăn qua hình ảnh là bài toán then chốt trong các ứng dụng ẩm thực thông minh, theo dõi chế độ dinh dưỡng cá nhân, tìm kiếm địa điểm du lịch và quản lý thực đơn số.

Dự án này xây dựng quy trình phân loại tự động 10 món ăn đường phố đặc trưng toàn cầu từ ảnh chụp đời thực với nhiều góc chụp, ánh sáng và cách bài trí đa dạng.

---

## 📁 Cấu trúc thư mục

```text
More Project/
└── Street Food Image Classification/
    ├── Data/                           # Thư mục dữ liệu (Train / Test)
    ├── Notebooks/
    │   └── solve.ipynb                 # Notebook hoàn chỉnh (EDA, Training, Grad-CAM, Inference)
    ├── Submission/
    │   └── street_food_submission.csv  # Kết quả dự đoán tập test nộp Kaggle
    └── README.md                       # Tài liệu hướng dẫn dự án
```

---

## 🍲 Danh sách 10 loại món ăn

Tập dữ liệu bao gồm 10 lớp món ăn đường phố nổi tiếng thế giới:

| STT | Tên lớp (Class Name) | Món ăn | Nguồn gốc / Đặc trưng |
| :---: | :--- | :--- | :--- |
| 1 | `burger` | Bánh mì kẹp thịt (Hamburger) | Bánh mì tròn kèm thịt nướng, rau xà lách, phô mai |
| 2 | `churros` | Bánh quẩy ngọt Tây Ban Nha | Bánh chiên rắc đường quế ăn kèm sốt sô-cô-la |
| 3 | `crepes` | Bánh kếp Pháp | Bánh mỏng gấp ngọt hoặc mặn |
| 4 | `falafel` | Chả đậu chiên Trung Đông | Viên đậu gà xay chiên giòn thơm thảo mộc |
| 5 | `hot_dog` | Bánh mì kẹp xúc xích | Bánh mì dài kẹp xúc xích kèm mù tạt / tương cà |
| 6 | `pad_thai` | Hủ tiếu xào Thái Lan | Hủ tiếu xào tôm, đậu phụ, trứng, đậu phộng, chanh |
| 7 | `pani_puri` | Bánh bóng tròn nhồi Ấn Độ | Vỏ bánh giòn rỗng ruột chứa khoai tây, nước sốt cay chua |
| 8 | `pretzel` | Bánh quy xoắn Đức | Bánh nướng thắt nút rắc muối hạt giòn |
| 9 | `shawarma` | Bánh cuộn thịt Trung Đông | Bánh pita cuộn thịt nướng xiên quay thái mỏng |
| 10 | `tacos` | Bánh kẹp vỏ ngô Mexico | Vỏ bánh tortilla kẹp thịt bò/gà, salsa, ngò rí |

---

## 🔬 Kiến trúc & Pipeline Huấn luyện

### 1. Phân chia dữ liệu & Tiền xử lý
- **Tập Train**: 1,216 ảnh ($80\%$)
- **Tập Validation**: 304 ảnh ($20\%$)
- **Tập Test**: 764 ảnh
- **Kích thước ảnh**: $224 \times 224$ pixels.
- **Augmentation**:
  - `RandomResizedCrop(224, scale=(0.8, 1.0))`
  - `RandomHorizontalFlip(p=0.5)`
  - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`
  - `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

### 2. Mô hình Backbone
- **Model**: `ResNet-18` sử dụng pre-trained weights `ResNet18_Weights.DEFAULT` trên ImageNet.
- **Classifier Head**: Thay thế fully connected layer cuối cùng (`fc`) bằng linear layer chuyển tiếp 512 features $\rightarrow$ 10 output classes.

### 3. Hyperparameters & Chiến lược tối ưu
- **Optimizer**: AdamW (Learning rate $= 2 \times 10^{-4}$, Weight Decay $= 10^{-3}$)
- **Loss Function**: `CrossEntropyLoss` kết hợp **Label Smoothing** ($0.15$) giúp giảm thiểu hiện tượng overconfidence và tăng tính tổng quát hóa (generalization).
- **Class Weights**: Tính trọng số cân bằng lớp nhằm khắc phục độ lệch số lượng mẫu giữa các món ăn.
- **Batch Size**: 32
- **Epochs**: 50 epochs

---

## 🔍 Trực quan hóa mô hình (Grad-CAM Explainability)

Dự án tích hợp thuật toán **Grad-CAM (Gradient-weighted Class Activation Mapping)** để giải thích quyết định của mô hình AI:
- Trích xuất gradients của score lớp dự đoán đối với feature map tại tầng tích chập cuối cùng (`layer4`).
- Tính trọng số trung bình toàn cục (Global Average Pooling) của gradients để tạo Heatmap.
- Chồng Heatmap lên ảnh gốc giúp xác minh mạng nơ-ron thực sự tập trung vào vùng đặc trưng của món ăn (như phần nhân bánh, sốt, hạt mè, cấu trúc sợi hủ tiếu) thay vì nền xung quanh.

```text
Input Image ───► ResNet-18 Feature Maps (layer4) ───► Target Class Score
                          │                                   │
                          ▼                                   ▼
                   Gradients Weight ◄───────────────── Backward Pass
                          │
                          ▼
             Heatmap (ReLU Activation) ──► Overlay on Original Image
```

---

## 📊 Kết quả thực nghiệm

- **Validation Accuracy**: Đạt độ chính xác cao trên tập kiểm thử nội bộ.
- Mô hình nhận diện xuất sắc các đặc trưng hình thái riêng biệt của từng món ăn, hội tụ ổn định sau 50 epochs mà không bị overfitting nhờ Data Augmentation và Label Smoothing.

---

## 🛠️ Hướng dẫn cài đặt & Thực thi

### 1. Cài đặt môi trường

```bash
# Clone repository
git clone https://github.com/CeatursHarmginton/my-machine-learning-study.git
cd "my-machine-learning-study/More Project/Street Food Image Classification"

# Cài đặt thư viện
pip install torch torchvision pandas numpy matplotlib pillow opencv-python
```

### 2. Tải & Chuẩn bị dữ liệu từ Hugging Face

Tập dữ liệu hoàn chỉnh đã được lưu trữ trên [Hugging Face](https://huggingface.co/datasets/ceatur/street-food-image-classification-dataset). Bạn có thể tải nhanh trực tiếp:

```bash
# Tải file zip dataset từ Hugging Face
wget -q -O data.zip "https://huggingface.co/datasets/ceatur/street-food-image-classification-dataset/resolve/main/street-food-image-classification-dataset.zip?download=true"

# Giải nén vào thư mục dự án
unzip -q data.zip -d ./
```

### 3. Huấn luyện & Chạy suy luận
Mở và thực thi file Jupyter Notebook:
```bash
jupyter notebook Notebooks/solve.ipynb
```
Notebook sẽ tự động:
1. Tải và giải nén dữ liệu.
2. Trực quan hóa mẫu ảnh 10 lớp.
3. Huấn luyện ResNet-18 với Transfer Learning & Validation Monitoring.
4. Trực quan hóa đường cong Loss / Accuracy và Grad-CAM Explainability.
5. Dự đoán nhãn cho 764 ảnh test và xuất file `Submission/street_food_submission.csv`.

---

## 📄 Định dạng Submission

File nộp bài `submission.csv` tuân theo chuẩn định dạng cuộc thi:
```csv
image_id,label
test_0001.jpg,pad_thai
test_0002.jpg,burger
test_0003.jpg,tacos
test_0004.jpg,pani_puri
...
```

---

## 📚 Tài liệu tham khảo

- [Kaggle Street Food Image Classification Competition](https://www.kaggle.com/competitions/street-food-image-classification)
- [Hugging Face Street Food Dataset](https://huggingface.co/datasets/ceatur/street-food-image-classification-dataset)
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. In CVPR.
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization*. In ICCV.
