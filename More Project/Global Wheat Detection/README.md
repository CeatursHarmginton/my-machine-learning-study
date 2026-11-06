# 🌾 Global Wheat Detection (Kaggle Competition)

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/global-wheat-detection)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/ceatur/global-wheat-detection-dataset)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-Object%20Detection-blue)](https://pytorch.org/vision/)

Dự án phát hiện và định vị bông lúa mì (Wheat Heads) trên ảnh chụp quang học độ phân giải cao từ đồng ruộng ngoài trời, thuộc cuộc thi **[Global Wheat Detection trên Kaggle](https://www.kaggle.com/competitions/global-wheat-detection)** do Global Institute for Food Security (GIFS) tại Đại học Saskatchewan cùng các viện nghiên cứu quốc tế (Arvalis, ETHZ, INRAE, Rothamsted Research,...) đồng tổ chức.

Tập dữ liệu hoàn chỉnh đã được lưu trữ và sẵn sàng tải về tại: **[Hugging Face Dataset - ceatur/global-wheat-detection-dataset](https://huggingface.co/datasets/ceatur/global-wheat-detection-dataset)**.

---

## 📌 Mục lục

- [Tổng quan dự án](#-tổng-quan-dự-án)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Đặc điểm tập dữ liệu](#-đặc-điểm-tập-dữ-liệu)
- [Phương pháp & Kỹ thuật](#-phương-pháp--kỹ-thuật)
- [Thước đo đánh giá (Evaluation Metric)](#-thước-đo-đánh-giá-evaluation-metric)
- [Hướng dẫn cài đặt & Thực thi](#-hướng-dẫn-cài-đặt--thực-thi)
- [Chiến lược tối ưu nâng cao](#-chiến-lược-tối-ưu-nâng-cao)
- [Tài liệu tham khảo](#-tài-liệu-tham-khảo)

---

## 🎯 Tổng quan dự án

Ước tính số lượng bông lúa mì từ ảnh chụp máy bay không người lái (UAV) hoặc thiết bị chụp thực địa là bài toán cốt lõi trong nông nghiệp kỹ thuật số (digital agriculture) nhằm:
1. Dự đoán năng suất cây trồng chính xác và sớm hơn.
2. Đánh giá sức khỏe và mật độ cây trồng tại các giai đoạn sinh trưởng khác nhau.
3. Hỗ trợ chọn giống lúa mì có khả năng thích nghi cao với biến đổi khí hậu.

**Thách thức chính**:
- Bông lúa mì thường bị che khuất một phần (overlap, occlusion).
- Biến động mạnh về góc chiếu sáng, bóng râm, gió làm nhòe chuyển động.
- Sự đa dạng về giống lúa mì, màu sắc qua các giai đoạn trưởng thành và nền đất/lá phức tạp.
- Dữ liệu thu thập từ nhiều quốc gia và điều kiện thời tiết khác nhau (domain shift).

---

## 📁 Cấu trúc thư mục

```text
More Project/
└── Global Wheat Detection/
    ├── Data/
    │   ├── train/                 # Tập ảnh huấn luyện (.jpg, 1024x1024)
    │   ├── test/                  # Tập ảnh kiểm thử (.jpg, 1024x1024)
    │   ├── train.csv              # Nhãn bounding box dạng [x_min, y_min, width, height]
    │   └── sample_submission.csv  # Định dạng nộp bài (image_id, PredictionString)
    ├── Notebooks/                 # Các Jupyter Notebooks huấn luyện & phân tích dữ liệu
    ├── Submissions/               # File kết quả dự đoán (.csv)
    └── README.md                  # Tài liệu giới thiệu dự án
```

---

## 📊 Đặc điểm tập dữ liệu

- **Định dạng ảnh**: JPEG kích thước tiêu chuẩn $1024 \times 1024$ pixels.
- **Tệp `train.csv`**:
  - `image_id`: Mã định danh duy nhất của mỗi bức ảnh.
  - `width`, `height`: Kích thước ảnh ($1024 \times 1024$).
  - `bbox`: Chuỗi toạ độ `[x_min, y_min, width, height]` tính theo pixel.
  - `source`: Nguồn thu thập dữ liệu (ví dụ: `usask_1`, `arvalis_1`, `arvalis_2`, `arvalis_3`, `inrae_1`, `rres_1`, `ethz_1`).

| Thuộc tính | Chi tiết |
| :--- | :--- |
| **Số lượng ảnh train** | ~3,373 ảnh |
| **Tổng số bounding box** | 147,700+ nhãn bông lúa mì |
| **Số lượng ảnh test** | Ẩn trên Kaggle (Public/Private Test) |
| **Độ phân giải** | $1024 \times 1024$ px (RGB) |
| **Kho lưu trữ Dataset** | [Hugging Face: ceatur/global-wheat-detection-dataset](https://huggingface.co/datasets/ceatur/global-wheat-detection-dataset) |

---

## 🔬 Phương pháp & Kỹ thuật

### 1. Tiền xử lý & Data Augmentation
Để đối phó với hiện tượng overlap và thay đổi tỷ lệ khung hình, pipeline áp dụng thư viện `Albumentations`:
- **Spatial Transforms**: Random Horizontal Flip, Random Vertical Flip, Random Rotate 90°, ShiftScaleRotate.
- **Color & Light Transforms**: Random Brightness / Contrast, HueSaturationValue, CLAHE (Contrast Limited Adaptive Histogram Equalization).
- **Advanced Augmentation**: CutMix, Mosaic (kết hợp 4 ảnh ngẫu nhiên để mô phỏng mật độ dày đặc), MixUp.

### 2. Kiến trúc mô hình đề xuất
- **Faster R-CNN (ResNet-50 / ResNeXt-101-FPN)**: Two-stage detector với độ chính xác cao trong nhận diện các đối tượng nhỏ và dày đặc.
- **YOLOv5 / YOLOv8 (Ultralytics)**: One-stage detector tối ưu về tốc độ suy luận và huấn luyện ổn định.
- **EfficientDet (D0 - D5)**: BiFPN và compound scaling cho cân bằng tốt giữa tài nguyên tính toán và mAP.

### 3. Hậu xử lý (Post-Processing)
- **Weighted Boxes Fusion (WBF)**: Thay thế Non-Maximum Suppression (NMS) truyền thống bằng cách tổng hợp và tính trung bình có trọng số các toạ độ box từ nhiều mô hình hoặc TTA.
- **Test-Time Augmentation (TTA)**: Đánh giá ảnh qua các phép biến đổi (Flip, Multi-scale) rồi gộp kết quả bằng WBF.

---

## 📐 Thước đo đánh giá (Evaluation Metric)

Cuộc thi sử dụng độ đo **mean Average Precision (mAP)** tính trên nhiều ngưỡng Intersection over Union (IoU):

$$\text{mAP} = \frac{1}{|T|} \sum_{t \in T} \text{AP}(t)$$

với tập ngưỡng IoU:

$$T = \{0.50, 0.55, 0.60, 0.65, 0.70, 0.75\}$$

Tại mỗi ngưỡng $t$:
- **True Positive (TP)**: Bounding box dự đoán có $\text{IoU} \ge t$ với một ground truth box chưa được match.
- **False Positive (FP)**: Bounding box dự đoán không match với ground truth box nào hoặc ground truth đã được match trước đó với độ tin cậy cao hơn.
- **False Negative (FN)**: Ground truth box không có dự đoán nào match đạt $\text{IoU} \ge t$.

---

## 🛠️ Hướng dẫn cài đặt & Thực thi

### 1. Cài đặt môi trường

```bash
# Clone repository
git clone https://github.com/CeatursHarmginton/my-machine-learning-study.git
cd "my-machine-learning-study/More Project/Global Wheat Detection"

# Cài đặt các thư viện cần thiết
pip install torch torchvision albumentations opencv-python pandas matplotlib ensemble-boxes tqdm
```

### 2. Tải & Chuẩn bị dữ liệu từ Hugging Face

Bạn có thể tải nhanh toàn bộ dataset đã đóng gói sẵn từ [Hugging Face](https://huggingface.co/datasets/ceatur/global-wheat-detection-dataset):

```bash
# Tải file zip dataset trực tiếp từ Hugging Face
wget -q -O data.zip "https://huggingface.co/datasets/ceatur/global-wheat-detection-dataset/resolve/main/global-wheat-detection-dataset.zip?download=true"

# Giải nén vào thư mục dự án
unzip -q data.zip -d ./
```

Cấu trúc sau khi giải nén:
```text
Data/
├── train.csv
├── sample_submission.csv
├── train/
└── test/
```

### 3. Chạy Notebook huấn luyện
Khởi động Jupyter Lab / Notebook:
```bash
jupyter notebook
```
Mở notebook trong thư mục `Notebooks/` để tiến hành EDA (Exploratory Data Analysis), huấn luyện mô hình, kiểm tra mAP và sinh file `submission.csv`.

---

## 💡 Chiến lược tối ưu nâng cao

1. **Stratified K-Fold theo Source**: Chia tập Train/Val dựa trên trường `source` và số lượng box trên ảnh để tránh rò rỉ dữ liệu cùng một điều kiện chụp vào tập validation.
2. **Loại bỏ nhãn rác (Data Cleaning)**: Lọc bỏ các box có kích thước bất thường (quá lớn chiếm trọn ảnh hoặc quá nhỏ $< 10$ px).
3. **Pseudo Labeling**: Sử dụng mô hình tốt nhất để gán nhãn cho tập Test, sau đó tái huấn luyện với tập dữ liệu mở rộng.
4. **Model Ensembling**: Kết hợp dự đoán của Faster R-CNN + EfficientDet + YOLO thông qua Weighted Boxes Fusion (WBF).

---

## 📚 Tài liệu tham khảo

- [Global Wheat Detection Competition Overview (Kaggle)](https://www.kaggle.com/competitions/global-wheat-detection)
- [Hugging Face Global Wheat Detection Dataset](https://huggingface.co/datasets/ceatur/global-wheat-detection-dataset)
- David, E. et al. (2020). *Global Wheat Head Detection (GWHD) dataset: a large and diverse dataset of high-resolution RGB-labelled images to develop and benchmark wheat head detection methods*. Plant Phenomics.
- Roman Solovyev, Weimin Wang, Tatiana Gabruseva. *Weighted boxes fusion: Ensembling methods for object detection*. Image and Vision Computing, 2021.
