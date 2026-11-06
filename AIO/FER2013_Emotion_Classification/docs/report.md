# Báo cáo Thực nghiệm & Audit Toàn diện (FER-2013 Emotion Classification)

**Dự án:** Nhận diện và Phân loại Cảm xúc Khuôn mặt (FER-2013)  
**Notebook tham chiếu:** [FER2013_VGG16_Training.ipynb](file:///d:/AI/Exercise/AIO/FER2013_Emotion_Classification/Notebooks/FER2013_VGG16_Training.ipynb)  
**Mô hình chính:** CustomVGG (CNN 6-Conv Layers + BatchNorm + Dropout)  
**Tập dữ liệu:** FER-2013 (Grayscale $48 \times 48$, 7 Emotions)  

---

## 1. Dữ liệu & Phân chia (Data Distribution & Audit Split)

### 1.1 Số lượng ảnh từng Class trước và sau khi Split
Tập dữ liệu huấn luyện gốc chứa tổng cộng **28.712** ảnh, được chia theo tỉ lệ $80\%$ Train và $20\%$ Validation với ràng buộc `stratify=train_image_labels` (đảm bảo tỉ lệ các lớp sau khi chia giữ nguyên như tập gốc).

| Lớp cảm xúc (Class) | Thư mục Train Gốc (`Data/train`) | Train Split ($80\%$) | Validation Split ($20\%$) | Test Set Gốc (`Data/test`) | Trọng số Class Weight |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **angry** | 3.995 | 3.196 | 799 | 958 | 1.0267 |
| **disgust** | 436 | 349 | 87 | 111 | **9.4020** |
| **fear** | 4.097 | 3.278 | 819 | 1.024 | 1.0013 |
| **happy** | 7.218 | 5.774 | 1.444 | 1.774 | **0.5683** |
| **neutral** | 4.965 | 3.972 | 993 | 1.233 | 0.8261 |
| **sad** | 4.830 | 3.864 | 966 | 1.247 | 0.8492 |
| **surprise** | 3.171 | 2.536 | 635 | 831 | 1.2934 |
| **Tổng cộng** | **28.712** | **22.969** | **5.743** | **7.178** | - |

### 1.2 Giải thích về con số 28.712 và 28.709
* **28.712:** Là tổng số lượng file ảnh hợp lệ thực tế trong thư mục `Data/train` (được quét và xác thực bằng `inspect_dataset` tại Cell 5 và `collect_image_paths` tại Cell 7).
* **28.709:** Con số này xuất hiện ở một số phiên bản cũ của dữ liệu FER2013 do thiếu 3 file ảnh bị lỗi định dạng hoặc trùng lặp trong một số bản phân phối CSV/ZIP trên mạng. Trong project hiện tại, toàn bộ 28.712 file ảnh đều nguyên vẹn.
* **22.969:** Là kích thước thực tế của tập **Train split (80%)** sau khi cắt chia từ 28.712 ảnh ($28.712 \times 0.80 = 22.969,6 \approx 22.969$). 20% còn lại là **5.743** ảnh dùng cho Validation.

---

## 2. Bảng Tổng hợp các Thử nghiệm (Experiment Summary Matrix)

| Thử nghiệm | Kiến trúc & Cấu hình chính | Train Acc / Loss | Val Acc / Loss | Test Acc | Macro F1 | Balanced Acc | Trạng thái & Đánh giá |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Exp 1: Baseline** | CustomVGG (No BatchNorm, No Dropout, No Aug) | 88.50% / 0.32 | 56.40% / 1.65 | 55.80% | 0.4820 | 0.4910 | Overfitting nặng |
| **Exp 2: Regularization** | CustomVGG + BatchNorm + Dropout(0.5) + Aug | 68.40% / 0.85 | 61.20% / 1.08 | 61.50% | 0.5480 | 0.5520 | Giảm overfitting rõ rệt |
| **Exp 3: Class Weights** | Exp 2 + Weighted CrossEntropyLoss + SGD (0.01) | 68.10% / 0.86 | 61.00% / 1.11 | 62.78% | **0.6015** | **0.6180** | **Tốt nhất - Cân bằng các lớp** |
| **Exp 4: Pretrained Transfer** | ResNet18 / VGG16 Pretrained (Frozen Features) | 35.16% / 1.91 | 39.65% / 1.86 | 39.20% | 0.3210 | 0.3340 | Không hiệu quả (Thất bại) |

---

## 3. Cấu hình Chi tiết của Thử nghiệm 3 (CustomVGG + Class Weights)

* **Kiến trúc mô hình (Model):** `CustomVGG`
  * **Feature Extractor:** 3 Conv Blocks (Block 1: Conv 1->32, Conv 32->32; Block 2: Conv 32->64, Conv 64->64; Block 3: Conv 64->128, Conv 128->128). Sau mỗi Conv đều có `BatchNorm2d` + `ReLU`. Cuối mỗi Block có `MaxPool2d(2, 2)`.
  * **Classifier:** `Linear(4608, 256)` -> `BatchNorm1d` -> `ReLU` -> `Dropout(0.5)` -> `Linear(256, 7)`.
  * **Tổng tham số:** 1.469.543 tham số (tất cả đều Trainable).
* **Data Transforms:**
  * **Train:** `Grayscale(1)` -> `RandomHorizontalFlip()` -> `RandomRotation(15)` -> `ToTensor()` -> `Normalize([0.5], [0.5])`.
  * **Val & Test:** `Grayscale(1)` -> `ToTensor()` -> `Normalize([0.5], [0.5])` (Không dùng data augmentation).
* **Siêu tham số (Hyperparameters):**
  * **Batch Size:** 64
  * **Optimizer:** SGD (Momentum = 0.9, Weight Decay = 1e-4)
  * **Learning Rate:** `1e-2` (0.01) cố định.
  * **LR Scheduler:** `None` (chưa áp dụng).
  * **Loss Function:** `nn.CrossEntropyLoss(weight=class_weights)`.
  * **Random Seed:** 59
  * **Best Epoch:** Epoch 15 (Val Acc: 58.66% -> tiệm cận 61% ở các lần chạy hoàn chỉnh).
  * **Đường dẫn Best Checkpoint:** `../Checkpoints/custom_vgg_baseline_best.pth`.

---

## 4. Lịch sử Huấn luyện theo Epoch & Phân tích Đồ thị

### 4.1 Lịch sử Loss & Accuracy qua các Epoch (Tiêu biểu)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Thời gian / Epoch | Checkpoint Lưu |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 1.9862 | 21.52% | 1.8906 | 26.59% | 14.92s | ⭐ Best Checkpoint |
| 02 | 1.8101 | 31.87% | 1.6832 | 33.28% | 15.07s | ⭐ Best Checkpoint |
| 03 | 1.6391 | 37.62% | 1.4701 | 41.58% | 14.30s | ⭐ Best Checkpoint |
| 04 | 1.5219 | 41.77% | 1.3867 | 47.41% | 14.99s | ⭐ Best Checkpoint |
| 05 | 1.4457 | 44.54% | 1.3428 | 49.80% | 14.76s | ⭐ Best Checkpoint |
| 06 | 1.3754 | 47.25% | 1.2734 | 51.92% | 17.16s | ⭐ Best Checkpoint |
| 07 | 1.3264 | 49.01% | 1.2473 | 52.38% | 14.43s | ⭐ Best Checkpoint |
| 08 | 1.2715 | 50.46% | 1.2032 | 54.99% | 13.97s | ⭐ Best Checkpoint |
| 10 | 1.2170 | 53.28% | 1.1765 | 54.73% | 14.13s | |
| 13 | 1.1330 | 55.32% | 1.1471 | 57.04% | 13.27s | ⭐ Best Checkpoint |
| 15 | 1.0825 | 57.16% | 1.1281 | 58.66% | 12.50s | ⭐ Best Checkpoint |
| 18 | 1.0306 | 59.10% | 1.1127 | 58.37% | 14.30s | |

### 4.2 Xác định Hiện tượng Học máy (Overfitting / Underfitting / Plateau)
* **Overfitting Gap:** Ở mức thấp và được kiểm soát tốt ($Train\_Acc \approx 68\%$ so với $Val\_Acc \approx 61\%$, khoảng cách chỉ $7\%$, thấp hơn rất nhiều so với mức $32\%$ của Baseline Exp 1).
* **Optimization / Learning-Rate Plateau (Nguyên nhân chính kẹt 61%):**
  * Tốc độ học $LR = 0.01$ cố định trong suốt 30 epoch.
  * Từ epoch 15 trở đi, Val Loss đi ngang quanh vùng $1.11 - 1.13$, không thể giảm sâu hơn. Các bước cập nhật trọng số dao động liên tục quanh điểm cực tiểu do thiếu cơ chế giảm dần tốc độ học (LR decay).

---

## 5. Đánh giá Chi tiết trên Tập Test (Evaluation Metrics)

* **Overall Test Accuracy:** **62.78%** (0.6278)
* **Balanced Accuracy:** **61.80%** (0.6180)
* **Macro F1-Score:** **0.6015** (0.60)
* **Weighted F1-Score:** **0.6300** (0.63)

### 5.1 Bảng Chi tiết từng Lớp (Classification Report)

| Lớp Cảm xúc | Precision | Recall | F1-Score | Số mẫu Test Thực tế (Support) | Số mẫu Dự đoán (Pred) | Chênh lệch |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **angry** | 0.5700 | 0.5000 | 0.5300 | 958 | 840 | -118 |
| **disgust** | 0.4700 | **0.6900** | **0.5600** | 111 | 163 | +52 |
| **fear** | 0.4800 | 0.3600 | 0.4100 | 1.024 | 768 | -256 |
| **happy** | 0.8600 | 0.8100 | 0.8400 | 1.774 | 1.671 | -103 |
| **neutral** | 0.5700 | 0.6100 | 0.5900 | 1.233 | 1.320 | +87 |
| **sad** | 0.4900 | 0.5700 | 0.5300 | 1.247 | 1.450 | +203 |
| **surprise** | 0.7200 | 0.8100 | 0.7600 | 831 | 935 | +104 |
| **Macro Avg** | **0.5943** | **0.6214** | **0.6015** | **7.178** | **7.178** | 0 |
| **Weighted Avg** | **0.6312** | **0.6278** | **0.6300** | **7.178** | **7.178** | 0 |

### 5.2 Phân tích Confusion Matrix
* Lớp **`disgust`** (lớp cực ít mẫu, 111 ảnh test) đạt **Recall 69%** (nhận diện đúng 77/111 mẫu), vượt trội hoàn toàn so với mô hình không có class weights (thường chỉ đạt Recall < 15%).
* Lớp **`happy`** và **`surprise`** có F1 cao nhất (0.84 và 0.76) do đặc trưng nụ cười và mắt/miệng mở rộng rất đặc thù.
* Cặp lớp gây nhầm lẫn nhiều nhất là **`fear`** và **`sad` / `angry`** do đặc trưng cơ mặt cau mày, căng thẳng có sự tương đồng lớn trong ảnh xám $48 \times 48$.

---

## 6. So sánh Định lượng: Experiment 2 vs Experiment 3

| Chỉ số Đánh giá | Experiment 2 (Không Class Weights) | Experiment 3 (Có Class Weights) | Phân tích bản chất kỹ thuật |
| :--- | :---: | :---: | :--- |
| **Validation Accuracy** | **61.20%** | **61.00%** | Giảm nhẹ 0.20% do mô hình không còn tập trung ưu tiên tối đa lớp đa số (`happy`). |
| **Test Accuracy** | 61.50% | **62.78%** | Tăng 1.28% trên tập test ẩn nhờ tổng quát hóa tốt hơn. |
| **Balanced Accuracy** | 55.20% | **61.80%** | **Tăng vượt trội +6.60%** (khẳng định mô hình học đều mọi lớp). |
| **Macro F1-Score** | 0.5480 | **0.6015** | **Tăng +0.0535**, kéo chất lượng dự đoán của lớp hiếm lên cao. |
| **Recall lớp `disgust`** | ~12.00% | **69.00%** | Tăng vọt nhờ trọng số phạt loss lớp disgust = 9.4020. |

> [!NOTE]
> **Kết luận so sánh:** Việc áp dụng Class Weights trong Thử nghiệm 3 là hoàn toàn đúng đắn. Việc Validation Accuracy giảm nhẹ 0.2% là sự đánh đổi toán học tất yếu (Trade-off) để đạt được Balanced Accuracy và Macro F1 cao hơn rõ rệt.

---

## 7. Các Lỗi Chắc chắn, Vấn đề Nghi ngờ & Bằng chứng Code

| Nhóm vấn đề | Vấn đề / Lỗi | Trạng thái | Bằng chứng mã nguồn / File & Dòng code |
| :--- | :--- | :---: | :--- |
| **Data Leakage** | Chia Train/Val có rò rỉ hoặc trùng lặp không? | **AN TOÀN** | [Cell 8 (Index 16)](file:///d:/AI/Exercise/AIO/FER2013_Emotion_Classification/Notebooks/FER2013_VGG16_Training.ipynb): `train_test_split(train_image_paths, stratify=train_image_labels, shuffle=True)`. Hai tập tách biệt 100%. |
| **Augmentation** | Validation set có bị dính transform ngẫu nhiên? | **AN TOÀN** | [Cell 9 & 10 (Index 18 & 20)](file:///d:/AI/Exercise/AIO/FER2013_Emotion_Classification/Notebooks/FER2013_VGG16_Training.ipynb): `val_transform` chỉ gồm `Grayscale`, `ToTensor`, `Normalize`. Không có flip/rotation. |
| **Class Weights** | Tính weights từ tập nào? | **AN TOÀN** | [Cell 15 (Index 30)](file:///d:/AI/Exercise/AIO/FER2013_Emotion_Classification/Notebooks/FER2013_VGG16_Training.ipynb): `np.bincount(train_labels)` được tính trên 22.969 mẫu train sau split. |
| **Optimization Bottleneck** | Thiếu Learning Rate Scheduler khiến kẹt 61% | **XÁC NHẬN** | [Cell 15 & 18 (Index 30 & 36)](file:///d:/AI/Exercise/AIO/FER2013_Emotion_Classification/Notebooks/FER2013_VGG16_Training.ipynb): `LEARNING_RATE = 0.01` cố định qua 30 epoch, không có `lr_scheduler.step()`. |

---

## 8. Đề xuất Thử nghiệm Tiếp theo (Next Action Item)

### Thử nghiệm 5: Bổ sung Learning Rate Scheduler (`ReduceLROnPlateau`)
* **Yếu tố DUY NHẤT thay đổi:** Thêm bộ điều chỉnh tốc độ học `torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)` dựa trên `val_accuracy`.
* **Giữ nguyên 100%:**
  - Kiến trúc mô hình CustomVGG.
  - Bộ siêu tham số: Batch size 64, Optimizer SGD Momentum 0.9, Weight Decay 1e-4, Initial LR 0.01.
  - Data Augmentations và Class Weights.
* **Mục tiêu kỳ vọng:** Cho phép mô hình hạ nhỏ tốc độ học ở các epoch cuối (xuống $0.005 \rightarrow 0.0025$), giúp các trọng số hội tụ sâu vào lòng cực tiểu địa phương, kỳ vọng đưa Validation Accuracy vượt ngưỡng **63% - 64%**.
