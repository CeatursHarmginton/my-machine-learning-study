# Lộ trình Computer Vision nâng cao (24 tuần)

**Mục tiêu:** Kết hợp AI Competition (OLP AI, VOAI, IOAI) và AI
Research.\
**Trình độ đầu vào:** Intermediate --- đã biết CNN, Classification,
EfficientNet, ViT, Augmentation, Ensemble và tối ưu training cơ bản.\
**Thời lượng:** Linh hoạt, dự kiến khoảng 6 tháng. Không cần ép tiến độ
nếu chưa nắm vững đầu ra từng tuần.

## 1. Mục tiêu đầu ra

Sau lộ trình, người học hướng tới khả năng:

-   Phân tích bản chất một bài toán Computer Vision mới.
-   Lựa chọn representation, loss, sampling strategy và evaluation
    metric phù hợp.
-   Xây dựng pipeline thực nghiệm bằng PyTorch.
-   Đọc, tái hiện và đánh giá paper.
-   Thiết kế ablation study, phân tích lỗi và đề xuất cải tiến có cơ sở.
-   Xử lý các bài toán vượt ra ngoài Classification, Object Detection và
    Segmentation cơ bản.

## 2. Phương pháp học và phân bổ thời gian

Phân bổ tham khảo cho mỗi tuần:

  Hoạt động                   Tỷ lệ
  ------------------------- -------
  Lý thuyết và toán             20%
  Đọc paper                     20%
  Implement bằng PyTorch        35%
  Experiment và phân tích       25%

Quy tắc học:

1.  **Understand:** Xác định bài toán, giả định và giới hạn của phương
    pháp cũ.
2.  **Implement:** Tự cài đặt phiên bản nhỏ trước khi dùng thư viện hoặc
    pretrained model.
3.  **Experiment:** Thay đổi từng yếu tố có kiểm soát.
4.  **Analyze:** Đánh giá lỗi, generalization, chi phí inference và độ
    ổn định.

Không chuyển module chỉ vì hết tuần. Nếu chưa tự giải thích và triển
khai được ý tưởng cốt lõi, hãy kéo dài thời gian.

## 3. Tổng quan 8 giai đoạn

  Giai đoạn       Tuần Chủ đề
  ----------- -------- -------------------------------------------
  1               1--3 Representation Learning & Metric Learning
  2               4--6 Domain Generalization & Robustness
  3               7--9 Self-Supervised Learning
  4             10--12 Video Understanding & Temporal Vision
  5             13--15 Geometric Vision & 3D Computer Vision
  6             16--18 Vision-Language & Multimodal Learning
  7             19--21 Few-shot Learning & Anomaly Detection
  8             22--24 Research & Competition Challenge

------------------------------------------------------------------------

# PHẦN I --- LỘ TRÌNH CHI TIẾT

## Module 1 --- Representation Learning & Metric Learning

**Thời gian:** Tuần 1--3\
**Ưu tiên:** Rất cao

### Mục tiêu

Chuyển từ tư duy phân loại từng ảnh sang học embedding, đo độ tương đồng
và truy xuất ảnh tương ứng.

### Tuần 1 --- Embedding và Siamese Network

-   Embedding space, vector representation.
-   Euclidean distance và cosine similarity.
-   Positive pairs và negative pairs.
-   Contrastive Loss.
-   Thực hành: xây dựng Siamese Network bằng PyTorch; tự implement
    Contrastive Loss.
-   Dataset khởi đầu: MNIST hoặc Fashion-MNIST.

### Tuần 2 --- Triplet Learning

-   Anchor, positive, negative.
-   Triplet Loss.
-   Hard Negative Mining và Semi-hard Negative Mining.
-   Thực hành: xây dựng Triplet Network; so sánh các chiến lược mining.

### Tuần 3 --- Image Retrieval

-   Image retrieval pipeline.
-   Nearest Neighbor Search.
-   FAISS.
-   Recall@K, mean Average Precision (mAP).
-   GeM Pooling và feature aggregation.
-   Thực hành: xây dựng image retrieval engine từ embedding đã học.

### Paper trọng tâm

1.  [FaceNet (2015)](https://arxiv.org/abs/1503.03832) --- Triplet Loss.
2.  [Deep Metric Learning using Triplet Network
    (2015)](https://arxiv.org/abs/1412.6622).
3.  [Supervised Contrastive Learning
    (2020)](https://arxiv.org/abs/2004.11362).

### Project 1 --- Cross-Season Image Retrieval

**Bài toán:** Dùng một ảnh mùa hè để tìm ảnh tương ứng về vị trí trong
database ảnh mùa đông.

Thử nghiệm ít nhất ba phương pháp:

1.  Pretrained ResNet/EfficientNet + cosine similarity.
2.  Siamese Network + Contrastive Loss.
3.  Triplet Network + Hard Negative Mining.

Sau này bổ sung DINOv2 để làm phương pháp so sánh.

**Dataset gợi ý:** Nordland; có thể tham khảo thêm Oxford RobotCar nếu
dữ liệu truy cập được.

**Đánh giá:**

-   Recall@1, Recall@5, Recall@10.
-   mAP.
-   Khả năng chịu thay đổi mùa và điều kiện môi trường.
-   Tốc độ truy vấn và dung lượng embedding.

**Kiểm soát leakage:** Chia dữ liệu theo vị trí/tuyến đường hợp lý;
tránh để các frame gần như trùng nhau xuất hiện đồng thời trong train và
test.

------------------------------------------------------------------------

## Module 2 --- Domain Generalization & Robustness

**Thời gian:** Tuần 4--6\
**Ưu tiên:** Rất cao cho AI competition

### Mục tiêu

Hiểu và xử lý trường hợp phân phối dữ liệu test khác với dữ liệu
training.

### Các khái niệm cần phân biệt

  -----------------------------------------------------------------------
  Khái niệm                           Ý nghĩa
  ----------------------------------- -----------------------------------
  Domain Shift                        Phân phối dữ liệu thay đổi

  Domain Adaptation                   Thích nghi với target domain,
                                      thường có sử dụng thông tin target

  Domain Generalization               Tổng quát hóa sang domain chưa thấy
                                      trong training

  Test-Time Adaptation                Thích nghi ở thời điểm inference
  -----------------------------------------------------------------------

### Tuần 4 --- Distribution Shift

-   Covariate shift, label shift, concept shift.
-   Out-of-Distribution (OOD) detection.
-   Corruption và robustness benchmark.
-   Thực hành: tạo các domain bằng thay đổi ánh sáng, màu sắc, thời
    tiết, blur và noise.

### Tuần 5 --- Domain Generalization

-   Domain-invariant representation.
-   Feature alignment.
-   Domain-Adversarial Training.
-   CORAL.
-   Thực hành: so sánh ERM, CORAL và DANN trên các domain khác nhau.

### Tuần 6 --- Adaptation và Robustness

-   Test-Time Adaptation.
-   Entropy minimization.
-   Robust augmentation.
-   Thực hành: so sánh ERM, DG và TTA riêng biệt; ghi rõ giả định dữ
    liệu được phép sử dụng.

### Paper trọng tâm

1.  [Domain Generalization: A Survey
    (2021)](https://arxiv.org/abs/2103.02503).
2.  [Domain-Adversarial Training of Neural Networks
    (2016)](https://arxiv.org/abs/1505.07818).
3.  [Deep CORAL (2016)](https://arxiv.org/abs/1607.01719).
4.  [Test-Time Training (2019)](https://arxiv.org/abs/1909.13231).

### Project 2 --- Cross-Domain Generalization Benchmark

**Dataset gợi ý:**

-   PACS.
-   Office-Home.
-   DomainNet.
-   DomainBed: https://github.com/facebookresearch/DomainBed

**Thiết kế:**

-   Train trên source domains.
-   Validation theo protocol rõ ràng.
-   Test trên một domain không được dùng trong training hoặc model
    selection.

**So sánh:**

1.  ERM baseline.
2.  ERM + augmentation.
3.  CORAL hoặc DANN.
4.  Một phương pháp DG nâng cao.
5.  TTA --- đánh giá riêng, không gộp kết quả với DG.

**Đầu ra:** Báo cáo domain gap, kết quả theo từng domain và phân tích
phương pháp nào cải thiện generalization thực sự.

------------------------------------------------------------------------

## Module 3 --- Self-Supervised Learning & Foundation Models

**Thời gian:** Tuần 7--9\
**Ưu tiên:** Rất cao cho AI research

### Mục tiêu

Học cách khai thác dữ liệu không nhãn và xây dựng visual representation
có thể tái sử dụng.

### Tuần 7 --- Contrastive Self-Supervised Learning

-   Self-supervised learning.
-   Positive/negative pairs.
-   SimCLR.
-   MoCo.
-   Temperature và batch size.
-   Thực hành: implement SimCLR phiên bản nhỏ.

### Tuần 8 --- Self-Distillation

-   BYOL.
-   Teacher--student learning.
-   EMA teacher.
-   DINO.
-   Thực hành: tìm hiểu và implement phiên bản DINO thu nhỏ.

### Tuần 9 --- Masked Image Modeling và Feature Evaluation

-   MAE.
-   Vision Transformer.
-   Foundation Models.
-   Frozen feature extraction.
-   Linear probing và k-NN evaluation.
-   Thực hành: so sánh feature pretrained với feature tự học.

### Paper trọng tâm

1.  [SimCLR (2020)](https://arxiv.org/abs/2002.05709).
2.  [MoCo (2020)](https://arxiv.org/abs/1911.05722).
3.  [DINO (2021)](https://arxiv.org/abs/2104.14294).
4.  [MAE (2021)](https://arxiv.org/abs/2111.06377).
5.  [DINOv2 (2023)](https://arxiv.org/abs/2304.07193).

### Project 3 --- Label-Efficient Visual Recognition

**Mục tiêu:** Xây dựng mô hình có hiệu quả khi chỉ có ít dữ liệu được
gán nhãn.

Thực hiện:

-   Dùng CIFAR-10 hoặc một subset phù hợp.
-   Train SimCLR trên dữ liệu không nhãn.
-   Trích xuất feature từ pretrained DINOv2.
-   So sánh linear probing và k-NN classification.
-   Đánh giá với 1%, 10% và 100% nhãn.
-   Kiểm tra khả năng chuyển domain nếu có điều kiện.

------------------------------------------------------------------------

## Module 4 --- Video Understanding & Temporal Vision

**Thời gian:** Tuần 10--12

### Tuần 10 --- Optical Flow

-   Motion estimation.
-   Lucas--Kanade Optical Flow.
-   Dense và sparse optical flow.
-   Thực hành: tính optical flow giữa các frame bằng OpenCV; tìm hiểu
    cách đánh giá flow.

### Tuần 11 --- Object Tracking

-   Data association.
-   Kalman Filter.
-   SORT và DeepSORT.
-   Tracking metrics.
-   Thực hành: xây dựng pipeline multi-object tracking.

### Tuần 12 --- Video Representation

-   Temporal information.
-   3D CNN.
-   Temporal Attention.
-   Video Transformer.
-   Action Recognition.
-   Thực hành: video classification với clip ngắn.

### Paper trọng tâm

1.  [RAFT (2020)](https://arxiv.org/abs/2003.12039).
2.  [SORT (2016)](https://arxiv.org/abs/1602.00763).
3.  [TimeSformer (2021)](https://arxiv.org/abs/2102.05095).

**Checkpoint:** Input là video; output là các đối tượng được tracking
xuyên suốt các frame cùng trajectory của chúng.

------------------------------------------------------------------------

## Module 5 --- Geometric Vision & 3D Computer Vision

**Thời gian:** Tuần 13--15

### Mục tiêu

Kết hợp Linear Algebra, Geometry, Optimization và Computer Vision truyền
thống.

### Tuần 13 --- Camera Geometry

-   Pinhole camera model.
-   Intrinsic và extrinsic parameters.
-   Camera calibration.
-   Homography.
-   Thực hành: camera calibration và image warping.

### Tuần 14 --- Multi-view Geometry

-   Feature detection và matching.
-   Fundamental Matrix.
-   Essential Matrix.
-   Epipolar Geometry.
-   Stereo Vision.
-   Triangulation.
-   Thực hành: stereo depth estimation.

### Tuần 15 --- 3D Reconstruction

-   Structure from Motion.
-   SLAM.
-   Point Cloud.
-   NeRF.
-   3D Gaussian Splatting.
-   Thực hành: reconstruct một scene nhỏ.

### Tài liệu và paper

1.  [Multiple View Geometry in Computer Vision --- Hartley &
    Zisserman](https://www.robots.ox.ac.uk/~vgg/hzbook/).
2.  [NeRF (2020)](https://arxiv.org/abs/2003.08934).
3.  [3D Gaussian Splatting (2023)](https://arxiv.org/abs/2308.04079).

### Bài thực hành tổng hợp

Xây dựng pipeline hai ảnh:

1.  Load hai ảnh từ hai góc nhìn.
2.  Detect và match keypoints.
3.  Estimate Fundamental Matrix.
4.  Recover camera pose.
5.  Triangulate 3D points.
6.  Visualize point cloud.

Lưu ý: Không phải bài toán Computer Vision nào cũng cần neural network.
Hình học và thuật toán truyền thống vẫn rất quan trọng.

------------------------------------------------------------------------

## Module 6 --- Vision-Language & Multimodal Learning

**Thời gian:** Tuần 16--18

### Tuần 16 --- CLIP

-   Image-text contrastive learning.
-   Joint embedding space.
-   Contrastive loss.
-   Thực hành: implement CLIP loss và thử trên batch ảnh-văn bản nhỏ.

### Tuần 17 --- Zero-shot và Open Vocabulary

-   Zero-shot learning.
-   Few-shot prompting.
-   Open-vocabulary recognition.
-   Open-vocabulary detection.
-   Thực hành: zero-shot image classification.

### Tuần 18 --- Grounding và Vision-Language

-   Image-text retrieval.
-   Visual grounding.
-   Visual Question Answering.
-   Vision-language models.
-   Thực hành: xây dựng image-text retrieval.

### Paper trọng tâm

1.  [CLIP (2021)](https://arxiv.org/abs/2103.00020).
2.  [BLIP (2022)](https://arxiv.org/abs/2201.12086).
3.  [Grounding DINO (2023)](https://arxiv.org/abs/2303.05499).

### Project

Xây dựng hệ thống tìm kiếm ảnh bằng ngôn ngữ tự nhiên.

Ví dụ input: `A red car parked beside a building`.

Output: Top-K ảnh gần nhất theo embedding ảnh-văn bản.

Đánh giá zero-shot performance trên các class không xuất hiện trong dữ
liệu training.

------------------------------------------------------------------------

## Module 7 --- Few-shot Learning & Anomaly Detection

**Thời gian:** Tuần 19--21

### Tuần 19 --- Few-shot Learning

-   N-way K-shot classification.
-   Prototypical Networks.
-   Matching Networks.
-   Meta-learning.
-   Thực hành: implement Prototypical Networks.

### Tuần 20 --- Anomaly Detection

-   Normal-only training.
-   Feature memory bank.
-   Nearest-neighbor anomaly detection.
-   Anomaly localization.
-   Thực hành: k-NN anomaly detection.

### Tuần 21 --- Advanced Anomaly Detection

-   PatchCore.
-   PaDiM.
-   Image-level và pixel-level evaluation.
-   Thực hành: benchmark anomaly detection.

### Paper trọng tâm

1.  [Prototypical Networks (2017)](https://arxiv.org/abs/1703.05175).
2.  [Matching Networks (2016)](https://arxiv.org/abs/1606.04080).
3.  [PatchCore (2022)](https://arxiv.org/abs/2106.08265).

### Project 4 --- Industrial Anomaly Detection

**Dataset:** [MVTec
AD](https://www.mvtec.com/company/research/datasets/mvtec-ad).

Yêu cầu:

-   Training chủ yếu bằng dữ liệu bình thường.
-   Phát hiện ảnh bất thường.
-   Xác định vùng lỗi.
-   Đánh giá image-level AUROC và pixel-level AUROC.
-   Phân tích false positives và false negatives.

------------------------------------------------------------------------

## Module 8 --- Research & Competition Challenge

**Thời gian:** Tuần 22--24

### Tuần 22 --- Paper Reproduction

-   Chọn một paper trong các module trước.
-   Đọc kỹ methodology.
-   Tái hiện phương pháp chính.
-   So sánh với kết quả gốc khi điều kiện cho phép.
-   Ghi lại khác biệt về dataset, preprocessing, training và hardware.

### Tuần 23 --- Independent Experiment

Tự đề xuất một thay đổi có cơ sở:

-   Thay đổi sampling strategy.
-   Thay đổi loss function.
-   Kết hợp representation.
-   Thử augmentation mới.
-   Thay đổi positive/negative mining.

Bắt buộc có ablation study để xác định thành phần nào thực sự có tác
dụng.

### Tuần 24 --- Open-ended AI Challenge

Tự thiết kế một bài thi CV nhỏ:

-   Không cho biết trước kiến trúc phù hợp.
-   Không cho biết trước loss function.
-   Có domain shift hoặc class imbalance.
-   Có giới hạn tài nguyên.
-   Có validation và test được tách biệt đúng cách.

Đầu ra:

1.  Problem analysis.
2.  Baseline.
3.  Proposed method.
4.  Experiment log.
5.  Ablation study.
6.  Error analysis.
7.  Final model và technical report.

------------------------------------------------------------------------

# PHẦN II --- TÀI LIỆU HỌC TẬP

## 4. Khóa học đại học

### Stanford CS231n

**Vai trò:** Tài liệu chính để củng cố và mở rộng Deep Learning for
Computer Vision.

-   [Official course website](https://cs231n.stanford.edu/)
-   [Assignments](https://cs231n.stanford.edu/)

Nên ưu tiên các nội dung về Transformer, Self-Supervised Learning,
Video, 3D Vision và Vision-Language.

### Stanford CS231A

**Vai trò:** Tài liệu chính cho Geometric Vision và 3D Computer Vision.

-   [Official course website](https://web.stanford.edu/class/cs231a/)
-   [Course resources](https://web.stanford.edu/class/cs231a/)

### MIT 6.S191

**Vai trò:** Củng cố Deep Learning và mở rộng góc nhìn về các ứng dụng
AI.

-   [Official website](https://introtodeeplearning.com/)

## 5. Nguồn theo dõi paper

  ---------------------------------------------------------------------------------
  Nguồn                                         Mục đích
  --------------------------------------------- -----------------------------------
  [arXiv                                        Paper mới về Computer Vision
  cs.CV](https://arxiv.org/list/cs.CV/recent)   

  [CVF Open                                     CVPR, ICCV, ECCV
  Access](https://openaccess.thecvf.com/)       

  [Papers with                                  Paper, benchmark và implementation
  Code](https://paperswithcode.com/)            

  [Hugging Face                                 Theo dõi nghiên cứu nổi bật
  Papers](https://huggingface.co/papers)        

  [GitHub](https://github.com/)                 Tìm implementation và kiểm tra
                                                reproducibility
  ---------------------------------------------------------------------------------

Mục tiêu đọc: 1--2 paper mỗi tuần, ưu tiên đọc sâu thay vì chạy theo số
lượng.

## 6. Bộ công cụ PyTorch

  Công cụ            Ứng dụng
  ------------------ ---------------------------------------
  PyTorch            Training, custom loss, custom model
  torchvision        Image processing và pretrained models
  timm               Backbone và pretrained weights
  FAISS              Vector similarity search
  OpenCV             Image processing và geometry
  Kornia             Differentiable computer vision
  Albumentations     Advanced augmentation
  scikit-learn       Metrics và clustering
  Weights & Biases   Theo dõi experiment

Tài liệu:

-   [PyTorch Tutorials](https://pytorch.org/tutorials/)
-   [timm Documentation](https://huggingface.co/docs/timm/)
-   [FAISS](https://github.com/facebookresearch/faiss)
-   [Kornia](https://kornia.readthedocs.io/)
-   [Albumentations](https://albumentations.ai/docs/)

## 7. Chiến lược theo phần cứng

Với GPU RTX 3050 Laptop 4GB VRAM:

  Công việc           Chiến lược
  ------------------- --------------------------------------
  SimCLR              ResNet18, batch nhỏ, mixed precision
  DINO                Implement mô hình ViT nhỏ
  DINOv2              Frozen feature extraction
  CLIP                Pretrained model, freeze encoder
  3D Vision           OpenCV, NumPy, PyTorch
  Anomaly Detection   PatchCore với feature memory bank
  Video               Clip ngắn, giảm resolution

Dùng Google Colab cho các thí nghiệm cần GPU mạnh hơn. Ưu tiên frozen
feature extraction khi mục tiêu là nghiên cứu downstream task, không cần
train foundation model từ đầu.

------------------------------------------------------------------------

# PHẦN III --- ĐÁNH GIÁ NĂNG LỰC

## 8. Thang đánh giá cuối lộ trình

  Năng lực                                   Trọng số
  ---------------------------------------- ----------
  Hiểu bản chất và toán học của bài toán          15%
  Lựa chọn phương pháp phù hợp                    15%
  Implement bằng PyTorch                          15%
  Thiết kế experiment                             15%
  Generalization và robustness                    15%
  Đọc và tái hiện paper                           15%
  Phân tích lỗi, đề xuất cải tiến                 10%
  **Tổng**                                   **100%**

### Các mức năng lực mục tiêu

  ------------------------------------------------------------------------
                          Điểm Mức độ                Diễn giải
  ---------------------------- --------------------- ---------------------
                      Dưới 50% Cần củng cố           Biết sử dụng một số
                                                     phương pháp nhưng
                                                     chưa hiểu sâu

                       50--69% Practitioner          Giải quyết được các
                                                     bài toán CV quen
                                                     thuộc

                       70--84% Advanced Practitioner Thiết kế pipeline,
                                                     experiment và giải
                                                     quyết bài toán mới

                      85--100% Research-oriented     Tái hiện nghiên cứu,
                                                     phân tích giới hạn và
                                                     đề xuất cải tiến có
                                                     bằng chứng
  ------------------------------------------------------------------------

Đây là thang đánh giá nội bộ đề xuất, không phải chuẩn đánh giá học
thuật chính thức.

## 9. Dataset tham khảo

  ------------------------------------------------------------------------------------------------------------------------
  Dataset                 Chủ đề                      Nguồn
  ----------------------- --------------------------- --------------------------------------------------------------------
  Nordland                Visual Place Recognition    Tìm nguồn dataset và protocol phù hợp

  PACS                    Domain Generalization       [DomainBed](https://github.com/facebookresearch/DomainBed)

  Office-Home             Domain                      [Official project](https://hemanthdv.org/OfficeHome-Dataset/)
                          Adaptation/Generalization   

  MVTec AD                Anomaly Detection           [Official
                                                      dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

  KITTI                   3D Vision, Optical Flow     [KITTI Benchmark](https://www.cvlibs.net/datasets/kitti/)
  ------------------------------------------------------------------------------------------------------------------------

Lưu ý: Kiểm tra quyền truy cập, license, phiên bản dataset và protocol
đánh giá trước khi bắt đầu. Không để test data tham gia training hoặc
model selection nếu luật cuộc thi không cho phép.

------------------------------------------------------------------------

# PHẦN IV --- TƯ DUY GIẢI BÀI TOÁN AI COMPETITION

## 10. Quy trình phân tích đề CV mới

### Bước 1 --- Problem Definition

-   Input là gì?
-   Output cần dự đoán là gì?
-   Đây là classification, retrieval, matching, localization hay một bài
    toán khác?
-   Có yếu tố thời gian hoặc không gian không?

### Bước 2 --- Data Analysis

-   Có nhãn không?
-   Có class imbalance không?
-   Có domain shift không?
-   Train/validation/test có nguy cơ leakage không?
-   Có dữ liệu trùng lặp hoặc gần trùng lặp không?

### Bước 3 --- Learning Strategy

-   Supervised learning?
-   Self-supervised learning?
-   Metric learning?
-   Transfer learning?
-   Domain generalization?

### Bước 4 --- Evaluation

-   Metric nào phù hợp?
-   Có constraint về inference time và memory không?
-   Validation có phản ánh đúng test distribution không?
-   Có cần đánh giá robustness không?

### Bước 5 --- Experiment

Baseline → Experiment → Ablation → Optimization → Final Solution.

Không thay đổi nhiều yếu tố cùng lúc nếu muốn xác định nguyên nhân cải
thiện.

## 11. Cấu trúc repository đề xuất

``` text
Advanced-Computer-Vision/
│
├── 01_metric_learning/
│   ├── siamese_network.py
│   ├── contrastive_loss.py
│   ├── triplet_loss.py
│   └── hard_negative_mining.py
│
├── 02_domain_generalization/
├── 03_self_supervised/
├── 04_video_understanding/
├── 05_3d_vision/
├── 06_vision_language/
├── 07_few_shot_anomaly/
│
├── projects/
│   ├── nordland_retrieval/
│   ├── domain_generalization/
│   ├── label_efficient_learning/
│   └── anomaly_detection/
│
└── research_notes/
```

## 12. Template ghi chú paper

Với mỗi paper, ghi lại:

1.  **Problem:** Bài toán tác giả giải quyết.
2.  **Motivation:** Vì sao phương pháp cũ chưa đủ.
3.  **Method:** Ý tưởng, kiến trúc, loss và optimization.
4.  **Experiments:** Dataset, baseline, metrics và kết quả.
5.  **Limitations:** Giới hạn của phương pháp.
6.  **Reproduction:** Những gì đã tự implement.
7.  **Ideas:** Hướng cải tiến có thể kiểm chứng.

------------------------------------------------------------------------

# PHẦN V --- BẮT ĐẦU NGAY

## Tuần 1 --- Checklist

-   [ ] Đọc FaceNet.
-   [ ] Hiểu Embedding Space và Cosine Similarity.
-   [ ] Tìm hiểu Positive Pairs và Negative Pairs.
-   [ ] Tự implement Contrastive Loss bằng PyTorch.
-   [ ] Xây dựng Siamese Network.
-   [ ] Train thử trên MNIST hoặc Fashion-MNIST.
-   [ ] Visualize embedding space.
-   [ ] Viết báo cáo ngắn về sự thay đổi khoảng cách embedding trong quá
    trình training.

### Câu hỏi tự kiểm tra

1.  Vì sao classification loss không trực tiếp tối ưu khoảng cách giữa
    hai ảnh tương ứng?
2.  Vì sao Contrastive Loss cần positive và negative pairs?
3.  Nếu tất cả negative samples đều quá dễ, mô hình sẽ gặp vấn đề gì?
4.  Hard Negative Mining cải thiện training như thế nào và có rủi ro gì?
5.  Vì sao Recall@K phù hợp với image retrieval hơn accuracy thông
    thường?

**Điều kiện hoàn thành tuần 1:** Có code chạy được, loss được tự
implement, embedding được trực quan hóa và có giải thích được kết quả.

------------------------------------------------------------------------

# KẾT LUẬN

Mục tiêu của 24 tuần không phải là học thuộc càng nhiều kiến trúc càng
tốt. Mục tiêu là phát triển năng lực phân tích bài toán, lựa chọn phương
pháp, thiết kế thí nghiệm và đánh giá bằng chứng.

Hai hướng nên đào sâu sau lộ trình:

1.  **Visual Representation Learning:** Metric Learning, Self-Supervised
    Learning, Foundation Models và Image Retrieval.
2.  **Domain Generalization:** Robustness, Distribution Shift, Domain
    Adaptation và Test-Time Adaptation.

Đây là hai hướng có tính bổ trợ cao, phù hợp để phát triển cả năng lực
AI competition lẫn nghiên cứu Computer Vision.
