# Lộ Trình Master Computer Vision Từ Bản Chất Gốc (First-Principles) — Chuẩn Thi Đấu OLP AI, VOAI & IOAI

**Triết lý cốt lõi của lộ trình:**
> Trong đề thi thật của **OLP AI, VOAI và IOAI**, Ban Tổ Chức **SẼ KHÔNG** cho sẵn tên bài báo (như SINet, PraNet, CosPlace), không giảng sẵn lý thuyết, và không đưa sẵn khung code của mô hình mục tiêu như trong các bộ đề do AI soạn.
> Đề thi thật chỉ cho bạn: **(1) Một tập dữ liệu thô đầy cạm bẫy + (2) Yêu cầu đầu ra + (3) Công thức chấm điểm (Metric) + (4) Giới hạn phần cứng/thời gian.**
> Vì vậy, nếu học theo kiểu *"sưu tầm tên các kiến trúc/paper"* thì khi gặp bài mới sẽ đứng hình. Lộ trình này được thiết kế lại hoàn toàn để dẫn bạn đi từ **5 NGUYÊN LÝ BẢN CHẤT (First Principles)** của Thị giác Máy tính — giúp bạn nhìn vào bất kỳ dữ liệu và bài toán mới nào cũng tự **chẩn đoán được bệnh, tự lắp ghép kiến trúc và tự chế tạo hàm Loss**.

---

## PHẦN I — 5 TRỤ CỘT BẢN CHẤT CỦA COMPUTER VISION (CẦN NẮM TRƯỚC KHI HỌC BẤT KỲ PAPER NÀO)

Mọi bài toán Computer Vision — dù cổ điển hay hiện đại — đều chỉ xoay quanh **5 câu hỏi nguyên lý gốc**. Khi học bất kỳ tuần nào trong 24 tuần, bạn không học thuộc kiến trúc mà phải trả lời được 5 câu hỏi này:

```
[1. Tín hiệu & Inductive Bias] ──► [2. Hình học Không gian Đặc trưng] ──► [3. Động học Gradient & Loss]
                                                                                     │
[5. Tối ưu Quyết định & Metric] ◄── [4. Phân phối Dữ liệu & Chống Học Vẹt] ◄─────────┘
```

### Trụ cột 1: Bản chất Tín hiệu Ảnh & Thiên kiến Quy nạp (Signal Nature & Inductive Bias)
- **Bản chất:** Bức ảnh không phải là một vector vô hồn mà là một lưới tín hiệu 2D/3D có **tính cục bộ không gian (Spatial Locality)**, **tính phân cấp tỷ lệ (Multi-scale Hierarchy)** và các **bất biến vật lý** (dịch chuyển, xoay, thay đổi ánh sáng, biến dạng góc chụp).
- **Tư duy chọn/thiết kế khối mạng (Building Blocks):**
  - Cần bắt **kết cấu cục bộ, biên cạnh sắc nét, ít dữ liệu** $\rightarrow$ Dùng **Tích chập (Convolution)** vì có sẵn *Locality* và *Translation Equivariance* (không cần học lại từ đầu).
  - Cần bắt **vật thể biến thiên kích thước cực mạnh** (lúc rất nhỏ, lúc rất to) $\rightarrow$ Phải có **Multi-scale Pyramid (FPN)**, **Tích chập giãn nở đa nhánh (Dilated/Atrous Conv)** hoặc **Chia nhóm kênh đa tỷ lệ (Res2Net)**.
  - Cần bắt **quan hệ ngữ cảnh xa trên toàn ảnh** hoặc kết hợp đa phương thức (Ảnh + Text) $\rightarrow$ Dùng **Self-Attention / Cross-Attention (ViT)** hoặc **Depthwise Conv kích thước lớn ($7\times 7$) مدرن (ConvNeXt)**.
  - Cần biết **vùng nào quan trọng, vùng nào cần xóa bỏ để tìm chi tiết ẩn** $\rightarrow$ Dùng **Cơ chế Chú ý (Spatial/Channel Attention)** hoặc **Chú ý Đảo ngược ($1 - \text{Sigmoid}(P)$)**.

### Trụ cột 2: Hình học của Không gian Biểu diễn (Geometry of Representation Space)
- **Bản chất:** Mạng Backbone thực chất là một hàm ánh xạ $f_\theta: \mathcal{X} \to \mathcal{Z}$ biến ảnh thành điểm hoặc tập điểm trong không gian $\mathcal{Z}$. Bản chất bài toán quyết định hình học của $\mathcal{Z}$:
  1. **Bài toán Tập đóng (Closed-Set Classification):** Chia không gian $\mathbb{R}^d$ thành $C$ vùng bằng các siêu phẳng tuyến tính ($\mathbf{W}^T \mathbf{z} + \mathbf{b}$). *Điểm yếu:* Sụp đổ khi gặp lớp mới (Open-set/Zero-shot) hoặc khi gặp vật thể lạ ngoài phân phối (OOD).
  2. **Bài toán Đo lường & Truy xuất (Metric Learning / Retrieval / Verification / Few-Shot):** Ép toàn bộ vector đặc trưng nằm trên **Mặt cầu đơn vị (Unit Hypersphere $\|\mathbf{z}\|_2 = 1$)** để khoảng cách Euclid tương đương với góc Cosine ($\|\mathbf{z}_1 - \mathbf{z}_2\|_2^2 = 2 - 2\cos\theta$). Khi đó ta có thể đo khoảng cách giữa bất kỳ lớp mới nào chưa từng thấy khi train!
  3. **Bài toán Dự đoán Dày đặc & Cấu trúc (Dense Prediction: Segmentation / Detection / Counting / Depth):** Giữ nguyên cấu trúc không gian $(H' \times W')$ của bản đồ đặc trưng, kết hợp thông tin **Ngữ nghĩa cấp cao (tầng sâu: "đây là con gì?")** với thông tin **Hình học biên cạnh cấp thấp (tầng nông: "ranh giới nằm ở pixel nào?")**.

### Trụ cột 3: Động học Gradient & Thiết kế Hàm Mất Mát (Loss & Gradient Dynamics)
- **Bản chất:** Mạng nơ-ron không hiểu "ý muốn" của bạn; nó chỉ đi theo vector đạo hàm $\nabla_\theta \mathcal{L} = \sum_{i} \frac{\partial \mathcal{L}_i}{\partial z_i} \frac{\partial z_i}{\partial \theta}$.
- **Tư duy chẩn đoán lỗi từ Đạo hàm:**
  - Nếu **99.9% mẫu (hoặc pixel/anchor) là mẫu dễ/nền**: Đạo hàm tổng $\sum_{i \in \text{easy}} \nabla \mathcal{L}_i$ sẽ nuốt chửng đạo hàm của 0.1% mẫu khó/quan trọng $\rightarrow$ **Giải pháp:** Nhân hệ số điều chế tự động triệt tiêu đạo hàm khi mô hình đã dự đoán đúng (như $(1 - p_t)^\gamma$ của Focal Loss, hoặc Hard Negative Mining).
  - Nếu **ranh giới vật thể bị nhòe**: Các pixel ở ruột vật thể chiếm đa số diện tích nên chi phối toàn bộ BCE/IoU Loss $\rightarrow$ **Giải pháp:** Tạo ma trận trọng số không gian $\mathbf{W}_{ij}$ phạt nặng gấp 5–10 lần tại những nơi có độ biến thiên nhãn cao (vùng biên cạnh).
  - Nếu **học đa nhiệm (Multi-Task) mà 2 nhiệm vụ kéo nhau tụt điểm**: Vector đạo hàm $\mathbf{g}_1$ và $\mathbf{g}_2$ tạo góc tù ($\mathbf{g}_1 \cdot \mathbf{g}_2 < 0$) $\rightarrow$ **Giải pháp:** Cân bằng lại trọng số loss động hoặc chiếu trực giao đạo hàm (PCGrad).

### Trụ cột 4: Phân phối Dữ liệu, Học Vẹt & Khái quát hóa (Data Distribution & Shortcut Prevention)
- **Bản chất:** Mạng sâu cực kỳ "lười biếng" — nó luôn chọn **con đường ngắn nhất (Shortcut / Spurious Correlation)** để giảm Loss trên tập Train (ví dụ: nhận diện con bò dựa vào bãi cỏ xanh, nhận diện bệnh phổi dựa vào ký hiệu bệnh viện trên phim X-quang, copy pixel từ frame kề bên trong video).
- **Tư duy chống học vẹt:**
  - **Thiết kế Validation Split chuẩn xác:** Không bao giờ chia ngẫu nhiên (`random_split`) nếu dữ liệu có tính nhóm (cùng 1 bệnh nhân, cùng 1 đoạn video, cùng 1 vị trí địa lý, hoặc khác miền phong cách). Phải dùng `GroupKFold` / `StratifiedGroupKFold`.
  - **Phá hủy Shortcut bằng Augmentation / Masking có chủ đích:** Nếu mô hình nhìn màu sắc giả mạo $\rightarrow$ trộn màu/chuyển xám/chuẩn hóa tần số; nếu mô hình chỉ nhìn 1 bộ phận nhỏ $\rightarrow$ dùng `CutMix`/`Mixup`/GridMask; nếu video bị copy frame kề $\rightarrow$ che dạng ống xuyên thời gian (`Tube Masking`).

### Trụ cột 5: Suy luận, Hiệu chỉnh Xác suất & Hậu xử lý theo Độ đo (Inference & Metric Alignment)
- **Bản chất:** Huấn luyện tối ưu hàm Loss khả vi (như Cross-Entropy), nhưng cuộc thi lại chấm điểm bằng **độ đo rời rạc không khả vi** (`Macro-F1`, `Balanced Accuracy`, `mAP`, `IoU`, `Recall@K`, `AUROC`, `FPR95`).
- **Tư duy tối ưu hóa bước cuối (ăn thêm 5–15% điểm mà không cần train lại):**
  - Tối ưu ngưỡng cắt (`Threshold Tuning`) cho từng lớp trên tập Validation khi chấm `Macro-F1` hoặc `Balanced Accuracy`.
  - Thay xác suất `Softmax` bằng `Free Energy Score` hoặc `Cosine Distance` tới tâm cụm khi cần bắt mẫu lạ (OOD/Anomaly).
  - Dùng `Test-Time Augmentation (TTA)`, `Weighted Boxes Fusion (WBF)` (cho Detection), `Morphological Closing / Connected Components Filtering` (xóa nhiễu đốm nhỏ cho Segmentation), hoặc `Query Expansion / k-iprocal Re-ranking` (cho Retrieval).

---

## PHẦN II — LỘ TRÌNH 24 TUẦN HỌC THEO BẢN CHẤT (6 GIAI ĐOẠN TƯ DUY)

Thay vì chia vụn theo từng bài báo như các đề AI soạn, 24 tuần dưới đây được cấu trúc lại thành **6 Giai đoạn Tư duy nối tiếp nhau**, đi từ **Nền tảng Biểu diễn & Thực nghiệm** $\rightarrow$ **Không gian Khoảng cách** $\rightarrow$ **Cấu trúc Không gian Dày đặc (Dense Prediction)** $\rightarrow$ **Khái quát hóa Miền & Tập mở** $\rightarrow$ **Tự giám sát, Đa phương thức & Sinh ảnh (Foundation/Generative)** $\rightarrow$ **Chuỗi Thời gian & Thực chiến Giải Đề Mở**.

*(Các đề thi từ 01 đến 12 của thầy bạn được đưa về đúng vai trò là **Bài tập Case Study** để bạn mổ xẻ xem một phương pháp cụ thể vận dụng nguyên lý gốc như thế nào, kèm bài tập biến thể để rèn phản xạ tự nghĩ).*

---

### GIAI ĐOẠN 1 (Tuần 1–4): Giải Phẫu Mạng Thị Giác, Động Học Huấn Luyện & Phân Loại Thực Chiến (FGVC / Imbalance / Multi-Task)

> **Mục tiêu bản chất:** Hiểu tường tận chuyện gì xảy ra bên trong các tầng của CNN và Vision Transformer; làm chủ cách thiết kế pipeline chống học vẹt, xử lý dữ liệu lệch, nhãn nhiễu, phân loại siêu chi tiết (FGVC) và đa nhiệm (Multi-Task).

#### Tuần 1 — Giải phẫu Inductive Bias: Từ ResNet, Res2Net đến ViT & ConvNeXt
- **Bản chất cần hiểu sâu:**
  - So sánh **Receptive Field (Trường thụ cảm hiệu dụng)** giữa tích chập $3\times 3$ xếp chồng, tích chập giãn nở (Dilated Conv), tích chập phân tách chiều sâu $7\times 7$ (Depthwise Separable Conv), và Self-Attention.
  - Tại sao `BatchNorm` hoạt động tốt khi batch lớn và đồng nhất, nhưng lại sụp đổ khi batch nhỏ hoặc khi có Domain Shift? Tại sao `LayerNorm` lại thống trị trong ViT và `ConvNeXt`?
- **Thực hành từ gốc (PyTorch):**
  - Tự viết và trực quan hóa **Effective Receptive Field (ERF)** và **Grad-CAM / Attention Rollout** để tận mắt nhìn xem mô hình đang nhìn vào đâu trên ảnh.
  - Tự cài đặt khối `Res2NetBottleneck` (chia nhóm kênh đa tỷ lệ) và `ConvNeXtBlock` ($7\times 7$ Depthwise Conv + Inverted Bottleneck + `LayerNorm2d`).

#### Tuần 2 — Bản chất của Hàm Mất Mát Phân Loại, Mất cân bằng Lớp & Nhãn Nhiễu
- **Bản chất cần hiểu sâu:**
  - Giải tích đạo hàm của `Cross-Entropy`, `Label Smoothing`, `Symmetric Cross-Entropy` (chống nhãn nhiễu), `Focal Loss` và `Class-Balanced Loss` (dựa trên Effective Number of Samples $E_n = \frac{1 - \beta^n}{1 - \beta}$).
  - **Sự khác biệt giữa Điều chỉnh Trọng số Loss (Loss Re-weighting), Lấy mẫu lại (Resampling), và Hiệu chỉnh Logit sau huấn luyện (Logit Adjustment / Post-hoc Thresholding):**
    $$\text{Logit Adjustment:} \quad \tilde{z}_y = z_y - \tau \log \pi_y$$
    *(Hiểu vì sao Logit Adjustment giữ nguyên chất lượng học đặc trưng của backbone tốt hơn nhiều so với Oversampling ngây thơ!).*
- **Thực hành từ gốc:**
  - Cho một tập dữ liệu ảnh có tỷ lệ lệch $1:100$ giữa các lớp và $15\%$ nhãn bị gán sai ngẫu nhiên. Tự thiết kế pipeline đạt điểm `Macro-F1` và `Balanced Accuracy` cao nhất.

#### Tuần 3 — Phân loại Siêu chi tiết (FGVC) & Cơ chế Phối trộn Dữ liệu (Mixup / CutMix)
- **Bản chất cần hiểu sâu:**
  - Khi hai lớp có hình dáng tổng thể giống hệt nhau (ví dụ: hai loài lá cây, hai dòng xe hơi, tranh cổ thật vs giả trong đề thi **IOAI 2025**), mạng CNN thường bị **Overfitting vào 1 vùng nhỏ có độ phân biệt cao nhất** và bỏ qua các vùng còn lại.
  - Tại sao `CutMix`, `Mixup`, **High-Resolution Fine-tuning**, và **Bilinear Pooling / Part-Attention** buộc mạng phải gom đủ bằng chứng trên toàn bộ bức ảnh?
- **Thực hành & Case Study:**
  - Tự code `MixupCutMixCollator` và hàm `SoftTargetCrossEntropy`.
  - *Đối chiếu Case Study:* Xem cách **Đề 08** vận dụng ConvNeXt + Mixup/CutMix cho bài toán phân loại 200 loài dược liệu.

#### Tuần 4 — Học Đa Nhiệm (Multi-Task Learning), Cấu trúc Phân cấp & Ràng buộc Logic
- **Bản chất cần hiểu sâu:**
  - Khi một mô hình phải dự đoán nhiều đầu ra cùng lúc (ví dụ: vừa phân loại nhóm lớn, vừa phân loại nhóm nhỏ; vừa phân loại vừa hồi quy tọa độ), làm sao thiết kế **Shared Representation** mà không bị hiện tượng **Negative Transfer / Xung đột Gradient ($\mathbf{g}_1 \cdot \mathbf{g}_2 < 0$)**?
  - **Uncertainty Weighting (Kendall et al., CVPR 2018):** Tự động học trọng số cân bằng giữa các hàm loss đa nhiệm dựa trên phương sai bất định đồng phương sai ($\sigma_1, \sigma_2$):
    $$\mathcal{L}_{\text{MTL}} = \frac{1}{2\sigma_1^2} \mathcal{L}_1 + \frac{1}{2\sigma_2^2} \mathcal{L}_2 + \log(\sigma_1 \sigma_2)$$
  - **Chiếu trực giao đạo hàm (PCGrad)** và **Hậu xử lý Nhất quán Phân cấp (Hierarchical Probability Marginalization)**.

---

### GIAI ĐOẠN 2 (Tuần 5–7): Hình Học Không Gian Nhúng, Metric Learning & Truy Xuất Quy Mô Lớn

> **Mục tiêu bản chất:** Chuyển tư duy từ *"gán nhãn vào các hộp cố định"* sang *"kiến tạo một không gian hình học nơi khoảng cách phản ánh đúng bản chất tương đồng ngữ nghĩa"*. Giúp bạn tự giải được mọi bài toán: Nhận dạng khuôn mặt/chữ ký, khớp biểu tượng (Icon Matching - IOAI 2025), tìm kiếm sản phẩm, định vị địa lý (VPR), và Few-shot Learning.

#### Tuần 5 — Hai Trường Phái Metric Learning: Pairwise/Triplet vs Proxy/Angular Margin
- **Bản chất cần hiểu sâu:**
  - **Trường phái 1 — So sánh Mẫu-với-Mẫu (Contrastive, Triplet, Multi-Similarity, SupCon Loss):**
    - Tại sao nếu chọn cặp âm ngẫu nhiên, sau 3 epoch đạo hàm bằng $0$?
    - Phân tích toán học của **Hard Negative** ($d(a,n) < d(a,p)$: dễ bị sụp đổ mô hình nếu có nhãn nhiễu) vs **Semi-Hard Negative** ($d(a,p) < d(a,n) < d(a,p) + m$: vùng học ổn định nhất).
  - **Trường phái 2 — So sánh Mẫu-với-Tâm Lớp trên Mặt cầu (CosFace, ArcFace, Sub-center ArcFace):**
    - Tại sao thêm một góc biên $m$ vào $\cos(\theta_y + m)$ (ArcFace) hoặc $\cos(\theta_y) - m$ (CosFace) lại ép các mẫu cùng lớp co cụm cực chặt mà **không cần tốn chi phí tìm mẫu âm khó $\mathcal{O}(N^3)$**?
    - **Tư duy đột phá:** Khi gặp bài toán có biến liên tục (như tọa độ GPS, góc quay, độ tuổi), làm sao rời rạc hóa khôn ngoan thành các lớp không kề nhau để biến bài toán Hồi quy/Retrieval khó thành bài toán Phân loại CosFace dễ hội tụ? *(Đây chính là bản chất gốc đằng sau bài báo CosPlace ở **Đề 02**!).*

#### Tuần 6 — Bản chất của Feature Aggregation: Từ GAP, GeM đến NetVLAD & Multi-scale Token Pooling
- **Bản chất cần hiểu sâu:**
  - Bản đồ đặc trưng đầu ra của Backbone có kích thước $(C, H, W)$, tức là tập hợp của $N = H \times W$ vector cục bộ $\mathbf{x}_i \in \mathbb{R}^C$. Làm sao nén $N$ vector này thành 1 vector duy nhất $\mathbf{v}$?
  - **Global Average Pooling (GAP):** Lấy trung bình cộng $\rightarrow$ bị nhiễu nền làm loãng nếu vật thể nhỏ hoặc nền thay đổi (như tuyết phủ mùa đông).
  - **Generalized Mean Pooling (GeM):** Điều chỉnh lũy thừa $p_k$ khả vi để tập trung vào các đỉnh kích hoạt.
  - **VLAD / NetVLAD (Vector of Locally Aggregated Descriptors):** Thay vì lấy trung bình toàn cục, hãy chia không gian đặc trưng thành $K$ cụm từ vựng thị giác (visual words: "cây cối", "đường ray", "mái nhà", "bầu trời") và **tính tổng độ lệch (residual)** của từng điểm ảnh so với tâm cụm. Sau đó dùng **Intra-normalization** để giảm trọng số của những cụm biến đổi mạnh theo mùa (như cụm "tuyết/lá cây")! *(Hiểu nguyên lý này, bạn tự khắc hiểu vì sao **Đề 01** dùng NetVLAD mà không cần học vẹt!).*

#### Tuần 7 — Tối ưu hóa Truy xuất Thực chiến: FAISS, GPU Search, Query Expansion & Re-ranking
- **Bản chất cần hiểu sâu:**
  - Khi đã có vector nhúng $\mathbf{q}$ và kho ảnh Gallery $\mathbf{G}$, làm sao đẩy điểm `Recall@K` và `mAP` lên mức kịch trần mà không cần train lại mạng?
  - **3 Vũ khí Hậu xử lý Không gian Nhúng (Cực mạnh khi đi thi):**
    1. **$\alpha$-Query Expansion ($\alpha$-QE):** Lấy vector $\mathbf{q}$ cộng với trung bình có trọng số của top-$k$ ảnh giống nhất trong Gallery rồi chuẩn hóa $\ell_2$ và tìm kiếm lại lần 2.
    2. **Database-side Augmentation (DBA):** Làm mượt chính các vector trong Gallery bằng láng giềng của chúng.
    3. **k-Reciprocal Re-ranking (Zhong et al., CVPR 2017):** Hai ảnh chỉ thực sự giống nhau nếu $A$ nằm trong top-$k$ của $B$ **VÀ** $B$ cũng nằm trong top-$k$ của $A$ (khử triệt để các hub/outliers gây dương tính giả).

---

### GIAI ĐOẠN 3 (Tuần 8–11): Bản Chất Dự Đoán Dày Đặc (Dense Prediction) — Segmentation, Detection & Counting

> **Mục tiêu bản chất:** Đây là mảng chiếm tỷ trọng cực lớn trong các đề thi **IOAI (2024–2026)**, **VOAI** và **OLP AI** nhưng lại bị thiếu trong lộ trình cũ của bạn. Mục tiêu là hiểu bản chất cách mạng nơ-ron định vị tọa độ, tách biên đối tượng ngụy trang/y sinh, phát hiện vật thể siêu nhỏ/mất cân bằng và đếm số lượng vật thể.

#### Tuần 8 — Kiến trúc Biểu diễn Đa Tỷ Lệ (U-Net, FPN, BiFPN) & Giải phẫu các Hàm Loss Phân Đoạn
- **Bản chất cần hiểu sâu:**
  - Mâu thuẫn nội tại của CNN: Càng đi sâu qua các lớp Pooling/Stride, mạng càng hiểu rõ ngữ nghĩa ("con gì") nhưng càng **mất thông tin vị trí không gian ("ở đâu")**.
  - Cách giải quyết bằng **Skip Connections (U-Net)**, **Top-Down + Lateral Pathway (FPN)**, và **Atrous Spatial Pyramid Pooling (ASPP / RFB - DeepLab)**.
  - **Bảng phân loại Bản chất Hàm Loss cho Segmentation (Khi nào dùng hàm nào?):**
    - *Distribution-based:* `Weighted Cross-Entropy`, `Focal Loss` (tối ưu từng pixel độc lập).
    - *Region-based:* `Dice Loss`, `IoU / Jaccard Loss`, `Tversky Loss` (tối ưu độ chồng lấn toàn vùng — cực mạnh khi đối tượng chỉ chiếm $1\%$ diện tích ảnh!).
    - *Boundary-based:* **Pixel-Adaptive Edge-Weighted Loss** ($\alpha_{ij} = 1 + \gamma |\text{AvgPool}(G)_{ij} - G_{ij}|$) — tự động phát hiện đường biên bằng bộ lọc trung bình lân cận và nhân hệ số phạt gấp nhiều lần tại biên.

#### Tuần 9 — Xử lý Biên Mờ & Đối tượng Ngụy trang / Y sinh (Coarse-to-Fine & Reverse Attention)
- **Bản chất cần hiểu sâu:**
  - Khi đối tượng có màu sắc và vân bề mặt giống hệt môi trường xung quanh (động vật ngụy trang, polyp nội soi phẳng, vết nứt trên bề mặt kim loại):
    - Nếu giải trực tiếp trong 1 bước: mạng bị nhiễu nền đánh lừa.
    - **Nguyên lý Coarse-to-Fine & Chú ý Đảo ngược (Reverse Attention):**
      1. Bước 1 (Search / Global Saliency): Dùng các tầng sâu nhất dự đoán vùng thô $P_{\text{coarse}}$ chứa vật thể.
      2. Bước 2 (Identification / Boundary Erasure): Tính mặt nạ đảo ngược $A = 1 - \sigma(P_{\text{coarse}})$ để **triệt tiêu vùng nhân bên trong đã biết chắc**, ép các tầng nông tiếp theo chỉ được phép học **phần dư sai số ở ranh giới ($\Delta P$)** rồi cộng dồn vào: $P_{\text{fine}} = P_{\text{coarse}} + \Delta P$.
  - *Đối chiếu Case Study:* Giải phẫu **Đề 03 (SINet)** và **Đề 04 (PraNet)** dưới góc nhìn nguyên lý Coarse-to-Fine residual refinement; thực hành mã hóa **Run-Length Encoding (RLE)** và các độ đo cấu trúc (`mDice`, `mIoU`, `S-measure`, `E-measure`).

#### Tuần 10 — Bản chất Object Detection: Từ Anchor-based (RetinaNet/YOLO) đến Anchor-free (FCOS) & Transformer (DETR)
- **Bản chất cần hiểu sâu:**
  - Ba bài toán con của mọi bộ Object Detector: **(1) Sinh ứng viên (Anchors / Points / Queries)** $\rightarrow$ **(2) Gán nhãn Dương/Âm (Label Assignment / Bipartite Matching)** $\rightarrow$ **(3) Tối ưu Classification Loss + Box Regression Loss (GIoU / DIoU / CIoU)**.
  - Tại sao One-Stage Detector bị mất cân bằng âm/dương $1:10.000$ và tại sao **Focal Loss + Class Prior Bias Init ($b = -\log((1-\pi)/\pi)$)** giải quyết triệt để vấn đề này? *(Case Study: **Đề 07**).*
  - **Chuẩn IOAI 2024–2026 — DETR (DEtection TRansformer):** Tại sao việc dùng $N$ `Object Queries` kết hợp thuật toán ghép cặp tối ưu **Hungarian Algorithm** giúp loại bỏ hoàn toàn Anchor Boxes thủ công và thuật toán hậu xử lý NMS?

#### Tuần 11 — Bài toán Đếm Đối tượng (Crowd / Object Counting) & Hồi quy Bản đồ Mật độ (Density Map Regression) *(ĐẶC SẢN ĐỀ THI IOAI)*
- **Bản chất cần hiểu sâu (Từ đề thi thật *Chicken Counting* của IOAI 2025):**
  - Khi ảnh có hàng trăm vật thể chồng lấn lên nhau (đàn gà, tế bào hiển vi, đám đông, cây trồng từ UAV) và nhãn chỉ là tọa độ tâm điểm $(x_i, y_i)$ hoặc tổng số lượng:
    - Nếu dùng Object Detection (YOLO/Faster R-CNN): NMS sẽ xóa nhầm các vật thể đứng sát nhau!
    - **Giải pháp Bản chất — Density Map Regression:** Biến mỗi điểm đánh dấu $(x_i, y_i)$ thành một hạt nhân Gaussian 2D $\mathcal{N}((x_i, y_i), \sigma^2)$ sao cho **tích phân (tổng toàn bộ giá trị pixel) trên bản đồ mật độ đúng bằng tổng số lượng vật thể trong ảnh**!
- **Thực hành:** Tự xây dựng bộ tạo nhãn Gaussian Density Map và huấn luyện mạng CSRNet / U-Net regression dự đoán số lượng vật thể dưới nhiễu và che khuất.

---

### GIAI ĐOẠN 4 (Tuần 12–14): Phá Vỡ Giả Định I.I.D — Domain Generalization, Open-Set & Anomaly Detection

> **Mục tiêu bản chất:** Trong thực tế thi đấu, tập `private_test` rất hay bị lệch phân phối so với tập `train` (chụp bằng máy ảnh khác, điều kiện sáng khác, thời tiết khác) hoặc xuất hiện các lớp lạ chưa từng có. Giai đoạn này trang bị tư duy toán học để mô hình đứng vững khi bước ra ngoài phân phối huấn luyện.

#### Tuần 12 — Bản chất Domain Shift & Học Biểu diễn Bất biến Miền (Deep CORAL, DANN, IRM, MixStyle)
- **Bản chất cần hiểu sâu:**
  - Phân biệt rõ 3 bối cảnh để chọn đúng vũ khí trong phòng thi:
    1. **Unsupervised Domain Adaptation (UDA):** Có ảnh không nhãn của miền đích ngay lúc train $\rightarrow$ Căn chỉnh phân phối giữa Train và Test.
    2. **Domain Generalization (DG):** Hoàn toàn không biết miền đích lúc train, nhưng có nhiều miền nguồn $\rightarrow$ Ép mạng học đặc trưng bất biến giữa các miền nguồn bằng **Căn chỉnh Hiệp phương sai bậc hai (Deep CORAL)**, **Đảo chiều Đạo hàm Đối kháng (DANN / GRL)**, hoặc **Trộn thống kê phong cách tầng nông (MixStyle — trộn $\mu, \sigma$ của Instance Normalization giữa các ảnh trong batch)**.
    3. **Test-Time Adaptation (TTA / TENT):** Thích nghi trực tiếp lúc chạy `private_test` bằng cách cập nhật $(\gamma, \beta)$ của BatchNorm để giảm Entropy dự đoán.
  - *Đối chiếu Case Study:* Giải phẫu **Đề 05 (PACS DomainBed)** và thử nghiệm xem thêm `MixStyle` vào tầng `layer1, layer2` của ResNet vượt xa Deep CORAL đơn thuần như thế nào!

#### Tuần 13 — Bản chất Open-Set Recognition & Out-of-Distribution (OOD) Detection (Energy, Mahalanobis, ViM)
- **Bản chất cần hiểu sâu:**
  - Tại sao hàm `Softmax` luôn quá tự tin (Overconfident) trên mẫu lạ OOD? (Do chuẩn hóa tổng bằng $1$ và tính bất biến cộng $\text{Softmax}(\mathbf{z} + c) = \text{Softmax}(\mathbf{z})$).
  - **3 Cách Chấm Điểm OOD từ Bản chất Toán học (Không cần hoặc có cần train lại):**
    1. **Trên không gian Logits — Helmholtz Free Energy Score:** $E(\mathbf{x}) = -T \log \sum_i e^{f_i(\mathbf{x})/T}$ (bảo toàn độ lớn tuyệt đối của năng lượng kích hoạt; có thể tinh chỉnh thêm với **Energy-Bounded Margin Loss** như **Đề 06**).
    2. **Trên không gian Đặc trưng (Feature Space) — Khoảng cách Mahalanobis:** Tính tâm $\boldsymbol{\mu}_c$ và ma trận hiệp phương sai chung $\boldsymbol{\Sigma}$ của các lớp đã biết, đo khoảng cách $(\mathbf{z} - \boldsymbol{\mu}_c)^T \boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu}_c)$ để bắt mẫu lạ mà **không cần tốn 1 giây train lại mô hình**!
    3. **Kết hợp Feature + Logit (Virtual-logit Matching - ViM):** Chiếu phần dư đặc trưng nằm ngoài không gian con chính (principal subspace) thành một logit ảo cho lớp "Unknown".

#### Tuần 14 — Phát hiện Bất thường Công nghiệp Không giám sát (Normal-Only Anomaly Detection: PaDiM & PatchCore)
- **Bản chất cần hiểu sâu:**
  - Khác với Đề 07 (có nhãn bounding box của 8 loại lỗi), nếu đề bài chỉ cho **toàn bộ ảnh trong tập Train là ảnh bình thường (100% Normal)** và yêu cầu phát hiện ảnh lỗi + tô vùng pixel bị lỗi trên tập Test thì làm thế nào?
  - **Nguyên lý Memory Bank của PatchCore / PaDiM:**
    - Dùng Backbone tiền huấn luyện (đóng băng), trích xuất bản đồ đặc trưng ở tầng trung gian (`layer2` + `layer3` — vừa có chi tiết cục bộ vừa có ngữ cảnh, tránh dùng `layer4` vì quá thiên vị ImageNet).
    - Lưu các patch vectors của ảnh bình thường vào một bộ nhớ nén (**Coreset Subsampling**).
    - Khi gặp ảnh Test, tính khoảng cách từ mỗi patch trên ảnh Test tới láng giềng gần nhất trong Memory Bank: giá trị lớn nhất toàn ảnh là `image_anomaly_score`, còn bản đồ khoảng cách nội suy lên kích thước gốc chính là `pixel_anomaly_map`!

---

### GIAI ĐOẠN 5 (Tuần 15–18): Tự Giám Sát, Foundation Models (DINOv2, CLIP) & Điều Khiển Mô Hình Sinh (Diffusion Steering)

> **Mục tiêu bản chất:** Làm chủ kỷ nguyên Foundation Models — nơi ta khai thác tri thức từ các mô hình tự giám sát / đa phương thức khổng lồ bằng Zero-shot, Linear Probing, Prompt Engineering, Adapter/LoRA, hoặc can thiệp trực tiếp vào ma trận trọng số (Weight Editing chuẩn IOAI).

#### Tuần 15 — Bản chất Học Tự Giám Sát (Self-Supervised Learning): Contrastive vs Self-Distillation vs Masked Modeling
- **Bản chất cần hiểu sâu:**
  - Khi không có nhãn, làm sao tạo ra "tín hiệu giám sát tự thân" (Pretext Task) mà không bị **Sụp đổ Biểu diễn (Representation Collapse — mọi ảnh đều ra cùng 1 vector hằng số)**?
    1. **Contrastive (SimCLR / MoCo):** Dùng lực đẩy của các mẫu âm (Negative pairs) qua hàm InfoNCE.
    2. **Self-Distillation (BYOL / DINO):** Không dùng mẫu âm, mà dùng bất đối xứng giữa mạng **Student** và mạng **Teacher cập nhật chậm bằng EMA** + **Centering & Sharpening** + **KoLeo Entropy Regularizer** (đẩy các vector phân bố đều trên mặt cầu).
    3. **Masked Image Modeling (MAE / iBOT):** Che 75% các patch ảnh và bắt tái tạo lại pixel đã chuẩn hóa hoặc token ngữ nghĩa.
- **Thực hành & Case Study (Đề 09):**
  - Hiểu vì sao đặc trưng của **DINOv2** có thể phân tách tuyến tính và chạy **Zero-Shot Weighted k-NN** trực tiếp mà không cần fine-tune.
  - Làm chủ kỹ thuật trích xuất đa tầng (ghép nối `[CLS]` + `Mean Patch Tokens` hoặc ghép 4 tầng cuối của ViT) và ràng buộc kích thước ảnh chia hết cho kích thước Patch ($14 \times 14$).

#### Tuần 16 — Bản chất Không gian Đa phương thức (Vision-Language Alignment & CLIP)
- **Bản chất cần hiểu sâu:**
  - Cách **CLIP** phá vỡ giới hạn của ma trận trọng số phân loại cố định $\mathbf{W} \in \mathbb{R}^{d \times C}$ bằng cách **sinh động ma trận trọng số từ mô tả ngôn ngữ tự nhiên** qua Text Encoder $g_\phi(\text{prompt}_c)$.
  - Toán học của **Prompt Ensembling trên Mặt cầu Đơn vị:** Tại sao việc lấy trung bình cộng nhiều vector văn bản đã chuẩn hóa $\ell_2$ từ các mẫu câu mô tả thuộc tính thị giác (màu sắc, chất liệu, hình khối, bối cảnh) rồi **chuẩn hóa $\ell_2$ lại lần thứ hai** giúp triệt tiêu phương sai nhiễu ngôn ngữ và đẩy biên quyết định về đúng tâm thực sự của lớp? *(Case Study: **Đề 10**).*
  - **Mở rộng nâng cao (Vượt xa Đề 10):** Kỹ thuật **Tip-Adapter (Training-free CLIP Adapter)** — kết hợp điểm Zero-shot của CLIP với bộ nhớ Cache Few-shot từ vài ảnh mẫu để tăng vọt độ chính xác mà không cần backpropagation!

#### Tuần 17 — Mô hình Sinh Khuếch tán (Diffusion Models) & Kỹ thuật Can thiệp Trọng số (Weight Steering / Model Hacking) *(TRỌNG TÂM IOAI 2024–2026)*
- **Bản chất cần hiểu sâu (Từ đề thi thật IOAI 2024 Scientific Round):**
  - Hiểu rõ giải tích và kiến trúc của **Latent Diffusion (Stable Diffusion / SDXL):**
    - Quá trình khuếch tán thuận $q(\mathbf{z}_t | \mathbf{z}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{z}_0, (1-\bar{\alpha}_t)\mathbf{I})$ và quá trình khử nhiễu ngược dự đoán nhiễu $\epsilon_\theta(\mathbf{z}_t, t, \mathbf{c})$.
    - Vai trò của các tầng **Cross-Attention** trong U-Net Denoiser: nơi đặc trưng không gian của ảnh $\mathbf{Q} = W_Q \boldsymbol{\phi}(\mathbf{z}_t)$ lấy thông tin ngữ nghĩa từ vector văn bản $\mathbf{K} = W_K \mathbf{c}, \;\mathbf{V} = W_V \mathbf{c}$.
  - **Bài toán IOAI Weight Manipulation (Can thiệp mô hình không sửa Prompt):**
    - Nếu đề bài khóa cố định câu prompt `"a photo of a giraffe"` nhưng yêu cầu bạn sửa trọng số mô hình để nó sinh ra `"a zebra"`:
    - **Nguyên lý giải quyết (Closed-Form Rank-1 / Least-Squares Concept Editing):** Tìm ma trận chiếu $W_{\text{new}}$ cho $W_K, W_V$ sao cho hình chiếu của vector từ khóa nguồn $\mathbf{c}_{\text{giraffe}}$ trùng với hình chiếu cũ của vector đích $\mathbf{c}_{\text{zebra}}$, đồng thời giữ nguyên các từ khác:
      $$W_{\text{new}} = \arg\min_W \|W \mathbf{c}_{\text{src}} - W_{\text{old}} \mathbf{c}_{\text{tgt}}\|_2^2 + \lambda \|W - W_{\text{old}}\|_F^2$$

#### Tuần 18 — Parameter-Efficient Fine-Tuning (LoRA, Visual Prompt Tuning, Linear Probing vs Full Fine-tuning)
- **Bản chất cần hiểu sâu:**
  - Khi đi thi chỉ có 6 giờ và vài nghìn ảnh nhưng mô hình nền tảng rất lớn (ViT-Base / ViT-Large / CLIP / Swin):
    - Nếu **Full Fine-tuning** với learning rate lớn $\rightarrow$ Phá hỏng đặc trưng tiền huấn luyện (**Catastrophic Forgetting**) và Overfit nặng trên tập Train nhỏ!
    - **Nguyên lý chọn chiến lược Fine-tune theo quy mô dữ liệu:**
      - Dữ liệu cực ít ($< 10$ ảnh/lớp): Dùng **Zero-shot + Tip-Adapter / Prototypical Cosine Classifier**.
      - Dữ liệu vừa + lệch miền mạnh: Dùng **Linear Probing trước (ấm đầu phân loại trong 3 epochs) rồi mới Unfreeze 2 block cuối hoặc gắn LoRA ($W + \frac{\alpha}{r} B A$) với learning rate phân tầng (Layer-wise LR Decay)**.

---

### GIAI ĐOẠN 6 (Tuần 19–24): Thị Giác Thời Gian (Video), Hình Học 3D & Luyện Phản Xạ Phòng Thi Đề Mở

#### Tuần 19 — Bản chất Tín hiệu Không-Thời Gian (Spatio-Temporal Vision): Optical Flow, 3D CNN & Dual-Pathway (SlowFast)
- **Bản chất cần hiểu sâu:**
  - Sự bất đối xứng vật lý giữa **Không gian (Spatial)** và **Thời gian (Temporal)** trong video:
    - Ngữ nghĩa vật thể ("ai, cái gì, ở đâu") biến đổi **chậm** nhưng cần độ phân giải không gian & số kênh đặc trưng **lớn**.
    - Chuyển động động học ("đang ngã, đang đấm, đang gắp linh kiện") biến đổi **cực nhanh** nhưng chỉ cần rất **ít kênh** để biểu diễn vector vận tốc/quỹ đạo.
  - Từ nguyên lý bất đối xứng này, suy ra thiết kế **Dual-Pathway (SlowFast Networks - Đề 11)** hoặc **Temporal Shift Module (TSM — dịch chuyển $1/8$ số kênh giữa frame $t-1, t, t+1$ hoàn toàn miễn phí $0$ FLOPs!)**.

#### Tuần 20 — Tự Giám Sát trên Video & Bài toán Rò rỉ Thời gian (Temporal Leakage & VideoMAE)
- **Bản chất cần hiểu sâu:**
  - Tại sao trong video 30 fps, hai khung hình liên tiếp giống nhau $> 95\%$? Nếu áp dụng che mặt nạ ngẫu nhiên từng frame, mạng tự mã hóa sẽ học thói quen "copy điểm ảnh từ frame trước/sau" thay vì học hiểu chuyển động.
  - Nguyên lý **Tube Masking với tỷ lệ che cực đại $90\% - 95\%$** và kiến trúc **Asymmetric Encoder-Decoder** (chỉ đưa 5–10% visible tokens qua ViT Encoder) để huấn luyện tự giám sát video nhanh gấp 10 lần trên tập video không nhãn. *(Case Study: **Đề 12**).*

#### Tuần 21 — Đa đối tượng theo Thời gian (Multi-Object Tracking) & Hình học Thị giác 3D Cốt lõi
- **Phần A — Tracking-by-Detection (SORT / DeepSORT / ByteTrack):**
  - Bản chất của **Kalman Filter** (dự đoán trạng thái động học tiếp theo của hộp bao) + **Thuật toán Hungarian** (ghép cặp tối ưu dựa trên ma trận chi phí kết hợp khoảng cách hình học `IoU` và khoảng cách ngoại hình `Re-ID Cosine`).
  - Tư duy của **ByteTrack:** Tại sao không được vứt bỏ ngay các bounding box có điểm confidence thấp ($0.1 < s < 0.5$)? Vì đó chính là những đối tượng đang bị che khuất một phần!
- **Phần B — Hình học Camera, Homography & RANSAC:**
  - Ma trận nội/ngoại thông số camera, phép biến đổi phối cảnh **Homography ($\mathbf{H}$)**, khớp điểm đặc trưng (`ORB`, `SuperPoint`, `LoFTR`) và thuật toán **RANSAC** lọc nhiễu ngoại lai — vũ khí bắt buộc khi đề thi yêu cầu căn chỉnh ảnh bị nghiêng/méo, ghép ảnh bản đồ UAV, hoặc đo khoảng cách hình học.

#### Tuần 22–24 — Luyện Phản Xạ "Giải Bài Toán Lạ Không Gợi Ý" (Blind Problem-Solving Bootcamp)
Trong 3 tuần cuối, **tuyệt đối không mở phần gợi ý kiến trúc của các đề mẫu**. Hãy tự thực hành quy trình **"Blind Diagnosis" (Chẩn đoán mù)**:
1. Chỉ lấy **Tập dữ liệu + Mô tả đầu vào/đầu ra + Công thức chấm điểm** (che hoàn toàn tên bài báo và code mẫu trong đề).
2. Tự thực hiện **Exploratory Data Analysis (EDA) trong 20 phút** để tìm ra các bẫy dữ liệu:
   - Phân phối kích thước ảnh, tỷ lệ khung hình, dải màu?
   - Tỷ lệ mất cân bằng lớp / kích thước vùng đối tượng?
   - Có sự trùng lặp (near-duplicates) hoặc phân nhóm ngầm dẫn đến Data Leakage không?
   - Có sự lệch miền giữa các tập con không?
3. Tự lập **Bảng Giả thuyết Thực nghiệm (Hypothesis Table)**:
   - *Baseline 0 (Chạy trong 30 phút đầu):* Backbone nhẹ (`resnet34` / `convnext_tiny` / `dinov2_vits14`) + Loss cơ bản + Xuất file nộp thử.
   - *Cải tiến 1 (Đánh vào Bottleneck dữ liệu):* Thay đổi chiến lược lấy mẫu, hàm Loss (Focal / Edge-Weighted / ArcFace), hoặc Augmentation.
   - *Cải tiến 2 (Đánh vào Kiến trúc):* Thêm Multi-scale / Attention / Feature Aggregation phù hợp bản chất tín hiệu.
   - *Cải tiến 3 (Đánh vào Suy luận & Độ đo):* TTA + Threshold Tuning / Query Expansion / Post-processing Consistency.

---

## PHẦN III — BẢNG TRA CỨU NHANH "CHẨN ĐOÁN BỆNH & KÊ ĐƠN" CHO MỌI DẠNG BÀI MỚI

Khi vào phòng thi thật và gặp một đề bài hoàn toàn lạ, hãy dùng bảng phản xạ nguyên lý gốc này để tự thiết kế giải pháp trong 15 phút đầu:

| Triệu chứng của Dữ liệu / Bài toán Mới trong Đề Thi | Nguyên nhân Bản chất (Root Cause) | "Đơn thuốc" Nguyên lý Gốc cần lắp vào Pipeline |
| :--- | :--- | :--- |
| **1. Các lớp trông rất giống nhau, chỉ khác ở chi tiết vi mô** *(lá cây, chữ viết tay, tranh giả/thật, lỗi vi mạch)* | Mô hình dùng độ phân giải thấp sẽ mất chi tiết tần số cao, hoặc chỉ học vẹt 1 vùng nhỏ (Overfitting cục bộ). | - Tăng độ phân giải đầu vào ($384^2$ hoặc $512^2$).<br>- Dùng backbone có trường thụ cảm tốt (`ConvNeXt`, `Swin`, `Res2Net`, `EVA-02/DINOv2`).<br>- Bắt buộc dùng **Mixup + CutMix + Label Smoothing** để ép mạng nhìn toàn cục.<br>- Dùng **GeM Pooling** thay cho GAP. |
| **2. Dữ liệu mất cân bằng cực đoan ($1:100$ đến $1:10.000$) và chấm bằng `Macro-F1` / `Balanced Acc`** | Mẫu đa số/mẫu dễ áp đảo hoàn toàn tổng đạo hàm $\nabla_\theta \mathcal{L}$, khiến mạng dự đoán toàn bộ về lớp đa số. | - Dùng **Focal Loss** (nhớ khởi tạo **Prior Bias** $b = -\log((1-\pi)/\pi)$!).<br>- Hoặc dùng **Logit Adjustment** ($\tilde{z}_y = z_y - \tau \log \pi_y$).<br>- Ở bước suy luận: **Tối ưu ngưỡng cắt (Threshold Tuning)** riêng cho từng lớp trên tập Val. |
| **3. Phân đoạn vật thể có ranh giới mờ, hòa lẫn vào nền hoặc có chi tiết mảnh** *(Camouflage, Polyp, mạch máu, vết nứt)* | Pixel ở phần ruột vật thể và nền chiếm 98% diện tích, làm lu mờ sai số ở 2% pixel vùng ranh giới. | - Kết hợp **Region Loss (`Dice` / `Weighted IoU`) + `Edge-Weighted BCE`** (nhân trọng số $\alpha_{ij} = 1 + \gamma |\text{AvgPool}(G) - G|$).<br>- Dùng cơ chế **Coarse-to-Fine + Reverse Attention** ($1 - \sigma(P_{\text{coarse}})$) để tinh chỉnh phần dư ở biên. |
| **4. Đếm số lượng vật thể dày đặc, chồng lấn lên nhau** *(đàn gia cầm, tế bào, cây cối từ UAV)* | Bounding Box + NMS sẽ xóa nhầm các vật thể đứng sát nhau khi IoU cao. | - Chuyển từ Object Detection sang **Density Map Regression** (mỗi tâm vật thể là 1 hạt nhân Gaussian 2D; tổng các pixel trên output map = số lượng vật thể) hoặc dùng **DETR / Point-based Crowd Counting**. |
| **5. Số lượng lớp rất lớn, hoặc tập Test có lớp mới chưa từng xuất hiện trong tập Train** *(Instance Retrieval, Re-ID, Icon/Product Matching)* | Tầng phân loại tuyến tính Softmax cố định không thể tổng quát hóa sang lớp mới. | - Chuyển sang **Metric Learning trên Mặt cầu Đơn vị** ($\ell_2$-normalized embeddings).<br>- Huấn luyện bằng **CosFace / ArcFace / SupCon / Triplet Loss**.<br>- Hoặc trích xuất đặc trưng từ **DINOv2 / CLIP** + **Weighted k-NN** + **$\alpha$-Query Expansion**. |
| **6. Tập Test bị đổi phong cách, ánh sáng, mùa, thiết bị chụp (Domain Shift) hoặc chứa vật thể lạ (OOD)** | Mạng học vẹt vào thống kê màu sắc/kết cấu bề mặt của tập Train (Spurious Shortcuts) và Softmax bị Overconfident. | - **Chống Domain Shift:** Augmentation mạnh về màu/tần số (`RandAugment`, `FDA`, `ColorJitter`), thêm `MixStyle` hoặc `Deep CORAL`.<br>- **Bắt mẫu lạ (OOD):** Dùng **Helmholtz Free Energy Score** (`-T * logsumexp(logits/T)`) hoặc **Mahalanobis Distance** trên không gian đặc trưng. |
| **7. Đề có nhiều nhiệm vụ liên kết (Multi-Task) hoặc có quy tắc logic giữa các cột nộp bài** | Hai nhánh dự đoán độc lập có thể ra kết quả mâu thuẫn nhau (ví dụ: nhánh 1 bảo "không có lỗi" nhưng nhánh 2 lại vẽ hộp lỗi). | - Huấn luyện chung Backbone với **Multi-Task Loss** (dùng Uncertainty Weighting hoặc PCGrad nếu xung đột).<br>- Bắt buộc viết hàm **Post-processing Logical Consistency** kiểm tra 100% các dòng trong file CSV trước khi nộp! |
| **8. Dữ liệu có nhãn rất ít ($1\% - 10\%$) nhưng có kèm kho ảnh/video không nhãn lớn** | Huấn luyện có giám sát từ đầu hoặc fine-tune mạnh trên ít nhãn sẽ bị Overfitting ngay lập tức. | - Tận dụng **Frozen Foundation Model (`DINOv2` / `CLIP`)** + **Linear Probing / LoRA / Tip-Adapter**.<br>- Hoặc thực hiện **Self-Supervised Pre-training** (MAE / VideoMAE / SimCLR) hoặc **Semi-Supervised Pseudo-Labeling (FixMatch)** trên tập không nhãn. |
