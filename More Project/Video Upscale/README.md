# 🎬 Video Upscale Project (Video2X Colab Launcher)

Dự án nâng cấp chất lượng video và tăng khung hình (FPS) sử dụng AI bằng framework **[Video2X (K4YT3X/video2x)](https://github.com/K4YT3X/video2x)** chạy trực tiếp trên máy chủ **Google Colab GPU**.

---

## 📁 Cấu trúc thư mục

```text
More Project/
└── Video Upscale/
    ├── README.md                     # Hướng dẫn chi tiết dự án
    └── video2x_colab_launcher.ipynb  # File Notebook Google Colab Launcher
```

---

## 🚀 Tính năng chính

- ⚡ **Tự động cấu hình môi trường**: Tự động tải Video2X v6.x (C++ engine), cài đặt driver Vulkan (`libvulkan1`, `mesa-vulkan-drivers`) và `ffmpeg` trên Colab.
- 📂 **Tích hợp Google Drive**: Đọc tệp nguồn và lưu trực tiếp tệp kết quả vào Google Drive của bạn mà không lo mất dữ liệu.
- 🎛️ **Form điều khiển trực quan**: Tùy chỉnh trực tiếp trên giao diện Colab Forms (chọn model, tỷ lệ scale 2x/3x/4x, denoise level, GPU ID).
- 🔄 **Xử lý đơn lẻ & hàng loạt**: Hỗ trợ nâng cấp 1 video hoặc xử lý tự động toàn bộ thư mục video.
- 📺 **Trình phát video xem trước (Preview Player)**: Xem kết quả trực tiếp ngay trong giao diện Colab trước khi tải về.

---

## 📖 Hướng dẫn sử dụng trên Google Colab

1. **Mở Colab**: Truy cập [Google Colab](https://colab.research.google.com/).
2. **Tải lên Notebook**: Chọn tab **Upload** và tải lên file [`video2x_colab_launcher.ipynb`](./video2x_colab_launcher.ipynb).
3. **Kích hoạt GPU**:
   - Vào menu: **Runtime** $\rightarrow$ **Change runtime type** (Thời gian chạy $\rightarrow$ Thay đổi loại thời gian chạy).
   - Chọn **Hardware accelerator** là **GPU** (T4, L4 hoặc A100) $\rightarrow$ bấm **Save**.
4. **Thực thi các bước**:
   - **Bước 1**: Kiểm tra GPU (`nvidia-smi`).
   - **Bước 2**: Kết nối Google Drive (`drive.mount`).
   - **Bước 3**: Tự động tải và cài đặt Video2X.
   - **Bước 4**: Điền đường dẫn video đầu vào/đầu ra và bắt đầu upscale.
   - **Bước 5** *(Tùy chọn)*: Nâng cấp hàng loạt toàn bộ thư mục video.
   - **Bước 6**: Xem trước video kết quả bằng HTML5 Player.

---

## 🧠 Bảng tham khảo Models & Engines

| Thể loại Video | Processor | Model khuyên dùng | Mục đích sử dụng |
| :--- | :--- | :--- | :--- |
| **Anime / Hoạt hình 2D** | `realesrgan` | `realesr-animevideov3` | Rất nhanh, viền nét, giảm mờ và khử nhiễu tối ưu cho hoạt hình. |
| **Anime cao cấp / Chi tiết sắc nét** | `realcugan` | `models-se` / `models-pro` | Tái tạo đường nét cực nét, giữ màu nguyên bản. |
| **Video người thật / Phong cảnh đời thực** | `realesrgan` | `realesrgan-x4plus` | Tối ưu hóa cho da, tóc, bề mặt và vật thể thực tế. |
| **Tăng độ mượt khung hình (FPS)** | `rife` | `rife-v4.6` | Nội suy khung hình AI (ví dụ: 24 FPS $\rightarrow$ 60 FPS). |
| **Anime cổ điển / Pixel art** | `waifu2x` | `cunet` | Thuật toán kinh điển khử noise và upscale tranh vẽ/anime. |

