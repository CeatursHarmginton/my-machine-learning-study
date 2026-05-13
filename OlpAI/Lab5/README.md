# MLP Neural Network Activity Pack

Gói này chứa chuỗi bài thực hành tiếng Việt về **perceptron nhiều lớp (Multilayer Perceptron - MLP)**, được thiết kế để nối tiếp các bài về hồi quy tuyến tính, hồi quy đa thức, Ridge, Lasso và Elastic Net.

## Nội dung

- `Huong_dan_hoat_dong_MLP_Mang_noron.docx`: tài liệu Word chính cho sinh viên và giảng viên.
- `HUONG_DAN_HOAT_DONG_MLP_MANG_NORON.md`: bản Markdown của tài liệu hướng dẫn.
- `datasets/`: dữ liệu CSV tổng hợp cho 6 hoạt động.
- `notebooks/mlp_assignment_solutions_vi.ipynb`: notebook lời giải mẫu, có code huấn luyện, đánh giá và trực quan hóa.
- `notebooks/mlp_assignment_starter_vi.ipynb`: notebook khởi đầu cho sinh viên, có cấu trúc và TODO.
- `figures/`: hình minh họa dùng trong tài liệu.
- `_equations/`: hình công thức dùng trong tài liệu Word.
- `requirements.txt`: thư viện Python cần cài đặt.

## Cài đặt nhanh

```bash
pip install -r requirements.txt
jupyter notebook notebooks/mlp_assignment_starter_vi.ipynb
```

Hoặc mở lời giải mẫu:

```bash
jupyter notebook notebooks/mlp_assignment_solutions_vi.ipynb
```

## Ghi chú

Tất cả dữ liệu là dữ liệu tổng hợp dùng cho mục đích giảng dạy. Các cột `y_clean` và file `activity_06_instructor_true_terms.csv` giúp giảng viên kiểm tra hàm thật, không nhất thiết phải cung cấp cho sinh viên trong bài kiểm tra chính thức.
