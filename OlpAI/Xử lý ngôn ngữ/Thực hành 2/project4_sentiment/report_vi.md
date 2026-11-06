# Báo cáo NLP Project 4  
Phân tích cảm xúc đánh giá phim với N-grams (BoW / TF-IDF)

## 1. Tổng quan

Dự án xây dựng bộ phân loại cảm xúc nhị phân cho đánh giá phim IMDb (tích cực vs tiêu cực). Unigram, bigram và trigram được dùng làm **đặc trưng phân loại** (không phải mô hình ngôn ngữ dự đoán từ kế tiếp). Chúng tôi so sánh Bag-of-Words đếm thô với trọng số TF-IDF dưới cùng một quy trình tiền xử lý và tách train/validation, rồi đánh giá mô hình đã chọn **một lần** trên tập test chính thức.

**Câu hỏi nghiên cứu.** Các đặc trưng bigram/trigram có cải thiện phân loại cảm xúc so với Bag-of-Words unigram không, và TF-IDF có thay đổi kết luận đó không?

## 2. Dữ liệu và quy trình

- **Bộ dữ liệu:** Stanford Large Movie Review Dataset (`aclImdb`), 25.000 train + 25.000 test, nhãn cân bằng.
- **Validation:** tách phân tầng 80/20 từ tập train chính thức (`random_state=42`).
- **Làm sạch:** giải mã HTML entity, loại bỏ `<br>`, chuẩn hóa khoảng trắng. **Không** loại stopwords để giữ tín hiệu phủ định như *not* / *never*.
- **Kiểm soát rò rỉ:** vectorizer chỉ fit trên văn bản train; chọn mô hình chỉ dựa trên validation; tập test chính thức chỉ dùng một lần cho mô hình cuối.

## 3. Các cấu hình

| ID | Biểu diễn | Phạm vi N-gram | Bộ phân loại |
|----|-----------|----------------|--------------|
| A | BoW counts | (1,1) | Logistic Regression |
| B | BoW counts | (1,2) | Logistic Regression |
| C | TF-IDF | (1,1) | Logistic Regression |
| D | TF-IDF | (1,2) | Logistic Regression |
| E | TF-IDF | (1,3) | Logistic Regression |
| F | BoW counts | (1,2) | Multinomial Naive Bayes |

Cài đặt chung: `min_df=2`, `max_features=150000`, `sublinear_tf=True` cho TF-IDF, Logistic Regression `max_iter=1000`.

## 4. Kết quả validation

| ID | # đặc trưng | Val accuracy | Val precision | Val recall | Val F1 |
|----|-------------|--------------|---------------|------------|--------|
| A | 40.909 | 0.8742 | 0.8690 | 0.8812 | 0.8751 |
| B | 150.000 | 0.8864 | 0.8824 | 0.8916 | 0.8870 |
| C | 40.909 | 0.8798 | 0.8713 | 0.8912 | 0.8812 |
| D | 150.000 | 0.8864 | 0.8773 | 0.8984 | 0.8877 |
| E | 150.000 | **0.8878** | 0.8789 | 0.8996 | **0.8891** |
| F | 150.000 | 0.8654 | 0.8748 | 0.8528 | 0.8637 |

**Mô hình được chọn:** cấu hình **E** (TF-IDF, phạm vi n-gram 1–3, Logistic Regression), F1 validation tốt nhất.

## 5. Đánh giá trên tập test chính thức

Sau khi huấn luyện lại E trên toàn bộ 25.000 đánh giá train chính thức:

| Chỉ số | Điểm trên test |
|--------|----------------|
| Accuracy | **0.8989** |
| Precision | 0.8948 |
| Recall | 0.9042 |
| F1 | 0.8994 |

Ma trận nhầm lẫn được lưu tại `outputs/confusion_matrix.png`.

## 6. Kiểm tra đặc trưng (hệ số Logistic Regression)

Các unigram tích cực mạnh gồm *great*, *excellent*, *wonderful*, *perfect*, *amazing*.  
Các unigram tiêu cực mạnh gồm *bad*, *worst*, *awful*, *boring*, *waste*.

Các đặc trưng đa từ hữu ích trong nhóm trọng số mạnh nhất gồm:

- **the best** / **of the best** (tích cực) — cụm khen ngợi thông dụng.
- **must see**, **very good**, **well worth**, **my favorite** (tích cực) — collocation và từ nhấn mạnh.
- **the worst** / **of the worst** (tiêu cực) — cụm chê bai thông dụng.
- **waste of**, **not even**, **not worth**, **at best** (tiêu cực) — phủ định và đánh giá tiêu cực thành ngữ.

Các cụm này minh họa vì sao N-gram quan trọng: cực tính thường đến từ tổ hợp từ, không phải từng từ riêng lẻ.

## 7. Phân tích lỗi (định tính)

Các lỗi còn lại điển hình của mô hình cổ điển gồm:

1. **Mỉa mai / khen mỉa** dùng từ vựng tích cực trong đánh giá tiêu cực.
2. **Cảm xúc lẫn** (ví dụ khen diễn xuất + chê cốt truyện).
3. **Đánh giá nặng phần tóm tắt cốt truyện** với từ vựng tối khiến mô hình nghĩ tiêu cực dù điểm tổng thể tích cực.
4. **Phủ định phạm vi xa** vượt cửa sổ bigram/trigram cục bộ.
5. **Tên riêng / thuật ngữ thương hiệu** tương quan với nhãn trong dữ liệu train nhưng không phải tín hiệu cảm xúc thật.

Các đoạn trích chi tiết được xem trong `notebooks/experiments.ipynb` (Mục 13).

## 8. Câu hỏi thảo luận

### 8.1 Việc thêm bigram có cải thiện hiệu năng validation không?

**Có.**  

- BoW: A → B accuracy tăng từ **0.8742 → 0.8864** (+1,22 điểm); F1 từ 0.8751 → 0.8870.  
- TF-IDF: C → D accuracy tăng từ **0.8798 → 0.8864** (+0,66 điểm); F1 từ 0.8812 → 0.8877.

Đặc trưng cụm từ giúp mã hóa thứ tự cục bộ như phủ định và collocation mà unigram bỏ sót.

### 8.2 Hai bigram mà nghĩa không rõ nếu chỉ nhìn từng unigram

1. **waste of** — *waste* tiêu cực và *of* trung tính; cùng nhau tạo thành lời phàn nàn thành ngữ (*waste of time/money*) mạnh hơn từng token.  
2. **not worth** — *not* đảo cực tính và *worth* thường tích cực; bigram mã hóa khuyến nghị tiêu cực mà bag-of-words unigram có thể làm xáo trộn.

Các cụm liên quan đã học được: **the worst**, **must see**, **at best**, **not even**.

### 8.3 TF-IDF thay đổi hiệu năng thế nào so với đếm thô?

TF-IDF giúp vừa phải trên unigram (**A 0.8742 → C 0.8798** accuracy) và gần như hòa với BoW trên unigram+bigram (**B và D cùng 0.8864** accuracy), với F1 nhỉnh hơn nhẹ ở TF-IDF D (0.8877 vs 0.8870).  

Diễn giải: trọng số giảm sự thống trị của từ rất phổ biến, nhưng khi đã có bigram giàu thông tin, Logistic Regression đã dùng đặc trưng đếm khá hiệu quả. TF-IDF hữu ích, không phải bước nhảy lớn.

### 8.4 Trigram có giúp không?

Một chút. **D → E**: accuracy **0.8864 → 0.8878**, F1 **0.8877 → 0.8891**.  

**Đánh đổi:** trigram thêm ngữ cảnh cục bộ phong phú hơn, nhưng cũng tăng thưa và dư thừa. Với `max_features=150000`, nhiều trigram hiếm cạnh tranh slot với unigram/bigram hữu ích. Có cải thiện nhưng nhỏ; nhiều đặc trưng hơn không tự động tốt hơn.

### 8.5 Những đánh giá nào vẫn khó?

Đánh giá có mỉa mai, ý kiến xung đột trong cùng một văn bản, tóm tắt cốt truyện dày đặc, phủ định tinh tế hoặc khoảng cách xa, và tương quan giả từ tên/thể loại vẫn khó với mô hình tuyến tính bag-of-N-grams. Các trường hợp này thường cần biểu diễn mức diễn ngôn hoặc ngữ cảnh vượt đặc trưng N-gram thưa.

## 9. Kết luận

Đặc trưng bigram cải thiện rõ so với BoW unigram. TF-IDF cho thêm một cải thiện nhỏ hơn, chủ yếu ở unigram. Trigram mang lại lợi ích biên dưới ngân sách đặc trưng cố định. Mô hình TF-IDF (1,3) + Logistic Regression đã chọn đạt khoảng **89,9% accuracy** trên tập test IMDb chính thức. Các lỗi còn lại nêu rõ giới hạn của đặc trưng N-gram cục bộ với cảm xúc tinh tế.

## Tài liệu tham khảo

- Maas, A. et al. (2011). Large Movie Review Dataset. https://ai.stanford.edu/~amaas/data/sentiment/
- Tài liệu scikit-learn cho `CountVectorizer` và `TfidfVectorizer`.
