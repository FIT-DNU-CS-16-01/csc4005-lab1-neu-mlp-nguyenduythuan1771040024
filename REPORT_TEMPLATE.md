# CSC4005 - Lab 1 Report

## 1. Mục tiêu
Mục tiêu của bài lab là huấn luyện mô hình MLP để phân loại lỗi bề mặt thép trên bộ dữ liệu NEU (6 lớp), sau đó so sánh tối thiểu 3 cấu hình huấn luyện để:

- quan sát learning curves,
- đánh giá hiện tượng overfitting/underfitting,
- chọn cấu hình tốt nhất theo validation (không dùng test để chọn model).

## 2. Cấu hình thí nghiệm
Đường dẫn dữ liệu dùng để chạy: `C:\DL\NEU-CLS.zip` (được resolve thành `C:\DL\NEU-CLS_extracted`).

Các cấu hình đã chạy (3 run chính + 1 run ablation):

1. `baseline_adamw`
- optimizer: `adamw`
- lr: `0.001`
- weight_decay: `0.0001`
- dropout: `0.3`
- epochs: `20`

2. `run_b_sgd`
- optimizer: `sgd`
- lr: `0.01`
- weight_decay: `0.0`
- dropout: `0.3`
- epochs: `20`

3. `run_c_strong_reg`
- optimizer: `adamw`
- lr: `0.0005`
- weight_decay: `0.001`
- dropout: `0.5`
- epochs: `20`

4. `ablation_dropout_only` (ablation có kiểm soát)
- optimizer: `adamw`
- lr: `0.001`
- weight_decay: `0.0001`
- dropout: `0.5`
- epochs: `20` (early stopping ở epoch 12)

Thiết kế kiểm soát cho ablation:
- Giữ nguyên toàn bộ tham số như baseline (`adamw`, `lr`, `weight_decay`, `batch_size`, `img_size`, `augment`, `seed`).
- Chỉ thay đổi duy nhất 1 yếu tố: `dropout` từ `0.3` -> `0.5`.

Tất cả run đều bật augmentation và ghi log W&B ở chế độ offline.

## 3. Kết quả

### 3.1 Bảng configs -> metrics

| Run | Optimizer | LR | Weight Decay | Dropout | Best Val Loss | Best Val Acc | Test Loss | Test Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_adamw | adamw | 0.0010 | 0.0001 | 0.3 | 1.5036 | 0.4037 | 1.5018 | 0.3593 |
| run_b_sgd | sgd | 0.0100 | 0.0000 | 0.3 | 1.4358 | 0.4185 | 1.3808 | 0.3926 |
| run_c_strong_reg | adamw | 0.0005 | 0.0010 | 0.5 | 1.6600 | 0.3185 | 1.6335 | 0.3259 |
| ablation_dropout_only | adamw | 0.0010 | 0.0001 | 0.5 | 1.7600 | 0.1815 | 1.7585 | 0.1815 |

### 3.2 Artifacts đã sinh ra

Mỗi run đều có đủ:

- `best_model.pt`
- `history.csv`
- `curves.png`
- `confusion_matrix.png`
- `metrics.json`

Vị trí:

- `outputs/baseline_adamw/`
- `outputs/run_b_sgd/`
- `outputs/run_c_strong_reg/`
- `outputs/ablation_dropout_only/`

### 3.3 Bảng ablation (kiểm soát 1 yếu tố)

| Cặp so sánh | Yếu tố thay đổi | Giá trị A | Giá trị B | Best Val Acc A | Best Val Acc B | Delta |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_adamw vs ablation_dropout_only | dropout | 0.3 | 0.5 | 0.4037 | 0.1815 | -0.2222 |

Nhận xét ablation:
- Khi chỉ tăng dropout từ `0.3` lên `0.5`, hiệu năng giảm mạnh trên cả validation và test.
- Điều này cho thấy mức regularization cao là nguyên nhân chính gây underfitting trong thí nghiệm này.

## 4. Phân tích

### 4.1 Cấu hình tốt nhất
Nếu chọn theo validation, cấu hình tốt nhất là `run_b_sgd` với:

- `best_val_acc = 0.4185` (cao nhất),
- `best_val_loss = 1.4358` (thấp nhất).

### 4.2 Dấu hiệu overfitting / underfitting

- `baseline_adamw`: train_acc và val_acc cùng tăng dần, khoảng cách không quá lớn. Có cải thiện ổn định, nhưng hiệu năng tổng thể chưa cao.
- `run_b_sgd`: val_acc đạt cao nhất, đồng thời test_acc cũng cao nhất. Có dao động giữa các epoch nhưng xu hướng chung là học tốt hơn baseline.
- `run_c_strong_reg`: dropout cao và weight decay lớn làm mô hình bị regularize mạnh, dẫn đến train_acc và val_acc thấp hơn, thể hiện xu hướng underfitting.

### 4.3 So sánh AdamW và SGD trong thí nghiệm này

- Với tập tham số đang dùng, SGD (`run_b_sgd`) cho kết quả validation và test tốt hơn cả hai run AdamW.
- AdamW baseline vẫn ổn định và dễ hội tụ, nhưng chưa vượt được SGD ở thiết lập hiện tại.
- AdamW với regularization mạnh (`run_c_strong_reg`) giảm hiệu năng rõ rệt, cho thấy mức regularization này quá mạnh với bài toán hiện tại.

### 4.4 Error cases và phân nhóm lỗi

Từ confusion matrix của run tốt nhất (`run_b_sgd`), các nhóm lỗi nổi bật:

- Nhóm 1: `Rolled-in_Scale -> Crazing` (29 mẫu). Đây là cặp nhầm lẫn lớn nhất.
- Nhóm 2: `Inclusion -> Scratches` (21 mẫu). Mô hình dễ nhầm các mẫu inclusion có vệt mảnh với scratches.
- Nhóm 3: `Crazing -> Pitted_Surface` (17 mẫu) và `Patches -> Pitted_Surface` (11 mẫu).

Ví dụ minh họa từ ma trận nhầm lẫn (run_b_sgd):
- `Rolled-in_Scale` đúng 45 mẫu nhưng chỉ nhận đúng 6, còn 29 bị dự đoán thành `Crazing`.
- `Inclusion` đúng 45 mẫu nhưng chỉ nhận đúng 1, còn 21 bị dự đoán thành `Scratches`.

### 4.5 Đề xuất cải tiến dựa trên lỗi

- Cải thiện dữ liệu cho các cặp dễ nhầm: tăng augmentation theo hướng texture/contrast cho `Rolled-in_Scale`, `Crazing`, `Inclusion`, `Scratches`.
- Dùng weighted loss hoặc focal loss để tăng độ nhạy cho lớp có recall thấp.
- Bổ sung phân tích lỗi mức ảnh (lưu top-k ảnh dự đoán sai) để tách lỗi do nhiễu ảnh và lỗi do đặc trưng lớp chồng lấp.
- Thử mô hình có inductive bias theo không gian (CNN nhỏ) ở vòng sau để giảm nhầm giữa các texture gần nhau.

## 5. Kết luận
Theo tiêu chí của lab, chọn mô hình theo validation thì `run_b_sgd` là cấu hình tốt nhất.

Lý do:

1. Có `best_val_acc` cao nhất (0.4185).
2. Có `best_val_loss` thấp nhất (1.4358).
3. Kết quả test cũng tốt nhất trong 3 run (0.3926), củng cố lựa chọn từ validation.

Liên hệ CLO3 / A2.3 / A3:
- Đã có bảng metric và so sánh baseline - tuned.
- Đã có ít nhất 1 ablation có kiểm soát 1 yếu tố (`dropout`).
- Đã có error analysis gồm phân nhóm lỗi, ví dụ định lượng và hướng cải tiến để đưa vào phần report/demo A3.

Ghi chú W&B: các run đã được ghi log offline. Cần hoàn tất bước xác thực W&B rồi sync lại để đính kèm link dashboard cloud trong báo cáo nộp.
