# Hướng Dẫn Cấu Trúc Dự Án

Repo này được tổ chức để làm thí nghiệm cho CSC4005 Lab 1.

## Thư mục cấp cao

- `src/`: mã nguồn huấn luyện và phần cốt lõi của mô hình.
- `configs/`: cấu hình thí nghiệm, ví dụ `baseline.json`.
- `ci/`: các script kiểm tra nhanh.
- `docs/`: tài liệu hướng dẫn lab và ghi chú hỗ trợ.
- `notebooks/`: notebook để demo hoặc khám phá dữ liệu.
- `outputs/`: nơi lưu kết quả sau mỗi lần chạy train.

## File quan trọng

- `src/dataset.py`: đọc dữ liệu và chia train/val/test.
- `src/model.py`: định nghĩa mô hình MLP.
- `src/train.py`: điểm vào chính để huấn luyện.
- `src/utils.py`: tiện ích tính metric, vẽ biểu đồ, lưu file, early stopping.
- `ci/smoke_train.py`: kiểm tra nhanh pipeline train bằng dữ liệu giả nhỏ.
- `ci/check_structure.py`: kiểm tra cấu trúc bắt buộc của repo.
- `configs/baseline.json`: tham số baseline của thí nghiệm.

## Sau quick test cần làm gì

1. Chạy cấu hình baseline.
2. Chạy thêm ít nhất 2 cấu hình khác.
3. So sánh metric trên validation và learning curves.
4. Chọn run tốt nhất theo validation để đưa vào báo cáo cuối.

## Thư mục output

Mỗi lần chạy sẽ tạo kết quả tại:

- `outputs/<run_name>/best_model.pt`
- `outputs/<run_name>/history.csv`
- `outputs/<run_name>/curves.png`
- `outputs/<run_name>/confusion_matrix.png`
- `outputs/<run_name>/metrics.json`

## Thứ tự chạy gợi ý

- `baseline_adamw`
- `run_b_sgd`
- `run_c_strong_reg`

## Lưu ý

Không thêm thư mục `data/` vào starter repo. Khi train, truyền trực tiếp đường dẫn dữ liệu qua tham số `--data_dir`.
