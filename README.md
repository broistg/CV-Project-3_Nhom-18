# Computer Vision - Project 3: Image Stitching & Panorama Creation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/broistg/CV-Project-3_Nhom-18/blob/main/notebooks/CV_Project_3_Demo.ipynb)

Bài tập lớn 3 - Computer Vision | HK 2025-2026 | Giảng viên: ThS. Võ Thanh Hùng

---

## Giới thiệu

Project này xây dựng một pipeline ghép ảnh toàn cảnh hoàn chỉnh, từ trích xuất đặc trưng đến căn chỉnh hình học và hòa trộn ảnh đầu ra. Mục tiêu là ghép các khung hình chồng lấn thành một ảnh panorama liền mạch, hạn chế đường nối và méo hình ở vùng giao nhau.

Các bước chính trong dự án:

1. Trích xuất đặc trưng bằng SIFT hoặc ORB.
2. So khớp đặc trưng bằng brute-force matcher và Lowe ratio test.
3. Ước lượng ma trận homography bằng RANSAC.
4. Warp ảnh về cùng hệ tọa độ và ghép hai ảnh lại.
5. Blend vùng chồng lấn bằng trọng số theo distance transform để mượt hơn.

---

## Tính năng chính

- Hỗ trợ hai bộ trích xuất đặc trưng: SIFT và ORB.
- So sánh trực quan keypoints và matches giữa hai ảnh.
- Tính homography ổn định hơn nhờ RANSAC.
- Ghép ảnh panorama với cơ chế dịch tọa độ và giới hạn kích thước an toàn.
- Hòa trộn vùng giao nhau để giảm đường nối rõ ràng giữa các ảnh.
- Có notebook demo để chạy thử toàn bộ pipeline.

---

## Cấu trúc thư mục

```
CV-Project-3_Nhom-18/
├── data/
│   ├── inputs/                  # Ảnh đầu vào cho demo
│   └── outputs/                 # Kết quả sau khi chạy pipeline
├── notebooks/
│   └── CV_Project_3_Demo.ipynb  # Notebook demo chính
├── report/
├── src/
│   ├── feature_extraction.py    # SIFT / ORB và hiển thị keypoints
│   ├── feature_matching.py      # So khớp đặc trưng và vẽ matches
│   ├── homography.py            # Tính homography bằng RANSAC
│   ├── image_stitching.py       # Warp và ghép ảnh panorama
│   └── utils.py                 # Hàm đọc ảnh, hiển thị, resize
├── requirements.txt
└── README.md
```

---

## Cài đặt

Yêu cầu: Python 3.8+.

```bash
git clone https://github.com/broistg/CV-Project-3_Nhom-18.git
cd CV-Project-3_Nhom-18
pip install -r requirements.txt
```

Các thư viện chính sử dụng trong dự án: numpy, opencv-python, matplotlib, scipy.

---

## Cách chạy

### Chạy trên Google Colab

1. Mở notebook demo bằng badge Colab ở đầu README.
2. Chạy lần lượt các cell hoặc chọn `Run all` để thực thi toàn bộ demo.

### Chạy local

```bash
jupyter notebook notebooks/CV_Project_3_Demo.ipynb
```

Hoặc mở file notebook bằng VS Code và chạy trực tiếp từng cell.

---

## Ghi chú sử dụng

- Ảnh đầu vào nên được đặt trong `data/inputs/`.
- Kết quả trung gian và ảnh panorama đầu ra có thể lưu vào `data/outputs/`.
- Nếu ảnh có kích thước quá lớn, pipeline ghép ảnh đã có cơ chế giới hạn kích thước để tránh tốn bộ nhớ quá mức.

---

## Tài liệu tham khảo trong code

- [src/feature_extraction.py](src/feature_extraction.py)
- [src/feature_matching.py](src/feature_matching.py)
- [src/homography.py](src/homography.py)
- [src/image_stitching.py](src/image_stitching.py)
- [src/utils.py](src/utils.py)
