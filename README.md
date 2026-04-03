# Computer Vision - Project 2: Gradient Domain Editing & Biến đổi hình học

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/broistg/CV-Project-2_Nhom-18/blob/main/notebooks/CV_Project_2_Demo.ipynb)

Bài tập lớn 2 - Computer Vision | HK 2025-2026 | Giảng viên: ThS. Võ Thanh Hùng

---

## 📝 Giới thiệu

Project này giải quyết hai bài toán cơ bản nhưng quan trọng trong Computer Vision:

1. **Gradient Domain Editing (Poisson Blending):** Kỹ thuật ghép ảnh dựa trên việc giải phương trình Poisson để hòa trộn miền gradient của ảnh nguồn vào ảnh đích, giúp loại bỏ biên và cân bằng ánh sáng tự nhiên hơn so với cắt ghép thông thường. So sánh với các phép ghép ảnh trực tiếp.
2. **Geometric Transformations:** Thực hiện và so sánh các phép biến đổi hình học (Affine vs. Projective). Ứng dụng Homography để dán ảnh quảng cáo lên bề mặt phẳng trong không gian 3D.

---

## 👥 Thành viên nhóm

| MSSV | Họ và Tên | Công việc thực hiện |
| :---: | :---: | :--- |
| 2111493 | Nguyễn Minh Khánh | Các phép biến đổi hình học cơ bản |
| 2233163 | Nguyễn Anh Duy | Gradient Domain Editing |
| 2011706 | Nguyễn Nhựt Nguyên | Phép biến đổi Projective & Ứng dụng Mở rộng |
| 2310653 | Lê Tiến Đạt | Thực nghiệm & Demo |

---

## 📂 Cấu trúc thư mục

```
CV-Project-2_Nhom-18/
├── data/                       # Chứa dữ liệu ảnh (Input/Output)
│   ├── inputs/                 # Ảnh gốc
│   │   ├── gde/                # Ảnh cho phần Gradient Domain Editing
│   │   └── geometry/           # Ảnh cho phần biến đổi hình học
│   └── outputs/                # Ảnh kết quả sau khi chạy code
├── notebooks/                  # Demo Colab chạy thử nghiệm
├── report/                     # Chứa file báo cáo cuối cùng
├── src/
│   ├── image_blending.py       # Module xử lý Poisson Blending và các phép ghép trực tiếp
│   ├── geometry_transform.py   # Module xử lý Affine, Rotation, Scaling...
│   ├── homography.py           # Module xử lý Projective & Dán ảnh tòa nhà
│   └── utils.py                # Các hàm hỗ trợ đọc/ghi/hiển thị ảnh
├── requirements.txt
└── README.md
```

---

## ⚙️ Cài đặt

**Yêu cầu:** Python 3.8+

```bash
# Clone repository
git clone https://github.com/broistg/CV-Project-2_Nhom-18.git
cd CV-Project-2_Nhom-18

# Cài đặt dependencies
pip install -r requirements.txt
```

**Thư viện sử dụng:** numpy, opencv-python, matplotlib, scipy

---

## 🚀 Hướng dẫn chạy

**Cách 1: Google Colab (Khuyên dùng)**

1. Truy cập vào link demo Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/broistg/CV-Project-2_Nhom-18/blob/main/notebooks/CV_Project_2_Demo.ipynb)
2. Nhấn nút "Run all" trong Colab để chạy demo dự án.

**Cách 2: Local**

```bash
jupyter notebook notebooks/CV_Project_2_Demo.ipynb
```
