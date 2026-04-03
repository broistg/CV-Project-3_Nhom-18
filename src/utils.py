import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def read_image(path, mode='rgb'):
    """
    Hàm load ảnh an toàn, kiểm tra file tồn tại.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh tại: {path}")
    
    if mode == 'gray':
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        # OpenCV đọc BGR, cần chuyển sang RGB
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def show_image(image, title="Image", cmap_type='gray'):
    """
    Hàm hiển thị ảnh đơn bằng Matplotlib.
    """
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')

def show_comparison(img1, img2, title1="Original", title2="Processed", cmap1='gray', cmap2='gray'):
    """
    Hiển thị 2 ảnh song song để so sánh trực quan (Before -> After).
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap=cmap1)
    plt.title(title1)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap=cmap2)
    plt.title(title2)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def resize_image_keep_aspect(img, max_width=800):
    """Resize ảnh nếu quá lớn để dễ hiển thị/xử lý."""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img