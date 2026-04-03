import cv2
import numpy as np

def calculate_homography(kp1, kp2, matches, ransac_thresh=5.0):
    """
    Tính toán ma trận Homography từ các điểm đã được so khớp.
    """
    if len(matches) < 4:
        raise ValueError("Cần ít nhất 4 cặp điểm để tính ma trận Homography.")
        
    # Trích xuất tọa độ (x, y) từ các điểm khớp
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Sử dụng RANSAC để loại bỏ các điểm khớp sai (outliers) và tính H
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    
    return H, mask