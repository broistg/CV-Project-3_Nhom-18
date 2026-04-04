import cv2
import numpy as np


def calculate_homography(kp1, kp2, matches, ransac_thresh=5.0, min_inliers=10):
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

    if H is None or mask is None:
        raise ValueError("Không thể ước lượng ma trận Homography từ các điểm khớp hiện tại.")

    inlier_count = int(mask.ravel().sum())
    if inlier_count < min_inliers:
        raise ValueError(
            f"Số lượng inlier quá ít ({inlier_count}). Cần ít nhất {min_inliers} inlier để ghép ảnh ổn định."
        )

    return H, mask