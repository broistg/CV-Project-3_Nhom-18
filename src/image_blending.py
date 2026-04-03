import cv2
import numpy as np

def get_safe_roi(source_img, target_img, mask_img, center):
    """
    Hàm tính toán ranh giới an toàn, chống lỗi tràn viền ảnh.
    Dùng chung cho cả Naive và Alpha Blending.
    Trả về vùng ảnh nguồn, vùng ảnh đích và vùng mask đã được cắt gọt hợp lệ.
    """
    x, y, w, h = cv2.boundingRect(mask_img)
    if w == 0 or h == 0:
        return None, None, None, None

    source_roi = source_img[y:y+h, x:x+w]
    mask_roi = mask_img[y:y+h, x:x+w]

    start_x = center[0] - w // 2
    start_y = center[1] - h // 2

    # Giới hạn tọa độ không cho vượt quá ảnh đích
    t_start_x = max(0, start_x)
    t_start_y = max(0, start_y)
    t_end_x = min(target_img.shape[1], start_x + w)
    t_end_y = min(target_img.shape[0], start_y + h)

    roi_w = t_end_x - t_start_x
    roi_h = t_end_y - t_start_y

    if roi_w <= 0 or roi_h <= 0:
        return None, None, None, None

    s_start_x = t_start_x - start_x
    s_start_y = t_start_y - start_y

    cropped_src = source_roi[s_start_y:s_start_y+roi_h, s_start_x:s_start_x+roi_w]
    cropped_mask = mask_roi[s_start_y:s_start_y+roi_h, s_start_x:s_start_x+roi_w]

    # Trả về các thông số cần thiết
    bounds = (t_start_x, t_end_x, t_start_y, t_end_y)
    return cropped_src, cropped_mask, bounds

def naive_copy_paste(source_img, target_img, mask_img, center):
    """
    Cắt dán trực tiếp đơn giản, không xử lý viền hay màu sắc.
    """
    result = target_img.copy()

    if len(mask_img.shape) == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    cropped_src, cropped_mask, bounds = get_safe_roi(source_img, target_img, mask_img, center)

    if cropped_src is None:
        return result

    tx1, tx2, ty1, ty2 = bounds
    target_roi = result[ty1:ty2, tx1:tx2]

    target_roi[cropped_mask > 0] = cropped_src[cropped_mask > 0]
    result[ty1:ty2, tx1:tx2] = target_roi

    return result

def alpha_blending(source_img, target_img, mask_img, center, blur_radius=15):
    """
    Phương pháp trộn theo độ trong suốt (Feathering/Alpha Blending).
    Sử dụng bộ lọc Gaussian để làm mờ viền mặt nạ, tạo hiệu ứng chuyển tiếp mềm mại.
    """
    result = target_img.copy()

    if len(mask_img.shape) == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    cropped_src, cropped_mask, bounds = get_safe_roi(source_img, target_img, mask_img, center)

    if cropped_src is None:
        return result

    tx1, tx2, ty1, ty2 = bounds
    target_roi = result[ty1:ty2, tx1:tx2].astype(np.float32)
    source_roi = cropped_src.astype(np.float32)

    # 1. Làm mờ mặt nạ để tạo vùng chuyển tiếp
    # blur_radius phải là số lẻ
    if blur_radius % 2 == 0:
        blur_radius += 1
    blurred_mask = cv2.GaussianBlur(cropped_mask, (blur_radius, blur_radius), 0)

    # 2. Chuẩn hóa mặt nạ về khoảng [0, 1] để làm kênh Alpha
    alpha = blurred_mask.astype(np.float32) / 255.0

    # Mở rộng chiều của alpha từ (H, W) thành (H, W, 1) để nhân được với ảnh màu 3 kênh
    alpha = np.expand_dims(alpha, axis=-1)

    # 3. Áp dụng công thức trộn: I = Alpha * Tới + (1 - Alpha) * Nền
    blended_roi = (alpha * source_roi) + ((1.0 - alpha) * target_roi)

    # Cập nhật kết quả
    result[ty1:ty2, tx1:tx2] = np.clip(blended_roi, 0, 255).astype(np.uint8)

    return result

def poisson_blending(source_img, target_img, mask_img, center, mode='normal'):
    """
    Ghép miền Gradient: Giải phương trình Poisson để hòa quyện màu sắc và ánh sáng.
    """
    if len(mask_img.shape) == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    _, mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

    try:
        if mode == 'normal':
            # NORMAL_CLONE triệt tiêu hoàn toàn viền và điều chỉnh màu sắc
            return cv2.seamlessClone(source_img, target_img, mask_img, center, cv2.NORMAL_CLONE)
        elif mode == 'mixed':
            # MIXED_CLONE giữ lại một phần chi tiết viền và điều chỉnh màu sắc nhẹ hơn
            return cv2.seamlessClone(source_img, target_img, mask_img, center, cv2.MIXED_CLONE)
        else:
            raise ValueError("Chế độ không hợp lệ. Chọn 'normal' hoặc 'mixed'.")
    except Exception as e:
        print(f"Lỗi Poisson Blending: {e}")
        return target_img