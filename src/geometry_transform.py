import cv2
import numpy as np

def adjust_bounding_box(rows, cols, M, keep_original_space=False):
    """
    Hàm dùng chung để tính toán lại hộp giới hạn và tự động bù đắp tịnh tiến.
    Giúp ảnh không bao giờ bị cắt xén khi bị đẩy ra khỏi tọa độ âm.
    """
    # 1. Xác định 4 góc của ảnh gốc
    corners = np.float32([
        [0, 0],
        [cols, 0],
        [cols, rows],
        [0, rows]
    ])

    # 2. Tính toán tọa độ 4 góc sau khi nhân với ma trận M
    new_corners = np.zeros_like(corners)
    for i in range(4):
        x, y = corners[i]
        new_corners[i][0] = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        new_corners[i][1] = M[1, 0] * x + M[1, 1] * y + M[1, 2]

    # 3. Tìm tọa độ biên mới
    if keep_original_space:
        # Giữ lại không gian gốc để trực quan hóa sự dịch chuyển
        min_x = min(0, np.min(new_corners[:, 0]))
        max_x = max(cols, np.max(new_corners[:, 0]))
        min_y = min(0, np.min(new_corners[:, 1]))
        max_y = max(rows, np.max(new_corners[:, 1]))
    else:
        # Ôm sát vào ảnh biến đổi mới
        min_x = np.min(new_corners[:, 0])
        max_x = np.max(new_corners[:, 0])
        min_y = np.min(new_corners[:, 1])
        max_y = np.max(new_corners[:, 1])

    # 4. Tính lượng tịnh tiến bù đắp nếu ảnh bị đẩy sang không gian âm
    tx_offset = -min_x if min_x < 0 else 0
    ty_offset = -min_y if min_y < 0 else 0

    # 5. Cập nhật ma trận an toàn
    M_safe = M.copy()
    M_safe[0, 2] += tx_offset
    M_safe[1, 2] += ty_offset

    # 6. Tính kích thước khung vẽ mới bao trọn được ảnh
    new_cols = int(np.ceil(max_x + tx_offset))
    new_rows = int(np.ceil(max_y + ty_offset))

    return M_safe, (new_cols, new_rows)

def scale_image(image, fx=1.0, fy=1.0):
    """Phép co giãn ảnh."""
    rows, cols = image.shape[:2]
    M_scale = np.float32([[fx, 0, 0], [0, fy, 0]])
    M_safe, new_size = adjust_bounding_box(rows, cols, M_scale)
    return cv2.warpAffine(image, M_safe, new_size)

def rotate_image(image, angle=0.0):
    """Phép xoay ảnh chuẩn toán học (Góc dương = Ngược chiều kim đồng hồ)."""
    rows, cols = image.shape[:2]
    theta = np.radians(angle)
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)

    M_rot = np.float32([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0]
    ])

    M_safe, new_size = adjust_bounding_box(rows, cols, M_rot)
    return cv2.warpAffine(image, M_safe, new_size)

def translate_image(image, tx=0, ty=0):
    """Phép tịnh tiến chuẩn toán học (ty dương = dịch lên trên)."""
    rows, cols = image.shape[:2]
    actual_ty = -ty

    M_trans = np.float32([
        [1, 0, tx],
        [0, 1, actual_ty]
    ])

    M_safe, new_size = adjust_bounding_box(rows, cols, M_trans, keep_original_space=True)
    return cv2.warpAffine(image, M_safe, new_size)

def mirror_image(image, direction='none'):
    """Phép lật ảnh."""
    rows, cols = image.shape[:2]
    if direction == 'horizontal':
        M_mirror = np.float32([[-1, 0, cols], [0, 1, 0]])
    elif direction == 'vertical':
        M_mirror = np.float32([[1, 0, 0], [0, -1, rows]])
    elif direction == 'both':
        M_mirror = np.float32([[-1, 0, cols], [0, -1, rows]])
    else:
        return image.copy()

    return cv2.warpAffine(image, M_mirror, (cols, rows))

def shear_image(image, shear_x=0.0, shear_y=0.0):
    """Phép trượt ảnh chuẩn toán học."""
    rows, cols = image.shape[:2]
    M_shear = np.float32([
        [1, -shear_x, 0],
        [-shear_y, 1, 0]
    ])

    M_safe, new_size = adjust_bounding_box(rows, cols, M_shear, keep_original_space=True)
    return cv2.warpAffine(image, M_safe, new_size)

def affine_transform(image, src_pts, dst_pts, return_matrix=False):
    """Phép biến đổi Affine dựa trên 3 điểm."""
    rows, cols = image.shape[:2]
    M_affine = cv2.getAffineTransform(src_pts, dst_pts)

    # Sử dụng hàm dùng chung để giải quyết hoàn toàn lỗi cắt xén ảnh của Affine
    M_safe, new_size = adjust_bounding_box(rows, cols, M_affine, keep_original_space=True)
    result = cv2.warpAffine(image, M_safe, new_size)

    if return_matrix:
        return result, M_safe
    return result

def projective_transform(image, src_pts, dst_pts, return_matrix=False):
    """
    Phép biến đổi phối cảnh dựa trên 4 điểm.
    Tự động tính toán khung chứa mới và nhân ma trận tịnh tiến để chống cắt xén ảnh.
    """
    rows, cols = image.shape[:2]

    # 1. Tính ma trận phối cảnh 3x3 ban đầu
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 2. Xác định 4 góc của ảnh gốc (Định dạng mảng 3 chiều cho OpenCV)
    corners = np.float32([[[0, 0], [cols, 0], [cols, rows], [0, rows]]])

    # 3. Ánh xạ 4 góc qua ma trận H
    # Hàm perspectiveTransform tự động xử lý phép chia cho tọa độ thuần nhất
    new_corners = cv2.perspectiveTransform(corners, H)[0]

    # 4. Tìm tọa độ biên
    min_x = min(0, np.min(new_corners[:, 0]))
    max_x = max(cols, np.max(new_corners[:, 0]))
    min_y = min(0, np.min(new_corners[:, 1]))
    max_y = max(rows, np.max(new_corners[:, 1]))

    tx_offset = -min_x if min_x < 0 else 0
    ty_offset = -min_y if min_y < 0 else 0

    # 5. Tạo ma trận tịnh tiến 3x3
    T = np.float32([
        [1, 0, tx_offset],
        [0, 1, ty_offset],
        [0, 0, 1]
    ])

    # 6. Cập nhật ma trận an toàn bằng phép nhân ma trận (T * H)
    H_safe = np.dot(T, H)

    # 7. Tính kích thước khung chứa mới
    new_cols = int(np.ceil(max_x + tx_offset))
    new_rows = int(np.ceil(max_y + ty_offset))

    # Thực hiện biến đổi với ma trận và kích thước đã an toàn
    result = cv2.warpPerspective(image, H_safe, (new_cols, new_rows))

    if return_matrix:
        return result, H_safe
    return result