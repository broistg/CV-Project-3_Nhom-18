import cv2
import numpy as np

def warp_and_stitch(img1, img2, H):
    """
    Bẻ cong img1 để khớp với góc nhìn của img2 và ghép chúng lại.
    Sử dụng thuật toán trộn ảnh để làm mượt đường giao nhau một cách tự nhiên.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 1. Lấy tọa độ 4 góc của cả hai ảnh
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # 2. Áp dụng ma trận H để tìm vị trí mới của 4 góc img1
    warped_corners1 = cv2.perspectiveTransform(corners1, H)

    # 3. Gộp tọa độ góc để tìm giới hạn khung hình lớn nhất
    all_corners = np.concatenate((warped_corners1, corners2), axis=0)
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # 4. Tính toán mức độ tịnh tiến để đưa các tọa độ âm về dương
    translation_dist = [-x_min, -y_min]
    
    T = np.array([[1, 0, translation_dist[0]],
                  [0, 1, translation_dist[1]],
                  [0, 0, 1]], dtype=np.float64)

    # Kích thước của ảnh toàn cảnh
    out_width = x_max - x_min
    out_height = y_max - y_min

    # 5. Khung hình thứ nhất: Bẻ cong img1 với ma trận mới
    warped_img1 = cv2.warpPerspective(img1, T.dot(H), (out_width, out_height))

    # 6. Khung hình thứ hai: Tạo mảng đen cùng kích thước và đặt img2 vào đúng vị trí tịnh tiến
    translated_img2 = np.zeros_like(warped_img1)
    
    y_start = translation_dist[1]
    y_end = h2 + translation_dist[1]
    x_start = translation_dist[0]
    x_end = w2 + translation_dist[0]
    
    translated_img2[y_start:y_end, x_start:x_end] = img2

    # 7. Hợp nhất hai khung hình bằng kỹ thuật trộn mượt
    result_img = blend_panoramas(warped_img1, translated_img2)

    return result_img

def blend_panoramas(warped_img, translated_img):
    """
    Trộn hai ảnh đã được căn chỉnh trên cùng một kích thước khung hình lớn.
    Sử dụng biến đổi khoảng cách để tạo trọng số Alpha mượt mà tại vùng giao nhau.
    """
    # 1. Tạo mặt nạ nhị phân xác định vùng có chứa dữ liệu ảnh
    gray_warped = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
    gray_translated = cv2.cvtColor(translated_img, cv2.COLOR_RGB2GRAY)
    
    mask_warped = (gray_warped > 0).astype(np.uint8)
    mask_translated = (gray_translated > 0).astype(np.uint8)

    # 2. Tính toán khoảng cách từ mỗi điểm ảnh đến phần rìa đen gần nhất
    dist_warped = cv2.distanceTransform(mask_warped, cv2.DIST_L2, 5)
    dist_translated = cv2.distanceTransform(mask_translated, cv2.DIST_L2, 5)

    # 3. Tính toán ma trận trọng số Alpha
    # Cộng thêm một lượng rất nhỏ epsilon để tránh lỗi chia cho 0
    epsilon = 1e-5
    alpha = dist_warped / (dist_warped + dist_translated + epsilon)

    # Mở rộng chiều của Alpha để có thể nhân với ma trận ảnh màu 3 kênh
    alpha = np.expand_dims(alpha, axis=-1)

    # 4. Trộn hai bức ảnh lại với nhau dựa trên trọng số đã tính
    blended_img = (warped_img * alpha) + (translated_img * (1.0 - alpha))
    
    return blended_img.astype(np.uint8)