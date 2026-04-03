import cv2
import numpy as np

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

def warp_and_stitch(img1, img2, H, max_dim=8000):
    """
    Bẻ cong img1 để khớp với góc nhìn của img2 và ghép chúng lại.
    Có bổ sung chốt chặn an toàn kích thước để chống tràn RAM.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    warped_corners1 = cv2.perspectiveTransform(corners1, H)

    all_corners = np.concatenate((warped_corners1, corners2), axis=0)
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    out_width = x_max - x_min
    out_height = y_max - y_min

    # CHỐT CHẶN AN TOÀN TRÁNH TRÀN BỘ NHỚ
    if out_width > max_dim or out_height > max_dim:
        print(f" -> CẢNH BÁO: Khung hình quá lớn ({out_width}x{out_height}). Đang thu hẹp lại mức {max_dim}x{max_dim}.")
        out_width = min(out_width, max_dim)
        out_height = min(out_height, max_dim)
        x_max = x_min + out_width
        y_max = y_min + out_height

    translation_dist = [-x_min, -y_min]
    
    T = np.array([[1, 0, translation_dist[0]],
                  [0, 1, translation_dist[1]],
                  [0, 0, 1]], dtype=np.float64)

    warped_img1 = cv2.warpPerspective(img1, T.dot(H), (out_width, out_height))

    translated_img2 = np.zeros_like(warped_img1)
    
    y_start = translation_dist[1]
    y_end = min(h2 + translation_dist[1], out_height)
    x_start = translation_dist[0]
    x_end = min(w2 + translation_dist[0], out_width)
    
    # Tính toán lại vùng ROI hợp lệ trong trường hợp khung hình bị giới hạn
    roi_h = y_end - y_start
    roi_w = x_end - x_start
    
    translated_img2[y_start:y_end, x_start:x_end] = img2[:roi_h, :roi_w]

    result_img = blend_panoramas(warped_img1, translated_img2)

    return result_img

def stitch_image_sequence(image_list, method='SIFT'):
    """
    Hàm ghép nối tiếp một danh sách ảnh từ trái sang phải.
    """
    # Khởi tạo bức ảnh toàn cảnh ban đầu là ảnh đầu tiên bên trái
    panorama = image_list[0]
    
    for i in range(1, len(image_list)):
        print(f"[{method}] Đang ghép ảnh {i} và ảnh {i+1}...")
        next_img = image_list[i]
        
        # 1. Trích xuất đặc trưng dựa trên phương pháp được chọn
        if method == 'SIFT':
            kp1, des1 = extract_features_sift(panorama)
            kp2, des2 = extract_features_sift(next_img)
        else:
            kp1, des1 = extract_features_orb(panorama)
            kp2, des2 = extract_features_orb(next_img)
            
        # 2. So khớp đặc trưng
        matches = match_features(des1, des2, method=method, ratio_thresh=0.75)
        print(f" -> Tìm thấy {len(matches)} điểm khớp hợp lệ.")
        
        if len(matches) < 4:
            print(f" -> CẢNH BÁO: Không đủ điểm khớp để ghép tiếp ảnh {i+1}. Dừng lại.")
            break
            
        # 3. Tính toán ma trận biến đổi
        H, mask = calculate_homography(kp1, kp2, matches)
        
        # 4. Thực hiện căn chỉnh và ghép
        panorama = warp_and_stitch(panorama, next_img, H)
        
    return panorama