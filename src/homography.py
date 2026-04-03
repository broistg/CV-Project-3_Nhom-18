import cv2
import numpy as np
from src import image_blending

def get_homography(src_pts, dst_pts, method=cv2.RANSAC):
    H, mask = cv2.findHomography(src_pts.astype(np.float32), dst_pts.astype(np.float32), method, 5.0)
    return H, mask

def apply_transform(image, H, size=None):
    h, w = image.shape[:2]
    return cv2.warpPerspective(image, H, size if size else (w, h))

def overlay_image(background, foreground, H, use_poisson=False):
    """
    Ghép ảnh phối cảnh. Tích hợp tùy chọn ghép trực tiếp hoặc dùng Poisson Blending.
    """
    h_fg, w_fg = foreground.shape[:2]
    h_bg, w_bg = background.shape[:2]

    # Bẻ cong vật thể và mặt nạ
    warped_fg = apply_transform(foreground, H, (w_bg, h_bg))
    mask = np.ones((h_fg, w_fg), dtype=np.uint8) * 255
    warped_mask = apply_transform(mask, H, (w_bg, h_bg))

    # Khử răng cưa cho viền mặt nạ do quá trình nội suy
    _, warped_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)

    x, y, w, h = cv2.boundingRect(warped_mask)
    center = (x + w // 2, y + h // 2)

    if not use_poisson:
        # Dùng Alpha Blending cho ghép trực tiếp
        result = image_blending.alpha_blending(warped_fg, background, warped_mask, center)
    else:
        # Dùng Poisson Blending
        try:
            result = background.copy()
            result = image_blending.poisson_blending(warped_fg, result, warped_mask, center)
        except Exception as e:
            print(f"Lỗi Poisson: {e}. Đang chuyển về Alpha Blending.")
            result = image_blending.alpha_blending(warped_fg, background, warped_mask, center)

    return result

def automatic_find_dst_pts(template_img, background_img, top_matches=50):
    orb = cv2.ORB_create(nfeatures=2000)
    
    kp1, des1 = orb.detectAndCompute(template_img, None)
    kp2, des2 = orb.detectAndCompute(background_img, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    
    # Chỉ lấy top các điểm tốt nhất để giảm nhiễu cho RANSAC
    good_matches = matches[:top_matches]
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, _ = get_homography(src_pts, dst_pts)

    if H is None:
        raise ValueError("Không thể tính toán ma trận Homography")
    
    h, w = template_img.shape[:2]
    template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    found_corners = cv2.perspectiveTransform(template_corners, H)
    
    return found_corners.reshape(4, 2), H