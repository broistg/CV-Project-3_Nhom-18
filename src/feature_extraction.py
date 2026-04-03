import cv2

def extract_features_sift(image):
    """
    Trích xuất đặc trưng bằng thuật toán SIFT.
    Tạo ra các điểm đặc trưng bất biến với tỷ lệ và góc xoay.
    """
    # Chuyển sang ảnh xám nếu ảnh đầu vào là ảnh màu
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def extract_features_orb(image, nfeatures=2000):
    """
    Trích xuất đặc trưng bằng thuật toán ORB.
    Tốc độ xử lý nhanh hơn SIFT, thích hợp so sánh hiệu năng.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def draw_found_keypoints(image, keypoints, color=(0, 255, 0)):
    """
    Vẽ các điểm đặc trưng lên ảnh để hiển thị trực quan trong báo cáo.
    """
    return cv2.drawKeypoints(image, keypoints, None, color=color, 
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)