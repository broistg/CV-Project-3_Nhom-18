import cv2

def match_features(desc1, desc2, method='SIFT', ratio_thresh=0.75):
    """
    So khớp các vector mô tả giữa 2 ảnh.
    Sử dụng kiểm tra tỷ lệ Lowe để lọc bỏ các cặp điểm không đáng tin cậy.
    """
    if desc1 is None or desc2 is None:
        return []

    if method == 'SIFT':
        # SIFT sử dụng khoảng cách Euclid (L2)
        bf = cv2.BFMatcher(cv2.NORM_L2)
    elif method == 'ORB':
        # ORB là vector nhị phân, cần sử dụng khoảng cách Hamming
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        raise ValueError("Phương pháp không hợp lệ. Chọn 'SIFT' hoặc 'ORB'.")

    # Tìm 2 điểm gần nhất cho mỗi đặc trưng
    raw_matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue

        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
            
    return good_matches

def draw_feature_matches(img1, kp1, img2, kp2, matches):
    """
    Vẽ các đường nối giữa các điểm đã khớp thành công.
    """
    return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                           matchColor = (255,255,0), singlePointColor = (255,0,0),
                           flags=cv2.DrawMatchesFlags_DEFAULT)