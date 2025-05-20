import cv2
import numpy as np

def is_valid_match(query_img, db_img):
    orb = cv2.ORB_create()
    q_img = np.array(query_img)
    d_img = np.array(db_img)
    kp1, des1 = orb.detectAndCompute(q_img, None)
    kp2, des2 = orb.detectAndCompute(d_img, None)

    if des1 is None or des2 is None:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 4:
        return False

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return mask is not None and mask.sum() / len(matches) > 0.3