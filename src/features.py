import cv2
import numpy as np

def detect_and_match(img1, img2, ratio_thresh=0.75):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)

    pts1, pts2, good_matches = [], [], []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
            
    return np.array(pts1), np.array(pts2), kp1, kp2, good_matches

def detect_and_match(img1, img2, ratio_thresh=0.75):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)

    pts1, pts2, good_matches = [], [], []
    
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
            
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    if len(pts1) > 0:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        pts1 = cv2.cornerSubPix(gray1, np.float32(pts1), (5, 5), (-1, -1), criteria)
        pts2 = cv2.cornerSubPix(gray2, np.float32(pts2), (5, 5), (-1, -1), criteria)
    
    return pts1, pts2, kp1, kp2, good_matches