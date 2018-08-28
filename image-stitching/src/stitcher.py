import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

sift = cv.xfeatures2d.SIFT_create()

def read_image(img_name, read_mode):
    
    img = cv.imread(img_name, read_mode)
    return img


def detect_features(img1, img2):
    
    global sift
    
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, 4)
     
    good_matches = []
    for m, n, o, p in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
        if o.distance < 0.75 * p.distance:
            good_matches.append(o)
        if m.distance < 0.75 * p.distance:
            good_matches.append(m)
        if o.distance < 0.75 * n.distance:
            good_matches.append(o)            

    print(len(good_matches))
    
#     MIN_MATCH_COUNT = 100
#     if len(good_matches) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    return [src_pts, dst_pts]         

#     return None


def stitch_image(src_pts, dst_pts, img1, img2):
    
    homography_matrix, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 6.0)
    stitched_img = cv.warpPerspective(img1, homography_matrix, (img1.shape[1], img1.shape[0]))
#     stitched_img = cv.warpPerspective(img1, homography_matrix, (img1.shape[1] + img2.shape[1], img1.shape[0] + img2.shape[0]))
#     stitched_img[0 : img2.shape[0], 0 : img2.shape[1]] = img2
    
    return stitched_img
    

img1_name = "IMG_0150.JPG"
img2_name = "IMG_0151.JPG"   

img1 = read_image(img1_name, cv.IMREAD_GRAYSCALE)
img2 = read_image(img2_name, cv.IMREAD_GRAYSCALE)

feature_pts = detect_features(img1, img2)

if feature_pts == None:
    print( "Not enough matches are found")
else:    
    stitched_img = stitch_image(feature_pts[0], feature_pts[1], img1, img2)
    
    plt.imshow(stitched_img, "gray")
    plt.show()