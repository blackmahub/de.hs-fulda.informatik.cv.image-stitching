import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import flag

def read_image(img_name, read_mode):
    img = cv.imread(img_name, read_mode)
    return img

def display_image(img):
#     print(img)
#     print(img.shape)
#     print(img.size)
#     print(img.ndim)
#     print(type(img))
#     print(img[100:105, ].shape)
#     print(img[100:, :].shape)

#     img_roi = img[1000:1010, 1000:1500]
#     cv.imshow("image roi", img_roi)    

    cv.imshow("image", img)
    
#     img_roll = np.roll(img, -500, axis = 0)
#     cv.imshow("image roll", img_roll)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

img1_name = "IMG_0149.JPG"
img2_name = "IMG_0150.JPG"
img3_name = "IMG_0151.JPG"
img4_name = "IMG_0152.JPG"    

img1 = read_image(img1_name, cv.IMREAD_GRAYSCALE)
img2 = read_image(img2_name, cv.IMREAD_GRAYSCALE)
img3 = read_image(img3_name, cv.IMREAD_GRAYSCALE)
img4 = read_image(img4_name, cv.IMREAD_GRAYSCALE)

# max_img_shape = [max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1])]
# print(max_img_shape)
# 
# img1 = img1.reshape(max_img_shape)
# print("Img1 shape: %s" % str(img1.shape))
# 
# img2 = img2.reshape(max_img_shape)
# print("Img2 shape: %s" % str(img2.shape))


sift = cv.xfeatures2d.SIFT_create()

kp1, desc1 = sift.detectAndCompute(img1, None)
img1 = cv.drawKeypoints(img1, kp1, img1)

print("Descriptor for Image 1")
print(type(desc1))
print(desc1.ndim)
print(desc1.shape)
print(desc1[0, ])

kp2, desc2 = sift.detectAndCompute(img2, None)
img2 = cv.drawKeypoints(img2, kp2, img2)

print("Descriptor for Image 2")
print(type(desc2))
print(desc2.ndim)
print(desc2.shape)
print(desc2[0, ])

# index_params = cv.flann.KDTreeIndexParams (trees=5)
# search_params = cv.flann.SearchParams(checks=50)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
# flann = cv.FlannBasedMatcher_create()
flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(desc1, desc2, 2)
# matches = flann.knnMatch(desc2, desc1, 2)
matchesMask = [[0,0] for i in range(len(matches))]
for i,(m,n) in enumerate(matches):
    if 0.55*n.distance<m.distance < 0.80*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
img_with_matches = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
# plt.imshow(img3, 'gray'), plt.show()
# print(len(matches))
# print(matches)
# print(matches[0][0].distance)
# print(matches[0][1].distance)
# 
good1 = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good1.append(m)
         
print("Good1 Matches Length: %d" % len(good1))        
# print(good1)   

good2 = []
for m, n in matches:
    if 0.55*n.distance<m.distance < 0.80*n.distance:
        good2.append(m)
         
print("Good2 Matches Length: %d" % len(good2))        
# print(good2)   

good = good1
MIN_MATCH_COUNT = 100
 
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#     
#     print(src_pts.ndim)
#     print(src_pts)
#     print(dst_pts.ndim)
#     print(dst_pts)
#     
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    
    print("Homography Matrix:")
    print(M)
    
    inv_matrix_M = np.linalg.inv(M)
    print("Inverse Homography Matrix:")
    print(inv_matrix_M)
#     inv_matrix_M = np.dot(in)
    
    matchesMask = mask.ravel().tolist()
#     h,w,d = img1.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv.perspectiveTransform(pts,M)
#     img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)

    stitched_img = cv.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0] + img2.shape[0]))
    stitched_img[0 : img2.shape[0], 0 : img2.shape[1]] = img2
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None     
# 
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img_with_matches2 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
# plt.imshow(img4, 'gray'),plt.show()
# fig, ax = plt.subplots(nrows=3)
# ax[0].imshow(img3, "gray")
# ax[1].imshow(img4, "gray")
# ax[2].imshow(img5, "gray")
plt.imshow(stitched_img, "gray")
plt.show()



# fig, axs = plt.subplots(ncols=2)
# print(axs.ndim)
# print(axs.shape)
# print(axs)
# axs[0].imshow(img1)
# axs[1].imshow(img2)
# plt.show()



# display_image(img2)

# img_grayscale = read_image(img_name, cv.IMREAD_GRAYSCALE)
# display_image(img_grayscale)

# img_color = read_image(img_name, cv.IMREAD_COLOR)
# display_image(img_color)

# img_unchanged = read_image(img_name, cv.IMREAD_UNCHANGED)
# display_image(img_unchanged)

# height, width = img_unchanged.shape[:2]
# pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
# pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
# pts1 = np.float32([[0,0], [width, 0], [0, height], [width, height]])
# pts2 = np.float32([[0,height / 2], [width / 2, 0], [width, height / 2], [width / 2, height]])
# print(type(pts1))
# perspective_transform_matrix = cv.getPerspectiveTransform(pts1, pts2)
# dst_img = cv.warpPerspective(img_unchanged, perspective_transform_matrix, (width, height))
# display_image(dst_img)
# cv.waitKey(0)
# cv.destroyAllWindows()
