import cv2 as cv
import numpy as np
import sys
from matplotlib import pyplot as plt

print(__name__)

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

img1_name = "IMG_0150.JPG"
img2_name = "IMG_0151.JPG"   

# img1 = read_image(img1_name, cv.IMREAD_GRAYSCALE)
# img2 = read_image(img2_name, cv.IMREAD_GRAYSCALE)

# max_img_shape = [max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1])]
# print(max_img_shape)
# 
# img1 = img1.reshape(max_img_shape)
# print("Img1 shape: %s" % str(img1.shape))
# 
# img2 = img2.reshape(max_img_shape)
# print("Img2 shape: %s" % str(img2.shape))

img1 = cv.imread(img1_name)
# plt.imshow(img1)
gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
edges1 = cv.Canny(gray1, 50, 100, apertureSize = 3)
lines = cv.HoughLinesP(edges1,1,np.pi/180,50,minLineLength=50,maxLineGap=10)
for line in lines:
    x1,y1,x2,y2 = line[0]
#     cv.line(edges1, (x1,y1), (x2,y2), (255,255,255), 10)
#     cv.line(gray1, (x1,y1), (x2,y2), (255,255,255), 10)
    cv.line(gray1, (x1,y1), (x2,y2), (0,0,0), 10)

img2 = cv.imread(img2_name)
# plt.imshow(img2)
gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)   
edges2 = cv.Canny(gray2, 50, 100, apertureSize = 3)
lines = cv.HoughLinesP(edges2,1,np.pi/180,50,minLineLength=50,maxLineGap=10)
for line in lines:
    x1,y1,x2,y2 = line[0]
#     cv.line(edges2, (x1,y1), (x2,y2), (255,255,255), 10)    
#     cv.line(gray2, (x1,y1), (x2,y2), (255,255,255), 10)
    cv.line(gray2, (x1,y1), (x2,y2), (0,0,0), 10)

# fig, ax = plt.subplots(ncols=2)
# fig.canvas.set_window_title('(50, 100) lineDetectionThreshold=50 minLineLength=50 maxLineGap=10') 
# ax[0].imshow(gray1)
# ax[1].imshow(gray2)
# plt.imshow(img1)
# plt.imshow(edges1) 

# ret1, thresh1 = cv.threshold(edges1,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
ret1, thresh1 = cv.threshold(gray1,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)  
# plt.imshow(thresh1)
# noise removal
kernel = np.ones((3,3),np.uint8)
opening1 = cv.morphologyEx(thresh1,cv.MORPH_OPEN,kernel, iterations = 2)
# plt.imshow(opening1, "gray")
# sure background area
sure_bg1 = cv.dilate(opening1,kernel,iterations=3)
# Finding sure foreground area
dist_transform1 = cv.distanceTransform(opening1,cv.DIST_L2,5)
ret1, sure_fg1 = cv.threshold(dist_transform1,0.7*dist_transform1.max(),255,0)
# plt.imshow(dist_transform1)
# plt.imshow(sure_bg1)
# Finding unknown region
sure_fg1 = np.uint8(sure_fg1)
unknown1 = cv.subtract(sure_bg1,sure_fg1)
# print("Unknown: ")
# print(unknown1)
# Marker labelling
ret1, markers1 = cv.connectedComponents(sure_fg1)
# print(markers1)
# plt.imshow(markers1)
# Add one to all labels so that sure background is not 0, but 1
markers1 = markers1+1
# print(markers1)
# plt.imshow(markers1)
# Now, mark the region of unknown with zero
markers1[unknown1==255] = 0
# print(markers1)
# plt.imshow(markers1)
markers1 = cv.watershed(img1,markers1)
# plt.imshow(markers1)
img1[markers1 == -1] = [255,0,0]
# plt.imshow(img1)

ret2, thresh2 = cv.threshold(gray2,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU) 
# noise removal
opening2 = cv.morphologyEx(thresh2,cv.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg2 = cv.dilate(opening2,kernel,iterations=3)
# Finding sure foreground area
dist_transform2 = cv.distanceTransform(opening2,cv.DIST_L2,5)
ret2, sure_fg2 = cv.threshold(dist_transform2,0.7*dist_transform2.max(),255,0)
# Finding unknown region
sure_fg2 = np.uint8(sure_fg2)
unknown2 = cv.subtract(sure_bg2,sure_fg2)
# Marker labelling
ret2, markers2 = cv.connectedComponents(sure_fg2)
# Add one to all labels so that sure background is not 0, but 1
markers2 = markers2+1
# Now, mark the region of unknown with zero
markers2[unknown2==255] = 0
markers2 = cv.watershed(img2,markers2)
# plt.imshow(markers2)
img2[markers2 == -1] = [255,0,0]
# plt.imshow(img2)

_, ax = plt.subplots(ncols=2)
ax[0].imshow(markers1)
ax[1].imshow(markers2)
 
plt.show()  
 
sys.exit()

################################################################################################################################################

sift = cv.xfeatures2d.SIFT_create()

kp1, desc1 = sift.detectAndCompute(gray1, None)
# img1 = cv.drawKeypoints(img1, kp1, img1)

# print("Descriptor for Image 1")
# print(type(desc1))
# print(desc1.ndim)
# print(desc1.shape)
# print(desc1[0, ])

kp2, desc2 = sift.detectAndCompute(gray2, None)
# img2 = cv.drawKeypoints(img2, kp2, img2)

# print("Descriptor for Image 2")
# print(type(desc2))
# print(desc2.ndim)
# print(desc2.shape)
# print(desc2[0, ])

# index_params = cv.flann.KDTreeIndexParams (trees=5)
# search_params = cv.flann.SearchParams(checks=50)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
# flann = cv.FlannBasedMatcher_create()
flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(desc1, desc2, 2)
# matches = flann.knnMatch(desc2, desc1, 2)
# matchesMask = [[0,0] for i in range(len(matches))]
# for i,(m,n) in enumerate(matches):
#     if 0.55*n.distance<m.distance < 0.80*n.distance:
#         matchesMask[i]=[1,0]
# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                     singlePointColor = None,
#                     matchesMask = matchesMask, # draw only inliers
#                     flags = 2)
# img_with_matches = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
# plt.imshow(img_with_matches, 'gray'), plt.show()
print("Matches Length: %d" % len(matches))
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
# good = good2
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
    
#     xh = np.linalg.inv(M)
#     print("Inverse Homography Matrix:")
#     print(xh)
#     # start_p is denoted by f1
#     f1 = np.dot(xh, np.array([0,0,1]))
#     f1 = f1/f1[-1]
#     # transforming the matrix 
#     xh[0][-1] += abs(f1[0])
#     xh[1][-1] += abs(f1[1])
#     ds = np.dot(xh, np.array([img1.shape[1], img1.shape[0], 1]))
#     offsety = abs(int(f1[1]))
#     offsetx = abs(int(f1[0]))
#     # dimension of warped image
#     dsize = (int(ds[0])+offsetx, int(ds[1]) + offsety)
#     print("image dsize =>", dsize)
#     tmp = cv.warpPerspective(img1, xh, dsize)
#     tmp[offsety:img2.shape[0]+offsety, offsetx:img2.shape[1]+offsetx] = img2
#     plt.imshow(tmp, "gray")
#     plt.show()
    
#     matchesMask = mask.ravel().tolist()
#     h,w,d = img1.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv.perspectiveTransform(pts,M)
#     img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)

    stitched_img = cv.warpPerspective(gray1, M, (gray1.shape[1] + gray2.shape[1], gray1.shape[0] + gray2.shape[0]))
#     stitched_img[0 : img2.shape[0], 0 : img2.shape[1]] = img2
    plt.imshow(stitched_img, "gray")
    plt.show()
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
#     matchesMask = None     
# 
# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
# img_with_matches2 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
# plt.imshow(img4, 'gray'),plt.show()
# fig, ax = plt.subplots(nrows=2)
# ax[0].imshow(img_with_matches, "gray")
# ax[1].imshow(img_with_matches2, "gray")
# ax[2].imshow(img5, "gray")
# plt.show()



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
