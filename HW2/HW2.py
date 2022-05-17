import cv2
import numpy as np
import random
import math
import queue 
import sys

# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    # print(type(img))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ",img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name,img):
    cv2.imshow(window_name,img)

# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ImplementSIFT(img_gray):
    SIFT_Detector = cv2.SIFT_create()   
    kp, des = SIFT_Detector.detectAndCompute(img_gray,None)    
    return kp, des


def FeatureMatching(kp1, des1, kp2, des2, ratio):    
    good_matches = []    
    for i in range(len(des1)):
        # find KNN
        knn = [(np.inf, -1), (np.inf, -1)] # each element in knn is (norm, idx)
        for j in range(len(des2)):
            norm = np.linalg.norm(des1[i] - des2[j])
            if norm < knn[0][0]:
                knn[1] = knn[0]
                knn[0] = (norm, j)
            elif norm < knn[1][0]:
                knn[1] = (norm, j)
        # Lowe's ratio test
        if knn[0][0] < knn[1][0]*ratio:
            good_matches.append((kp1[i].pt, kp2[knn[0][1]].pt))        
    return good_matches

             
def Homography(P1, P2):
    # transform P1 to P2
    A = []  
    for point in range(len(P1)): 
        A.append([P1[point,0], P1[point,1], 1, 0, 0, 0, -P2[point,0]*P1[point,0], -P2[point,0]*P1[point,1], -P2[point,0]])
        A.append([0, 0, 0, P1[point,0], P1[point,1], 1, -P2[point,1]*P1[point,0], -P2[point,1]*P1[point,1], -P2[point,1]])
    A = np.array(A)
    U, S, VT = np.linalg.svd(A) # Solve system of linear equations Ah = 0 using SVD
    H = np.reshape(VT[-1], (3,3))        
    H = H / H[2,2] # normalization H33 to 1
    return H
    
def RANSAC(matches):   
    P1 = []
    P2 = []
    for kp1, kp2 in matches:
        P1.append(list(kp1)) 
        P2.append(list(kp2)) 
    P1 = np.array(P1)
    P2 = np.array(P2)
    
    matches_num = len(matches)
    iteration = 1000
    threshold = 1      
    max_inlier = 0
    for _ in range(iteration):
        sample_idx = random.sample(range(matches_num), 4)
        H = Homography(P1[sample_idx], P2[sample_idx])
        
        inlier_num = 0 
        for i in range(matches_num):
            kp1 = np.append(P1[i], 1)
            kp1_transform = np.dot(H, kp1.T)
            if kp1_transform[2] == 0: # avoid divide zero
                continue
            kp1_transform = kp1_transform[:2] / kp1_transform[2]
            if (np.linalg.norm(kp1_transform - P2[i]) < threshold):
                inlier_num += 1
                
        if (inlier_num > max_inlier):
            max_inlier = inlier_num
            best_H = H
    return best_H


def WarpAndBlend(H, image1,image2):
    (height1, width1) = image1.shape[:2]
    (height2, width2) = image2.shape[:2]
        
    corners = np.array([[0,0,1], [width1,0,1], [0,height1,1], [width1,height1,1]]).T 
    new_corners = np.dot(H, corners)
    new_corners = new_corners/new_corners[2]

    corner1_x =min(new_corners[:, 0][0], new_corners[:, 1][0], new_corners[:, 2][0], new_corners[:, 3][0], 0) 
    corner1_y =min(new_corners[:, 0][1], new_corners[:, 1][1], new_corners[:, 2][1], new_corners[:, 3][1], 0)     
    new_width = math.ceil(width2 + abs(corner1_x) + 1)
    new_height = math.ceil(height2 + abs(corner1_y) + 1)
    
    A = np.array([[1, 0, -corner1_x], [0, 1, -corner1_y], [0, 0, 1]])
    
    warped1 = cv2.warpPerspective(src = image1, M = np.dot(A, H), dsize = (new_width, new_height)) 
    warped2 = cv2.warpPerspective(src = image2, M = A, dsize = (new_width, new_height))
    
    overlap = np.where((np.sum(warped1, axis = 2)>0) & (np.sum(warped2, axis = 2)>0), 1, 0)      
    mask1 = np.ones((new_height, new_width))
    mask2 = np.ones((new_height, new_width))
    for i in range(len(overlap)):
        overlap_idx = np.where(overlap[i])[0]
        if (len(overlap_idx)>0):
            first = overlap_idx[0]
            last = overlap_idx[-1]
            width = last-first+1
            factor = 1/width            
            for j in range(first, last+1):
                mask1[i][j] = mask1[i][j-1]-factor
                mask2[i][j] = 1 - mask1[i][j]    
    mask1 = np.resize(np.repeat(mask1, 3), (new_height, new_width, 3))
    mask2 = np.resize(np.repeat(mask2, 3), (new_height, new_width, 3))    

    warp1 = warped1*mask1
    warp2 = warped2*mask2      
    blend_image = warp1+warp2    
    return blend_image.astype('uint8')
    

def StitchImage(image1, image1_gray, image2, image2_gray):
    kp1, des1 = ImplementSIFT(image1_gray)
    kp2, des2 = ImplementSIFT(image2_gray)   
    good_matches = FeatureMatching(kp1, des1, kp2, des2, 0.7)
    H = RANSAC(good_matches)
    stitched_image = WarpAndBlend(H, image1, image2)
    return stitched_image  
    

if __name__ == '__main__':
    imgs_path = ["./test/m" + str(i) + ".jpg" for i in range(1,11)]    
    all_images = []
    all_images_gray = []
    for img_path in imgs_path:
        img, img_gray = read_img(img_path)
        all_images.append(img)
        all_images_gray.append(img_gray)
    
    image_1_2 = StitchImage(all_images[0], all_images_gray[0], all_images[1], all_images_gray[1])    
    image_3_4 = StitchImage(all_images[2], all_images_gray[2], all_images[3], all_images_gray[3])
    image_1_2_3_4 = StitchImage(image_1_2, img_to_gray(image_1_2), image_3_4, img_to_gray(image_3_4))
    
    creat_im_window("image_1_2_3_4", image_1_2_3_4)
    im_show()