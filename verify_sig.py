# -*- coding: utf-8 -*-

"""
Spyder Editor

"""
# importing required packages/libraries
import cv2
import os

#for getting the test image 
print("\n")
print("\n")
test_img = input("Please enter test image name:  ")

#path of test image folder
test_img_path = "./test_sig"


#path of master signature folder
master_dataset_fold = "./master_sig_dataset"

#listing images in master signature folder
master_dataset_list =  os.listdir(master_dataset_fold)

#reading test image in gray scale
img1 = cv2.imread(os.path.join(test_img_path,test_img), 0)

#finding test image dimension
height, width = img1.shape

print("type img1: ",type(img1))

#calling orb function from opencv library

#nfeatures=500

orb = cv2.ORB_create()   
kp1, des1 = orb.detectAndCompute(img1, None)
print(len(kp1))


temp = 0
match_list = []
img_list = []

#checking matching test signature with each master signature images

for i in master_dataset_list:
    print("\n")
    print((os.path.join(master_dataset_fold,i)))
    img2 = cv2.imread(os.path.join(master_dataset_fold,i), 0)
    img2 = cv2.resize(img2,(width,height))
    
    print(type(img2))
    
    kp2, des2 = orb.detectAndCompute(img2, None)
    print(len(kp2))
    
    # matcher takes normType, 
    # which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    try:
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        print("len matches: ",len(matches))
        match_list.append(len(matches))
        img_list.append(os.path.join(master_dataset_fold,i))
        
        # drawing first 50 matches

        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
        
        #match_img = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow('Matches', cv2.resize(match_img,(600,600)))
        
        cv2.waitKey()
        cv2.destroyAllWindows()
    except:
        print("Matching error in image feature points")



print("\n")
print(match_list)

print("\n")
print(img_list)

#Return the max value of the list

max_value = max(match_list)   
max_index = match_list.index(max_value)

matching_image= img_list[max_index]

print("\n")
print("Test images matches with: ", matching_image)
print("\n")

