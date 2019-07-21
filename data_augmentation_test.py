#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 20:46:03 2019

@author: ymguo

Combine image crop, color shift, rotation and perspective transform together 
to complete a data augmentation script.

"""

import skimage.io as io
import numpy as np
import cv2
import random
import os
import glob
from matplotlib import pyplot as plt
#from skimage import data_dir
#from PIL import Image


def data_augmentation(f): 
#    img = io.imread(f)    # 依次读取rgb图片
    img = f
    # image crop
    img_crop = img[0:300, 0:450]    
    
    
    # color shift
    def random_light_color(img):
        # brightness
        B, G, R = cv2.split(img)
    
        b_rand = random.randint(-50, 50)
        if b_rand == 0:
            pass
        elif b_rand > 0:
            lim = 255 - b_rand
            B[B > lim] = 255         # 防止超过255 越界
            B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
        elif b_rand < 0:
            lim = 0 - b_rand
            B[B < lim] = 0            # 防止小于0 越界
            B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)
    
        g_rand = random.randint(-50, 50)
        if g_rand == 0:
            pass
        elif g_rand > 0:
            lim = 255 - g_rand
            G[G > lim] = 255
            G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
        elif g_rand < 0:
            lim = 0 - g_rand
            G[G < lim] = 0
            G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)
    
        r_rand = random.randint(-50, 50)
        if r_rand == 0:
            pass
        elif r_rand > 0:
            lim = 255 - r_rand
            R[R > lim] = 255
            R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
        elif r_rand < 0:
            lim = 0 - r_rand
            R[R < lim] = 0
            R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)
    
        img_merge = cv2.merge((B, G, R))     # 融合
        # img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)  ?
        return img_merge
    
    img_color_shift = random_light_color(img_crop)
    
    
    # rotation
    M = cv2.getRotationMatrix2D((img_color_shift.shape[1] / 2, img_color_shift.shape[0] / 2), 30, 0.85) # center, angle, scale
    img_rotate = cv2.warpAffine(img_color_shift, M, (img_color_shift.shape[1], img_color_shift.shape[0]))    # warpAffine函数：把旋转矩阵作用到图形上
    
    
    # perspective transform
    def random_warp(img, row, col):
        height, width, channels = img.shape
        # warp:
        random_margin = 60
        x1 = random.randint(-random_margin, random_margin)
        y1 = random.randint(-random_margin, random_margin)
        x2 = random.randint(width - random_margin - 1, width - 1)
        y2 = random.randint(-random_margin, random_margin)
        x3 = random.randint(width - random_margin - 1, width - 1)
        y3 = random.randint(height - random_margin - 1, height - 1)
        x4 = random.randint(-random_margin, random_margin)
        y4 = random.randint(height - random_margin - 1, height - 1)
    
        dx1 = random.randint(-random_margin, random_margin)
        dy1 = random.randint(-random_margin, random_margin)
        dx2 = random.randint(width - random_margin - 1, width - 1)
        dy2 = random.randint(-random_margin, random_margin)
        dx3 = random.randint(width - random_margin - 1, width - 1)
        dy3 = random.randint(height - random_margin - 1, height - 1)
        dx4 = random.randint(-random_margin, random_margin)
        dy4 = random.randint(height - random_margin - 1, height - 1)
    
        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
        M_warp = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(img, M_warp, (width, height))
        return img_warp
    img_warp = random_warp(img_rotate, img_rotate.shape[0], img_rotate.shape[1])
    
    return img_warp


# 获取待处理文件夹下的所有图片
# glob.glob 返回所有匹配的文件路径列表，只有一个参数pathname。
paths = glob.glob(os.path.join('/Users/ymguo/CVsummer/jpg_before/','*.jpg'))
paths.sort()    # 排序
print(paths)

i = 0
for path in paths:
    im = cv2.imread(path)    # 依次读取图片
#    pic_after = []
    pic_after = data_augmentation(im)
    print(i)
    plt.imshow(pic_after)
    plt.show()
    # 依次存储处理后并重命名的图片到新的文件夹下
    io.imsave("/Users/ymguo/CVsummer/pic_after/"+np.str(i)+'.jpg',pic_after)
    i += 1
#print(pic_after.dtype)
#print(pic_after.shape)






'''一些不太正确的尝试'''
#def file_name(file_dir):
#	for root, dirs, files in os.walk(file_dir):
#		count = 1
#		#当前文件夹所有文件
#		for i in files:
#			im=Image.open(i)
#			out=data_augmentation(im)
#			out.save('/Users/ymguo/CVsummer/image/'+str(count)+'.png','PNG')
#			count+=1
#			print(i)
# 
#file_name("/Users/ymguo/CVsummer/coll_after/")#当前文件夹
#file_name('./')#当前文件夹


    
#srcImgFolder = "/Users/ymguo/CVsummer/coll_after"
#def data(dir_proc):
#    for file in os.listdir(dir_proc):
#        fullFile = os.path.join(dir_proc, file)
#        if os.path.isdir(fullFile):
#            data_augmentation(fullFile)
#            
#
#if __name__ == "__main__":
#    data(srcImgFolder)           



 
#str=data_dir+'/*.png'
#coll_before = io.ImageCollection(str)
#coll_after = io.ImageCollection(str,load_func=data_augmentation)
# coll = io.ImageCollection(str)
# skimage.io.ImageCollection(load_pattern,load_func=None)
# 回调函数默认为imread(),即批量读取图片。

#print(len(coll_after))    # 处理后的图片数量
#print(coll_before[1].shape)
#
#plt.imshow(coll_before[1])
#plt.show()
#plt.imshow(coll_after[1])
#plt.show()
#io.imshow(coll_before[10])
#io.imshow(coll_after[10])
#cv2.imshow('raw pic', coll_before[10])
#cv2.imshow('pic after data augmentation', coll_after[10])
#key = cv2.waitKey(0)
#if key == 27:
#    cv2.destroyAllWindows()

# 循环保存c处理后的图片
#for i in range(len(coll_after)):
#    io.imsave("/Users/ymguo/CVsummer/coll_after/"+np.str(i)+'.png',coll_after[i])  








