import cv2
import os
import sep
import time
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
import sys
import math



# 读取仿图目标真值坐标
def get_gtCoords(path_txt, if_float=False):
    '''根据txt在rgb图像上绘制真值检测框'''
    with open(path_txt, 'r') as f:
        lines = f.readlines()
        coords_gt = [line[:-1].split(' ') for line in lines]
    if if_float:
        coords_gt = np.array(coords_gt, np.float64)
    else:
        coords_gt = np.array(coords_gt, np.int64)
    return coords_gt

# 获取指定文件夹所有指定后缀文件的完整路径
def get_fileList(path_fileDir, postfix):
    """    获取指定文件夹所有指定后缀文件的完整路径"""
    fileList = [os.path.join(path_fileDir, i) for i in os.listdir(path_fileDir) if i.endswith(postfix)]
    return fileList


def med_and_impressBkg(img, ks=3):
    '''
    对整图进行中值滤波与背景建模，返回背景抑制后的图片和背景RMS图
    '''
    img = np.array(img, np.float32)
    img_med = cv2.medianBlur(img, ks)
    bkg = sep.Background(img_med)
    bkg_image = bkg.back()
    bkg_rms = bkg.rms()
    img_impressed = img_med - bkg
    return img_impressed, bkg_rms

def search_target_outline(x,y,peak_target,mean_background,img,radius,T_target):
    w_r , h_u, w_l, h_d= 0,0,0,0
    for x_r in np.arange(radius):
        if img[y,x+x_r] < T_target or (peak_target-img[y,x+x_r])/(peak_target-mean_background) > 0.8:
            break
        w_r += 1
    for y_u in np.arange(radius):
        if img[y+y_u,x] < T_target or (peak_target-img[y+y_u,x])/(peak_target-mean_background) > 0.8:
            break
        h_u += 1 # 下
    for x_l in np.arange(radius):
        if img[y,x-x_l] < T_target or (peak_target-img[y,x-x_l])/(peak_target-mean_background) > 0.8: 
            break
        w_l += 1
    for y_d in np.arange(radius):
        if img[y-y_d,x] < T_target or (peak_target-img[y-y_d,x])/(peak_target-mean_background) > 0.8:
            break
        h_d += 1 # 上
    return   x-w_l,x+w_r,y-y_d, y+y_u

# ================================== 区域生长 ==============================================
class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
 
    def getX(self):
        return self.x
    def getY(self):
        return self.y
 
def getGrayDiff(img,currentPoint,tmpPoint):
    return img[currentPoint.x,currentPoint.y] - img[tmpPoint.x,tmpPoint.y]
 
def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [ Point(0, -1),  Point(1, 0),Point(0, 1), Point(-1, 0)]
    return connects

# 区域生长（向上）
def regionUpGrow(img, seeds, p = 1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    imgMax = img[seeds[0].x, seeds[0].y]
    maxX = seeds[0].y
    maxY = seeds[0].x
    seedList = []
    if p == 1:
        num = 8
    else:
        num = 4
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)
 
        # seedMark[currentPoint.x,currentPoint.y] = img[currentPoint.x,currentPoint.y]
        seedMark[currentPoint.x,currentPoint.y] = 1
        for i in range(num):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            # grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
            grayDiff = img[currentPoint.x,currentPoint.y] - img[tmpX,tmpY]
            # print(grayDiff)
            if grayDiff < 0  and seedMark[tmpX,tmpY] != 1:  # 灰度上升
                # seedMark[tmpX,tmpY] = img[tmpX, tmpY]
                seedMark[tmpX,tmpY] = 1
                seedList.append(Point(tmpX,tmpY))
                if img[tmpX,tmpY] > imgMax:
                    imgMax = img[tmpX,tmpY]
                    maxX = tmpY
                    maxY = tmpX
    return imgMax, maxX, maxY


# 区域生长（向下、去粘连）
def RegionGrow(img,seeds,thresh,p = 1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    imgMax = img[seeds[0].x, seeds[0].y]
    seedList = []
    seedListInc = []  # 其他凸峰
    if p == 1:
        num = 8
    else:
        num = 4
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)
 
        # seedMark[currentPoint.x,currentPoint.y] = img[currentPoint.x,currentPoint.y]
        seedMark[currentPoint.x,currentPoint.y] = 1
        for i in range(num):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            if img[tmpX, tmpY] > thresh and seedMark[tmpX,tmpY] != 1:  # 灰度下降，大于背景
                # seedMark[tmpX,tmpY] = img[tmpX, tmpY]
                seedMark[tmpX,tmpY] = 1
                seedList.append(Point(tmpX,tmpY))
    return seedMark
# ============================================================================================

"""计算梯度特征"""
def calGradMap(img):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # 计算图像的 x 方向和 y 方向的梯度
    gx = cv2.filter2D(img, -1, sobel_x)
    gy = cv2.filter2D(img, -1, sobel_y)
    # 计算图像梯度的大小和方向
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    return mag, ang

# 根据矩计算主方向及纵横比
def calTarPropertyWithMoments(img):
    """输入目标所在区域，根据矩快速计算目标主方向"""
    mX = cv2.moments(img-img.min())
    u20 = mX['mu20']
    u02 = mX['mu02']
    u11 = mX['mu11']
    # u00 = mX['m00'] if mX['m00']!=0 else 0.1
    if mX['m00']!=0:
        u00 = mX['m00']
    else:
        u00 = 0.1
    u20N = u20/u00
    u02N = u02/u00
    u11N = u11/u00
    a = (u20N+u02N)/2 + (4*u11N**2+(u20N-u02N)**2)**0.5/2
    b = (u20N+u02N)/2 - (4*u11N**2+(u20N-u02N)**2)**0.5/2
    theta = cv2.fastAtan2(2*u11N, u20N-u02N)/2
    # print("a:", a, " b:", b, " theta:", theta)
    return (a,b,theta)

# 背景估计
def backgroundEstimation(roi):
    roi_mean=np.mean(roi)
    roi_std=np.std(roi)
    means = np.mean(roi, axis=1)
    T=roi_mean-roi_std*2
    minMean = np.where(means==np.min(means[means > T]))[0][0]
    rol = roi[minMean, :]
    std = np.std(rol[rol > 1])
    mean = np.mean(rol[rol > 1])
    return mean, std

# 背景均值和标准差估计
def BackgroundModeling(img, width):
    iN = width//32
    backMean = np.zeros([iN,iN])
    backStd = np.zeros([iN,iN])
    for i in range(iN):
        for j in range(iN):
            roi = img[i*32:(i+1)*32, j*32:(j+1)*32]
            if not np.max(roi > 0.1):
                continue
            mean, std = backgroundEstimation(roi)
            backMean[i,j] = mean
            backStd[i,j] = std
    return backMean, backStd

# 计算配准精度
def registratPrecision(img_rectify, img0, bit):        
    subImg = img_rectify.astype(int) - img0.astype(int)  # 帧差图
    subList = subImg[img_rectify>0]
    MSE = (np.linalg.norm(subList, ord=2))**2/subList.shape[0]  # 计算均方误差
    PSNR = math.log10((2**bit-1)**2/MSE)
    return PSNR

# 越界处理
def crossLine(x, y, a, L):
    y0 = 0 if y-a<0 else y-a
    y1 = L if y+a+1>L else y+a+1
    x0 = 0 if x-a<0 else x-a
    x1 = L if x+a+1>L else x+a+1
    return round(y0), round(y1), round(x0), round(x1)

## 去除疑似目标中的恒星点
def deleteStars(points, img, imgRectify, width, height, H,FP1, FP2,rms,T_res):
    susPt = []
    tmap = np.zeros_like(img)
    tid=1
    for i in range(np.shape(points)[0]):
        x = round(points[i][0])
        y = round(points[i][1])
        scale = 2  # 恒星核大小
        scale1 = 5  # 噪声核大小
        sca_max = max(scale,scale1)+6
        H_x = H[0,2]
        H_y = H[1,2]
        if H_x < 0:
            H_x-=5
        else:
            H_x+=5
        if H_y < 0:
            H_y-=5
        else:
            H_y+=5
        if imgRectify[y, x]  == 0:
            continue
        if x<0+H_x or x>width-1+H_x or y<0+H_y or y>width-1+H_y:
            continue
        if x < sca_max or x > width - sca_max - 1 or y < sca_max or y > height - sca_max - 1:
            continue
        # # 估计背景均值（21*21）
        # roiDst = img[y-10:y+11, x-10:x+11].copy()
        # roiRef = imgRectify[y-10:y+11, x-10:x+11].copy()
        # # 背景估计
        # meanDst, stdDst = backgroundEstimation(roiDst)
        # meanRef, stdRef = backgroundEstimation(roiRef)

        
        # # 背景估计
        # meanDst = backMean[y//32, x//32]
        # meanRef = backMeanRectify[y//32, x//32]
        # # 减去背景估计值
        # roiDst = roiDst.astype('float')
        # roiRef = roiRef.astype('float')
        # roiDst -= meanDst
        # roiRef -= meanRef
        
        roiDst = img[y-scale:y+scale+1, x-scale:x+scale+1].copy()
        gaussianKernel = cv2.getGaussianKernel(2*scale+1, 1) * cv2.getGaussianKernel(2*scale+1,1).T
        # gaussianKernel = np.array([[2, 2,2,2,2],[2,5,5,5,2],[2,5,5,5,2],[2,5,5,5,2],[2,2,2,2,2]])
        # res = signal.convolve2d(roiDst, gaussianKernel, mode='valid')
        responseDst = cv2.filter2D(roiDst.astype('float32'), -1, gaussianKernel.astype('float32'))
        res = responseDst[scale,scale]
        # roiRef = imgRectify[y-scale-1:y+scale+2, x-scale-1:x+scale+2].copy()
        roiRef = imgRectify[y-scale:y+scale+1, x-scale:x+scale+1].copy()
        responseDiff = cv2.filter2D(roiRef.astype('float32'), -1, gaussianKernel.astype('float32'))
        # temp = responseDiff[scale:scale+3,scale:scale+3].max()
        temp = responseDiff[scale,scale]
        resDiff = abs(res - temp)/max(res,temp)  # 分布差异
        # resDiff = abs(responseDst[scale,scale] - responseDiff[scale,scale])/max(responseDst[scale,scale],responseDiff[scale,scale])  # 分布差异
        SE = 1 - resDiff  # 响应相似度

        roiDst = img[y-scale1:y+scale1+1, x-scale1:x+scale1+1].copy()
        roiRef = imgRectify[y-scale1:y+scale1+1, x-scale1:x+scale1+1].copy()
        # # 减去背景估计值
        # roiDst = roiDst.astype('float32')
        # roiRef = roiRef.astype('float32')
        # roiDst -= meanDst
        # roiRef -= meanRef
        img_difference_roi = roiDst - roiRef
        gaussianKernel_3 = cv2.getGaussianKernel(2*scale1+1, 1) * cv2.getGaussianKernel(2*scale1+1,1).T            
        # # 能量域
        # energeDst = int(np.sum(roiDst))
        # energeRef = int(np.sum(roiRef))
        # energeDiff = (energeDst - energeRef) / energeDst
        # SE = 1 - energeDiff  # 能量相似度
        # # 分布域
        # roiDiff = roiDst - roiRef  # 差值图
        # # scale = 2
        # gaussianKernel = cv2.getGaussianKernel(2*scale+1, 1) * cv2.getGaussianKernel(2*scale+1,1).T
        # responseDst = cv2.filter2D(roiDst.astype('float32'), -1, gaussianKernel.astype('float32'))
        # responseDiff = cv2.filter2D(roiDiff.astype('float32'), -1, gaussianKernel.astype('float32'))
        # resDiff = (responseDst[scale,scale] - responseDiff[scale,scale])/responseDst[scale,scale]  # 分布差异
        # SR = 1 - resDiff  # 响应相似度
        # 判断是否为 SE < 0.5 and SR > 0.7 and SM < 0.1 疑似目标  and SR > 0.7

        if SE < FP1: #0.4 and res>100
            # susPt.append([points[i][0], points[i][1], 0])
            distribution_val_diff = cv2.matchTemplate(img_difference_roi.astype('float32'), gaussianKernel_3.astype('float32'), cv2.TM_CCOEFF_NORMED)
            # res1 = distribution_val_diff.max()
            # roi_CNN = cv2.matchTemplate(roiDst.astype('float32'), gaussianKernel_3.astype('float32'), cv2.TM_CCOEFF_NORMED)
            # res2 = roi_CNN.max()
            # res = max(res1,res2)
            res= distribution_val_diff[0,0]
            ret = cv2.filter2D(img_difference_roi.astype('float32'), -1, gaussianKernel.astype('float32'))
            ret =ret[scale1,scale1]
            # distribution_val = cv2.matchTemplate(roiDst.astype('float32'), gaussianKernel_3.astype('float32'), cv2.TM_CCOEFF_NORMED)[0][0]
            # roiRef = imgRectify[y-4:y+4+1, x-4:x+4+1].copy()
            # distribution_val0 = cv2.matchTemplate(roiRef.astype('float32'), gaussianKernel.astype('float32'), cv2.TM_CCOEFF_NORMED)[0][0]
            # gradDst = cv2.filter2D(img_difference_roi[2:5,2:5].astype('float32'), -1, gradKenel.astype('float32'))
            # DS = 1 - abs(distribution_val_diff - distribution_val) / max(distribution_val_diff, distribution_val)
            if  res > FP2:    # and DS > 0.7distribution_val0 < 0.4 and and gradDst[1,1] < 0    ret> T_res*rms[y,x] and
                susPt.append([points[i][0], points[i][1], 0])
                tmap[y,x]=tid
                tid = tid+1
                # print(distribution_val0.max())
                # print(gradDst[1,1])
            # susPt.append([x, y, 0])
            # susPt.append([points[i][0], points[i][1], 0])
    susPt = np.array(susPt)
    return susPt,tmap

## 去除疑似目标中的恒星点
def deleteStarsV1(points, img, imgRectify, width, height, H,FP1, FP2):
    susPt = []
    for i in range(np.shape(points)[0]):
        x = round(points[i][0])
        y = round(points[i][1])
        scale = 3  # 恒星核大小
        scale1 = 6  # 噪声核大小
        a = 1
        sca_max = max(scale,scale1)+20
        H_x = H[0,2]
        H_y = H[1,2]
        if H_x < 0:
            H_x-=a
        else:
            H_x+=a
        if H_y < 0:
            H_y-=a
        else:
            H_y+=a
        # if img[y, x] == 0 or imgRectify[y, x]  == 0:
        #     continue
        if x<0+H_x or x>4095+H_x or y<0+H_y or y>4095+H_y:
            continue
        if x < sca_max or x > width - sca_max - 1 or y < sca_max or y > height - sca_max - 1:
            continue        
        roiDst = img[y-scale:y+scale+1, x-scale:x+scale+1].copy()
        gaussianKernel = cv2.getGaussianKernel(2*scale+1, 1) * cv2.getGaussianKernel(2*scale+1,1).T
        # gaussianKernel = np.array([[2, 2,2,2,2],[2,5,5,5,2],[2,5,5,5,2],[2,5,5,5,2],[2,2,2,2,2]])
        # res = signal.convolve2d(roiDst, gaussianKernel, mode='valid')
        responseDst = cv2.filter2D(roiDst.astype('float32'), -1, gaussianKernel.astype('float32'))
        res = responseDst[scale,scale]
        # roiRef = imgRectify[y-scale-1:y+scale+2, x-scale-1:x+scale+2].copy()
        roiRef = imgRectify[y-scale:y+scale+1, x-scale:x+scale+1].copy()
        responseDiff = cv2.filter2D(roiRef.astype('float32'), -1, gaussianKernel.astype('float32'))
        # temp = responseDiff[scale:scale+3,scale:scale+3].max()
        temp = responseDiff[scale,scale]
        resDiff = abs((res - temp)/max(res,temp))  # 分布差异
        # resDiff = abs(responseDst[scale,scale] - responseDiff[scale,scale])/max(responseDst[scale,scale],responseDiff[scale,scale])  # 分布差异
        SE = 1 - resDiff  # 响应相似度
        roiDst = img[y-scale1:y+scale1+1, x-scale1:x+scale1+1].copy()
        roiRef = imgRectify[y-scale1:y+scale1+1, x-scale1:x+scale1+1].copy()
        img_difference_roi = roiDst - roiRef
        gaussianKernel_3 = cv2.getGaussianKernel(2*scale1-1, 1) * cv2.getGaussianKernel(2*scale1-1,1).T            
        # 判断是否为 SE < 0.5 and SR > 0.7 and SM < 0.1 疑似目标  and SR > 0.7
        if SE < FP1: #0.4
            distribution_val_diff = cv2.matchTemplate(img_difference_roi.astype('float32'), gaussianKernel_3.astype('float32'), cv2.TM_CCOEFF_NORMED)
            res1 = distribution_val_diff.max()
            roi_CNN = cv2.matchTemplate(roiDst.astype('float32'), gaussianKernel_3.astype('float32'), cv2.TM_CCOEFF_NORMED)
            res2 = roi_CNN.max()
            res = max(res1,res2)
            if res > FP2:    # and DS > 0.7distribution_val0 < 0.4 and and gradDst[1,1] < 0
                susPt.append([points[i][0], points[i][1], points[i][2], points[i][3], points[i][4], points[i][5], points[i][6], i+1])
    susPt = np.array(susPt)
    return susPt

## 去除疑似目标中的恒星点
def deleteStars1(points, img, imgRectify, width, height, backMean):
    susPt = []
    backMeanRectify, backStdRectify = BackgroundModeling(imgRectify, width)
    for i in range(np.shape(points)[0]):
        x = int(points[i][1])
        y = int(points[i][0])
        scale1 = 1
        if img[y, x] < 1 or imgRectify[y, x] < 1:
            continue
        if x < scale1 or x > width - scale1 - 1 or y < scale1 or y > height - scale1 - 1:
            continue
        # # 估计背景均值（21*21）
        # roiDst = img[y-10:y+11, x-10:x+11].copy()
        # roiRef = imgRectify[y-10:y+11, x-10:x+11].copy()
        # # 背景估计
        # meanDst, stdDst = backgroundEstimation(roiDst)
        # meanRef, stdRef = backgroundEstimation(roiRef)

        roiDst = img[y-2:y+3, x-2:x+3].copy()
        roiRef = imgRectify[y-2:y+3, x-2:x+3].copy()
        # 背景估计
        meanDst = backMean[y//32, x//32]
        meanRef = backMeanRectify[y//32, x//32]
        # 减去背景估计值
        roiDst = roiDst.astype('float')
        roiRef = roiRef.astype('float')
        roiDst -= meanDst
        roiRef -= meanRef
        scale = 2
        gaussianKernel = cv2.getGaussianKernel(2*scale+1, 1) * cv2.getGaussianKernel(2*scale+1,1).T
        responseDst = cv2.filter2D(roiDst.astype('float32'), -1, gaussianKernel.astype('float32'))
        gradKenel = np.array([[-1,-1,-1],[-1,1,-1],[-1,-1,-1]])

        responseDiff = cv2.filter2D(roiRef.astype('float32'), -1, gaussianKernel.astype('float32'))
        resDiff = abs(responseDst[scale,scale] - responseDiff[scale,scale])/max(responseDst[scale,scale],responseDiff[scale,scale])  # 分布差异
        SE = 1 - resDiff  # 响应相似度
        roiDst = img[y-scale1:y+scale1+1, x-scale1:x+scale1+1].copy()
        roiRef = imgRectify[y-scale1:y+scale1+1, x-scale1:x+scale1+1].copy()
        # 减去背景估计值
        roiDst = roiDst.astype('float32')
        roiRef = roiRef.astype('float32')
        roiDst -= meanDst
        roiRef -= meanRef
        img_difference_roi = roiDst - roiRef
        gaussianKernel_3 = cv2.getGaussianKernel(2*scale1+1, 1) * cv2.getGaussianKernel(2*scale1+1,1).T            
        # # 能量域
        # energeDst = int(np.sum(roiDst))
        # energeRef = int(np.sum(roiRef))
        # energeDiff = (energeDst - energeRef) / energeDst
        # SE = 1 - energeDiff  # 能量相似度
        # # 分布域
        # roiDiff = roiDst - roiRef  # 差值图
        # # scale = 2
        # gaussianKernel = cv2.getGaussianKernel(2*scale+1, 1) * cv2.getGaussianKernel(2*scale+1,1).T
        # responseDst = cv2.filter2D(roiDst.astype('float32'), -1, gaussianKernel.astype('float32'))
        # responseDiff = cv2.filter2D(roiDiff.astype('float32'), -1, gaussianKernel.astype('float32'))
        # resDiff = (responseDst[scale,scale] - responseDiff[scale,scale])/responseDst[scale,scale]  # 分布差异
        # SR = 1 - resDiff  # 响应相似度
        # 判断是否为 SE < 0.5 and SR > 0.7 and SM < 0.1 疑似目标  and SR > 0.7
        if SE < 0.4: #0.4
                # print(distribution_val_diff)
                # print(gradDst[1,1])
            susPt.append([x, y, 0])
    susPt = np.array(susPt)
    return susPt

def decode(img, alpha1 = 2, alpha2 =2):
    decodeImg = img.copy()
    u = img.mean()
    v = img.std()

    maxVal = u + alpha1 * v
    minVal = u - alpha2 * v
    decodeImg[decodeImg < minVal] = minVal
    decodeImg[decodeImg > maxVal] = maxVal
    decodeImg = (decodeImg - minVal) / (maxVal - minVal) * 255
    return decodeImg.astype(np.uint8)

# 恒星定位(带阈值的质心法)
def starLocation(img, mask, T):
    # gaussianKernel = cv2.getGaussianKernel(9, 1) * cv2.getGaussianKernel(9,1).T
    # # # draw3D(gaussianKernel)
    # # result1 = cv2.matchTemplate(roi1.astype('float32'), gaussianKernel.astype('float32'),  cv2.TM_CCOEFF_NORMED)
    # img1 = cv2.filter2D(img.astype('float32'), -1, gaussianKernel.astype('float32')) 
    # M = cv2.moments(img)  # 求二阶矩
    # cx = M["m10"]/M["m00"]
    # cy = M["m01"]/M["m00"]
    # img1 = img.copy()
    # cv2.medianBlur(img1.astype('uint16'), 3, img1)

    gray_list = img[mask] - T # 恒星灰度列表
    list_y, list_x = np.where(mask==True)  # 恒星像素坐标列表
    sum_gray = np.sum(gray_list)  # 灰度求和
    x = np.sum(list_x * gray_list)/sum_gray
    y = np.sum(list_y * gray_list)/sum_gray
    if gray_list.shape[0]<9:
        mask[int(y)-2:int(y)+3,int(x)-2:int(x)+3] = 1
        gray_list = img[mask] - T
        list_y, list_x = np.where(mask==True)  # 恒星像素坐标列表
        sum_gray = np.sum(gray_list)  # 灰度求和
        x = np.sum(list_x * gray_list)/sum_gray
        y = np.sum(list_y * gray_list)/sum_gray
    return x, y

# 同帧疑似目标聚类
def targetCluster(susPts, img, tmap):
    susPtsNum = np.shape(susPts)[0]
    suspect_targets = []
    target_map = np.zeros_like(img)
    idx = 1
    for i in range(susPtsNum):
        x = susPts[i][0]
        y = susPts[i][1]
        # gray_value = np.max(img[y-1:y+2, x-1:x+2])
        gray_value = np.max(img[y, x])
        if gray_value <=0:
            continue
        a = 100
        y0, y1, x0, x1 = crossLine(x, y, a, np.shape(img)[0])
        img_roi = img[y0:y1,x0:x1].copy()
        map_roi = tmap[y0:y1,x0:x1].copy()
        # back_locality_mean = backMean[y//32, x//32]
        # back_locality_std = backStd[y//32, x//32]
        target_region_threshold = img_roi.mean() + (gray_value-img_roi.mean())*0.5 # 2*global_rms+
        # target_region_threshold = min(target_region_threshold,5*global_rms)
        # target_region_threshold = 5*global_rms
        # 如果目标已被聚类，则忽略
        if susPts[i][2] < 0:
            continue
        # target_peak_mask = RegionGrow(img_roi, [Point(y-y0, x-x0)], target_peak_threshold, 1)
        # 同帧同类目标聚类
        # num, labels, stats, centroids = cv2.connectedComponentsWithStats((img_roi>target_region_threshold).astype('uint8'), 8)
        # target_region_mask = (labels==labels[y-y0, x-x0]).astype('uint8')
        target_region_mask = RegionGrow(img_roi, [Point(y-y0, x-x0)], target_region_threshold, 1)
        ids = map_roi[target_region_mask>0][map_roi[target_region_mask>0]>0].astype('int')
        susPts[ids-1,2]=-1
        # for j in range(susPtsNum):
        #     x2 = susPts[j][0]
        #     y2 = susPts[j][1]
        #     if x2 >= x0 and x2 < x1 and y2 >= y0 and y2 < y1 and target_region_mask[y2-y0, x2-x0] > 0:
        #         # peak_gray.append(img[y2, x2])
        #         susPts[j][2] = -1    # 聚类标志 -1
        img_roi[target_region_mask<0.5] = 0
        # 找到轮廓
        contours, _ = cv2.findContours(target_region_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 假设最大的轮廓为主要前景对象
        if contours:
            # 获取最大的轮廓
            max_contour = max(contours, key=cv2.contourArea)
            # 计算最小外接矩形
            rect = cv2.minAreaRect(max_contour)
            (center, (width, height), angle) = rect
        M = cv2.moments(img_roi)
        if M["m00"] != 0:
            # 计算质心
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            # 计算主方向
            theta = 0.5 * np.arctan2(2 * M["mu11"], (M["mu20"] - M["mu02"]))  
        else:
            theta = angle

        # 计算位移
        d = abs(height-width)
        vx = d*np.cos(theta)
        vy = d*np.sin(theta)
        rx = round(cx + x0)
        ry = round(cy + y0)
        if rx>=0 and rx<img.shape[1] and ry>=0 and ry<img.shape[0]:
            target_map[ry,rx]=idx
            suspect_targets.append([cx + x0, cy + y0, vx, vy, 0])
            idx = idx + 1
    suspect_targets = np.array(suspect_targets,dtype=object)
    return suspect_targets, target_map

def EstimateSpeedV1(susPts, img, labels_img):
    pts_new = []
    for pts in susPts:
        cx = pts[0] - pts[2]
        cy = pts[1] - pts[3]
        pts = pts.astype(int)
        img_roi = img[pts[3]:pts[3]+pts[5],pts[2]:pts[2]+pts[4]].copy()
        labels_roi = labels_img[pts[3]:pts[3]+pts[5],pts[2]:pts[2]+pts[4]].copy()
        labels_roi = labels_roi==int(pts[-1])
        img_roi[~labels_roi] = 0
        # 找到轮廓
        contours, _ = cv2.findContours(labels_roi.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 假设最大的轮廓为主要前景对象
        if contours:
            # 获取最大的轮廓
            max_contour = max(contours, key=cv2.contourArea)
            # 计算最小外接矩形
            rect = cv2.minAreaRect(max_contour)
            (center, (width, height), angle) = rect
        M = cv2.moments(img_roi)
        if M["m00"] != 0:
            # 计算质心
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            # 计算主方向
            theta = 0.5 * np.arctan2(2 * M["mu11"], (M["mu20"] - M["mu02"]))  
        else:
            theta = angle

        # 计算位移
        d = abs(height-width)
        vx = d*np.cos(theta)
        vy = d*np.sin(theta)
        pts_new.append(np.array([cx+pts[2],cy+pts[3],vx,vy]))
    return np.array(pts_new)


        

def EstimateSpeed(susPts, img, segmap):
    suspect_targets = []
    for pt in susPts:
        img_roi = img[int(pt[4]):int(pt[5])+1,int(pt[2]):int(pt[3])+1].copy()
        vxmax = pt[3] - pt[2] + 1
        vymax = pt[5] - pt[4] + 1
        target_region_mask = segmap[int(pt[4]):int(pt[5])+1,int(pt[2]):int(pt[3])+1]==pt[-1]
        gray_value = pt[-2]
        kernel = np.ones((5,5), np.float32)
        dilate_img = cv2.dilate(img_roi, kernel)
        mask_peak = dilate_img==img_roi
        mask_peak = mask_peak * target_region_mask
        coords = np.where(mask_peak > 0)
        coords = np.array(coords).T  # 第一列为行，第二列为列
        if np.shape(coords)[0] > 1:
            gray_value = np.mean(img_roi[mask_peak])
        target_peak_threshold = gray_value * 0.6
        target_peak_mask = img_roi > target_peak_threshold
        target_peak_mask = target_peak_mask * target_region_mask
        target_peak_list = np.where(target_peak_mask > 0)
        left = np.array([min(target_peak_list[1]), (target_peak_list[0][min(np.where(target_peak_list[1] == min(target_peak_list[1]))[0])])])
        right = np.array([max(target_peak_list[1]), target_peak_list[0][max(np.where(target_peak_list[1] == max(target_peak_list[1]))[0])]])
        leftTop = np.array([min(target_peak_list[1]), min(target_peak_list[0])])
        rightBut = np.array([max(target_peak_list[1]), max(target_peak_list[0])])
        vTarget = rightBut - leftTop + 1
        vx = np.array([vTarget[0] - 4, vTarget[0] + 4])
        vy = np.array([vTarget[1] - 4, vTarget[1] + 4])
        # vx[0] = max(vx[0], 44
        # vy[0] = max(vy[1], vymax + 4)
        vx[1] = max(vx[1], vxmax + 2)
        vy[1] = max(vy[1], vymax + 2)
        direction = right -  left   # 速度方向
        if direction[1] < 0:
            vy = np.array([-vy[1], -vy[0]])
            vTarget[1] = -vTarget[1]
        Vtc = np.array([[vx[0], vy[0]], [vx[1], vy[1]]])
        Vtc = np.array([[Vtc[0,0], Vtc[0,1]], [Vtc[1,0], Vtc[1,1]], [-Vtc[0,0], -Vtc[0,1]], [-Vtc[1,0], -Vtc[1,1]]])
        suspect_targets.append([pt[0], pt[1], vTarget, Vtc, 0])
    suspect_targets = np.array(suspect_targets,dtype=object)
    return suspect_targets

# 计算两块区域的高斯响应相似度
def ResponseSimilarity(roi_dst, roi_ref, scale):
    gaussianKernel = cv2.getGaussianKernel(2*scale+1, 1) * cv2.getGaussianKernel(2*scale+1,1).T
    responseDst = cv2.filter2D(roi_dst.astype('float32'), -1, gaussianKernel.astype('float32'))
    responseRef = cv2.filter2D(roi_ref.astype('float32'), -1, gaussianKernel.astype('float32'))
    resDiff = abs(responseDst[scale,scale] - responseRef[scale,scale])/responseDst[scale,scale]  # 分布差异
    SE = 1 - resDiff  # 响应相似度
    return SE

# 在原图上绘制跟踪结果并保存
def SaveTrajectoryImg(img, targets, img_name):
    # imgback = np.ones_like(img)
    plt.imshow(decode(img),'gray')
    # 绘图
    plt.rcParams['font.sans-serif'] = ['STZhongsong']
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(targets[:, 0], targets[:, 1], s = 0.01, c='r', marker='o')
    plt.savefig(img_name + '.pdf', dpi=1000, bbox_inches='tight') # 解决图片不清晰，不完整的问题
    plt.show()


def openreadtxt(file_name):
    data = []
    file = open(file_name,'r')  #打开文件
    file_data = file.readlines() #读取所有行
    for row in file_data:
        tmp_list = row.split(' ') #按‘，’切分每行的数据
        #tmp_list[-1] = tmp_list[-1].replace('\n',',') #去掉换行符
        data.append(tmp_list) #将每行数据插入data中
    return data

def makeDir(pathSave):
    os.makedirs(os.path.dirname(pathSave), exist_ok=True)

# 导向滤波
def guideFilter(I, p, winSize, eps):

    mean_I = cv2.blur(I, winSize)      # I的均值平滑
    mean_p = cv2.blur(p, winSize)      # p的均值平滑

    mean_II = cv2.blur(I * I, winSize) # I*I的均值平滑
    mean_Ip = cv2.blur(I * p, winSize) # I*p的均值平滑

    var_I = mean_II - mean_I * mean_I  # 方差
    cov_Ip = mean_Ip - mean_I * mean_p # 协方差

    a = cov_Ip / (var_I + eps)         # 相关因子a
    b = mean_p - a * mean_I            # 相关因子b

    mean_a = cv2.blur(a, winSize)      # 对a进行均值平滑
    mean_b = cv2.blur(b, winSize)      # 对b进行均值平滑

    q = mean_a * I + mean_b
    return q