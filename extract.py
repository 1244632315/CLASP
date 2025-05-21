import numpy as np
import cv2
import sep

from tool import get_roi

def get_logKernel(k=11):        
    g = np.zeros((k,k))
    for x in range(k):
        for y in range(k):
            sigma = (k-1)/6
            x0 = k//2
            y0 = k//2
            f = np.exp(((x-x0)**2+(y-y0)**2)/(-2*sigma**2))
            g[x,y] = f
    return g.astype(np.float32)

def extract(img, thSeg=0.3, kz=11, flagRetImg=False):
    thArea = 4
    img = np.array(img, np.float32)
    h, w = img.shape[:2]
    k = get_logKernel(kz).astype(np.float32)
    salientMap = cv2.matchTemplate(img, k, 5)
    u, v = salientMap.mean(), salientMap.std()
    # thSeg = (u+ratioSeg*v) if (u+ratioSeg*v) <= thSegTop else thSegTop
    mask = (salientMap > thSeg).astype(np.uint8)
    ret, labels, stats, cens = cv2.connectedComponentsWithStats(mask)
    maskarea = stats[:, -1] > thArea
    coords = cens[maskarea][1:]
    coords = (np.array(coords)+0.5+kz//2).astype(np.int64)
    if flagRetImg:
        ret = np.zeros_like(img)
        ret[kz//2:-kz//2+1, kz//2:-kz//2+1] = salientMap
        return coords, ret
    else:
        return coords

def extract_sep(img, th=5, deblend_cont=1.0):
    img = np.asarray(img, np.float32)
    bkg = sep.Background(img, bw=32, bh=32, fw=3, fh=3)
    imgBkg = np.array(bkg)
    img -= imgBkg
    kernel =  np.array([[1,2,1], [2,4,2], [1,2,1]], np.float32)
    objects = sep.extract(img, th, err=bkg.globalrms, filter_kernel=kernel, filter_type='conv', deblend_cont=deblend_cont)
    coords = [[obj['x'], obj['y']] for obj in objects]
    # coords = (np.array(coords)+0.5).astype(np.int32)
    return np.array(coords)

def extract_NTH(img, kz=(5,9), ratio=1, flagRetMap=False):
    img = img.astype(np.float32)
    # white top hat
    kz1, kz2 = kz
    thArea = 2
    k1 = np.ones((kz2, kz2), np.uint8)
    k1[kz2//2-kz1//2:kz2//2+kz1//2+1, kz2//2-kz1//2:kz2//2+kz1//2+1] = 0
    k2 = np.ones((kz2, kz2), np.uint8)
    ero = cv2.erode(img, k1)
    dil = cv2.dilate(ero, k2)
    wth = img - dil
    thSeg = wth.mean() + ratio * wth.std()
    segMap = (wth > thSeg).astype(np.uint8)
    ret, labels, stats, cens = cv2.connectedComponentsWithStats(segMap)
    maskarea = stats[:, -1] > thArea
    coords = cens[maskarea][1:]
    if flagRetMap:
        return coords, segMap
    else:
        return coords


def nms(img, coords, r):
    mask = np.zeros_like(img)
    coords_nms = []
    coords_int = np.array(coords+0.5, dtype=np.int32)
    for (x,y) in coords_int:
        mask[y, x] = img[y, x]
    for i, (x,y) in enumerate(coords):
        roi = get_roi(mask, x, y, r)
        gray = img[int(y+0.5), int(x+0.5)]
        if gray == roi.max():
            coords_nms.append([x, y])
    return np.array(coords_nms)


def deleteStars(points, img, imgRectify,  H, FP1, FP2):
    susPt = []
    tmap = np.zeros_like(img)
    tid=1
    height, width = img.shape[:2]
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
        
        roiDst = img[y-scale:y+scale+1, x-scale:x+scale+1].copy()
        gaussianKernel = cv2.getGaussianKernel(2*scale+1, 1) * cv2.getGaussianKernel(2*scale+1,1).T
        responseDst = cv2.filter2D(roiDst.astype('float32'), -1, gaussianKernel.astype('float32'))
        res = responseDst[scale,scale]
        roiRef = imgRectify[y-scale:y+scale+1, x-scale:x+scale+1].copy()
        responseDiff = cv2.filter2D(roiRef.astype('float32'), -1, gaussianKernel.astype('float32'))
        temp = responseDiff[scale,scale]
        resDiff = abs(res - temp)/max(res,temp)  # 分布差异
        SE = 1 - resDiff  # 响应相似度

        roiDst = img[y-scale1:y+scale1+1, x-scale1:x+scale1+1].copy()
        roiRef = imgRectify[y-scale1:y+scale1+1, x-scale1:x+scale1+1].copy()
        img_difference_roi = roiDst - roiRef
        gaussianKernel_3 = cv2.getGaussianKernel(2*scale1+1, 1) * cv2.getGaussianKernel(2*scale1+1,1).T            


        if SE < FP1: #0.4 and res>100
            distribution_val_diff = cv2.matchTemplate(img_difference_roi.astype('float32'), gaussianKernel_3.astype('float32'), cv2.TM_CCOEFF_NORMED)
            res= distribution_val_diff[0,0]
            ret = cv2.filter2D(img_difference_roi.astype('float32'), -1, gaussianKernel.astype('float32'))
            ret =ret[scale1,scale1]
            if  res > FP2:    # and DS > 0.7distribution_val0 < 0.4 and and gradDst[1,1] < 0    ret> T_res*rms[y,x] and
                susPt.append([points[i][0], points[i][1], 0])
                tmap[y,x]=tid
                tid = tid+1
    susPt = np.array(susPt)
    return susPt, tmap



if __name__ == '__main__':
    from tool import load_img, show_img, draw3D, get_roi
    import matplotlib.pyplot as plt
    img = load_img('./imgs/test/SKYMAPPER0015-CAM1-20221130001035774.fits')
    h, w = img.shape
    nh, nw = int(h*0.1), int(w*0.1)
    sub = cv2.resize(img, (nw, nh))
    _, coords = extract_sep(img)
    show_img(img)
    plt.plot(coords[:, 0], coords[:, 1], 'ro', ms=1)
    draw3D(get_roi(img, 6047, 833, 80))
    plt.show()