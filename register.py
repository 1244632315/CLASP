import cv2
import sep
import time
import numpy as np
import matplotlib.tri as tri
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from extract import extract_sep

"""点单应变换"""
def registerPoints(points, H12):
    if isinstance(points, tuple):
        points = np.array([points[0], points[1]])
        points = points.reshape(-1, 2)
        coords = np.ones((points.shape[0], 3))
        coords[:, :2] = points
        coordsR = np.dot(H12, coords.T)
        coordsR /= coordsR[-1, :]
        coordsR = coordsR[:2, :].T
        return coordsR[0][0], coordsR[0][1]
    else:
        if points.ndim == 1:
            points = points.reshape(-1, 2)
        coords = np.ones((points.shape[0], 3))
        coords[:, :2] = points
        coordsR = np.dot(H12, coords.T)
        coordsR /= coordsR[-1, :]
        coordsR = coordsR[:2, :].T
        return coordsR



"""Delaunay三角配准"""
class TriAngleRectifyWithDelaunay:
    def __init__(self, img, r=5, numStars=25, ratio=0.05, angleRatio=10, thLen=100) -> None:
        super().__init__()
        self.r = r                      # 质心定位的r
        self.numStars = numStars        # 提取星点数量
        self.lengthRatio = ratio        # 长度相对变换量
        self.angleRatio = angleRatio    # 角度相对变换量
        self.img = img
        self.thLen = thLen
        # 星点提取->精定位->构建配准三角形
        self.withdrawStars(img)
        self.starsPrecision(img)
        self.buildDelaunayTriAngs()
    
    # 输入图像，返回面积前x%大的连通域中心坐标，用于检测星点
    def withdrawStars(self, img):
        bkg = sep.Background(img, bw=32, bh=32, fw=3, fh=3)
        objects, imgBi = sep.extract(img, 3, err=bkg.globalrms, filter_type='conv', deblend_cont=1, segmentation_map=True)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.asarray(imgBi>0, np.uint8))
        sortIdx = np.argsort(stats[:, -1])
        self.pointsStars = centroids[sortIdx][-self.numStars-1:-1]
        
        # showImg(img, figName='Triangle')
        # plt.plot(self.pointsStars[:, 0], self.pointsStars[:, 1], 'ro')
        # print("num of stats: ", ret)

    # 星点精定位
    def starsPrecision(self, img):
        r = self.r
        h, w = img.shape[:2]
        for num, pointStar in enumerate(self.pointsStars):
            x0 = pointStar[0]
            y0 = pointStar[1]
            x1 = x0-r if x0>r else 0
            y1 = y0-r if y0>r else 0
            x2 = x0+r if x0+r<w else w
            y2 = y0+r if y0+r<h else h
            roi = img[int(y1):int(y2), int(x1):int(x2)]
            roiBi = roi>(roi.mean()+0.5*roi.std())
            if roiBi.sum() == 0:
                roiBi = roi>(roi.mean()+0.1*roi.std())
            x, y = 0, 0
            for row in range(roi.shape[0]):
                for col in range(roi.shape[1]):
                    if roiBi[row, col]:
                        x += roi[row, col]*col
                        y += roi[row, col]*row
            valSum = roi[roiBi].sum()
            x = x/valSum-r+x0
            y = y/valSum-r+y0
            # print("before:", x0, y0, " precisionLocate:", x, y)
            self.pointsStars[num, 0] = x
            self.pointsStars[num, 1] = y

    # 输入三个点，输出角度
    def calAngles(self, a, b, c):
        la = np.linalg.norm(b-c, 2)
        lb = np.linalg.norm(a-c, 2)
        lc = np.linalg.norm(a-b, 2)
        cosa = (la+lb-lc)/(2*(la*lb)**0.5)



    # 构建delaunay三角形
    def buildDelaunayTriAngs(self):
        coords = self.pointsStars
        # 生成给定平面点组凸包的 Delauney 三角剖分.
        triObj = tri.Triangulation(coords[:, 0], coords[:, 1])	
        # 构建线索矩阵
        triangles = triObj.triangles  # 三角形列表
        cueMatrix_length = np.zeros(triangles.shape)  # 用三角形的三边长构建边长线索矩阵
        cueMatrix_angle = np.zeros(triangles.shape)   # 用三角形的三角构建边长线索矩阵
        # 计算边长
        points1 = coords[triangles[:,0],:]
        points2 = coords[triangles[:,1],:]
        points3 = coords[triangles[:,2],:]
        cueMatrix_length[:, 0] = np.linalg.norm((points3 - points2), ord=2, axis=1)
        cueMatrix_length[:, 1] = np.linalg.norm((points3 - points1), ord=2, axis=1)
        cueMatrix_length[:, 2] = np.linalg.norm((points2 - points1), ord=2, axis=1)
        tri_ = np.stack((triangles, triangles[:,[1,2,0]], cueMatrix_length), axis=2)  # 边的端点集合
        # 按边长从小到大排序三角形的灰度和恒星点
        sortIdx = np.argsort(tri_[:,:,2])
        arraySorted = tri_[np.arange(len(tri_))[:,np.newaxis], sortIdx]
        cueMatrix_length = arraySorted[:,:,2]
        triangles_sort = arraySorted[:,:,0:2].astype(int) 
        (triangles_sort[:,1])[:,[0,1]]=(triangles_sort[:,1])[:,[1,0]]
        triangles[:,1] = (triangles_sort[:,0])[triangles_sort[:,0] == triangles_sort[:,1]]
        triangles[:,0] = (triangles_sort[:,0])[triangles_sort[:,0] != triangles_sort[:,1]]
        triangles[:,2] = (triangles_sort[:,2])[triangles_sort[:,2] == triangles_sort[:,1]]
        self.pointsTriAngle = coords[triangles]
        self.sidesTriAngle = cueMatrix_length

    # 获得单应矩阵，本帧——>目标帧
    def getH(self, ta2):
        pointsRectify = []
        for idx, side in enumerate(self.sidesTriAngle):
            ratio = self.lengthRatio
            sideMask = (abs(ta2.sidesTriAngle - side) < (ratio*side)).sum(axis=1) == 3
            if sideMask.sum() == 0:
                continue
            elif sideMask.sum() == 1:
                p1 = self.pointsTriAngle[idx]
                p2 = ta2.pointsTriAngle[sideMask].squeeze()
                pointsRectify.append(np.array([p1, p2]))
            else:
                idxTa2 = abs(ta2.sidesTriAngle-side).sum(axis=1).argmin()
                p1 = self.pointsTriAngle[idx]
                p2 = ta2.pointsTriAngle[idxTa2]
                pointsRectify.append(np.array([p1, p2]))

        # 一阶配准
        pointsRectify = np.array(pointsRectify)
        pointsRectify = pointsRectify.transpose(1,0,2,3).reshape(2,-1,2)
        pointsRectify = np.hstack([pointsRectify[0], pointsRectify[1]])
        pointsRectify = np.unique(pointsRectify, axis=0)
        p1 = pointsRectify[:,:2]
        p2 = pointsRectify[:,2:]

        # 初筛一下
        filter = abs(p1-p2).sum(axis=1)<self.thLen  # 帧间偏移过大时对应调整
        p1 = p1[filter]
        p2 = p2[filter]

        if p1.shape[0] < 6 or p2.shape[0] < 6:
            raise ValueError("Not enough points to compute homography.")
        
        H, _ = cv2.findHomography(p1, p2, cv2.RANSAC)
        return H, pointsRectify

    # 配准（本帧——>目标帧） + 帧差（目标帧-配准后）
    def getFrameDiff(self, ta2, H12):
        imgR12 = cv2.warpPerspective(self.img, H12, self.img.shape)
        imgSub = np.array(ta2.img, dtype=np.float32) - np.array(imgR12, dtype=np.float32)
        return imgSub

    # 点配准，本帧——>目标帧
    def registerPoints(self, points, H12):
        coords = np.ones((points.shape[0], 3))
        coords[:, :2] = points
        coordsR = np.dot(H12, coords.T)
        coordsR /= coordsR[-1, :]
        coordsR = coordsR[:2, :].T
        return coordsR
    
    # 图像配准，本帧——>目标帧
    def registerImg(self, ta2):
        t1 = time.time()
        H12, p12 = self.getH(ta2)
        h, w = self.img.shape
        imgR12 = cv2.warpPerspective(self.img, H12, (w, h))
        return imgR12



"""两阶段及分块三角配准"""
class TriAngleRectifyWithTwoStageAndBlock:
    def __init__(self, img, r=5, numStars=36, ratio=0.05, secondFlag=False, err=0.5, plot=(2,2)) -> None:
        super().__init__()
        self.r = r                  # 质心定位的r
        self.numStars = numStars//(plot[0]*plot[1])    # 单个plot提取星点数量
        self.lengthRatio = ratio    # 长度相对变换量
        self.img = img
        self.err = err
        self.plot = plot
        # 星点提取->精定位->构建配准三角形
        self.withdrawStars(img)
        self.starsPrecision(img)
        self.buildTriAngle()
        self.secondFlag = secondFlag
    
    # 输入图像，返回面积前x%大的连通域中心坐标，用于检测星点
    def withdrawStars(self, img):
        m, n = self.plot
        h, w = img.shape[:2]
        ratio = 0.995
        allcoords = []
        for i in range(m):
            for j in range(n):
                y0 = h//m*i
                x0 = w//n*j
                roi = img[h//m*i:h//m*(i+1), w//n*j:w//n*(j+1)]
                N = roi.size
                data = roi.flatten()
                data.sort()
                th = data[int(ratio*N)]
                bi = roi > th
                ret, labels, stats, cens = cv2.connectedComponentsWithStats(bi.astype(np.uint8))
                index = stats[:, -1].argsort()[-1-self.numStars:-1]
                coords = cens[index] + np.array([x0, y0])
                allcoords.append(coords)
        self.pointsStars = allcoords
        # showImg(img, figName='Triangle')
        # for coords in allcoords:
        #     plt.plot(coords[:, 0], coords[:, 1], 'ro')
        # print("num of stats: ", ret)

    # 星点精定位
    def starsPrecision(self, img):
        r = self.r
        h, w = img.shape[:2]
        for pointsStars in self.pointsStars:
            for num, pointStar in enumerate(pointsStars):
                x0 = pointStar[0]
                y0 = pointStar[1]
                x1 = x0-r if x0>r else 0
                y1 = y0-r if y0>r else 0
                x2 = x0+r if x0+r<w else w
                y2 = y0+r if y0+r<h else h
                roi = img[int(y1):int(y2), int(x1):int(x2)]
                roiBi = roi>(roi.mean()+0.5*roi.std())
                if roiBi.sum() == 0:
                    roiBi = roi>(roi.mean()+0.1*roi.std())
                x, y = 0, 0
                for row in range(roi.shape[0]):
                    for col in range(roi.shape[1]):
                        if roiBi[row, col]:
                            x += roi[row, col]*col
                            y += roi[row, col]*row
                valSum = roi[roiBi].sum()
                x = x/valSum-r+x0
                y = y/valSum-r+y0
                # print("before:", x0, y0, " precisionLocate:", x, y)
                pointsStars[num, 0] = x
                pointsStars[num, 1] = y

    # 构建三角形
    def buildTriAngle(self):
        self.pointsTriAngle = []
        self.sidesTriAngle = []
        for pointsStars in self.pointsStars:
            numStars = pointsStars.shape[0]
            pointsTriAngle, sidesTriAngle = [], []
            for i in range(numStars-2):
                p1 = pointsStars[i]
                for j in range(i+1, numStars-1):
                    p2 = pointsStars[j]
                    for k in range(j+1, numStars):
                        p3 = pointsStars[k]
                        coordStars = np.array([p3, p2, p1]) 
                        a1 = np.linalg.norm(p1-p2, 2)
                        a2 = np.linalg.norm(p1-p3, 2)
                        a3 = np.linalg.norm(p2-p3, 2)
                        side = np.array([a1, a2, a3])
                        indexSort = np.argsort(side)
                        coordStarsSort = coordStars[indexSort]
                        sidesSort = side[indexSort]
                        sidesTriAngle.append(sidesSort)
                        pointsTriAngle.append(coordStarsSort)
            self.pointsTriAngle.append(np.array(pointsTriAngle))
            self.sidesTriAngle.append(np.array(sidesTriAngle))
    
    # 构建delaunay三角形
    def buildDelaunayTriAngs(self):
        coords = self.pointsStars
        # 生成给定平面点组凸包的 Delauney 三角剖分.
        triObj = tri.Triangulation(coords[:, 0], coords[:, 1])	
        # 构建线索矩阵
        triangles = triObj.triangles  # 三角形列表
        cueMatrix_length = np.zeros(triangles.shape)  # 用三角形的三边长构建边长线索矩阵
        # 计算边长
        points1 = coords[triangles[:,0],:]
        points2 = coords[triangles[:,1],:]
        points3 = coords[triangles[:,2],:]
        cueMatrix_length[:, 0] = np.linalg.norm((points2 - points1), ord=None, axis=1)
        cueMatrix_length[:, 1] = np.linalg.norm((points3 - points2), ord=None, axis=1)
        cueMatrix_length[:, 2] = np.linalg.norm((points1 - points3), ord=None, axis=1)
        tri_ = np.stack((triangles, triangles[:,[1,2,0]], cueMatrix_length), axis=2)  # 边的端点集合
        # 按边长从小到大排序三角形的灰度和恒星点
        sortIdx = np.argsort(tri_[:,:,2])
        arraySorted = tri_[np.arange(len(tri_))[:,np.newaxis], sortIdx]
        cueMatrix_length = arraySorted[:,:,2]
        triangles_sort = arraySorted[:,:,0:2].astype(int) 
        (triangles_sort[:,1])[:,[0,1]]=(triangles_sort[:,1])[:,[1,0]]
        triangles[:,1] = (triangles_sort[:,0])[triangles_sort[:,0] == triangles_sort[:,1]]
        triangles[:,0] = (triangles_sort[:,0])[triangles_sort[:,0] != triangles_sort[:,1]]
        triangles[:,2] = (triangles_sort[:,2])[triangles_sort[:,2] == triangles_sort[:,1]]
        self.pointsTriAngle = coords[triangles]
        self.sidesTriAngle = cueMatrix_length

    # 获得单应矩阵，本帧——>目标帧
    def getH(self, ta2):
        pointsRectify = []
        for i in range(len(self.sidesTriAngle)):
            for idx, side in enumerate(self.sidesTriAngle[i]):
                ratio = self.lengthRatio
                sideMask = (abs(ta2.sidesTriAngle[i] - side) < (ratio*side)).sum(axis=1) == 3
                if sideMask.sum() == 0:
                    continue
                elif sideMask.sum() == 1:
                    p1 = self.pointsTriAngle[i][idx]
                    p2 = ta2.pointsTriAngle[i][sideMask].squeeze()
                    pointsRectify.append(np.array([p1, p2]))
                else:
                    idxTa2 = abs(ta2.sidesTriAngle[i]-side).sum(axis=1).argmin()
                    p1 = self.pointsTriAngle[i][idx]
                    p2 = ta2.pointsTriAngle[i][idxTa2]
                    pointsRectify.append(np.array([p1, p2]))
        # 一阶配准
        pointsRectify = np.array(pointsRectify)
        pointsRectify = pointsRectify.transpose(1,0,2,3).reshape(2,-1,2)
        pointsRectify = np.hstack([pointsRectify[0], pointsRectify[1]])
        pointsRectify = np.unique(pointsRectify, axis=0)
        # print("1st pointsRectify: ", pointsRectify.shape[0])
        p1 = pointsRectify[:,:2]
        p2 = pointsRectify[:,2:]
        filter = abs(p1-p2).sum(axis=1)<500
        p1 = p1[filter]
        p2 = p2[filter]
        pointsRectify = pointsRectify[filter]

        H, _ = cv2.findHomography(p1, p2, cv2.RANSAC)

        # 二阶配准
        p12 = np.hstack([p1, np.ones((p1.shape[0], 1))]).T
        p12 = np.dot(H, p12)
        p12 = p12/p12[-1]
        p12 = p12[:-1].T
        err = np.linalg.norm(p2 - p12, axis=1)
        mErr = err.mean()
        if mErr <= 0.5:
            return H, pointsRectify
        for _ in range(100):
            index = err.argsort()
            p12 = p12[index[:int(0.95*index.size)]]
            mask = index[:int(0.95*index.size)]
            p1 = p1[mask]
            p2 = p2[mask]
            H, _ = cv2.findHomography(p1, p2, cv2.RANSAC)
            p12 = np.hstack([p1, np.ones((p1.shape[0], 1))]).T
            p12 = np.dot(H, p12)
            p12 = p12/p12[-1]
            p12 = p12[:-1].T
            err = np.linalg.norm(p2 - p12, axis=1)
            mErr = err.mean()
            if mErr <= self.err:
                return H, np.hstack([p1, p2])
        raise ValueError("Not enough points to compute homography.")    

    # 配准（本帧——>目标帧） + 帧差（目标帧-配准后）
    def getFrameDiff(self, ta2, H12):
        # H12 = self.getH(ta2, self.secondFlag)
        imgR12 = cv2.warpPerspective(self.img, H12, self.img.shape)
        imgSub = np.array(ta2.img, dtype=np.float32) - np.array(imgR12, dtype=np.float32)
        return imgSub

    # 点配准，本帧——>目标帧
    def registerPoints(self, points, H12):
        # H12 = self.getH(ta2, self.secondFlag)
        coords = np.ones((points.shape[0], 3))
        coords[:, :2] = points
        coordsR = np.dot(H12, coords.T)
        coordsR /= coordsR[-1, :]
        coordsR = coordsR[:2, :].T
        return coordsR
    
    # 图像配准，本帧——>目标帧
    def registerImg(self, ta2):
        t1 = time.time()
        H12 = self.getH(ta2, self.secondFlag)
        print('Proposed: %.3f ms'%(time.time()-t1))

        imgR12 = cv2.warpPerspective(self.img, H12, self.img.shape)
        return imgR12




if __name__ == '__main__':
    from tool import load_img, show_img, draw3D, ax_imshow, get_fileList
    from extract import extract, extract_sep, extract_NTH, nms
    from tqdm import tqdm
    list_imgs = get_fileList('./imgs/20220607')
    sum = 0
    for idx in tqdm(range(20)):
        img1 = load_img(list_imgs[idx])
        img2 = load_img(list_imgs[idx+1])
        numStars = 25
        ta1 = TriAngleRectifyWithDelaunay(img1, numStars=numStars)
        ta2 = TriAngleRectifyWithDelaunay(img2, numStars=numStars)
        img12 = ta1.registerImg(ta2)
        sub = img12 - img2
        sum  += sub

        plt.ion()
        u, v = sub.mean(), sub.std()
        show_img(sub)
        show_img(sub > (u+v))
        pass
        

    plt.ion()
    show_img(sum)
    plt.show()
    pass