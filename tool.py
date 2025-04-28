import os
import cv2
import time
import numpy as np
from astropy.io import fits
import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# 读取Raw图片
def load_rawImg(pathImg, numHead=0, numRow=0, h=4096, w=4096, flagInvert=True, ratio_trunc=2):
    numData = numHead + h*w*2 + numRow*h                # 文件字节数
    arr = np.fromfile(pathImg, np.uint8, numData)      # 读取
    img = np.zeros((h, w), np.uint16)
    data = arr[numHead:]
    if data.size != (h*(w*2+numRow)):
        numPad = h*(w*2+numRow)-data.size
        arr = np.hstack([arr, np.zeros((numPad))])
    data = arr[numHead:].reshape(h, w*2+numRow)
    data = data[:, :w*2]
    if flagInvert:
        img = data[:, 0::2]*256 + data[:, 1::2]
    else:
        img = data[:, 0::2] + data[:, 1::2]*256
    return img

# 读取fits星图
def load_fitsImg(path_img, ratio_down=2, ratio_top=2):
    raw_picture=fits.open(path_img, ignore_missing_simple=True)   #读取fits文件
    # raw_picture.info()                     #看文件的信息有几层
    # header = raw_picture[0].header             #看头文件
    # print(f'exp:{header['EXPTIME']}, temp:{header['CAM-TEMP']}')
    # raw_picture[0].data                 #看文件数据 
    img = raw_picture[0].data
    if "LVT01" in path_img:
        img = img[::-1, :]
    u = np.mean(img)
    v = np.std(img)
    return img


# 读取tif星图
def load_tifImg(path_img, ratio_trunc =2):
    """读取tif星图, 截断标准差默认为2"""
    img = cv2.imread(path_img, cv2.IMREAD_UNCHANGED)
    return img


# 加载星图
def load_img(pathImg):
    if pathImg.endswith('.tif'):
        img = load_tifImg(pathImg)
    elif pathImg.endswith('.fits'):
        img = load_fitsImg(pathImg)
    elif pathImg.endswith('.fit'):
        img = load_fitsImg(pathImg)
    elif pathImg.endswith('.FIT'):
        img = load_fitsImg(pathImg)
    elif pathImg.endswith('.raw'):
        img = load_rawImg(pathImg)
    else:
        img = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)
    return img.astype(np.float32)


# 获取指定文件夹所有指定后缀文件的完整路径
def get_fileList(path_fileDir, postfix=None):
    """    获取指定文件夹所有指定后缀文件的完整路径"""
    if postfix == None:
        fileList = []
        for postfix in ['.tif', '.fits', '.FIT', '.raw', '.png']:
            fileList +=  [os.path.join(path_fileDir, i) for i in os.listdir(path_fileDir) if i.endswith(postfix)]
    else:
        fileList = [os.path.join(path_fileDir, i) for i in os.listdir(path_fileDir) if i.endswith(postfix)]
    return fileList


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

# 获取当前日期
def get_date():
    t = time.localtime(time.time())
    day = t.tm_mday
    month = t.tm_mon
    date = '%d_%d'%(month, day)
    return date


# 读取人卫站fits头文件
def load_fitsHeader(data):
    """加载fits图头文件信息，此函数针对人卫站9-24数据进行解析，返回接口数据1-5行"""
    header = data[0].header.cards
    header_bitpix = header[1]           # 图像位数
    header_naxis = header[2]            # 图像通道数
    header_naxis1 = header[3]           # 图像高
    header_naxis2 = header[4]           # 图像宽
    header_bzero = header[8]            # 数据偏移量
    header_bscale = header[9]           # 数据缩放量
    header_obsTime = header[10]         # 曝光中间时刻   
    header_sysTime = header[11]         # 系统时间
    header_expTime = header[12]         # 曝光时间
    header_expTime = header[13]         # 设定曝光时间
    header_camSeialNum = header[14]     # 相机序列号
    header_extName = header[15]         # 望远镜名称
    header_HA = header[17]              # 时角，赤经(RA)与恒星时(LST)差值
    header_DE = header[18]              # 赤纬？
    header_obsMode = header[19]         # 观测模式
    # 测站经纬度和海拔高度
    longitude = '126.331389'
    latitude = '43.824167'
    height = '299.0'
    line1 = [longitude, latitude, height]
    # 望远镜视场和焦距
    fov = '6.5'
    f = '324.0'
    line2 = [fov, f]
    # 像元数和像元尺寸, 默认9.0
    line3 = [header_naxis1[1], header_naxis2[1], '9.0', '9.0']
    # 时间
    sysTime = header_obsTime[1]
    date, t = sysTime.split('T')
    date = date.split('-')
    t = t.split(':')
    line4 = date+t[:-1]+[str(round(float(t[-1]), 4))]
    # 望远镜指向
    DE = str(header_DE[1])
    HA = str(header_HA[1])
    line5 = [HA, DE]
    # 前五行数据
    lines = [line1, line2, line3, line4, line5]
    return lines  

# 创建文件夹
def makeDir(pathDir):
    if not os.path.exists(pathDir):
        os.makedirs(pathDir)
    else:
        print("%s has been built"%pathDir)


# 图像保存为Fits文件
def saveImgToFits(img, pathSaveImg):
    from astropy.io import fits
    # 创建一个头部对象
    # header = fits.Header()
    # header['OBJECT'] = 'Example Object'
    # header['DATE'] = '2023-00-00'
    # 保存为 FITS 文件（包含头部信息）
    fits.writeto(pathSaveImg, img, header=None, overwrite=True)


# 保存16图为rgb图像
def convert_16_to_RGB(img):
    u, v = img.mean(), img.std()
    vmax, vmin = u+2*v, u-2*v
    dec = np.clip(img, vmin, vmax)
    oup = ((dec-vmin)/(vmax-vmin)*255)
    oup = np.array([oup, oup, oup], np.uint8).transpose(1,2,0)
    return np.ascontiguousarray(oup)


# 将指定路径下的所有图片转为Fits文件并保存
def transformImgsToFits(pathImgDir, postfix='.tif'):
    listImgs = get_fileList(pathImgDir, postfix)
    pathSaveImgDir = os.path.join(pathImgDir, 'Fits')
    makeDir(pathSaveImgDir)
    for pathImg in listImgs:
        nameImg = os.path.basename(pathImg)
        img, _ = load_img(pathImg)
        pathSaveImg = os.path.join(pathSaveImgDir, nameImg.replace(postfix, '.fits'))
        saveImgToFits(img, pathSaveImg)
        print(f"Save img to: {pathSaveImg}")


# 将16位灰度图转为可视化RGB图像，便于标注矩形框、文字等
def cvt_16uimg2RGB(img, alpha=2):
    u = np.mean(img)
    v = np.std(img)
    img_show = np.array(img.copy(), np.float64)
    img_show[img_show>u+alpha*v] = u+4*v
    img_show[img_show<u-alpha*v] = u-alpha*v
    img_show = np.array((img_show - img_show.min())/(img_show.max()-img_show.min())*255, np.uint8)
    img_show = cv2.cvtColor(img_show, cv2.COLOR_GRAY2BGR)    
    return img_show



# 整批图像解码
def decodeImgDir(pathImgDir, suffix, ratioTrunc=4, flagDecode=True, flagShow=True, scale=1):
    from tqdm import tqdm
    if '.' not in suffix:
        suffix = '.'+suffix
    pathDecodeDir = os.path.join(pathImgDir, 'decode')
    pathShowDir = os.path.join(pathImgDir, 'show')
    if flagDecode: makeDir(pathDecodeDir)
    if flagShow: makeDir(pathShowDir)
    imgList = get_fileList(pathImgDir, suffix)
    for pathImg in tqdm(imgList, desc='img'):
        nameImg = os.path.basename(pathImg)
        pathDecodeImg = os.path.join(pathDecodeDir, nameImg.replace(suffix, '.tif'))
        pathShowImg = os.path.join(pathShowDir, nameImg.replace(suffix, '.png'))
        imgTrunc = None
        if "raw" in suffix and suffix in nameImg:
            img = load_rawImg(pathImg, ratio_trunc=ratioTrunc)
        elif "fits" in suffix and suffix in nameImg:
            img = load_fitsImg(pathImg, ratioTrunc, 2)
        elif "fit" in suffix and suffix in nameImg:
            img = load_fitsImg(pathImg, ratioTrunc, 2)
        elif "FIT" in suffix and suffix in nameImg:
            img = load_fitsImg(pathImg, ratioTrunc, 2)
        elif "tif" in suffix and suffix in nameImg:
            img = load_tifImg(pathImg, ratio_trunc=ratioTrunc)
        elif "png" in suffix and suffix in nameImg:
            img = cv2.imread(pathImg, 0)
        else:
            print("Imgs don't need decode or something wrong!")
            break
        if scale != 1:
            h, w = img.shape[:2]
            nh, nw = int(h*scale), int(w*scale)
            # img = cv2.medianBlur(img, 5)
            img = cv2.resize(img, (nw, nh))
        imgShow = cvt_16uimg2RGB(img, ratioTrunc)
        if flagDecode: cv2.imwrite(pathDecodeImg, img)
        if flagShow: cv2.imwrite(pathShowImg, imgShow)
        # print("show: %s"%pathShowImg)
        # print("decode: %s\n"%pathDecodeImg)
    

# 显示单张图像
def show_img(img, if_trunc=True, figNum=None, figName=None):
    if figNum is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig, ax = plt.subplots(1,1)
    u = img.mean()
    v = img.std()
    if if_trunc:
        # plt.imshow(img, vmax=(u+2*v), vmin=(u-2*v), cmap='gray')
        plt.imshow(img, vmax=(u+4*v), vmin=(u-2*v), cmap='gray')
    else:
        plt.imshow(img, cmap='gray')
    if figName != None:
        plt.title(figName)
    return fig, ax


# 获取坐标点灰度值
def get_gray(img, x, y):
    gray = img[y, x]
    return gray
    

# 绘制三维灰度图
def draw3D(img, figName='3D', cmap='viridis', return_fig=False, figSize=(6,6)):
    """配色：jet, camp, terrain, viridis, rainbow"""
    h, w = img.shape[:2]
    rstride, cstride = 1 if h<100 else 5, 1 if w<100 else 5
    xtick, ytick = w//3 if w//3 > 1 else 3, h//3 if h//3 > 1 else 3
    fig = plt.figure(figName, figsize=figSize, dpi=120)
    plt.clf()
    ax = fig.add_subplot(projection='3d')
    X = np.arange(0, h, 1)
    Y = np.arange(0, w, 1)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, img.T, cmap=cmap, rstride=rstride, cstride=cstride,
                      linewidth=0.5, antialiased=True)
    # 设置坐标轴刻度
    ax.set_xlim(0, w-1)
    ax.set_ylim(0, h-1)
    ax.set_xticks(np.arange(0, h, xtick))
    ax.set_yticks(np.arange(0, w, ytick))
    ax.set_zticks(np.arange(img.min(), img.max(), (img.max()-img.min())/3))
    # 设置坐标轴标签
    ax.set_ylabel("x", fontname="Times New Roman", fontsize=25)
    ax.set_xlabel("y", fontname="Times New Roman", fontsize=25)
    # ax.set_zlabel("value", fontname="Times New Roman", fontsize=25, labelpad=7)
    ax.tick_params(axis='x', which='major', labelsize=20, direction='in', pad=3)
    ax.tick_params(axis='y', which='major', labelsize=20, direction='in', pad=3)
    ax.tick_params(axis='z', which='major', labelsize=20, direction='in', pad=10)
    # 设置三维图默认朝向(方位角azim、俯仰角elev)
    ax.view_init(elev=15) 
    if return_fig:
        return fig, ax


# ax绘图
def ax_imshow(ax, img, rMax=4, rMin=2, flagBI=False, cmap='gray'):
    h, w = img.shape[:2]
    s = h//20
    # img = img[h//2-s:h//2+s+1, w//2-s:w//2+s+1]
    u, v = img.mean(), img.std()
    if flagBI:
        im = ax.imshow(img, cmap=cmap)
    else:
        im = ax.imshow(img, cmap=cmap, vmax=u+rMax*v, vmin=u-rMin*v)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis('off')
    return im



# 获取roi
def get_roi(img, x, y, r, flagPadding=False, flatCopy=True):
    """输入图像及目标坐标点，返回目标区域，可选择是否padding"""
    h, w = img.shape[:2]
    if x>=w or y>=h or x<0 or y<0:
        # print("获取roi时x,y超出图像范围")
        return np.zeros((2*r+1, 2*r+1), img.dtype)
    x1 = int(x-r) if x-r>=0 else 0
    y1 = int(y-r) if y-r>=0 else 0
    x2 = int(x+r+1) if (x+r+1) < w else w
    y2 = int(y+r+1) if (y+r+1) < h else h
    if not flagPadding:
        if flatCopy:
            roi = img[y1:y2, x1:x2].copy()
        else:
            roi = img[y1:y2, x1:x2]
    else:
        xp, yp = 2*r+1-(x2-x1), 2*r+1-(y2-y1)
        roi = np.zeros((2*r+1, 2*r+1), img.dtype)
        if flatCopy:
            roi[yp:, xp:] = img[y1:y2, x1:x2].copy()
        else:
            roi[yp:, xp:] = img[y1:y2, x1:x2]
    return roi


if __name__ == '__main__':
    decodeImgDir('./imgs/20221025', 'fits', scale=0.5, ratioTrunc=2)