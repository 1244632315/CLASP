import os
import cv2
import sep
import astroalign as aa
import numpy as np
import matplotlib.pyplot as plt

from tool import load_img, show_img, draw3D, ax_imshow, get_fileList, get_roi
from extract import extract, extract_sep, extract_NTH, nms




def view_img():
    path_img = 'imgs/SKYMAPPER0015-CAM1-20221130001035774.fits'
    img = load_img(path_img)

    img = cv2.medianBlur(img, 3)
    fore, coords = extract_sep(img)

    show_img(fore)
    plt.plot(coords[:, 0], coords[:, 1], 'ro', ms=2)
    plt.show()


def main():
    list_imgs = get_fileList('./imgs/test')
    for idx in range(len(list_imgs)-1):
        img1 = load_img(list_imgs[idx])
        img2 = load_img(list_imgs[idx+1])

        img1 = cv2.medianBlur(img1, 5)
        source1, res_map1 = extract_NTH(img1, 2, True)
        source1_nms = nms(img1, source1, 100)

        source2, res_map2 = extract_NTH(img2, 2, True)
        source2_nms = nms(img2, source2, 100)

        plt.ion()
        show_img(img1)
        plt.plot(source1_nms[:, 0], source1_nms[:, 1], 'ro', ms=2)
        plt.plot(source2_nms[:, 0], source2_nms[:, 1], 'g*', ms=2)

        p12, ret = aa.register(source1_nms, source2_nms)
        plt.figure()
        plt.plot(source1_nms[:, 0], source1_nms[:, 1], 'ro', ms=2)
        plt.plot(p12[:, 0], p12[:, 1], 'g*', ms=2)



if __name__ == '__main__':
    img = load_img('./imgs/test/SKYMAPPER0015-CAM1-20221130001035774.fits')
    
    plt.ion()
    show_img(img)
    roi = get_roi(img, 1160, 2445, 80)
    draw3D(roi)
    
    
    bkg = sep.Background(img, bw=32, bh=32, fw=3, fh=3)
    objects, imgBi = sep.extract(img, 1.5, err=bkg.globalrms, filter_type='conv', deblend_cont=1, segmentation_map=True)
    
    plt.show()

