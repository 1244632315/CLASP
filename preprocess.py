import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import sigmaclip

from tool import show_img, load_img, ax_imshow


def cal_med_img(path_dir, prefix, idx0, idx1):
    list_files = [os.path.join(path_dir, x) for x in os.listdir(path_dir) if x.startswith(prefix)][idx0:idx1]
    img_sum = []
    for path in tqdm(list_files, desc=prefix):
        img = load_img(path).astype(np.float32)
        img_sum.append(img)
    img_sum = np.stack(img_sum, axis=0)
    med = np.median(img_sum, axis=0)
    return med


def save_bias_dark():
    path_bias = 'E:/CLASP/20220607/'
    prefix_bias = 'bias-'
    path_dark = 'E:/CLASP/20220607/'
    prefix_dark = 'dark-'
    dark = cal_med_img(path_dark, prefix_dark, 420, 440)
    bias = cal_med_img(path_bias, prefix_bias, 0, 20)

    cv2.imwrite('./dark.tif', dark)
    cv2.imwrite('./bias.tif', bias)


def save_flat():
    path_flat = 'E:/CLASP/20220825/'
    prefix_flat = 'evening-flat'
    list_files = [os.path.join(path_flat, x) for x in os.listdir(path_flat) if x.startswith(prefix_flat)][::5]
    img_sum = []
    for path in tqdm(list_files, desc=prefix_flat):
        img = load_img(path).astype(np.float32)
        img_sum.append(img)
    img_sum = np.stack(img_sum, axis=0)
    flat = np.median(img_sum, axis=0)
    cv2.imwrite('./flat.tif', flat)




def show_saved_img():
    bias = cv2.imread('./bias.tif', cv2.IMREAD_UNCHANGED)
    dark = cv2.imread('./dark.tif', cv2.IMREAD_UNCHANGED)
    flat = cv2.imread('./flat.tif', cv2.IMREAD_UNCHANGED)

    fig, axes = plt.subplots(1,3,figsize=(20,5))
    ax_imshow(axes[0], bias[10:])
    axes[0].set_title('bias')
    ax_imshow(axes[1], dark[10:])
    axes[1].set_title('dark')
    ax_imshow(axes[2], flat[10:])
    axes[2].set_title('flat')
    plt.tight_layout()
    plt.show()



def correct_CLASP_img(img):
    bias = cv2.imread('./bias.tif', cv2.IMREAD_UNCHANGED)
    dark = cv2.imread('./dark.tif', cv2.IMREAD_UNCHANGED)
    flat = cv2.imread('./flat.tif', cv2.IMREAD_UNCHANGED)
    flat_processed = flat-bias
    flat_norm = flat_processed/np.mean(flat_processed[2200:4200, 3800:5800])
    a, b = img-bias, flat_norm
    img_cor = np.divide(a, b, out=np.zeros_like(a, dtype=float), where=(b!=0)) 
    # oup = np.clip(img_cor, 0, 65535)
    return img_cor



if __name__ == '__main__':

    # save_bias_dark()

    img = load_img('./imgs/20220929/SKYMAPPER0031-CAM1-20220929220911268.fits').astype(np.float32)
    # show_saved_img()
    cor = correct_CLASP_img(img)

    plt.ion()
    show_img(img)
    show_img(cor)
    plt.show()