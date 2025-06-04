import numpy as np
import cv2
import matplotlib.pyplot as plt


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


def detectWithTM(img, thf=0.3, kz=11, thArea=2):
    """
    extract with LoG kernel + template matching                             \\
    :param img: input image, should be a grayscale image (float32)          \\
    :param thf: segmentation threshold, 0.2-0.5 is recommended              \\
    :param kz: kernel size, should be odd number, 9 and 11 are recommended  \\
    :param thArea: minimum area of connected components, 2-4 is recommended \\
    return: coordinates of extracted suspected objects, shape (N, 2)                   \\
    """
    def get_logKernel(kz=11):      
        kernel = np.zeros((kz,kz))
        for x in range(kz):
            for y in range(kz):
                sigma = (kz-1)/6
                x0 = kz//2
                y0 = kz//2
                f = np.exp(((x-x0)**2+(y-y0)**2)/(-2*sigma**2))
                kernel[x,y] = f
        return kernel.astype(np.float32)
    img = np.array(img, np.float32)
    log = get_logKernel(kz=11).astype(np.float32)
    salientMap = cv2.matchTemplate(img, log, 5)
    mask = (salientMap > thf).astype(np.uint8)
    ret, labels, stats, cens = cv2.connectedComponentsWithStats(mask)
    maskarea = stats[:, -1] > thArea
    coords = cens[maskarea][1:]
    coords = (np.array(coords)+0.5+kz//2).astype(np.int64)
    return coords



def detectWithNTH(img, ratio=2, kz=(5,9)):
    """
    extract with new Tophat filter                                      \\
    :param img: input image, should be a grayscale image (float32)      \\
    :param ratio: threshold ratio, 1-3 is recommended                   \\
    :param kz: kernel size, should be a tuple (kz1, kz2), kz2 > kz1,    \\
    :return: coordinates of extracted suspected objects, shape (N, 2)   \\
    """
    img = img.astype(np.float32)
    # white top hat
    kz1, kz2 = kz
    k1 = np.ones((kz2, kz2), np.uint8)
    k1[kz2//2-kz1//2:kz2//2+kz1//2+1, kz2//2-kz1//2:kz2//2+kz1//2+1] = 0
    k2 = np.ones((kz2, kz2), np.uint8)
    ero = cv2.erode(img, k1)
    dil = cv2.dilate(ero, k2)
    wth = img - dil
    thSeg = wth.mean() + ratio * wth.std()
    segMap = (wth > thSeg).astype(np.uint8)
    ret, labels, stats, cens = cv2.connectedComponentsWithStats(segMap)
    coords = (cens[1:, :] + 0.5).astype(np.int32)
    return coords


if __name__ == '__main__':
    for idx in range(1,4):
        # img = cv2.imread(f'{idx}.tif', cv2.IMREAD_UNCHANGED)
        img = np.ones((100, 100), np.float32) * idx  # Dummy image for testing
        coords_tm = detectWithTM(img, thf=0.4, kz=11, thArea=4)
        coords_nth = detectWithNTH(img, ratio=2, kz=(5, 9))

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ax_imshow(axes[0], img)
        axes[0].scatter(coords_tm[:, 0], coords_tm[:, 1], s=2, c='r', label='Template Matching')
        axes[0].legend()
        ax_imshow(axes[1], img)
        axes[1].scatter(coords_nth[:, 0], coords_nth[:, 1], s=2, c='b', label='New Tophat Filter')
        axes[1].legend()
        plt.show()