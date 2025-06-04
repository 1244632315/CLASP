import os
import sep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.segmentation import flood

from tool import get_roi, show_img

ratio = 0.2
path_dir = 'imgs/flux'
task = 'SKYMAPPER0030-CAM1'
track_csv = f'{path_dir}/results/{task}/{task}_tracks.csv'
df = pd.read_csv(track_csv)
info = df[['IMG', 'T5', 'x5', 'y5']]

for idx, row in info.iterrows():
    if row.isna().sum(): continue
    name_img, idx, x, y = row.values
    path_fits = os.path.join(path_dir, name_img)
    hudl = fits.open(path_fits)[0]
    img = hudl.data
    header = hudl.header

    # extract local roi 
    x0, y0 = x/ratio, y/ratio
    r = 100
    roi = np.ascontiguousarray(get_roi(img, x0, y0, r), np.float32)
    bkg = sep.Background(roi, bw=32, bh=32, fw=3, fh=3)
    sub = roi - bkg.back()
    objects = sep.extract(sub, 3, err=bkg.globalrms, deblend_cont=1)
    dist = np.hypot(objects['x'] - r, objects['y'] - r)
    target_idx = np.argmin(dist)
    target = objects[target_idx]

    # 获取 flux 值（sep 已返回 flux，若没有也可单独测光）
    flux = target['flux']
    x, y = target['x'], target['y']
    print(f"目标位置: ({x:.1f}, {y:.1f}), flux: {flux:.2f}")

    plt.ion()
    show_img(sub)
    plt.scatter(objects['x'], objects['y'], s=30, facecolors='none', edgecolors='red')
    plt.show()




