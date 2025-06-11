import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.io import fits
from astropy.stats import sigma_clip


path_dir = './imgs/flux/output/'
list_fits = [path_dir+x for x in os.listdir(path_dir) if x.endswith('.fits')][1:]
fluxs = {x:{} for x in range(1,11)}
idxs = []
for i, path_fits in enumerate(list_fits):
    hdu_list = fits.open(path_fits)
    tar_info = hdu_list[2].data
    for row in tar_info:
        fluxs[row['idx']][i] = row['flux'].item()
        idxs.append(row['idx'].item())
fluxs = {k: v for k, v in fluxs.items() if k in set(idxs)}
num_tracks = fluxs.__len__()

plt.ion()
fig, axes = plt.subplots(num_tracks, 1, figsize=(14, num_tracks), sharex=True)
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X', 'h']
cmap = cm.get_cmap('tab20', 8)
for track_id, info in fluxs.items():
    if len(info) == 0: continue
    frame = np.asarray(list(info.keys()))
    flux = np.asarray(list(info.values()))
    flux_clipped = sigma_clip(flux, sigma=3, maxiters=3)
    flux_clean = flux_clipped.data[~flux_clipped.mask]
    frame_clean = frame[~(flux_clipped.mask)]

    color = cmap(track_id)
    marker = markers[track_id % len(markers)]
    ax = axes[track_id-1]
    ax.scatter(frame_clean, flux_clean, color=color, label=str(track_id), edgecolors='k')
    ax.legend()
fig.tight_layout()
plt.savefig(f'{path_dir}/flux.png', dpi=500)