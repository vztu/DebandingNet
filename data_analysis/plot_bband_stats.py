import os
import glob
import matplotlib.pyplot as plt
import pylab as py
import seaborn as sns
from matplotlib import rc
import numpy as np
# plt.style.use("seaborn")

ticks_size = 12
label_size = 14

ORG_DIR = 'result/Waterloo1K'
VP9_DIR = 'result/Waterloo1K_vp9'

files = sorted(glob.glob(os.path.join(ORG_DIR, '*.txt')))
bband_stats_org = []
names_org = []
for file in files:
    with open(file) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        for line in lines:
            path, score = line.split(',')
            bband_stats_org.append(np.float32(score))
            names_org.append(path)
bband_stats_org = np.asarray(bband_stats_org)
print(bband_stats_org.shape)
fig_path = "./bband_hist_orig.pdf"
fig, ax = plt.subplots(figsize=(6, 5.5))
plot=ax.hist(bband_stats_org)
ax.set_xlabel(r'BBAND', fontsize=label_size)
ax.set_ylabel(r'Counts', fontsize=label_size)
ax.legend(fontsize=label_size)
plt.xticks(fontsize=ticks_size)
plt.yticks(fontsize=ticks_size)
# plt.tight_layout()
# plt.show()
py.savefig(fig_path)


files = sorted(glob.glob(os.path.join(VP9_DIR, '*.txt')))
bband_stats_vp9 = []
names_vp9 = []
for file in files:
    with open(file) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        for line in lines:
            path, score = line.split(',')
            bband_stats_vp9.append(float(score))
            names_vp9.append(path)
print(len(bband_stats_vp9))
bband_stats_vp9 = np.asarray(bband_stats_vp9, dtype=np.float32)
# print(bband_stats_org)

# PLOT 
fig_path = './scatter_plots_bband_stats.pdf'
fig_path_png = './scatter_plots_bband_stats.png'
nan_mask = np.isnan(bband_stats_org) | np.isnan(bband_stats_vp9)
nan_mask[371] = True
# These two videos have extremely large BBAND scores!! Exclude them
# ../data/Waterloo1K/source04/0372_fps25.mp4
# ../data/Waterloo1K_vp9/source04/0372_fps25.webm


print(f"number of nans: {np.sum(nan_mask)}")
# fig, ax = plt.subplots(figsize=(6, 8))
plot=sns.jointplot(x=bband_stats_org[~nan_mask], y=bband_stats_vp9[~nan_mask],
kind='scatter', s=50, color='m', edgecolor="white", linewidth=1)
ax = plot.ax_joint
# ax.set_label('Sample')
ax.set_xlabel(r'BBAND$\downarrow$ (Original video)', fontsize=label_size)
ax.set_ylabel(r'BBAND$\downarrow$ (VP9 compressed)', fontsize=label_size)
plot.ax_marg_x.set_xlim(0, 2.5)
plot.ax_marg_y.set_ylim(0, 2.5)

plot.ax_joint.plot([0,2.5], [0,2.5], 'tab:gray', linestyle='dashed', linewidth = 2)

# plot=ax.scatter(bband_stats_org[~nan_mask], bband_stats_vp9[~nan_mask],
#  c='tab:gray', marker='+')
# plot.set_label('Sample')
# ax.set_xlabel(r'BBAND (Original video)', fontsize=label_size)
# ax.set_ylabel(r'BBAND (VP9 compressed)', fontsize=label_size)
ax = plt.gca()
# ax.legend()
# plot.ax_joint.xticks(fontsize=ticks_size)
# plot.ax_joint.set_yticks(fontsize=ticks_size)
plt.tight_layout()
# plt.show()
py.savefig(fig_path)
py.savefig(fig_path_png)

print(np.nanmax(bband_stats_vp9))  # 371!
print(np.nanmax(bband_stats_org))  # 
print(names_org[np.nanargmax(bband_stats_org)])
print(names_vp9[np.nanargmax(bband_stats_vp9)])

### filter videos that don't increase BBAND scores after compression.
with open('filtered_video_names.txt', 'w') as f:
    for ii in range(len(bband_stats_org)):
        if nan_mask[ii] is True:
            continue
        if bband_stats_org[ii] < bband_stats_vp9[ii]:
            f.write('%s\n' % names_org[ii])
