import os
import glob
import matplotlib.pyplot as plt
import pylab as py
import seaborn as sns
from matplotlib import rc
import numpy as np
import pandas as pd
import random

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
fig_path = './scatter_plots_bband_stats_after_filtering.pdf'
fig_path_png = './scatter_plots_bband_stats_after_filtering.png'

nan_mask = np.isnan(bband_stats_org) | np.isnan(bband_stats_vp9)
with open('train_samples.txt') as f:
    train_names = f.readlines()
    train_names = [line.rstrip() for line in train_names]
for idx, name in enumerate(names_org):
    if name not in train_names:
        nan_mask[idx] = True
df1 = pd.DataFrame(
    np.stack([bband_stats_org[~nan_mask],bband_stats_vp9[~nan_mask]], axis=1),
    columns=['BBAND (Original video)', 'BBAND (VP9 compressed)'] )
print(df1.shape)
nan_mask = np.isnan(bband_stats_org) | np.isnan(bband_stats_vp9)
with open('val_samples.txt') as f:
    val_names = f.readlines()
    val_names = [line.rstrip() for line in val_names]
for idx, name in enumerate(names_org):
    if name not in val_names:
        nan_mask[idx] = True
df2 = pd.DataFrame(
    np.stack([bband_stats_org[~nan_mask],bband_stats_vp9[~nan_mask]], axis=1),
    columns=['BBAND (Original video)', 'BBAND (VP9 compressed)'] )

df1['kind'] = 'Train'
df2['kind'] = 'Validation'
df = pd.concat([df1, df2])


def colored_scatter(x, y, **kwargs):
    def scatter(*args):
        args = (x, y)
        plt.scatter(*args, **kwargs)

    return scatter

# print(f"number of nans: {np.sum(nan_mask)}")
fig, ax = plt.subplots(figsize=(6, 8))
plot = sns.JointGrid(
        x=df['BBAND (Original video)'],
        y=df['BBAND (VP9 compressed)'])
colors = ['blue', 'red']
legends=[]
color_cnt = 0
for name, df_group in df.groupby('kind'):
    legends.append(name)
    plot.plot_joint(
        colored_scatter(df_group['BBAND (Original video)'],
            df_group['BBAND (VP9 compressed)'], color=colors[color_cnt],
            edgecolor="white", s=50, linewidth=1)
    )

    sns.histplot(
        x=df_group['BBAND (Original video)'].values,
        ax=plot.ax_marg_x,
        color=colors[color_cnt],
        bins=20,
        stat='count',
        line_kws={"linewidth": 1,
        "edgecolor": 'black'}
    )
    sns.histplot(
        y=df_group['BBAND (VP9 compressed)'].values,
        ax=plot.ax_marg_y,
        color=colors[color_cnt],
        bins=20,
        stat='count',
        # vertical=True,
        line_kws={"linewidth": 1,
        "edgecolor": 'black'}
    )
    color_cnt += 1

plt.legend(legends, fontsize=label_size)
# jointplot(x=bband_stats_org[~nan_mask], y=bband_stats_vp9[~nan_mask],
#     kind='scatter', s=50, color='b', edgecolor="white", linewidth=1)
ax = plot.ax_joint
# ax.set_label('Sample')
ax.set_xlabel(r'BBAND (Original video)', fontsize=label_size)
ax.set_ylabel(r'BBAND (VP9 compressed)', fontsize=label_size)
plot.ax_marg_x.set_xlim(0, 2.5)
plot.ax_marg_y.set_ylim(0, 2.5)

plot.ax_joint.plot([0,2.5], [0,2.5], 'tab:gray', linestyle='dashed', linewidth = 2)
# plot.ax_joint.set_xticks(fontsize=ticks_size)
plot.ax_joint.set_yticklabels(plot.ax_joint.get_yticks(), size = ticks_size)

# plot.ax_joint.set_yticks(fontsize=ticks_size)
plt.tight_layout()
# # plt.show()
py.savefig(fig_path)
py.savefig(fig_path_png)