
import os
import glob
import matplotlib.pyplot as plt
import pylab as py
import seaborn as sns
from matplotlib import rc
import numpy as np
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

nan_mask = np.isnan(bband_stats_org) | np.isnan(bband_stats_vp9)
### filter videos that don't increase BBAND scores after compression.
cnt = 0
with open('banding_study_candidates.txt', 'w') as f:
    for ii in range(len(bband_stats_org)):
        if nan_mask[ii] is True:
            continue
        if (bband_stats_org[ii] + 0.1 < bband_stats_vp9[ii]) and (
            bband_stats_org[ii] < 1.0) and (bband_stats_vp9[ii] > 1.0):
            f.write('%s\n' % names_org[ii])
            cnt += 1
print(f"Total counts: {cnt}")
