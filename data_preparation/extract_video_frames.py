r"""
Need to run this in the (base) conda env.
"""

import os
import argparse
import subprocess
from tqdm import tqdm
from PIL import Image
from subprocess import check_output
import numpy as np 
import cv2
import glob

# TMP_YUV_PATH = './tmp_yuv'
# if not os.path.exists(TMP_YUV_PATH):
#     os.makedirs(TMP_YUV_PATH)

# class VideoCaptureYUV:
#     def __init__(self, filename, size):
#         self.height, self.width = size
#         self.frame_len = self.width * self.height * 3 / 2
#         self.f = open(filename, 'rb')
#         self.shape = (int(self.height*1.5), self.width)

#     def read_raw(self):
#         try:
#             raw = self.f.read(self.frame_len)
#             yuv = np.frombuffer(raw, dtype=np.uint8)
#             yuv = yuv.reshape(self.shape)
#         except Exception as e:
#             print(str(e))
#             return False, None
#         return True, yuv

#     def read(self):
#         ret, yuv = self.read_raw()
#         if not ret:
#             return ret, yuv
#         bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
#         return ret, bgr


def extract_frames(filename, basepath):
    dirname, basename = os.path.split(filename)
    name, ext = os.path.splitext(basename)

    frame_name = os.path.join(basepath, name+f"_%04d.png")
    cmd = f"ffmpeg -loglevel error -y -i {filename}"
    cmd += f" -filter:v fps=1 {frame_name}"
    os.system(cmd)


def main(args):
    file_subdir = os.path.join(args.target_dir, 'target')
    file_vp9_subdir = os.path.join(args.target_dir, 'input')
    if not os.path.exists(args.target_dir):
        os.makedirs(file_subdir)
        os.makedirs(file_vp9_subdir)

    with open(args.filepath) as file:
        files = file.readlines()
        files = [line.rstrip() for line in files]

    for file in files:
        dirname, basename = os.path.split(file)
        name, ext = os.path.splitext(basename)
        file_vp9 = os.path.join(args.vp9_dir, os.path.split(dirname)[-1],
                                name + '.webm')
        print(file)
        print(file_vp9)
        framerate = int(name[-2:])
        sampling_step = framerate
        extract_frames(file, file_subdir)
        extract_frames(file_vp9, file_vp9_subdir)


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--filepath', type=str, default='../data_analysis/train_samples.txt')
    parser.add_argument('--vp9_dir', type=str, default='../data/Waterloo1K_vp9')
    parser.add_argument('--target_dir', type=str, default='../data/frames/train')
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--width', type=int, default=1920)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)