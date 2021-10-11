r"""
Need to run this in the (base) conda env.
"""

import os
import subprocess
from subprocess import check_output
import numpy as np 
import glob

DATA_DIR = "../data/Waterloo1K/"
RES_DIR = "../data/Waterloo1K_vp9/"

if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

input_files = sorted(glob.glob(DATA_DIR+"source05/*.mp4"))
print(f"found {len(input_files)} files")



for file in input_files:
    basepath, fname = os.path.split(file)
    basename, ext = os.path.splitext(fname)
    outdir = os.path.join(RES_DIR, basepath.split(os.sep)[-1])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, basename+".webm")
    if os.path.exists(outfile): continue
    encode_cmd = f"ffmpeg -i {file} -vf scale=1920x1080 -b:v 1800k"
    encode_cmd += f" -minrate 900k -maxrate 2610k -tile-columns 2 -g 240 -threads 8"
    encode_cmd += f" -quality good -crf 31 -c:v libvpx-vp9 -c:a libopus"
    encode_cmd += f" -pass 1 -speed 4 {outfile} &&"
    encode_cmd += f" ffmpeg -i {file} -vf scale=1920x1080 -b:v 1800k"
    encode_cmd += f" -minrate 900k -maxrate 2610k -tile-columns 3 -g 240 -threads 8"
    encode_cmd += f" -quality good -crf 31 -c:v libvpx-vp9 -c:a libopus"
    encode_cmd += f" -pass 2 -speed 4 -y {outfile}"
    print(f"{encode_cmd}")
    os.system(encode_cmd)

    