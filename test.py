"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils
import cv2

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from data_RGB import get_test_data, get_validation_data
from MPRNet import MPRNet
from skimage import img_as_ubyte
from pdb import set_trace as stx
from runpy import run_path
parser = argparse.ArgumentParser(description='Image Debanding using MPRNet')

parser.add_argument('--input_dir', default='./datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--model_file', default='./UNet.py', type=str, help='Model file path')
parser.add_argument('--model_variant', default='UNet-32', type=str, help='Model variant')
parser.add_argument('--weights', default='./pretrained_models/model_deblurring.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='VP9', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
parser.add_argument('--gpus', default='0,1', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--crop_size', default=0, type=int, help='Patch size for cropping. O: no cropping')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

######### Model ###########
load_file = run_path(args.model_file)
model_restoration = load_file['model'](args.model_variant)
model_restoration.cuda()

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

dataset = args.dataset
rgb_dir_test = os.path.join(args.input_dir, dataset, 'val', 'input')
rgb_dir_test_gt = os.path.join(args.input_dir, dataset, 'val', 'target')
test_dataset = get_test_data(rgb_dir_test, img_options={})
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

result_dir  = os.path.join(args.result_dir, dataset)
utils.mkdir(result_dir)

with open(os.path.join(args.result_dir, 'metrics.txt'), 'w') as f:
  f.write('filename,psnr_pred,ssim_pred,psnr_input,ssim_input\n')
  with torch.no_grad():
    total_psnr, input_psnr = [], []
    total_ssim, input_ssim = [], []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        # if ii<738: continue
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        # target = data_test[0].cuda()
        input_    = data_test[0].cuda()
        filenames = data_test[1]
        target = cv2.cvtColor(
            cv2.imread(
                os.path.join(rgb_dir_test_gt, filenames[0]+'.png')), cv2.COLOR_BGR2RGB)
        input = cv2.cvtColor(
            cv2.imread(
                os.path.join(rgb_dir_test, filenames[0]+'.png')), cv2.COLOR_BGR2RGB)
        # Padding in case images are not multiples of 8
        if dataset == 'RealBlur_J' or dataset == 'RealBlur_R':
            factor = 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
        if args.crop_size > 0:
            input_ = input_[:,:,:args.crop_size,:args.crop_size]
            input = input[:args.crop_size,:args.crop_size,:]
            target = target[:args.crop_size,:args.crop_size,:]

        restored = model_restoration(input_)
        restored = torch.clamp(restored,0,1)

        # Unpad images to original dimensions
        if dataset == 'RealBlur_J' or dataset == 'RealBlur_R':
            restored = restored[:,:,:h,:w]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        
        restored_img = img_as_ubyte(restored[0])
        psnr_val = psnr(target, restored_img)
        ssim_val = ssim(target, restored_img, multichannel=True)
        total_psnr.append(psnr_val)
        total_ssim.append(ssim_val)
        psnr_val_input = psnr(target, input)
        ssim_val_input = ssim(target, input, multichannel=True)
        input_psnr.append(psnr_val_input)
        input_ssim.append(ssim_val_input)
        print(f"input: psnr:{psnr_val_input}, ssim:{ssim_val_input}")
        print(f"pred: psnr:{psnr_val}, ssim:{ssim_val}")
        f.write(f'{filenames[0]},{psnr_val},{ssim_val},{psnr_val_input},{ssim_val_input}\n')

        utils.save_img((os.path.join(result_dir, filenames[0]+'.png')), restored_img)
        utils.save_img((os.path.join(result_dir, filenames[0]+'_input.png')), input)
        utils.save_img((os.path.join(result_dir, filenames[0]+'_target.png')), target)

    total_psnr, total_ssim = np.array(total_psnr), np.array(total_ssim)
    input_psnr, input_ssim = np.array(input_psnr), np.array(input_ssim)

    valid_idx = np.argwhere(np.isfinite(input_psnr))
    total_psnr, total_ssim = total_psnr[valid_idx], total_ssim[valid_idx]
    input_psnr, input_ssim = input_psnr[valid_idx], input_ssim[valid_idx]
    f.write(f"Average,{np.mean(total_psnr)},{np.mean(total_ssim)},{np.mean(input_psnr)},{np.mean(input_ssim)}\n")
    print(f"Average PSNR: {np.mean(total_psnr)}, SSIM: {np.mean(total_ssim)}\n")
    print(f"Average_of_inputs PSNR: {np.mean(input_psnr)}, SSIM: {np.mean(input_ssim)}\n")
