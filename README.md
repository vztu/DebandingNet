# DebandingNet


## Prepare data

* We started with the original videos used in [ICASSP'21] [BandingDataset](https://github.com/akshay-kap/Meng-699-Image-Banding-detection). By inquirying Prof. Zhou Wang, the original videos are stored as [Waterlook1KVideo](http://ivc.uwaterloo.ca/database/Waterloo1KVideo/), which containing a total of 1000 1080p@24-30fps videos. Please refer to `data/README` to prepare the source videos `data/Waterloo1K/source{02d}/*.mp4` and VP9-compressed versions `data/Waterlook1K_vp9/source{02d}/*.webm`.

## Analyze data

* Run `data_analysis/bband_stats.m` to gather the BBAND stats of source videos and compressed ones (have to change path and run twice). Move the results to `data_analysis/result/Waterloo1K/source{02d}*.txt` and `data_analysis/result/Waterloo1K_vp9/source{02d}*.txt` respectively. Then you can run the following script to plot the BBAND stats:

```
$ python3 plot_bband_stats.py
```

* Excluding one outlier video (../data/Waterloo1K/source04/0372_fps25.mp4) that has extremely large BBAND (source: 3.629773, vp9: 3.496615), we got the stats as follows:

![BBAND stats for Waterloo1K: compressed vs. original](https://github.com/vztu/DebandingNet/blob/main/data_analysis/scatter_plots_bband_stats.png)

* We only kept videos whose BBAND score increases after compression. We also observed that some videos already have noticeable banding artifacts, so we remove those videos whose original video has BBAND larger than 1.0. which results in a total of 477 videos (383 train, 94 val). We run `$ python3 filter_and_split_videos.py` to filter these critirion and generate train-test split, where training set has 383 and test set has 94 videos.

* After filtering, the distribution is like:

![BBAND stats for Waterloo1K (Filtered 477 videos): compressed vs. original](https://github.com/vztu/DebandingNet/blob/main/data_analysis/scatter_plots_bband_stats_after_filtering.png)

## Extract frames

* We use FFmpeg to extract frames automatically. The following script extract 1 frame every 1 second and stores them in the `data/frames` directory:

```
$ python3 extract_video_frames.py --filepath "../data_analysis/train_samples.txt" --target_dir "../data/frames/train" --vp9_dir "../data/Waterloo1K_vp9" --height 1080 --width 1920
```

```
$ python3 extract_video_frames.py --filepath "../data_analysis/val_samples.txt" --target_dir "../data/frames/val" --vp9_dir "../data/Waterloo1K_vp9" --height 1080 --width 1920
```

* Move the dataset `data/frames` and rename as `datasets/VP9BandingDataset`.

<!-- ## Prepare training data -->

<!-- First set up basicsr: `python setup.py develop --no_cuda_ext`. Install python-lmdb: `conda install -c conda-forget python-lmdb`

Move the dataset `data/frames` and rename as `HINet/datasets/VP9BandingDataset`. Then prepare data:

* `python3 scripts/data_preparation/vp9bandingdataset.py` -->
<!-- 
## Train 

* `python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/VP9BandingDataset/HINet.yml --launcher pytorch` -->

## Prepare training 

* Install yacs: `conda install -c conda-forge yacs`
* Install natsort: `pip install natsort`
* Install skimage: `pip install scikit-image`
* Install tqdm: `pip install tqdm`
* Install cv2: `conda install -c conda-forge opencv`
* Install warmup scheduler 

```
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```

## Start training

* Carefully modify `training.yml` and then run:

```
python3 train.py
```

## Testing on validation set 

* Download images of validation set and place them in `datasets/VP9BandingDataset/val/`

* Run

```
python test.py --input_dir datasets/ --result_dir results/MPRNet_epochs100_100epochs_charfreqloss/ --dataset VP9BandingDataset --weights checkpoints/Debanding/models/MPRNet/model_best.pth 
```

## Todos:

[ ] Integrate BBAND and DBI metrics.
[ ] Implemented BandingEdgeLoss (no need to be same to BBAND). Can refer to Canny.py


## Experiment logs

### 12.5

* Summary:

| UNet    | Epochs | Train size  | Loss |  PSNR  |  SSIM  | Pars | MACs |
|---|---|---|---|---|---|---|---|
| UNet32  | 300 | 128x128  |  l1  | 37.33  | 0.9439  | 1.82M | 19.84G |
| UNet32  | 300 | 256x256  |  l1  | 37.50  | 0.9459  | 1.82M | 19.84G |
| UNet32  | 300 | 256x256  |  l2  | 37.48  | 0.9456  | 1.82M | 19.84G |
| UNet64  | 300 | 128x128  |  l1  | 37.49  | 0.9456  | 7.27M | 79.01G |
| UNet64  | 300 | 256x256  |  l1  | 37.60  | 0.9467  | 7.27M | 79.01G |
| UNet64  | 300 | 256x256  |  l2  | 37.60  | 0.9468  | 7.27M | 79.01G |

### 11.10

* Trained the `UNet-32` model (1.82M, 19.84GMAC) with 100 epochs.

* Test:

```
python test.py --input_dir datasets/ --model_file "UNet.py" --model_variant "UNet-32" --result_dir results/UNet32_epochs100_l2loss/ --dataset VP9BandingDataset --weights checkpoints/Debanding/models/UNet/model_best.pth --gpus='0,1' --crop_size=512
```

* Results: `(PSNR, SSIM) = ()`

### 12.2

* Trained (on Odin) the `UNet-32` model (1.82M, 19.84 GMAC) with 300 epochs on 256x256 using L1Loss().

* Test:

```
python test.py --input_dir datasets/ --model_file "UNet.py" --model_variant "UNet-32" --result_dir results/UNet32_epochs300_l1loss_train256x256/ --dataset VP9BandingDataset --weights checkpoints/Debanding/models/UNet32_epochs300_l1loss_train256x256/model_latest.pth --gpus='0,1' --crop_size=0
```

* Results on full-res: `(PSNR, SSIM) = (37.50, 0.9459)`

### 12.4

* Trained (on Odin) the `UNet-32` model (1.82M, 19.84 GMAC) with 300 epochs on 256x256 using MSELoss().

```
python test.py --input_dir datasets/ --model_file "UNet.py" --model_variant "UNet-32" --result_dir results/UNet32_epochs300_l2loss_train256x256/ --dataset VP9BandingDataset --weights checkpoints/Debanding/models/UNet32_epochs300_l2loss_train256x256/model_latest.pth --gpus='0,1' --crop_size=0
```

* Results on full-res: `(PSNR, SSIM) = (36.67, 0.9413)`



### 12.4

* Trained (on Exx-1) the `UNet-64` model (M, GMAC) with 300 epochs on 256x256 using L1Loss().

```
python test.py --input_dir datasets/ --model_file "UNet.py" --model_variant "UNet-64" --result_dir results/UNet64_epochs300_l1loss_train256x256/ --dataset VP9BandingDataset --weights checkpoints/Debanding/models/UNet64_epochs300_l1loss_train256x256/model_latest.pth --gpus='0,1' --crop_size=0
```


### 12.4

* Trained (on Odin) the `UNet-64` model (M, GMAC) with 300 epochs on 256x256 using MSELoss().

* Test:

```
python test.py --input_dir datasets/ --model_file "UNet.py" --model_variant "UNet-64" --result_dir results/UNet64_epochs300_l2loss_train256x256/ --dataset VP9BandingDataset --weights checkpoints/Debanding/models/UNet64_epochs300_l2loss_train256x256/model_latest.pth --gpus='0,1' --crop_size=0
```

* Results on full-res: `(PSNR, SSIM) = (36.67, 0.9413)`

### 12.5

* Trained (on Odin) the `UNet-32` model (M, GMAC) with 300 epochs on 128x128 using L1Loss().
            
* Test:

```
python test.py --input_dir datasets/ --model_file "UNet.py" --model_variant "UNet-32" --result_dir results/UNet32_epochs300_l1loss_train128x128/ --dataset VP9BandingDataset --weights checkpoints/Debanding/models/UNet32_epochs300_l1loss_train128x128/model_latest.pth --gpus='0,1' --crop_size=0
```

### 1.15

* Extract BBAND maps
