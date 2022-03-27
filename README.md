# DebandingNet

## Installation

* Install yacs: `conda install -c conda-forge yacs`
* Install natsort: `pip install natsort`
* Install skimage: `pip install scikit-image`
* Install tqdm: `pip install tqdm`
* Install cv2: `conda install -c conda-forge opencv`
* Install warmup scheduler 

```
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```

## Download dataset

TBD

## Start training

* Carefully modify `training.yml` and then run:

```
python3 train.py
```

## Testing on validation set 

* Download images of validation set and place them in `datasets/VP9BandingDataset/val/`

* Run:

```
python test.py --input_dir datasets/ --model_file "UNet.py" --model_variant "UNet-32" --result_dir results/test_figure5_AdaDeband_UNet/ --dataset FFMPEG --weights checkpoints/Debanding/models/UNet32_epochs300_l1loss_train256x256/model_latest.pth --gpus='0,1' --crop_size=0
```
