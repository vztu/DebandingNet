# DebandingNet


## Prepare data

We started with the original videos used in [ICASSP'21] [BandingDataset](https://github.com/akshay-kap/Meng-699-Image-Banding-detection). By inquirying Prof. Zhou Wang, the original videos are stored as [Waterlook1KVideo](http://ivc.uwaterloo.ca/database/Waterloo1KVideo/), which containing a total of 1000 1080p@24-30fps videos. Please refer to `data/README` to prepare the source videos `data/Waterloo1K/source{02d}/*.mp4` and VP9-compressed versions `data/Waterlook1K_vp9/source{02d}/*.webm`.

## Analyze data

Run `data_analysis/bband_stats.m` to gather the BBAND stats of source videos and compressed ones (have to change path and run twice). Move the results to `data_analysis/result/Waterloo1K/source{02d}*.txt` and `data_analysis/result/Waterloo1K_vp9/source{02d}*.txt` respectively. Then you can run the following script to plot the BBAND stats:

```
$ python3 plot_bband_stats
```

Excluding one outlier video (../data/Waterloo1K/source04/0372_fps25.mp4) that has extremely large BBAND (source: 3.629773, vp9: 3.496615), we got the stats as follows:

