Download datasets from [Waterloo1KVideo](http://ivc.uwaterloo.ca/database/Waterloo1KVideo/), then unzip the files. The directory should look like this:

`Waterloo1K` <br/>
  `├──`source01/\*.mp4
  `├──`source02/\*/mp4
  `├──`source03/\*.mp4
  `...`
  `└──`source10/\*.mp4

By running the compression scripts:

```
$ python3 data_preparation/encode_videos_vp9.py
```

You should get the compressed versions of source videos in a different directory like this:

`Waterloo1K_vp9` <br/>
  `├──`source01/\*.webm
  `├──`source02/\*/webm
  `├──`source03/\*.webm
  `...`
  `└──`source10/\*.webm