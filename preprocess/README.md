## Preprocess Raw Videos

**You can use our preprocessed features above or process by yourself as follows:**

1. Rescale each raw video into 1600 frames and extract the middle frame of every 16 frames (100 middle frames will be extracted). 

    ```
   # msrvtt
    python preprocess/rescale_video.py --video-root data/msrvtt/raw_videos/*.mp4 --output-root preprocess/msrvtt/rescaled --frame-dir preprocess/msrvtt/middle_frames
   
   # msvd
    python preprocess/rescale_video.py --video-root data/msvd/raw_videos/*.avi --output-root preprocess/msvd/rescaled --frame-dir preprocess/msvd/middle_frames
    ```
   **Note:** If you want extract 15 frames, you can rescale each raw video into 1500 frames and extract the middle frame of every 100 frames (15 middle frames will be extracted)

2. To extract the visual features from the rescaled videos, run the following commands.

    ```
   # msrvtt
    python preprocess/extract_frame_feat.py --frame-root preprocess/msrvtt/middle_frames --output-root preprocess/msrvtt/[clip_l14/clip_b16/clip_b32]/frame_feature --dset_name msrvtt
   
   # msvd
    python preprocess/extract_frame_feat.py --frame-root preprocess/msvd/middle_frames --output-root preprocess/msvd/[clip_l14/clip_b16/clip_b32]/frame_feature --dset_name msvd
   ```


