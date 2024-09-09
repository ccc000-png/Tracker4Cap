import os.path
import os
import random

import decord
import numpy as np
import torch
import cv2

def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]:
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices

    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if 0 < max_num_frames < len(frame_indices):
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    elif "interval" in sample:
        if num_frames == 1:
            frame_indices = [random.randint(0, vlen - 1)]
        else:
            # transform FPS
            interval = 8
            clip_length = num_frames * interval * input_fps / 30
            max_idx = max(vlen - clip_length, 0)
            start_idx = random.uniform(0, max_idx)
            end_idx = start_idx + clip_length - 1

            frame_indices = torch.linspace(start_idx, end_idx, num_frames)
            frame_indices = torch.clamp(frame_indices, 0, vlen - 1).long().tolist()
    else:
        raise ValueError
    return frame_indices


def get_frame_indices_start_end(num_frames, vlen, fps, start_time, end_time):
    start_idx = max(int(fps * start_time), 0)
    end_idx = min(int(fps * end_time), vlen)
    clip_len = end_idx - start_idx

    acc_samples = min(num_frames, clip_len)
    # split the video into `acc_samples` intervals, and sample from each interval.
    intervals = np.linspace(start=start_idx, stop=end_idx, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))

    try:
        frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
    except:
        frame_indices = np.random.permutation(list(range(start_idx, end_idx)))[:acc_samples]
        frame_indices.sort()
        frame_indices = list(frame_indices)

    if len(frame_indices) < num_frames:  # padded with last frame
        padded_frame_indices = [frame_indices[-1]] * num_frames
        padded_frame_indices[:len(frame_indices)] = frame_indices
        frame_indices = padded_frame_indices

    return frame_indices


def read_frames_decord(video_path, data_root, video_id, width=None, height=None, sample_frames=8, sample='rand',
                       fix_start=None, max_num_frames=-1, start_time=None, end_time=None):
    '''save frame and get frame use PIL.Image.open(f)'''
    in_video = cv2.VideoCapture(video_path)
    num_frames = int(in_video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(in_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = in_video.get(cv2.CAP_PROP_FPS)
    if start_time and end_time:
        frame_indices = get_frame_indices_start_end(
            sample_frames, num_frames, fps, start_time, end_time
        )
    else:
        frame_indices = get_frame_indices(
            sample_frames, num_frames, sample=sample, fix_start=fix_start, input_fps=fps, max_num_frames=max_num_frames
        )

    frame_count=0
    selected_count=0
    frames = []
    while frame_count < num_frames:
        success, frame = in_video.read()
        if not success:
            print(f'Error reading frame {frame_count}/{num_frames}')
            break

        if frame_count in frame_indices:
            frame_filename = os.path.join(data_root, 'sample_frames')
            if not os.path.exists(frame_filename):
                os.mkdir(frame_filename)
            video_frame_files = os.path.join(frame_filename, video_id)
            if not os.path.exists(video_frame_files):
                os.mkdir(video_frame_files)
            cv2.imwrite(os.path.join(video_frame_files,f'frame_{selected_count}.jpg'), frame)
            f = os.path.join(video_frame_files,f'frame_{selected_count}.jpg')
            frames.append(f)
            selected_count += 1

        frame_count += 1

    in_video.release()

    # frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    return frames
