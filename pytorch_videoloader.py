

import os
import time
import sys
import glob
import random
from typing import List
from pathlib import Path
from decord import VideoReader
from decord import cpu
import numpy as np

import torch
from torch.utils.data import IterableDataset, DataLoader, dataloader, distributed

VID_FORMATS = 'avi', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg' # include video suffixes

def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloader(path: str,
                      bs: int, 
                      prefix: str, 
                      workers: int, 
                      interval: int=3, 
                      shuffle: bool=True):

    dataset = VideoDataset(path=path,
                            prefix=prefix, 
                            interval=interval)
    sampler = None
    generator = torch.Generator()
    generator.manual_seed(0)

    return InfiniteDataLoader(dataset,
                batch_size=bs, 
                num_workers=workers, 
                shuffle=shuffle,
                pin_memory=False,
                sampler=sampler,
                worker_init_fn=seed_worker, 
                generator=generator)


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler:
    """ Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def video2label_paths(video_path: List) -> List:        
    return [int(Path(x).parts[-2]) for x in video_path]


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
                path: str, 
                prefix: str, 
                interval: int=3):
        super(VideoDataset).__init__()
        
        try:
            f = [] # video files 
            for p in path if isinstance(path, list) else [path]:
                p = Path(p) 
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
        except Exception as e:
            raise Exception(f'{prefix} Error loading data from {path}: {e}\n')
        self.vd_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in VID_FORMATS)
        assert self.vd_files, f'{prefix} No videos found'
        self.labels = video2label_paths(self.vd_files)
        assert len(self.vd_files) == len(self.labels), f'{prefix} The number of video files are not matched with label files'
        self.total_files_num = len(self.vd_files)
        self.indices = range(self.total_files_num)
        self.interval = interval

    def __getitem__(self, index):
        index = self.indices[index]
        vd_file, label = self.vd_files[index], self.labels[index]
        video = get_video(video_path=vd_file, interval=self.interval)
        return video, label
    
    def __len__(self):
        return self.total_files_num

def get_video(video_path: str, 
              interval: int=1):
    vr = VideoReader(str(video_path), ctx=cpu(0)) # load video using cpu
    frames_idx = [frame for frame in range(0, len(vr), interval)]
    video = vr.get_batch(frames_idx).asnumpy() # get specified frames
    video = torch.from_numpy(video)
    video = video.permute(0, 3, 1, 2) # change to [frame, channel, height, width]

    return video
    
if __name__ == "__main__":
    start_time = time.time()
    train_loader  = create_dataloader("./videos", 
                                    bs=8, 
                                    prefix='train', 
                                    workers=8, 
                                    interval=5)
    for video, label in train_loader:
        print(f'video shape: {video.shape}')
        print(f'label shape: {label.shape}')
    print(f'[Pytorch Videoloader] time: {time.time() - start_time}')