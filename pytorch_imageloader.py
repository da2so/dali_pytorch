

import os
import time
import glob
import random
from typing import List
from pathlib import Path
import cv2
import numpy as np

import torch
from torch.utils.data import IterableDataset, DataLoader, dataloader, distributed

IMG_FORMATS = 'jpg', 'png', 'jpeg'

def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloader(path: str,
                      bs: int, 
                      prefix: str, 
                      workers: int, 
                      shuffle: bool=True):

    dataset = ImageDataset(path=path,
                            prefix=prefix)
    sampler = None
    generator = torch.Generator()
    generator.manual_seed(0)

    return InfiniteDataLoader(dataset,
                batch_size=bs, 
                num_workers=workers, 
                shuffle=shuffle,
                pin_memory=True, # Error when pin_memory is True
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

def img2label_paths(img_path: List) -> List:        
    return [int(Path(x).parts[-2]) for x in img_path]


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
                path: str, 
                prefix: str):
        super(ImageDataset).__init__()
        
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
        self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
        assert self.img_files, f'{prefix} No Images found'
        self.labels = img2label_paths(self.img_files)
        assert len(self.img_files) == len(self.labels), f'{prefix} The number of image files are not matched with label files'
        self.total_files_num = len(self.img_files)
        self.indices = range(self.total_files_num)

    def __getitem__(self, index):
        index = self.indices[index]
        img_file, label = self.img_files[index], self.labels[index]
        image = get_image(image_path=img_file)
        return image, label
    
    def __len__(self):
        return self.total_files_num

def get_image(image_path: str,
              image_size: int=640):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    interp = cv2.INTER_LINEAR
    img = cv2.resize(img, (image_size, image_size), interpolation=interp)
    img = torch.from_numpy(img)

    return img
    
if __name__ == "__main__":
    start_time = time.time()
    train_loader = create_dataloader('/usr/src/app/da2so/datasets/VOC/images', 
                                    bs=32, 
                                    prefix='train', 
                                    workers=8)
    for image, label in train_loader:
        print(f'image shape: {image.shape}')
        print(f'label shape: {label.shape}')
    print(f'[Pytorch Imageloader] time: {time.time() - start_time}')