import os
import argparse
import time
import glob
from typing import List
from pathlib import Path

import torch
import torch.distributed as dist

from nvidia.dali import pipeline_def
from nvidia.dali.plugin import pytorch
import nvidia.dali.fn as fn
import nvidia.dali.types as types

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1)) 
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

VID_FORMATS = 'avi', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg' # include video suffixes

@pipeline_def
def video_pipe(filenames: List[str],
               labels: List[int], 
               sequence_length: int,
               stride: int,
               local_rank: int,
               world_size: int):

    videos, label = fn.readers.video(device="gpu", 
                              filenames=filenames,
                              labels=labels, 
                              sequence_length=sequence_length,
                              normalized=False, 
                              random_shuffle=True, 
                              image_type=types.RGB,
                              dtype=types.UINT8,
                              initial_fill=16,
                              num_shards=world_size,
                              shard_id=local_rank,
                              stride=stride,
                              name="Reader")
    return videos, label[0]

def video2label_paths(video_path: List) -> List:        
    return [int(Path(x).parts[-2]) for x in video_path]


class DALIVideoLoader():
    def __init__(self, 
                 path: str, 
                 batch_size: int, 
                 num_threads: int,
                 sequence_length: int,
                 stride: int,
                 local_rank: int,
                 world_size: int):
        
        try:
            f = [] # video files 
            for p in path if isinstance(path, list) else [path]:
                p = Path(p) 
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
        except Exception as e:
            raise Exception(f'Error loading data from {path}: {e}\n')
        self.vd_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in VID_FORMATS)
        assert self.vd_files, f'No videos found'
        self.labels = video2label_paths(self.vd_files)
        assert len(self.vd_files) == len(self.labels), f'The number of video files are not matched with label files'

        pipe = video_pipe(batch_size=batch_size, 
                          num_threads=num_threads, 
                          device_id=local_rank,
                          local_rank=local_rank,
                          world_size=world_size, 
                          filenames=self.vd_files,
                          labels=self.labels,
                          stride=stride,
                          sequence_length=sequence_length,
                          seed=123456)
        pipe.build()

        self.dali_iterator = pytorch.DALIGenericIterator(pipe,
                                                         ["data", "label"],
                                                         reader_name="Reader",
                                                         last_batch_policy=pytorch.LastBatchPolicy.PARTIAL,
                                                         auto_reset=True)
    def __len__(self):
        return int(self.epoch_size)

    def __iter__(self):
        return self.dali_iterator.__iter__()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--device', default='6,7', help='cuda device, i.e. 0 or 0,1,2,3')
    parser.add_argument('--batch_size', type=int, default=8, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--num_threads', type=int, default=8, help='number of threads')
    parser.add_argument('--frame_num', type=int, default=60, help='Frame nums to load')
    parser.add_argument('--frame_interval', type=int, default=5, help='Frame interval')
    parser.add_argument('--data_dir', type=str, default='/usr/src/app/da2so/dali_pytorch/videos', help='dataset directory')    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device 
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device('cuda', LOCAL_RANK)
    os.environ['NCCL_BLOCKING_WAIT'] = '1'  # set to enforce timeout
    dist.init_process_group('nccl' if dist.is_nccl_available() else 'gloo')

    daliloader = DALIVideoLoader(path=args.data_dir,
                                 sequence_length=args.frame_num,
                                 stride=args.frame_interval,
                                 batch_size=args.batch_size,
                                 num_threads=args.num_threads,
                                 local_rank=LOCAL_RANK,
                                 world_size=WORLD_SIZE)
    if RANK == 0:
        start_time = time.time()
        for inp in daliloader:
            print(f'video shape: {inp[0]["data"].shape}')
            print(f'label shape: {inp[0]["label"].shape}')
        print(f'[Multi-GPU {args.device} DALI Videoloader] time: {time.time() - start_time}')