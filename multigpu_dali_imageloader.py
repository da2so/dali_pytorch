import os
import argparse
import time

import torch
import torch.distributed as dist

from nvidia.dali import pipeline_def
from nvidia.dali.plugin import pytorch
import nvidia.dali.fn as fn
import nvidia.dali.types as types

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1)) 
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

@pipeline_def
def image_pipe(file_root: str,
               local_rank: int,
               world_size: int,
               image_size: int=640):

    jpegs, labels = fn.readers.file(file_root=file_root,
                                    initial_fill=1024,
                                    random_shuffle=True,
                                    shard_id=local_rank,
                                    num_shards=world_size,
                                    name="Reader")
    images = fn.decoders.image(jpegs, 
                               device="mixed", 
                               output_type=types.RGB)
    
    images = fn.resize(images, 
                       device="gpu", 
                       size=[image_size, image_size],
                       interp_type=types.INTERP_LINEAR)
    return images, labels[0]

class DALIImageLoader():
    def __init__(self, 
                 path: str, 
                 batch_size: int, 
                 num_threads: int,
                 local_rank: int,
                 world_size: int):
        pipe = image_pipe(batch_size=batch_size,
                          num_threads=num_threads, 
                          device_id=local_rank,
                          local_rank=local_rank,
                          world_size=world_size,
                          file_root=path,
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
    parser.add_argument('--batch_size', type=int, default=32, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--num_threads', type=int, default=8, help='number of threads')
    parser.add_argument('--data_dir', type=str, default='/usr/src/app/da2so/datasets/VOC/images', help='dataset directory')    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device 
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device('cuda', LOCAL_RANK)
    os.environ['NCCL_BLOCKING_WAIT'] = '1'  # set to enforce timeout
    dist.init_process_group('nccl' if dist.is_nccl_available() else 'gloo')

    daliloader = DALIImageLoader(path=args.data_dir,
                                 batch_size=args.batch_size,
                                 num_threads=args.num_threads,
                                 local_rank=LOCAL_RANK,
                                 world_size=WORLD_SIZE)
    if RANK == 0:
        start_time = time.time()
        for idx, inp in enumerate(daliloader):
            print(f'image shape: {inp[0]["data"].shape}')
            print(f'label shape: {inp[0]["label"].shape}')
        print(f'[Multi-GPU {args.device} DALI Imageloader] time: {time.time() - start_time}')

