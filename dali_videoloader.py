import os
import time
import glob
from typing import List
from pathlib import Path

from nvidia.dali import pipeline_def
from nvidia.dali.plugin import pytorch
import nvidia.dali.fn as fn
import nvidia.dali.types as types

VID_FORMATS = 'avi', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg' # include video suffixes

@pipeline_def
def video_pipe(filenames: List[str],
               labels: List[int], 
               sequence_length: int,
               stride: int):

    videos, label = fn.readers.video(device="gpu", 
                              filenames=filenames,
                              labels=labels, 
                              sequence_length=sequence_length,
                              normalized=False, 
                              random_shuffle=True, 
                              image_type=types.RGB,
                              dtype=types.UINT8, 
                              initial_fill=16,
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
                 stride: int):
        
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
                          device_id=0, 
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
    start_time = time.time()
    daliloader = DALIVideoLoader(path='./videos',
                                 sequence_length=60,
                                 stride=5,
                                 batch_size=8,
                                 num_threads=8)
    for inp in daliloader:
        print(f'video shape: {inp[0]["data"].shape}')
        print(f'label shape: {inp[0]["label"].shape}')
    print(f'[DALI Videoloader] time: {time.time() - start_time}')