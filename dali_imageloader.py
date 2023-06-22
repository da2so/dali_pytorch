import time

from nvidia.dali import pipeline_def
from nvidia.dali.plugin import pytorch
import nvidia.dali.fn as fn
import nvidia.dali.types as types

@pipeline_def
def image_pipe(file_root: str,
               image_size: int=640):

    jpegs, labels = fn.readers.file(file_root=file_root,
                                    initial_fill=1024,
                                    random_shuffle=True,
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
                 num_threads: int):
        pipe = image_pipe(batch_size=batch_size,
                          num_threads=num_threads, 
                          device_id=0, 
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
    daliloader = DALIImageLoader(path='/usr/src/app/da2so/datasets/VOC/images',
                                 batch_size=32,
                                 num_threads=8)
    start_time = time.time()
    for idx, inp in enumerate(daliloader):
        print(f'image shape: {inp[0]["data"].shape}')
        print(f'label shape: {inp[0]["label"].shape}')
    print(f'[DALI Imageloader] time: {time.time() - start_time}')

