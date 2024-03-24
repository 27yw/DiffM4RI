# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

import random
import os

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset

def load_data_yield(loader):
    while True:
        yield from loader

def load_data_inpa(
    *,
    gt_path=None,
    mask_path=None,
    batch_size,
    image_size,
    deterministic=False,

    return_dataloader=False,
    return_dict=False,
    max_len=None,
    drop_last=True,

    offset=0,

    **kwargs
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """

    gt_dir = os.path.expanduser(gt_path)
    mask_dir = os.path.expanduser(mask_path)

    gt_paths = _list_npy_files_recursively(gt_dir)
    mask_paths = _list_npy_files_recursively(mask_dir)

    assert len(gt_paths) == len(mask_paths)


    dataset = NpyDatasetInpa(
        image_size,
        gt_paths=gt_paths,
        mask_paths=mask_paths,
        shard=0,
        num_shards=1,
        return_dict=return_dict,
        max_len=max_len,
        offset=offset
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=drop_last
        )

    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=drop_last
        )
    print(len(loader))
    if return_dataloader:
        return loader
    else:
        return load_data_yield(loader)


def _list_npy_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() == "npy":
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_npy_files_recursively(full_path))
    return results



class NpyDatasetInpa(Dataset):
    def __init__(
        self,
        resolution,
        gt_paths,
        mask_paths,
        shard=0,
        num_shards=1,
        return_dict=False,
        max_len=1,
        offset=0
    ):
        super().__init__()
        self.resolution = resolution

        gt_paths = sorted(gt_paths)[offset:]
        mask_paths = sorted(mask_paths)[offset:]

        self.local_gts = gt_paths[shard:][::num_shards]
        self.local_masks = mask_paths[shard:][::num_shards]
        self.max_len = max_len
        self.return_dict = return_dict



    def __len__(self):
        if self.max_len is not None:
            return self.max_len

        return len(self.local_gts)

    def __getitem__(self, idx):
        gt_path = self.local_gts[idx]
        arr_gt = np.load(gt_path)

        mask_path = self.local_masks[idx]
        arr_mask = np.load(mask_path)

        if self.return_dict:
            name = os.path.basename(gt_path)
            return {
                'GT': arr_gt.squeeze(0),
                'GT_name': name,
                'gt_keep_mask': arr_mask.squeeze(0),
            }
        else:
            print("NotImplementedError here")
            raise NotImplementedError()

