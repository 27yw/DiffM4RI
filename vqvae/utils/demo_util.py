import numpy as np
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

import torch
import torchvision.utils as vutils

from models.base_model import create_model

from utils.util import seed_everything

def tensor_to_pil(tensor):
    # """ assume shape: c h w """
    if tensor.dim() == 4:
        tensor = vutils.make_grid(tensor)

    return Image.fromarray( (rearrange(tensor, 'c h w -> h w c').cpu().numpy() * 255.).astype(np.uint8) )

############ START: all Opt classes ############

class BaseOpt(object):
    def __init__(self, gpu_ids=0, seed=None):
        # important args
        self.isTrain = False
        self.gpu_ids = [gpu_ids]
        # self.device = f'cuda:{gpu_ids}'
        self.device = 'cuda'
        self.debug = '0'

        # default args
        self.serial_batches = False
        self.nThreads = 4
        self.distributed = False

        # hyperparams
        self.batch_size = 1

        # dataset args
        self.max_dataset_size = 10000000
        self.trunc_thres = 0.2

        if seed is not None:
            seed_everything(seed)
            
        self.phase = 'test'

    def name(self):

        return 'BaseOpt'

class VQVAEOpt(BaseOpt):
    def __init__(self, gpu_ids=0, seed=None):
        super().__init__(gpu_ids)
        # some other custom args here
        self.log_dir = 'results'  # TensorBoard log directory
        self.save_epoch_freq = 1  # Save checkpoint every N epochs
        self.print_freq = 10
        self.num_epochs = 100  # Set the number of epochs
        self.batch_size = 1
        print(f'[*] {self.name()} initialized.')

    def name(self):
        return 'VQVAETestOpt'
    
    def init_model_args(
            self,
            vq_ckpt_path=None,
            isTrain=True,
            lr=4.5e-4
        ):
        self.model = 'vqvae'
        self.vq_cfg = 'configs/vqvae_snet.yaml'
        self.ckpt = vq_ckpt_path
        self.dset = 'snet'
        self.cat = 'all'
        self.isTrain=isTrain
        self.lr=lr
    
    def init_dset_args(self, dataroot='data', dataset_mode='snet', cat='all', res=64, cached_dir=None):
        # dataset - snet
        self.dataroot = dataroot
        self.cached_dir = cached_dir
        self.ratio = 1.0
        self.res = res
        self.dataset_mode = dataset_mode
        self.cat = cat




