
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter # 第1步
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import SimpleITK as sitk
from models import DiT_models
from diffusion import create_diffusion



#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################
class NpyDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = [file for file in os.listdir(folder_path) if file.endswith(".npy")]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.folder_path, file_name)
        data = np.load(file_path)
        tensor_data = torch.from_numpy(data).float()
        tensor_data = tensor_data.squeeze(0)
        tensor_data = tensor_data.to(memory_format=torch.contiguous_format).float()
        # print(torch.max(tensor_data))
        # print(torch.min(tensor_data))
        #print(tensor_data)
        #print(tensor_data.shape)
        return tensor_data,file_name
def extract_label_from_filename(filename):
    # 假设文件名的格式是 "Brats2021_00000_t1.nii.gz"
    parts = filename.split('_')
    if len(parts) >= 3 and parts[-1].endswith('.nii.gz'):
        label = parts[-2]  # 文件名的倒数第二部分即为标签信息
        return label
    return None

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled


class NiFTIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.nii.gz')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        nii_path = os.path.join(self.root_dir, self.file_list[idx])
        # print(nii_path)

        # data = load_nifti_file(nii_path)
        nii_img = sitk.ReadImage(nii_path)
        # 获取数据和元数据
        nii_data = sitk.GetArrayFromImage(nii_img)
        nrrd_img = sitk.GetImageFromArray(nii_data)
        nrrd_img = resize_image_itk(nrrd_img, (32, 32, 32),
                                    resamplemethod=sitk.sitkLinear)
        nii_data = sitk.GetArrayFromImage(nrrd_img)
        nii_data = torch.tensor(nii_data, dtype=torch.float32)
        nii_data = nii_data.unsqueeze(0)  # 添加批次维度
        # print(nii_data)
        label = extract_label_from_filename(self.file_list[idx])
        return nii_data, label
def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=12
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0)
    
    # Setup data:
    dataset = NpyDataset(args.data_path)

    loader = DataLoader(dataset, batch_size=args.global_batch_size, shuffle=True, pin_memory=True, num_workers=8)

    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    #diffusion.train()
    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Check if a checkpoint path is provided
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device(device))
        model.module.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        start_epoch = checkpoint["epoch"]
        train_steps = checkpoint["train_steps"]
        log_steps = train_steps
        logger.info(f"Resuming training from checkpoint: {args.resume_checkpoint}")
    else:
        start_epoch = 0
        train_steps = 0
        log_steps = 0
    writer = SummaryWriter(log_dir="/root/tf-logs/")
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch,args.epochs):
        # sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            opt.zero_grad()
            #print(x.shape)
            # print(y)
            x = x.to(device)
            # y = y.to(device)
            # with torch.no_grad():
            #     # Map input images to latent space + normalize latents:
            #     x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict,output = diffusion.training_losses(model, x, t)
            #print(output.shape)
            loss = loss_dict["loss"].mean()
            
            loss.backward()

            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                writer.add_scalar(f"loss_dit_sdf_{experiment_index:03d}_{model_string_name}", avg_loss, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "epoch":epoch,
                        "train_steps":train_steps,
                        "log_steps":log_steps,
                        "experiment_index":experiment_index
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:09d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
        #print("output.shape",output.shape)
        np.save(f'./sample_dit/{y[0]}.npy',
                        np.array(output.cpu().detach()))
        # writer.add_scalar("1loss_dit_8", avg_loss, train_steps)

    #model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/root/autodl-tmp/dit/npy_all_after")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/4")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=32)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10000000)
    parser.add_argument("--global-batch-size", type=int, default=10)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--resume-checkpoint", type=str, default=None)  # New argument for resuming training

    args = parser.parse_args()
    main(args)
