# first set up which gpu to use
import os
gpu_ids = 0
lr = 4.5e-4
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"
# import libraries
import numpy as np
from termcolor import colored, cprint
# for display
from IPython.display import Image as ipy_image
from IPython.display import display
from torch.utils.data import random_split
import numpy as np
import SimpleITK as sitk
import os
from torch.utils.data import DataLoader, Dataset
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
from tensorflow_mri import ssim3d
import SimpleITK as sitk
from tqdm import tqdm
from models.base_model import create_model

# %load_ext autoreload
# %autoreload 2

from utils.demo_util import VQVAEOpt

seed = 2023
opt = VQVAEOpt(gpu_ids=gpu_ids, seed=seed)
device = opt.device
# initialize SDFusion model
ckpt_path = '/root/autodl-tmp/vqvae/results/t2.pth'
dset="snet"
opt.init_model_args(ckpt_path)
opt.init_dset_args(dataset_mode=dset)
print(opt.model)
vqvae = create_model(opt)
# vqvae.initialize(opt)
cprint(f'[*] "{vqvae.name()}" loaded.', 'cyan')

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int16)
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
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.nii.gz') or f.endswith('.nii')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 获取数据和元数据
        nii_path = os.path.join(self.root_dir, self.file_list[idx])
        nii_img = sitk.ReadImage(nii_path)
        nii_data = sitk.GetArrayFromImage(nii_img)
        
        # z-score 归一化化
        mean_value = nii_data.mean()
        std_dev = nii_data.std()
        normalized_data = (nii_data - mean_value) / std_dev
        
        # resize 大小
        nrrd_img = sitk.GetImageFromArray(normalized_data)
        nrrd_img = resize_image_itk(nrrd_img, (128, 128, 128),
                                    resamplemethod=sitk.sitkLinear)
        nii_data = sitk.GetArrayFromImage(nrrd_img)
        
        # 转化成tensor 添加批次维度
        nii_data = torch.tensor(nii_data, dtype=torch.float32)
        nii_data = nii_data.unsqueeze(0)  # 添加批次维度
        # print(nii_data.shape)
        return nii_data, self.file_list[idx]
# dataset = NiFTIDataset("./data_brats")
path="./data/t2"
modality = path.split("/")[-1] + "_val"
dataset = NiFTIDataset(path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
print("Load data ",len(dataloader))

# 划分数据集
total_samples = len(dataset)
train_size = int(0.001 * total_samples)
val_size = total_samples - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# 创建 DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

def train(model, train_dataloader, val_dataloader, opt):
    model.train()

    # 创建 TensorBoard 的 SummaryWriter，用于记录训练过程
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    current_time += modality
    log_dir = os.path.join('./results', current_time)
    writer = SummaryWriter(log_dir)

    ckpt_dir = os.path.join(log_dir, 'ckpt')
    img_dir = os.path.join(log_dir, 'img')
    loss_dir = os.path.join(log_dir, 'loss')
    npy_dir = os.path.join(log_dir, 'npy')
    #print(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    for epoch in range(1):
        total_ssim = 0
        total_mse = 0

        model.eval()
        with torch.no_grad():
            total_ssim = 0
            total_mse = 0
            for val_step, (inputs, filename) in enumerate(val_dataloader):
                inputs = inputs.to("cuda")
                z,reconstructions = vqvae.inference(inputs)
                np_filename = os.path.join(npy_dir, f'{filename[0].replace(f"_{modality}.nii.gz","").replace(f"_{modality}.nii","")}.npy')
                np.save(np_filename,z.cpu().detach().numpy())
                out_img = sitk.GetImageFromArray(reconstructions.cpu().detach())
                output_filename = os.path.join(img_dir, f"{filename[0]}.nii.gz")
                sitk.WriteImage(out_img, output_filename)

                mse = torch.nn.functional.mse_loss(reconstructions, inputs)
                ssim = ssim3d(inputs.contiguous().clone().detach().cpu(),
                              reconstructions.contiguous().clone().detach().cpu(), filter_size=1).numpy()

                print(f'filename {filename[0]}, ssim: {ssim} , mse: {mse.item()}')

                total_ssim += ssim
                total_mse += mse.item()
            for val_step, (inputs, filename) in enumerate(train_dataloader):
                inputs = inputs.to("cuda")
                z,reconstructions = vqvae.inference(inputs)
                np_filename = os.path.join(npy_dir, f'{filename[0].replace(f"_{modality}.nii.gz","").replace(f"_{modality}.nii","")}.npy')
                np.save(np_filename,z.cpu().detach().numpy())
                out_img = sitk.GetImageFromArray(reconstructions.cpu().detach())
                output_filename = os.path.join(img_dir, f"{filename[0]}.nii.gz")
                sitk.WriteImage(out_img, output_filename)

                mse = torch.nn.functional.mse_loss(reconstructions, inputs)
                ssim = ssim3d(inputs.contiguous().clone().detach().cpu(),
                              reconstructions.contiguous().clone().detach().cpu(), filter_size=1).numpy()

                print(f'filename {filename[0]}, ssim: {ssim} , mse: {mse.item()}')

                total_ssim += ssim
                total_mse += mse.item()
            # 记录验证集上的平均 SSIM 和 MSE 到 TensorBoard
            avg_ssim_val = total_ssim / (len(val_dataloader)+len(train_dataloader))
            avg_mse_val = total_mse / (len(val_dataloader)+len(train_dataloader))
            writer.add_scalar('Metrics/Average_ssim_val', avg_ssim_val, epoch)
            writer.add_scalar('Metrics/Average_mse_val', avg_mse_val, epoch)

            print(f'Validation : Average_ssim: {avg_ssim_val} , Average_mse: {avg_mse_val}')

        model.train()

    # 训练结束后关闭 TensorBoard 的 SummaryWriter
    writer.close()

# 开始训练
train(vqvae, train_dataloader, val_dataloader, opt)

