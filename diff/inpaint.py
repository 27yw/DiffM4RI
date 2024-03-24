# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""
from tensorflow_mri import ssim3d
import os
import argparse
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
import time
import conf_mgt
import tqdm
import yaml

from diffusion import create_diffusion
from models import DiT_models

# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

def txtread(path):
    path = os.path.expanduser(path)
    with open(path, 'r') as f:
        return f.read()
def yamlread(path):
    return yaml.safe_load(txtread(path=path))

import os
import numpy as np
from torch.utils.data import Dataset,DataLoader

class NpyDataset(Dataset):
    def __init__(self, folder1, folder2):
        self.folder1 = folder1
        self.folder2 = folder2

        # 获取两个文件夹中相同文件名的列表
        self.common_files = list(set(os.listdir(folder1)) & set(os.listdir(folder2)))

    def __len__(self):
        return len(self.common_files)

    def __getitem__(self, index):
        # 获取相同文件名的数据路径
        filename = self.common_files[index]
        file_path1 = os.path.join(self.folder1, filename)
        file_path2 = os.path.join(self.folder2, filename)

        # 加载数据
        data1 = np.load(file_path1)
        data2 = np.load(file_path2)
        data1 = data1.squeeze(0)
        data2 = data2.squeeze(0)
        # 返回数据的元组
        return data1, data2, filename


def main(conf: conf_mgt.Default_Conf):

    print("Start", conf['name'])

    device = "cuda"

    model = DiT_models["DiT-B/4"](
        input_size=32,
        in_channels=4
    )
    checkpoint = torch.load(conf["model_path"], map_location=torch.device(device))
    model.load_state_dict(checkpoint["model"])
    diffusion = create_diffusion(timestep_respacing="",conf=conf,diffusion_steps=750)  # default: 1000 steps, linear noise schedule

    model.to(device)
    model.eval()

    show_progress = conf.show_progress

    cond_fn = None

    def model_fn(x, t, **kwargs):
        # assert y is not None
        return model(x, t)
        # return model(x, t, None, gt=gt)

    print("sampling...")

    dset = 'eval'
    # folder1_path = "./npy_2019_all"
    # folder2_path = "./data/brats2019/flair_t1ce_missing/gt_keep_masks"
    folder1_path = "/root/autodl-tmp/npy_2019_all"
    folder2_path = "/root/autodl-tmp/data/brats2019/flair_t1ce_missing/gt_keep_masks"
    result_path = "./inpaint_results_" + folder2_path.split("/")[-2]
    os.makedirs(result_path, exist_ok=True)
    eval_name = conf.get_default_eval_name()
    dataset = NpyDataset(folder1=folder1_path, folder2=folder2_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)
    print(f'Load dataset : {len(loader)}')
    # dl = conf.get_dataloader(dset=dset, dsName=eval_name)
    ssim_list = []
    mse_list = []
    i = 0
    for  x, y,filename in loader:
        if True:
            print(i,filename)
            i+=1
            
            #x = gts
            #y = gt_keep_mask
            #print(x.shape)
            model_kwargs = {}

            model_kwargs["gt"] = x.to("cuda").float()
            # print(batch['GT_name'])
            # print(model_kwargs["gt"].shape)
            # print(batch['GT_name'])
            gt_keep_mask = y.to("cuda")
            if gt_keep_mask is not None:
                model_kwargs['gt_keep_mask'] = gt_keep_mask.float()

            batch_size = model_kwargs["gt"].shape[0]

            result,final_list = diffusion.inpaint_sample_loop(
                model_fn,
                (batch_size, 4 ,conf.image_size, conf.image_size, conf.image_size),
                clip_denoised=conf.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=device,
                progress=show_progress,
                return_all=True,
                conf=conf
            )
            # print(result)
            srs = result['sample']
            gts = result['gt']
            lrs = result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask'))
            mse = torch.nn.functional.mse_loss(srs, x.to("cuda"))
            ssim = ssim3d(x.contiguous().clone().detach().cpu(),
                              srs.contiguous().clone().detach().cpu(), filter_size=1).numpy()
            ssim_list.append(ssim)
            mse_list.append(mse.item())
            gt_keep_masks = (model_kwargs.get('gt_keep_mask') * 2 - 1)
            # print(batch['GT_name'])
            # print(type(srs))
            os.makedirs(f'{result_path}/{filename[0].replace(".npy","")}/', exist_ok=True)
            np.save(f'{result_path}/{filename[0].replace(".npy","")}/{filename[0]}',srs.cpu().detach().numpy())
            print(f"{result_path}/{filename[0]}_mse{mse}_ssim{ssim}")
            # np.save("./result/gt",gts.cpu().detach().numpy())
            # np.save("./result/gt_keep_masks",gt_keep_masks.cpu().detach().numpy())
            # np.save("./result/lrs",lrs.cpu().detach().numpy())
            # conf.eval_imswrite(
            #     srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
            #     img_names=batch['GT_name'][0], dset=dset, name=eval_name, verify_same=False)

        print("=================================sampling complete============================")
        print("Max ssim",max(ssim_list))
        print("Min ssim",min(ssim_list))
        print("average ssim",sum(ssim_list)/len(ssim_list))
        print("Max mse",max(mse_list))
        print("Min mse",min(mse_list))
        print("average mse",sum(mse_list)/len(mse_list))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default="./config/example.yml")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/4")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=32)
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    main(conf_arg)
