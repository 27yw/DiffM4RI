import os
import numpy as np
import torch
# 源文件夹路径
source_folder = './npy'

# 新文件夹路径
target_folder = './npy_normalized_after'
os.makedirs(target_folder, exist_ok=True)

# 子文件夹列表
subfolders = ['flair','t1', 't1ce', 't2']

# 遍历每个子文件夹
for filename in os.listdir(os.path.join(source_folder, subfolders[0])):
    # 构建文件名的完整路径
    file_paths = [os.path.join(source_folder, subfolder, filename) for subfolder in subfolders]
    print(file_paths)
    # 读取每个文件的数据
    data = []
    for file_path in file_paths:
        x_start1 = torch.tensor(np.load(file_path))
        reshaped_data = x_start1.view(x_start1.size(0), -1)

        # 使用min-max归一化
        min_vals, _ = torch.min(reshaped_data, dim=1, keepdim=True)
        max_vals, _ = torch.max(reshaped_data, dim=1, keepdim=True)
        scaled_data = (reshaped_data - min_vals) / (max_vals - min_vals)
 
        # 将归一化后的数据重新reshape为3D形状
        x_start = scaled_data.view(x_start1.shape)
        data.append(x_start.numpy())
    # data = [np.load(file_path) for file_path in file_paths]
    #print(data.shape) torch.Size([1, 64, 16, 16, 16])
    # 在第一个维度进行拼接
    concatenated_data = np.concatenate(data, axis=1)
    print(concatenated_data.shape)
    
    # 构建新文件名的完整路径
    target_filepath = os.path.join(target_folder, filename)
    
    # 保存拼接后的数据到新文件夹
    np.save(target_filepath, concatenated_data)

print("数据拼接完成并保存到新文件夹。")
