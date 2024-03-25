# DiffM4RI \[[Paper]()]

<div align="center">
  <img src="imgs/model.jpg">
</div>

## 🌟 Highlights
- 📚 A 3D generative model based on diffusion models with multimodal transformers that learns the joint distribution in latent space over all modalities.
- 🚤 An modality inpainting method filling randomly missing modalities and generating semantically coherent and high-resolution images efficiently without training and resampling steps.
- 🏆 Experiments on BraTs 2018, BraTs 2019 and BraTs 2021 outperform other methods.

## 🔨 Usage
### Training
To train the DiffM4RI model on your own data, follow these steps:
#### 1. **Prepare Your Training Data**:
Ensure that your training data is split according to its modality. The target file structures should be like the following:
```
data
 ├── FLAIR
 │ ├── BraTS2021_00621_flair.nii.gz
 │ ├── ...
 ├── T1
 │ ├── BraTS2021_00621_t1.nii.gz
 │ ├── ...
 ├── T1ce
 │ ├── BraTS2021_00621_t1ce.nii.gz
 │ ├── ...
 ├── T2
 │ ├── BraTS2021_00621_t2.nii.gz
 │ ├── ...
 ├── ...
```
Then, modify the `path` in `vqvae/train.py` according to your own data.
```
path = "../data/t1"
```

#### 2. **Run the Training Script of 3D VQVAE for each modality**: 
Run `train.py` to execute the following command in your terminal:

```
cd vqvae
python train.py
```
This will start the training process of the 3DVQVAE model on your prepared data.

#### 3. **Run the latent representation Script of 3D VQVAE to get the latents of each modality**: 
Modify the `path` and `ckpt_path` in `vqvae/train.py` according to your own data and checkpoint.
```
ckpt_path = './results/t2.pth'
path="../data/t2"
```
Run `test.py` to execute the following command in your terminal:
```
cd vqvae
python test.py
```

#### 4. **Prepare multimodal Data for diffusion**: 
With a series of .npy folders, you can place them in a whole folders following structures below:
```
data
 ├── FLAIR
 │ ├── BraTS2021_00621.npy
 │ ├── ...
 ├── T1
 │ ├── BraTS2021_00621.npy
 │ ├── ...
 ├── T1ce
 │ ├── BraTS2021_00621.npy
 │ ├── ...
 ├── T2
 │ ├── BraTS2021_00621.npy
 │ ├── ...
 ├── ...
```
Modify the `source_folder` in `diff/data_operation.py` according to your own data.
Run `data_operation.py` to execute the following command in your terminal:
```
cd diff
python data_operation.py
```
After run `data_operation.py`, you will get:
```
npy_data
 ├── BraTS2021_00621.npy
 ├── BraTS2021_00622.npy
 ├── ...
```

#### 5. **Train diffusion**: 
Run `train.py` to execute the following command in your terminal:

```
cd diff
torchrun train.py
```
This will start the training process of the diffusion model on your prepared data.

#### 6. **Generation**: 
Run `sample.ipynb` to generate images.

#### 7. **Modality Inpainting**: 
Run `inpaint.py` to generate images.
```
cd diff
python  inpaint.py
```

## 🔗 Checkpoint
checkpoints are available with these links: [Baidu NetDisk Download (pwd:dm4r)](https://pan.baidu.com/s/11VvF0_8rhvq7PYBv5q4enA?pwd=dm4r) and [Google Drive]()