import os
import nibabel as nib
import numpy as np
from PIL import Image
import SimpleITK as sitk

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

def process_niigz(file_path, save_folder):
    # Read niigz file
    nii_img = sitk.ReadImage(file_path)
    nii_data = sitk.GetArrayFromImage(nii_img)
    nrrd_img = sitk.GetImageFromArray(nii_data)

    nrrd_img = resize_image_itk(nrrd_img, (240, 240, 155), resamplemethod=sitk.sitkLinear)
    data = sitk.GetArrayFromImage(nrrd_img)
    # data = data+0.8
    data[data<0]=0
    # Get middle indices for each dimension
    middle_indices = [dim // 2 for dim in data.shape]

    # Extract slices from the middle of each dimension
    slices = [data[middle_indices[0], :, :],  # Slice in the d dimension
              data[:, middle_indices[1], :],  # Slice in the w dimension
              data[:, :, middle_indices[2]]]  # Slice in the h dimension

    # Normalize data to [0, 255]
    # slice_data=slice_data -0.4
    slices = [(slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255.0 for slice_data in slices]

    # Convert to uint8 NumPy arrays
    slices = [slice_data.astype(np.uint8) for slice_data in slices]

    # Resize to 128x128
    resized_slices = [Image.fromarray(slice_data, 'L').resize((128, 128), Image.ANTIALIAS) for slice_data in slices]

    # Get the original file name and dimension names
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    dimension_names = ['dim_{}'.format(idx) for idx in range(len(middle_indices))]

    # Create the save folder
    os.makedirs(save_folder, exist_ok=True)

    # Save processed grayscale images to the new folder
    for idx, resized_slice in enumerate(resized_slices):
        print(np.asarray(resized_slice).shape)
        save_path = os.path.join(save_folder, '{}_{}.png'.format(file_name, dimension_names[idx]))
        resized_slice.save(save_path)
        print('Saved:', save_path)
if __name__ == "__main__":
    niigz_folder = "/root/autodl-tmp/vqvae/recon/4fid/0_missing"  # niigz文件所在文件夹的路径
    save_folder = "/root/autodl-tmp/gen"  # 保存处理后的图片的文件夹路径

    # 处理文件夹下的每个niigz文件
    for file_name in os.listdir(niigz_folder):
        if file_name.endswith(".nii.gz"):
            file_path = os.path.join(niigz_folder, file_name)
            process_niigz(file_path, save_folder)
