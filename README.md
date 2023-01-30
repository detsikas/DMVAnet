# DMVAnet
The DMVAnet architecture is a deep learning network that achieves state of the art performance for the document image binarization task. The architecture is shown in the following image

![](/images/DMVAnet.png?raw=true)

The architecture is described in detail in the paper "A Dilated MultiRes Visual Attention U-Net for Historical Document Image Binarization" by Nikolaos Detsikas , Nikolaos Mitianoudis and Nikolaos Papamarkos (Electrical and Computer Engineering Department, Democritus University of Thrace, University Campus Xanthi-Kimmeria, Xanthi 67100, Greece).

## Other networks
Apart from the proposed DMVAnet architecture, the repository creates, trains and examines many more deep learning network models. The paper describes in detail the architectures that led to the DMVAnet, while the repository also contains architectures that were tried but did not produce competitive results.

Here is a list of the other networks contained in the repository

| Model | Description |
| ----- | ----------- |
| Basic unet | The funcamental unet architecture |
| Residual unet | The base unet architecture extended with residual connections in the convolution blocks |
| Visual attention residual unet | The residual unet architecture extended with visual attention blocks |
| Multires visual attention unet | The visual attention architecture extended with multires blocks and residual paths |
| Dense unet | The original Dense unet architecture |
| Unet++ | The unet++ architecture extended with residual connections, visual attention blocks, multires blocks and residual paths |
| DeeplabV3+ | The original DeeplavV3+ architecture |
| Residual MobileNet V2 | The residual unet with MobileNet V2 ImageNet pretrained encoder |
| Residual VGG19 | The residual unet with VGG19 ImageNet pretrained encoder |
| MutliRes VGG19 | The Multires unet with VGG19 ImageNet pretrained encoder |

## Code structure
The code consists of 4 directories. 

| Directory | Description |
| --------- | ----------- |
| common/   | Common code to the rest of the project that includes loss functions, metrics, scoring functions, preprocessing functions etc. |
| prediction/ | Scripts for predicting with a designated model and performing statistical operations on the results |
| preprocessing/ | Code that constructs tf record datasets, performs verification on the constructed datasets, dataset operations such as merge, split, dataset augmentation scripts etc. |
| training | Scripts that train all the networks that were investigated |

## Dataset construction process
The original datasets are in the form of original and ground truth images, separeted in different folders. Many different datasets were used for the project experiments. Even though their file system structure is similar, they cannot be handled with a single common piece of loading code. In addition to that, we opted to perform all the necessary augmentations of line and not included them in the dataset processing pipeline at training time.

For these reasons, we are first constructing the tf record files for all datasets, augment them and then use them in the training of the networks. The following describe the dataset creation process in more detail:

1. Split the source images into patches and save them as numpy files. This is done with the preprocess.py script.

    Example usage: 
    
        python preprocess.py --input-path <path_to>/102_msi/S_MSI_1_0/S_MSI_1/orig/ --output-path smi/1/npy/oring --border-type reflect_101
        
        python preprocess.py --input-path <path_to>/S_MSI_1_0/S_MSI_1/GT/ --output-path smi/1/npy/gt --border-type reflect_101 --is_binary
        
2. Create a tf record file out of the store numpy files. THis is done with the dataset_to_tfrecord.py script

    Example usage: 
    
        python dataset_to_tfrecord.py --dataset-training-path smi/1/npy/orig/ --dataset-gt-path smi/1/npy/gt/ --output-dataset-file sm1_1.tfrecord
        
3. Keep one channel from the label (y) images, as they are binary. fix_tf_record_y_dim.py is the script that is used for that.

    Example usage: 
    
        python fix_tf_record_y_dim.py --tfrecord-file smi/smi_1.tfrecord --output-dataset-path .
        
4. Ensure that the label images are binary with the binarize_tf_record_y.py script.

    Example usage: 
    
        python binarize_tfrecord_y.py --tfrecord smi/smi_1.tfrecord --output-dataset-path .
        
        
The following steps are optional and can be run any time to verify the tf record file contents        
1. Verify that the label data is binary with the is_y_binary.py script.

    Example usage: 
    
        python is_y_binary.py --tfrecord smi/smi_1.tfrecord
        
2. Display the tf record contents. This reconstruct the images from the numpy patches. It is particularly usefule to view and verify the border policy. The display_dataset.py is used.

    Example usage:
    
        python display_dataset.py smi/smi_1.tfrecord
        
3. Reconstruct the images and labels from the tf record numpy patches, compare them to the stored numpy patches of the first step of the dataset creation process and verify that all samples have been included. This is done with the verify_tfrecord.py

    Example usage:
    
        python verify_tfrecord.py --tfrecord smi/smi_1.tfrecord --x-images-root-path smi/1/npy/orig/ --y-images-root-path smi/1/npy/gt/
        

