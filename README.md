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
