## FLSCNet: full-level supervision complementary network for infrared small target detection


## Prerequisite

* Trained and Tested on Ubuntu 20.04.6, with Python 3.8, PyTorch 1.12.1, Torchvision 0.13.1, CUDA 11.6, and 1x NVIDIA 3090

* [The NUAA-SIRST download dir](https://github.com/YimianDai/sirst)

* [The IRSTD-1k download dir](https://github.com/RuiZhang97/ISNet)


## Results and Trained Models

on NUAA-SIRST

|  Model  | IoU (x10(-2)) | nIoU (x10(-2)) | PD (x10(-2)) | FA (x10(-6)) |                                                                                                 |
|:-------:|:-------------:|:--------------:|:------------:|:------------:|:-----------------------------------------------------------------------------------------------:|
| FLSCNet |     81.24     |     78.90      |    100.0     |     4.79     | [[Weights]](https://drive.google.com/file/d/1oRyJFU9bypLWF0lhlcPRBduRa6TK0SVl/view?usp=sharing) |

on IRSTD-1k

|  Model  | IoU (x10(-2)) | nIoU (x10(-2)) | PD (x10(-2)) | FA (x10(-6)) |                                                                                                 |
|:-------:|:-------------:|:--------------:|:------------:|:------------:|:-----------------------------------------------------------------------------------------------:|
| FLSCNet |     72.45     |     65.35      |    94.56     |     1.21     | [[Weights]](https://drive.google.com/file/d/1RlGMfnAsBom4L967UqSIJRYbimn9yQzH/view?usp=sharing) |

*This code is highly borrowed from [DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Thanks to Boyang Li et al..


## References

B. Li, C. Xiao, L. Wang, Y. Wang, Z. Lin, M. Li, W. An, Y. Guo, Dense nested attention network for infrared small target detection, IEEE Transactions on Image Processing 32 (2022) 1745–1758. [[code]](https://github.com/YeRen123455/Infrared-Small-Target-Detection)