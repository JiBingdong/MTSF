# MTSF: Multi-level Temporal-Spatial Fusion Network for Driver Attention Prediction
Recently, there have been certain important advances in the field of driver attention prediction in dynamic scenarios for the bionic perception of intelligent vehicles. Given the visual characteristics of experienced and skilled drivers, who can rapidly and accurately perceive their environment and effectively identify the significance of environmental stimuli, it is believed that an effective driver attention prediction model should not only extract temporal-spatial features comprehensively but also flexibly highlight the importance of these features. In light of this challenge, this paper proposing an improved multi-scale temporal-spatial fusion network model, which adopts an encoder-fusion-decoder architecture and can fully use the scale, spatial, and temporal information in video data. First, in the encoder, two independent feature extraction backbones, one 2D and another 3D, are used to extract four temporal-spatial features of different scales from the input video clip and align them in the feature dimension. Then, in the hierarchical temporal-spatial feature fusion, features from different levels are added to the channel and fused using an attention mechanism to achieve the 3D-2D soft combination effect guided by spatial features. Finally, in the hierarchical decoder and prediction module, hierarchical decoding and prediction are performed on temporal-spatial features of different branches, and the results of multiple branches are fused to generate saliency maps. Experiments on three challenging datasets show that the proposed method is superior to the state- of-the-art methods regarding several saliency evaluation metrics and can predict driver attention more accurately. By using an effective temporal-spatial fusion strategy based on attention mechanism, the proposed driver attention prediction method can detect important targets and identify risk areas for a human-like autonomous driving system.
![overall(1)_24](https://user-images.githubusercontent.com/68813286/226500933-4e3207d1-6728-40c7-abf8-4e9a01769346.png)

# How-To

This repository was used throughout the whole work presented in the paper so it contains quite a large amount of code. Nonetheless, it should be quite easy to navigate into. In particular:

- **Main Directory**: The Python files in the main directory are used for training and testing, corresponding to different datasets (DR(eye)VE, DADA-2000, TDV).

- **model**: Contains the MTSF model files.

- **data**: Contains files used for loading data during training and testing, corresponding to different datasets.

- **saved_models**: Contains the weight files saved during training.

Please note that you need to check the paths in the above scripts and change them to your own paths.

All Python code has been developed and tested with Pytorch.

## Pre-trained weights:
Pre-trained weights of the MTSF model can be downloaded from this link(链接: https://pan.baidu.com/s/1EXLK_GemaG0h36A9VncELw 提取码: b8ny).

# Vidoe Demo
In order to be able to clearly demonstrate the contribution of our proposed MTSF, we made a video demo, you can find it from [here](https://www.bilibili.com/video/BV1cx4y1N7Vq/?spm_id_from=333.999.0.0&vd_source=52c141951b0d1bd188be6c941f796841).
![image](https://user-images.githubusercontent.com/68813286/226501296-7b7f3a1c-36f3-41be-96e3-4c5404db832d.png)

# Citation
If you find this work useful for your research, please cite our paper:  
```  
@ARTICLE{10803899,  
  author={Jin, Lisheng and Ji, Bingdong and Guo, Baicang and Wang, Huanhuan and Han, Zhuotong and Liu, Xingchen},  
  journal={IEEE Transactions on Intelligent Transportation Systems},   
  title={MTSF: Multi-Scale Temporal–Spatial Fusion Network for Driver Attention Prediction},   
  year={2024},  
  volume={},  
  number={},  
  pages={1-16},  
  keywords={Feature extraction;Vehicles;Three-dimensional displays;Predictive models;Convolution;Long short term memory;Data mining;Solid modeling;Attention mechanisms;Vehicle dynamics;Driver attention prediction;saliency prediction;temporal-spatial fusion;3D convolution;attention mechanism},  
  doi={10.1109/TITS.2024.3510116}}
```  

