# DRFRNet:A Dual-Resolution Network with Feature Rectification for Real time semantic segmentation

## Introduction  
This is the code of DRFRNet:A Dual-Resolution Network with Feature Rectification for Real time semantic segmentation.  DRFRNet yields 76.4% mIoU at 142 FPS on Cityscapes test set and 77.2% mIoU at 230 FPS on CamVid test set.
![fig2](https://github.com/user-attachments/assets/2dd8f7e4-7cf3-4775-a97f-52b5db3da2fc)
## requirements  
Here I list the software and hardware used in my experiment  
pytorch==1.9.1  
4090*1   
cuda==12.3  
## Quick start  
### Data preparation  
You need to download the [Cityscapes]（https://www.cityscapes-dataset.com/） datasets. and rename the folder cityscapes, then put the data under data folder.  
```
└── data
    ├── cityscapes
       └── list
```
### TRAIN  
train the model with 1 nvidia-4090  
```
python tools/train.py --cfg configs/cityscapes/DRFRnet - S.yaml
```
### SPEED
The test speed follows the DDRNet and PIDNet test methodology, and the speed test is in speed_test.py
### Heat Map Image (SINGLE CLASS)
In image segmentation tasks, heat maps can be used to represent the probability of which category or object each pixel belongs to. Each category has a corresponding heat map that shows the distribution of pixels in that category.
![train2-truck-new-211555](https://github.com/user-attachments/assets/1e2eac93-03fc-455b-950b-41408afccaf2)
### Segmentation Image 
Use "generate Segmentation Image" to generate the segmentation results.
![frankfurt_000000_014480_leftImg8bit](https://github.com/user-attachments/assets/6c7a6f46-c5bc-4e3a-b907-e78dc2e99e82)
### Intermediate feature maps of the model  
Use the "Intermediate feature maps of the model" file to generate feature maps.
![right](https://github.com/user-attachments/assets/4ed9b19f-a934-4bfa-a6ce-9e6a54d7528b)    ![00056](https://github.com/user-attachments/assets/bb8ca363-14ca-44e6-946a-4a4d2e43b35b)
