import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from  your model import your model
#图片预处理
def img_preprocess(img_in):
    img = img_in.copy()
    img = img[:, :, ::-1]   				# 1
    img = np.ascontiguousarray(img)			# 2
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize(1024),
        # transforms.CenterCrop(1024),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img = transform(img)
    img = img.unsqueeze(0)					# 3
    return img

#获取想可视化的某一层
# 1：定义用于获取网络各层输入输出tensor的容器
# 并定义module_name用于记录相应的module名字
module_name = []
features_in_hook = []
features_out_hook = []
# 2：hook函数负责将相应的module名字、获取的输入输出 添加到feature列表中
def hook(module, fea_in, fea_out):
    print("hooker working")
    module_name.append(module.__class__)
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None

#定义模型
model = your model

print(model)
#加载权重
# model_weight_path="PIDNet_S_Cityscapes_val.pth"
model_weight_path="best_ddr23.pth"

pretrained_dict=torch.load(model_weight_path)
if 'state_dict' in pretrained_dict:
    pretrained_dict = pretrained_dict['state_dict']
model_dict = model.state_dict()
pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                   if k[6:] in model_dict.keys()}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.eval()
print("加载成功okokkokokookokookookoookokokokokok")

# 3，想获取哪一层的特征图
#3.1直接用层的名字来进行特定特征层获取
layername=model.final_layer.relu
print("==========================++++++++++++++++++++",type(layername))
layer5_= layername.register_forward_hook(hook)
imagename='png'
#模型处理图片
origin_img = cv2.imread(imagename)
print("输入的图像尺寸",origin_img.shape)
crop_img=img_preprocess(origin_img)
print("处理过后的尺寸",crop_img.shape)
output=model(crop_img)
print("模型处理之后的image",output.shape)


# print("获取到的数据==============================================================================================")
# print("----------------",features_out_hook[0].shape)  #orch.Size([1, 128, 128, 128])
heatmap=features_out_hook[0]

# print("==========================++++++++++++++++++++",type(heatmap))
# print("heatmap",heatmap.shape)  #1,128,128,128
#是否进行上采样
# heatmap=F.interpolate(heatmap,
#                       size=[128, 256],
#                       mode='bilinear', align_corners=False)
# print("上采样之后的heatmap尺寸",heatmap.shape)  #128,128,128
# print(type(heatmap))   #tensor
# 将特征图转换成热力图（将各个通道特征图相加求和）
def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:, 0, :, :] * 0 #生成全0的向量
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap += feature_map[:, c, :, :] #将每个通道累加到heatmap中
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps

# 绘制特征图
print("================绘制特征图=========================")

#将所有通道加和结果
def draw_feature_map1(features, save_dir='', name=""):
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    idx = 0
    if isinstance(features, torch.Tensor):
        for heat_maps in features:
            heat_maps = heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, 4)   #编号为 4  9 20 效果较好
                superimposed_img = heatmap
                plt.imshow(superimposed_img, cmap='gray')
                # plt.title('M')
                plt.axis("off")
                if save_dir and name:
                    save_path = os.path.join(save_dir, name)
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # 保存图像，去掉白边
                plt.show()
draw_feature_map1(heatmap,'../可视化图片/m','bus22s')

