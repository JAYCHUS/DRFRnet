import os
import warnings
import  cv2
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
# 应用类激活
from pytorch_grad_cam import GradCAM
import time

# 更换运行时路径：设置父类为运行时路径
os.chdir("..")


# 更换任务名
save_father_path = "../segmentation_image"
# task_name = "motorcycle_pid_test"
task_name = "heat_map"
path = os.path.join(save_father_path, task_name)
# 更换模型
from your model import your model
model_state_file = "your weight"
# from models.ddrnet_23_slim2 import get_pred_model
# model_state_file = "best/ddrnet_23_slim.pth"



# 更换图片
# defaultPath = "data/cityscapes/leftImg8bit/test/"
defaultPath = "../data/cityscapes/leftImg8bit/"
imgPaths = os.listdir(defaultPath)
# imgPaths = ["bus1", "bus2", "bus3", "bus4","bus5","bus6"]
for (i, img) in enumerate(imgPaths):
    imgPaths[i] = defaultPath + img
# 更换类别
# 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
#         'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
#         'train', 'motorcycle', 'bicycle'
class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
               'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
               'train', 'motorcycle', 'bicycle']
# class_names = ['road']

#将模型包装起来并使用向前传播方法 在输入x的时候调用前向传播并返回输出
class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

#用于处理目标计算  在分割模型的输出上针对特定类被和掩码进行计算 通过掩码计算特定类别的像素值得总和
#特定类别在指定掩码区域内的激活值总和  用于检查模型在特定类别和区域的表现
class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category #目标类别
        self.mask = torch.from_numpy(mask)#numoy转为tensor
        if torch.cuda.is_available():#如果gpu可用
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum() #从模型输出中提取指定类别激活值


def main():
    # 载入权重
    model = your model
    model = model.eval()
    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    print(f'载入的权重数:{len(pretrained_dict)}')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if torch.cuda.is_available():
        model = model.cuda()
    print("权重加载ok")
    # model = SegmentationModelOutputWrapper(model)

    for (index, imgPath) in enumerate(imgPaths):
        """ 返送最后一层热力图 """
        file_path, file_name = os.path.split(imgPath)  #拿到路径和图片名字
        short_name, extension = os.path.splitext(file_name)#
        image = np.array(Image.open(imgPath).convert('RGB')) #将图像转化为RGB格式
        rgb_img = np.float32(image) / 255  #归一化图像
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]) #标准化
        # Taken from the torchvision tutorial
        # https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html

        # 创建任务目录
        if not os.path.exists(path):
            os.makedirs(path)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        output = model(input_tensor)  #获取输出
        # print(type(output))
        # print(output.keys())

        normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()#对输出进行归一化 将每一类别的概率归一化到【0，1】范围  包含每个类被的概率 并将结果移动到CPU上
        sem_classes = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
            'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
            'train', 'motorcycle', 'bicycle']
        sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}  #将类别名称到索引的映射字典 可以将类别名称转换为索引
        #遍历类别名称列表  并对每个累并生成预测结果和热力图 然后将其保存为图像文件
        for (idx, class_name) in enumerate(class_names):
            car_category = sem_class_to_idx[class_name]  # 比如说car是第一个类别
            car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()  #对每个像素点选择概率最高的类别 （预测的类别 并转为numpy）
            print("car_mask_shape",car_mask.shape)
            # car_mask_uint8 = 255 * np.uint8(car_mask == car_category)  #为了可视化将其*255
            car_mask_float = np.float32(car_mask == car_category)   #用于grad-cam

            # 查看预测类别结果
            # print(image.shape)
            # print(car_mask_uint8[:, :, None].shape)
            #car_mask_uint8[:, :, None] 为了拼接用None增加一个维度 仍是单通道图像
            #np.repeat(car_mask_uint8[:, :, None], 3, axis=-1) 为了与3通道进行拼接 对最后一个维度进行扩展 扩展为3维度的
            # print(np.repeat(car_mask_uint8[:, :, None], 3, axis=-1).shape)
            # both_images = np.hstack((image, cv2.resize(np.repeat(car_mask_uint8[:, :, None], 3, axis=-1),
            #                                            (image.shape[1], image.shape[0]),
            #                                            interpolation=cv2.INTER_NEAREST)))  # 将原始图像和掩码水平拼接
            # both_images = Image.fromarray(both_images) #转换为PIL图像
            time_str = time.strftime('%d%H%M')  #生成当前时间的字符串
            img_path = os.path.join(path, short_name + "-" + class_name + "-预测类别结果-" + time_str + '.png')
            # both_images.save(img_path) #保存为png图像

            target_layers = [model.final_layer]  # 目标层
            targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
            with GradCAM(model=model,
                         target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=input_tensor,
                                    targets=targets)[0, :]
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)  # 将热力图蝶姐到原图上
                print(type(cam_image))  # <class 'numpy.ndarray'>
                plt.show()
            res = Image.fromarray(cam_image)
            img_path = os.path.join(path, short_name + "-" + class_name + "-new-" + time_str + '.png')
            res.save(img_path)


if __name__ == '__main__':
    main()
