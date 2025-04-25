import os
import numpy as np
from PIL import Image

ignored_ids = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]).astype(np.uint8)
color_map = [(  0,  0,  0),
             (  0,  0,  0),
             (  0,  0,  0),
             (  0,  0,  0),
             (  0,  0,  0),
             (111, 74,  0),
             ( 81,  0, 81),
             (128, 64,128),
             (244, 35,232),
             (250,170,160),
             (230,150,140),
             ( 70, 70, 70),
             (102,102,156),
             (190,153,153),
             (180,165,180),
             (150,100,100),
             (150,120, 90),
             (153,153,153),
             (153,153,153),
             (250,170, 30),
             (220,220,  0),
             (107,142, 35),
             (152,251,152),
             ( 70,130,180),
             (220, 20, 60),
             (255,  0,  0),
             (  0,  0,142),
             (  0,  0, 70),
             (  0, 60,100),
             (  0,  0, 90),
             (  0,  0,110),
             (  0, 80,100),
             (  0,  0,230),
             (119, 11, 32),
             (  0,  0,142)]
a = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
id_set = np.array(a).astype(np.uint8)
id_extra = np.array([-1]).astype(np.uint8)

pred_root = '/root/autodl-tmp/mmsegmentation/data/seg/'
save_root = '/root/autodl-tmp/mmsegmentation/data/results/'
gt_list = '/root/autodl-tmp/mmsegmentation/data/val.lst'
gt_root = '/root/autodl-tmp/mmsegmentation/data/cityscapes/'
img_list = [line.strip().split() for line in open(gt_list)]
gt_files = []
for item in img_list:
    image_path, label_path = item
    name = os.path.splitext(os.path.basename(image_path))[0]
    gt_files.append({
        "label": label_path,
        "name": name,
    })
   
for gt_file in gt_files:
    sv_img = np.zeros((1024,2048,3)).astype(np.uint8)
    label = np.array(
        Image.open(gt_root+gt_file["label"]).convert('P')
    )
    pred = np.array(
        Image.open(pred_root+gt_file["name"]+'.png').convert('P')
    )
    for ig_id in ignored_ids:
        pred[label==ig_id] = ig_id
    for idx in id_set:
        for j in range(3):
            sv_img[:,:,j][pred==idx] = color_map[idx][j]
    for idx in id_extra:
        for j in range(3):
            sv_img[:,:,j][pred==idx] = color_map[34][j]
    
    sv_img = Image.fromarray(sv_img)
    sv_img.save(save_root+gt_file["name"]+'.png')
