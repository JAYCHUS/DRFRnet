# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


import _init_paths
import models
import datasets
from ..lib.config import config
from ..lib.config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, speed_test
from ..lib.utils.utils import speed_test

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/cityscapes/ddrnet23_slim_fusion.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    print("inininin")
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):#解决版本问题
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    # dump_input = torch.rand(
    #     (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'best.pth')      #配置文件 默认else
        # model_state_file = os.path.join(final_output_dir, 'final_state.pth')      
    logger.info('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load(model_state_file) #拿到训练好的权重
    #检查加载的与训练权重中是否包含state_dict选项
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']

    model_dict = model.state_dict() #获取当前模型的参数字典
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    #对预训练模型参数字典进行处理，去掉键名中的前6个字符（通常是'module.'）以匹配当前模型的参数字典，只保留在当前模型中存在的键值对。
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
        #遍历处理后的预训练参数字典，打印日志信息，表示正在从预训练模型中加载对应的参数。
    model_dict.update(pretrained_dict)#将处理后的与训练参数字典更新到当前模型的参数字典中 以便将权重应用到当前模型中
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    
    start = timeit.default_timer()
    #验证集
    mean_IoU, IoU_array, pixel_acc, mean_acc = testval(config, 
                                                        test_dataset, 
                                                        testloader, 
                                                        model,
                                                        sv_pred=False)

    msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
        Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, 
        pixel_acc, mean_acc)
    logging.info(msg)
    logging.info(IoU_array)

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int((end-start)/60))
    logger.info('Done')


if __name__ == '__main__':
    main()
