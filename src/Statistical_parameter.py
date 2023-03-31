from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os

import torch
import torch.utils.data
from opts import opts
from model.model import create_model, load_model, save_model
from model.data_parallel import DataParallel
from logger import Logger
from dataset.dataset_factory import get_dataset
from trainer import Trainer
from AutomaticWeightedLoss import AutomaticWeightedLoss

from torchstat import stat

def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        # print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)


if __name__ == '__main__':
    opt = opts().parse()
    torch.manual_seed(opt.seed)
    Dataset = get_dataset(opt.dataset)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    opt.training_seg = False
    opt.training_attention = False
    opt.attention_swith = 'ECA'
    opt.head_switch = 'None'
    opt.load_model = '../models/kitti_no_seg_head.pth'
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)

    model = load_model(model, opt.load_model, opt)
    # stat打印完整信息
    model_structure(model)
    # stat(model, (3, 224, 224))
    # 模型的总参数量
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))

    # ------------------------------------------------------------------------------------
    opt.training_seg = True
    opt.training_attention = False
    opt.attention_swith = 'ECA'
    opt.head_switch = 'None'
    opt.load_model = '../models_dla/kitti_half_dla34.pth'
    # opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)

    model = load_model(model, opt.load_model, opt)
    # stat打印完整信息
    model_structure(model)
    # stat(model, (3, 224, 224))
    # 模型的总参数量
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))

# ------------------------------------------------------------------------------------
    opt.training_seg = True
    opt.training_attention = True
    opt.attention_swith = 'ECA'
    opt.head_switch = 'None'
    opt.load_model = '../models_dla/kitti_merge_dla34_ECA_20+.pth'
    # opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)

    model = load_model(model, opt.load_model, opt)
    # stat打印完整信息
    model_structure(model)
    # stat(model, (3, 224, 224))
    # 模型的总参数量
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))

    # ------------------------------------------------------------------------------------
    opt.training_seg = True
    opt.training_attention = True
    opt.attention_swith = 'ECA'
    opt.head_switch = 'merge-attention'
    opt.load_model = '../models_dla/kitti_merge_dla34_ECA_attentionmerge_40.pth'
    # opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)

    model = load_model(model, opt.load_model, opt)
    # stat打印完整信息
    model_structure(model)
    # stat(model, (3, 224, 224))
    # 模型的总参数量
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))