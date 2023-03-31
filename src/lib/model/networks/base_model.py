from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .DCNv2.dcn_v2 import DCN_TraDeS
import numpy as np
from ..utils import _sigmoid
from .position_encoding import build_position_encoding
from ..transformer import Transformer
from .ECA_attention import ECABottleneck
from .ECA_attention import ECABasicBlock
from .DA_attention import DABasicBlock
from typing import Optional, List
import cv2

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

BN_MOMENTUM = 0.1
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class BaseModel(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModel, self).__init__()
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        self.num_stacks = num_stacks

        if opt.training_attention:
            if opt.attention_swith == 'self':
                # 使用自注意力的dropout
                self.dropout = 0.1
                # 输入特征图谱的维度
                self.d_model = 128
                # 自注意力的多头任务并行执行
                self.nhead = 1
                self.position_embedding = build_position_encoding(opt)

                self.transformer = Transformer(self.d_model,self.nhead,self.dropout)
                self.cov = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
                self.upsample = nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1,output_padding=0,bias=False)
            if opt.attention_swith == 'ECA':
                self.ECA_attention = ECABottleneck(64, 64)
                # if opt.head_switch == 'attention':
                #     self.ECA_attention_merge = ECABottleneck(256,256)
                if opt.head_switch == 'merge-attention':
                    self.conv1 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0, bias=False)
                    self.bn1 = self.bn = nn.BatchNorm2d(64)
                    self.relu1 = nn.ReLU(inplace=True)
            if opt.attention_swith == 'DA':
                self.DA_attention = DABasicBlock(64,64)

        if opt.trades:
            h = int(opt.output_h / 2)
            w = int(opt.output_w / 2)
            off_template_w = np.zeros((h, w, w), dtype=np.float32)
            off_template_h = np.zeros((h, w, h), dtype=np.float32)
            for ii in range(h):
                for jj in range(w):
                    for i in range(h):
                        off_template_h[ii, jj, i] = i - ii
                    for j in range(w):
                        off_template_w[ii, jj, j] = j - jj
            self.m = np.reshape(off_template_w, newshape=(h * w, w))[None, :, :] * 2
            self.v = np.reshape(off_template_h, newshape=(h * w, h))[None, :, :] * 2
            self.embed_dim = 128
            self.maxpool_stride2 = nn.MaxPool2d(2, stride=2)
            self.avgpool_stride4 = nn.AvgPool2d(4, stride=4)
            self.tempature = 5

            self.embedconv = nn.Sequential(
                nn.Conv2d(64, self.embed_dim, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(self.embed_dim, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
                nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(self.embed_dim, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
                nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1, stride=1, padding=0, bias=True))
            self._compute_chain_of_basic_blocks()
            self.attention_cur = nn.Conv2d(64, 1, kernel_size=(opt.deform_kernel_size, opt.deform_kernel_size), stride=1, dilation=(1, 1), padding=(1, 1), bias=True)
            self.attention_prev = nn.Conv2d(64, 1, kernel_size=(opt.deform_kernel_size, opt.deform_kernel_size), stride=1, dilation=(1, 1), padding=(1, 1), bias=True)
            self.conv_offset_w = nn.Conv2d(129, opt.deform_kernel_size * opt.deform_kernel_size, kernel_size=(opt.deform_kernel_size, opt.deform_kernel_size),
                                      stride=1, dilation=(1, 1), padding=(1, 1), bias=True)
            self.conv_offset_h = nn.Conv2d(129, opt.deform_kernel_size * opt.deform_kernel_size, kernel_size=(opt.deform_kernel_size, opt.deform_kernel_size),
                                      stride=1, dilation=(1, 1), padding=(1, 1), bias=True)
            self.dcn1_1 = DCN_TraDeS(64, 64, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=1)

        self.heads = heads

        if opt.head_switch == '+' or opt.head_switch == 'cat':

            self.conv_hm = nn.Conv2d(last_channel, 256, kernel_size=head_kernel, padding=head_kernel // 2,
                                        bias=True)
            self.conv_reg = nn.Conv2d(last_channel, 256, kernel_size=head_kernel, padding=head_kernel // 2,
                                     bias=True)
            self.conv_wh = nn.Conv2d(last_channel, 256, kernel_size=head_kernel, padding=head_kernel // 2,
                                     bias=True)
            self.conv_conv_weight = nn.Conv2d(last_channel, 256, kernel_size=head_kernel, padding=head_kernel // 2,
                                     bias=True)
            self.conv_seg_feat = nn.Conv2d(last_channel, 256, kernel_size=3, padding=1, bias=False)
            self.conv_seg_feat_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(256)
            self.bn2 = nn.BatchNorm2d(8)

            if opt.head_switch == '+':
                self.conv_ltrb_amodal = nn.Conv2d(last_channel, 256, kernel_size=head_kernel, padding=head_kernel // 2,
                                         bias=True)
                self.out_hm = nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0, bias=True)
                self.out_reg = nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True)
                self.out_wh = nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True)
                self.out_conv_weight = nn.Conv2d(256, 169, kernel_size=1, stride=1, padding=0, bias=True)
                self.out_seg_feat = nn.Conv2d(256, 8, kernel_size=1, stride=1, padding=0, bias=False)
                self.out_ltrb_amodal = nn.Conv2d(256, 4, kernel_size=1, stride=1, padding=0, bias=True)
                self.relu = nn.ReLU(inplace=True)
            elif opt.head_switch == 'cat':
                self.conv_ltrb_amodal = nn.Conv2d(last_channel, 256, kernel_size=head_kernel, padding=head_kernel // 2,
                                                  bias=True)
                self.out_hm = nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0, bias=True)
                self.out_reg = nn.Conv2d(512, 2, kernel_size=1, stride=1, padding=0, bias=True)
                self.out_wh = nn.Conv2d(512, 2, kernel_size=1, stride=1, padding=0, bias=True)
                self.out_conv_weight = nn.Conv2d(256, 169, kernel_size=1, stride=1, padding=0, bias=True)
                self.out_seg_feat = nn.Conv2d(512, 8, kernel_size=1, stride=1, padding=0, bias=False)
                self.out_ltrb_amodal = nn.Conv2d(512, 4, kernel_size=1, stride=1, padding=0, bias=True)
                self.relu = nn.ReLU(inplace=True)
            self.parameter_initialization(opt)

        else:
            for head in self.heads:
                classes = self.heads[head]
                head_conv = head_convs[head]
                if len(head_conv) > 0:
                    out = nn.Conv2d(head_conv[-1], classes,
                          kernel_size=1, stride=1, padding=0, bias=True)
                    conv = nn.Conv2d(last_channel, head_conv[0],
                                     kernel_size=head_kernel,
                                     padding=head_kernel // 2, bias=True)
                    convs = [conv]
                    for k in range(1, len(head_conv)):
                        convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k],
                                     kernel_size=1, bias=True))
                    if len(convs) == 1:
                      fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
                    elif len(convs) == 2:
                      fc = nn.Sequential(
                        convs[0], nn.ReLU(inplace=True),
                        convs[1], nn.ReLU(inplace=True), out)
                    elif len(convs) == 3:
                      fc = nn.Sequential(
                          convs[0], nn.ReLU(inplace=True),
                          convs[1], nn.ReLU(inplace=True),
                          convs[2], nn.ReLU(inplace=True), out)
                    elif len(convs) == 4:
                      fc = nn.Sequential(
                          convs[0], nn.ReLU(inplace=True),
                          convs[1], nn.ReLU(inplace=True),
                          convs[2], nn.ReLU(inplace=True),
                          convs[3], nn.ReLU(inplace=True), out)
                    if head == "seg_feat":
                        fc = nn.Sequential(
                        nn.Conv2d(last_channel, head_conv[0],
                                  kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(head_conv[0]),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv[0], head_conv[0],
                                  kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(head_conv[0]),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv[0], classes,
                                  kernel_size=1, padding=0, bias=False),
                        nn.BatchNorm2d(classes),
                        nn.ReLU(inplace=True))
                    if 'hm' in head:
                        fc[-1].bias.data.fill_(opt.prior_bias)
                    else:
                        fill_fc_weights(fc)
                else:
                    fc = nn.Conv2d(last_channel, classes,
                        kernel_size=1, stride=1, padding=0, bias=True)
                    if 'hm' in head:
                      fc.bias.data.fill_(opt.prior_bias)
                    else:
                      fill_fc_weights(fc)
                self.__setattr__(head, fc)

    def img2feats(self, x):
      raise NotImplementedError
    
    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
      raise NotImplementedError

    def forward(self, x, pre_img=None, pre_hm=None, addtional_pre_imgs=None, addtional_pre_hms=None, inference_feats=None):
      cur_feat = self.img2feats(x)

      assert self.num_stacks == 1
      if self.opt.trades:
          feats, embedding, tracking_offset, dis_volume, h_volume_aux, w_volume_aux \
              = self.TraDeS(cur_feat, pre_img, pre_hm, addtional_pre_imgs, addtional_pre_hms, inference_feats)
      else:
          feats = [cur_feat[0]]

      feat1 = feats[0]
      # feat_all = torch.sum(feat1,dim=1)
      #
      # max1 = torch.max(feat_all[0, :, :])
      # min1 = torch.min(feat_all[0, :, :])
      # feat_all[0, :, :] = (feat_all[0, :, :] - min1) / (max1 - min1)
      # cv2.imwrite('/home/actl/data/liaocunyi/TraDeS-master/src/lib/model/' + 'feat.jpg',
      #             feat_all[0, :, :].cpu().numpy() * 255)

      if self.opt.training_attention:
          if self.opt.attention_swith == 'self':
            feat_attention_2d, feat_attention_seg = self.attention_layer(feats[0])
          if self.opt.attention_swith == 'ECA':
            feat_attention_2d = self.ECA_attention(feats[0])
            feat_attention_seg = self.ECA_attention(feats[0])
          if self.opt.attention_swith == 'DA':
            feat_attention_2d = self.DA_attention(feats[0])
            feat_attention_seg = self.DA_attention(feats[0])


      # feat_all = torch.sum(feat_attention_2d, dim=1)
      # max1 = torch.max(feat_all[0, :, :])
      # min1 = torch.min(feat_all[0, :, :])
      # feat_all[0, :, :] = (feat_all[0, :, :] - min1) / (max1 - min1)
      # cv2.imwrite('/home/actl/data/liaocunyi/TraDeS-master/src/lib/model/' + 'feat_2d.jpg',
      #             feat_all[0, :, :].cpu().numpy() * 255)
      #
      # feat_all = torch.sum(feat_attention_seg, dim=1)
      # max1 = torch.max(feat_all[0, :, :])
      # min1 = torch.min(feat_all[0, :, :])
      # feat_all[0, :, :] = (feat_all[0, :, :] - min1) / (max1 - min1)
      # cv2.imwrite('/home/actl/data/liaocunyi/TraDeS-master/src/lib/model/' + 'feat_seg.jpg',
      #             feat_all[0, :, :].cpu().numpy() * 255)

      if self.opt.head_switch == 'attention':
          dd = feat_attention_2d
          seg = feat_attention_seg
          feat_attention_2d = self.ECA_attention(feat_attention_2d)
          feat_attention_seg = feat_attention_2d + feat_attention_seg
          seg = self.ECA_attention(seg)
          feat_attention_2d = seg + dd


      if self.opt.head_switch == 'merge-attention':
          feat = torch.cat([feats[0],feat_attention_2d,feat_attention_seg],1)
          feat = self.conv1(feat)
          feat = self.bn1(feat)
          feat = self.relu1(feat)
          feat_2d = self.ECA_attention(feat)
          feat_seg = self.ECA_attention(feat)
          feat_attention_2d = feat_attention_2d + feat_2d
          feat_attention_seg = feat_attention_seg + feat_seg

          feat_all = torch.sum(feat_attention_2d, dim=1)
          max1 = torch.max(feat_all[0, :, :])
          min1 = torch.min(feat_all[0, :, :])
          feat_all[0, :, :] = (feat_all[0, :, :] - min1) / (max1 - min1)
          cv2.imwrite('/home/actl/data/liaocunyi/TraDeS-master/src/lib/model/' + 'feat_2d_merge.jpg',
                      feat_all[0, :, :].cpu().numpy() * 255)

          feat_all = torch.sum(feat_attention_seg, dim=1)
          max1 = torch.max(feat_all[0, :, :])
          min1 = torch.min(feat_all[0, :, :])
          feat_all[0, :, :] = (feat_all[0, :, :] - min1) / (max1 - min1)
          cv2.imwrite('/home/actl/data/liaocunyi/TraDeS-master/src/lib/model/' + 'feat_seg_merge.jpg',
                      feat_all[0, :, :].cpu().numpy() * 255)

      out = []
      if self.opt.model_output_list:
        for s in range(self.num_stacks):
          z = []

          if self.opt.head_switch == '+' or self.opt.head_switch == 'cat':
              if self.opt.training_attention:
                  z,_ = self.head_layer(feat_attention_seg,feat_attention_2d,merge=self.opt.head_switch)
              else:
                  z,_ = self.head_layer(feats[s],merge=self.opt.head_switch)
          else:
              for head in sorted(self.heads):
                  if self.opt.training_attention:
                      if head == 'seg_feat' or head == 'conv_weight':
                        z.append(self.__getattr__(head)(feat_attention_seg))
                      else:
                        z.append(self.__getattr__(head)(feat_attention_2d))
                  else:
                      z.append(self.__getattr__(head)(feats[s]))
          out.append(z)

      else:
        for s in range(self.num_stacks):
          z = {}

          if self.opt.head_switch == '+' or self.opt.head_switch == 'cat':
              if self.opt.training_attention:
                  _,z = self.head_layer(feat_attention_seg,feat_attention_2d,merge=self.opt.head_switch)
              else:
                  _,z = self.head_layer(feats[s],merge=self.opt.head_switch)
          else:
              for head in self.heads:
                  if self.opt.training_attention:
                      if head == 'seg_feat' or head == 'conv_weight':
                          z[head] = self.__getattr__(head)(feat_attention_seg)
                      else:
                          z[head] = self.__getattr__(head)(feat_attention_2d)
                  else:
                      z[head] = self.__getattr__(head)(feats[s])

          if self.opt.trades:
              z['embedding'] = embedding
              z['tracking_offset'] = tracking_offset
              if not self.opt.inference:
                  z['h_volume'] = dis_volume[0]
                  z['w_volume'] = dis_volume[1]
                  assert len(h_volume_aux) == self.opt.clip_len - 2
                  for temporal_id in range(2, self.opt.clip_len):
                      z['h_volume_prev{}'.format(temporal_id)] = h_volume_aux[temporal_id-2]
                      z['w_volume_prev{}'.format(temporal_id)] = w_volume_aux[temporal_id-2]

          out.append(z)
      if self.opt.inference:
          return out, cur_feat[0].detach().cpu().numpy()
      else:
          return out

    def TraDeS(self, cur_feat, pre_img, pre_hm, addtional_pre_imgs, addtional_pre_hms, inference_feats):
        feat_list = []
        feat_list.append(cur_feat[0])  # current feature
        support_feats = []
        if self.opt.inference:
            for prev_feat in inference_feats:
                feat_list.append(torch.from_numpy(prev_feat).to(self.opt.device)[:, :, :, :])
            while len(feat_list) < self.opt.clip_len:  # only operate in the initial frame
                feat_list.append(cur_feat[0])

            for idx, feat_prev in enumerate(feat_list[1:]):
                pre_hm_i = addtional_pre_hms[idx]
                pre_hm_i = self.avgpool_stride4(pre_hm_i)
                support_feats.append(pre_hm_i * feat_prev)
        else:
            feat2 = self.img2feats_prev(pre_img)
            pre_hm_1 = self.avgpool_stride4(pre_hm)
            feat_list.append(feat2[0])
            support_feats.append(feat2[0] * pre_hm_1)
            for ff in range(len(addtional_pre_imgs) - 1):
                feats_ff = self.img2feats_prev(addtional_pre_imgs[ff])
                pre_hm_i = self.avgpool_stride4(addtional_pre_hms[ff])
                feat_list.append(feats_ff[0][:, :, :, :])
                support_feats.append(feats_ff[0][:, :, :, :]*pre_hm_i)

        return self.CVA_MFW(feat_list, support_feats)

    def CVA_MFW(self, feat_list, support_feats):
        prop_feats = []
        attentions = []
        h_max_for_loss_aux = []
        w_max_for_loss_aux = []
        feat_cur = feat_list[0]
        batch_size = feat_cur.shape[0]
        h_f = feat_cur.shape[2]
        w_f = feat_cur.shape[3]
        h_c = int(h_f / 2)
        w_c = int(w_f / 2)

        prop_feats.append(feat_cur)
        embedding = self.embedconv(feat_cur)
        # embedding_plot = embedding
        # embedding_all = torch.sum(embedding,dim=1)
        # for i in range(embedding.shape[1]):
        #     max1 = torch.max(embedding_plot[0,i,:,:])
        #     min1 = torch.min(embedding_plot[0,i,:,:])
        #     embedding_plot[0, i, :, :] = (embedding_plot[0, i, :, :] - min1)/ (max1 - min1)
        #     cv2.imwrite('/home/actl/data/liaocunyi/TraDeS-master/src/lib/model/img/' + str(i)+'.jpg',embedding_plot[0,i,:,:].cpu().numpy() * 255)
        #
        # max1 = torch.max(embedding_all[0, :, :])
        # min1 = torch.min(embedding_all[0, :, :])
        # embedding_all[0, :, :] = (embedding_all[0, :, :] - min1) / (max1 - min1)
        # cv2.imwrite('/home/actl/data/liaocunyi/TraDeS-master/src/lib/model/img/' + 'all.jpg',
        #             embedding_all[0, :, :].cpu().numpy() * 255)
        embedding_prime = self.maxpool_stride2(embedding)
        # (B, 128, H, W) -> (B, H*W, 128):
        embedding_prime = embedding_prime.view(batch_size, self.embed_dim, -1).permute(0, 2, 1)
        attention_cur = self.attention_cur(feat_cur)
        attentions.append(attention_cur)
        for idx, feat_prev in enumerate(feat_list[1:]):
            # Sec. 4.1: Cost Volume based Association
            c_h, c_w, tracking_offset = self.CVA(embedding_prime, feat_prev, batch_size, h_c, w_c)

            # tracking offset output and CVA loss inputs
            if idx == 0:
                tracking_offset_output = tracking_offset
                h_max_for_loss = c_h
                w_max_for_loss = c_w
            else:
                h_max_for_loss_aux.append(c_h)
                w_max_for_loss_aux.append(c_w)


            # Sec. 4.2: Motion-guided Feature Warper
            prop_feat = self.MFW(support_feats[idx], tracking_offset, feat_cur, feat_prev, batch_size, h_f, w_f)
            prop_feats.append(prop_feat)
            attentions.append(self.attention_prev(prop_feat))

        attentions = torch.cat(attentions, dim=1)  # (B,T,H,W)
        adaptive_weights = F.softmax(attentions, dim=1)
        adaptive_weights = torch.split(adaptive_weights, 1, dim=1)  # 3*(B,1,H,W)
        # feature aggregation (MFW)
        enhanced_feat = 0
        for i in range(len(adaptive_weights)):
            enhanced_feat += adaptive_weights[i] * prop_feats[i]

        return [enhanced_feat], embedding, tracking_offset_output, [h_max_for_loss, w_max_for_loss], h_max_for_loss_aux, w_max_for_loss_aux

    def CVA(self, embedding_prime, feat_prev, batch_size, h_c, w_c):
        embedding_prev = self.embedconv(feat_prev)
        # embedding_plot = embedding_prev
        # embedding_all = torch.sum(embedding_prev, dim=1)
        # for i in range(embedding_prev.shape[1]):
        #     max1 = torch.max(embedding_plot[0, i, :, :])
        #     min1 = torch.min(embedding_plot[0, i, :, :])
        #     embedding_plot[0, i, :, :] = (embedding_plot[0, i, :, :] - min1) / (max1 - min1)
        #     cv2.imwrite('/home/actl/data/liaocunyi/TraDeS-master/src/lib/model/img1/' + str(i) + '.jpg',
        #                 embedding_plot[0, i, :, :].cpu().numpy() * 255)
        #
        # max1 = torch.max(embedding_all[0, :, :])
        # min1 = torch.min(embedding_all[0, :, :])
        # embedding_all[0, :, :] = (embedding_all[0, :, :] - min1) / (max1 - min1)
        # cv2.imwrite('/home/actl/data/liaocunyi/TraDeS-master/src/lib/model/img1/' + 'all.jpg',
        #             embedding_all[0, :, :].cpu().numpy() * 255)

        _embedding_prev = self.maxpool_stride2(embedding_prev)
        _embedding_prev = _embedding_prev.view(batch_size, self.embed_dim, -1)
        # Cost Volume Map
        # 代价关联矩阵
        c = torch.matmul(embedding_prime, _embedding_prev)  # (B, H*W/4, H*W/4)
        # c_test = c
        # max1 = torch.max(c_test[0, :, :])
        # min1 = torch.min(c_test[0, :, :])
        # c_test[0, :, :] = (c_test[0, :, :] - min1) / (max1 - min1)
        # cv2.imwrite('/home/actl/data/liaocunyi/TraDeS-master/src/lib/model/' + 'guanlian.jpg',
        #                         c_test[0, :, :].cpu().numpy() * 255)
        c = c.view(batch_size, h_c * w_c, h_c, w_c)  # (B, H*W, H, W)

        # 最大池化操作得到物体出现在指定宽高的最大概率
        c_h = c.max(dim=3)[0]  # (B, H*W, H)
        # c_test = c_h
        # max1 = torch.max(c_test[0, :, :])
        # min1 = torch.min(c_test[0, :, :])
        # c_test[0, :, :] = (c_test[0, :, :] - min1) / (max1 - min1)
        # cv2.imwrite('/home/actl/data/liaocunyi/TraDeS-master/src/lib/model/' + 'ch.jpg',
        #             c_test[0, :, :].cpu().numpy() * 255)
        c_w = c.max(dim=2)[0]  # (B, H*W, W)
        # c_test = c_w
        # max1 = torch.max(c_test[0, :, :])
        # min1 = torch.min(c_test[0, :, :])
        # c_test[0, :, :] = (c_test[0, :, :] - min1) / (max1 - min1)
        # cv2.imwrite('/home/actl/data/liaocunyi/TraDeS-master/src/lib/model/' + 'cw.jpg',
        #             c_test[0, :, :].cpu().numpy() * 255)
        c_h_softmax = F.softmax(c_h * self.tempature, dim=2)
        c_w_softmax = F.softmax(c_w * self.tempature, dim=2)
        v = torch.tensor(self.v, device=self.opt.device)  # (1, H*W, H)
        m = torch.tensor(self.m, device=self.opt.device)
        off_h = torch.sum(c_h_softmax * v, dim=2, keepdim=True).permute(0, 2, 1)
        off_w = torch.sum(c_w_softmax * m, dim=2, keepdim=True).permute(0, 2, 1)
        off_h = off_h.view(batch_size, 1, h_c, w_c)
        off_w = off_w.view(batch_size, 1, h_c, w_c)
        off_h = nn.functional.interpolate(off_h, scale_factor=2)
        off_w = nn.functional.interpolate(off_w, scale_factor=2)

        tracking_offset = torch.cat((off_w, off_h), dim=1)

        return c_h, c_w, tracking_offset

    def MFW(self, support_feat, tracking_offset, feat_cur, feat_prev, batch_size, h_f, w_f):
        # deformable conv offset input
        off_deform = self.gamma(tracking_offset, feat_cur, feat_prev, batch_size, h_f, w_f)
        mask_deform = torch.tensor(np.ones((batch_size, 9, off_deform.shape[2], off_deform.shape[3]),
                                           dtype=np.float32)).to(self.opt.device)
        # feature propagation
        prop_feat = self.dcn1_1(support_feat, off_deform, mask_deform)

        return prop_feat

    def gamma(self, tracking_offset, feat_cur, feat_prev, batch_size, h_f, w_f):
        feat_diff = feat_cur - feat_prev
        feat_offs = self.offset_feats(feat_diff)
        feat_offs_h = torch.cat((tracking_offset[:, 1:2, :, :], feat_offs), dim=1)
        feat_offs_w = torch.cat((tracking_offset[:, 0:1, :, :], feat_offs), dim=1)

        off_h_deform = self.conv_offset_h(feat_offs_h)[:, :, None, :, :]  # (B, 9, H, W)
        off_w_deform = self.conv_offset_w(feat_offs_w)[:, :, None, :, :]
        off_deform = torch.cat((off_h_deform, off_w_deform), dim=2)  # (B, 9, 2, H, W)
        off_deform = off_deform.view(batch_size, 9 * 2, h_f, w_f)

        return off_deform

    def _compute_chain_of_basic_blocks(self):
        """
        "Learning Temporal Pose Estimation from Sparsely-Labeled Videos" (NeurIPS 2019)
        """
        num_blocks = 4
        block = BasicBlock
        in_ch = 128
        out_ch = 128
        stride = 1
        nc = 64
        ######
        downsample = nn.Sequential(
            nn.Conv2d(
                nc,
                in_ch,
                kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm2d(
                in_ch,
                momentum=BN_MOMENTUM
            ),
        )
        ##########3
        layers = []
        layers.append(
            block(
                nc,
                out_ch,
                stride,
                downsample
            )
        )
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_ch,
                    out_ch
                )
            )
        self.offset_feats = nn.Sequential(*layers)
        return
    def attention_layer(self,feats):
        feats = self.cov(feats)
        # 取出特征图谱
        feat_attention_in = feats

        # 获得特征图谱的维度信息
        bs, c, h, w = feat_attention_in.shape

        # 建立mask矩阵（遮盖补0的位置矩阵）
        device = feat_attention_in.device
        mask = torch.zeros((bs, h, w), dtype=torch.bool, device=device)

        # 获得位置编码
        x = NestedTensor(feat_attention_in, mask)
        pos_embed = self.position_embedding(x).to(x.tensors.dtype)

        feat_attention_in = feat_attention_in.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        feat_attention_2d = self.transformer(feat_attention_in, src_key_padding_mask=mask, pos=pos_embed)
        feat_attention_2d = feat_attention_2d.permute(1, 2, 0).view(bs, c, h, w)

        feat_attention_seg = self.transformer(feat_attention_in, src_key_padding_mask=mask, pos=pos_embed)
        feat_attention_seg = feat_attention_seg.permute(1, 2, 0).view(bs, c, h, w)

        feat_attention_2d = self.upsample(feat_attention_2d)
        feat_attention_seg = self.upsample(feat_attention_seg)

        return feat_attention_2d,feat_attention_seg

    def parameter_initialization(self,opt):
        fill_fc_weights(self.conv_hm)
        fill_fc_weights(self.conv_reg)
        fill_fc_weights(self.conv_wh)
        fill_fc_weights(self.conv_conv_weight)
        fill_fc_weights(self.conv_seg_feat)
        fill_fc_weights(self.conv_seg_feat_2)
        fill_fc_weights(self.conv_ltrb_amodal)
        fill_fc_weights(self.out_reg)
        fill_fc_weights(self.out_wh)
        fill_fc_weights(self.out_conv_weight)
        fill_fc_weights(self.out_seg_feat)
        fill_fc_weights(self.out_ltrb_amodal)

        self.out_hm.bias.data.fill_(opt.prior_bias)

    def head_layer(self,feat_attention_2d,feat_attention_seg=None,merge=None):

        z = []
        z_d = {}
        if feat_attention_seg == None:
            feat_attention_seg = feat_attention_2d

        # 计算热力图头的结果
        hm_conv = self.conv_hm(feat_attention_2d)
        hm_con = hm_conv
        hm_conv = self.relu(hm_conv)
        if merge == 'attention':
            hm_conv_attention = self.ECA_attention_merge(hm_conv)
        hm_out = self.out_hm(hm_conv)
        z.append(hm_out)
        z_d['hm'] = hm_out

        # 计算背景回归头的结果
        reg_conv = self.conv_reg(feat_attention_2d)
        reg_conv = self.relu(reg_conv)
        if merge == 'cat':
            reg_conv = torch.cat([hm_conv, reg_conv], 1)
        elif merge == '+':
            reg_conv = hm_conv + reg_conv
        elif merge == 'attention':
            reg_conv = hm_conv_attention + reg_conv
        reg_out = self.out_reg(reg_conv)
        z.append(reg_out)
        z_d['reg'] = reg_out

        # 计算宽高回归头的结果
        wh_conv = self.conv_wh(feat_attention_2d)
        wh_conv = self.relu(wh_conv)
        if merge == 'cat':
            wh_conv = torch.cat([hm_conv, wh_conv], 1)
        elif merge == '+':
            wh_conv = hm_conv + wh_conv
        elif merge == 'attention':
            wh_conv = hm_conv_attention + wh_conv
        wh_out = self.out_wh(wh_conv)
        z.append(wh_out)
        z_d['wh'] = wh_out

        # 计算conv_weight头的结果
        conv_weight_conv = self.conv_conv_weight(feat_attention_seg)
        conv_weight_conv = self.relu(conv_weight_conv)
        if merge == 'cat':
            conv_weight_conv = torch.cat([hm_conv, conv_weight_conv], 1)
        elif merge == '+':
            conv_weight_conv = hm_conv + conv_weight_conv
        elif merge == 'attention':
            conv_weight_conv = hm_conv_attention + conv_weight_conv
        conv_weight_out = self.out_conv_weight(conv_weight_conv)
        z.append(conv_weight_out)
        z_d['conv_weight'] = conv_weight_out

        # 计算实例分割头的结果
        seg_feat = self.conv_seg_feat(feat_attention_seg)
        seg_feat = self.bn(seg_feat)
        seg_feat = self.relu(seg_feat)
        seg_feat = self.conv_seg_feat_2(seg_feat)
        seg_feat = self.bn(seg_feat)
        seg_feat = self.relu(seg_feat)
        hm_con = self.bn(hm_con)
        hm_con = self.relu(hm_con)
        if merge == 'cat':
            seg_feat = torch.cat([hm_con,seg_feat],1)
        elif merge == '+':
            seg_feat = hm_con + seg_feat
        elif merge == 'attention':
            seg_feat = hm_conv_attention + seg_feat
        seg_out = self.out_seg_feat(seg_feat)
        seg_out = self.bn2(seg_out)
        seg_out = self.relu(seg_out)
        z.append(seg_out)
        z_d['seg_feat'] = seg_out

        # 计算中心点回归头的结果
        ltrb_amodal_conv = self.conv_reg(feat_attention_2d)
        ltrb_amodal_conv = self.relu(ltrb_amodal_conv)
        if merge == 'cat':
            ltrb_amodal_conv = torch.cat([hm_conv, ltrb_amodal_conv], 1)
        elif merge == '+':
            ltrb_amodal_conv = hm_conv + ltrb_amodal_conv
        elif merge == 'attention':
            ltrb_amodal_conv = hm_conv_attention + ltrb_amodal_conv
        ltrb_amodal_out = self.out_ltrb_amodal(ltrb_amodal_conv)
        z.append(ltrb_amodal_out)
        z_d['ltrb_amodal'] = ltrb_amodal_out

        return z,z_d
