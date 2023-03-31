from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np
from progress.bar import Bar
import torch.nn as nn
from model.data_parallel import DataParallel
from utils.utils import AverageMeter
from model.utils import _tranpose_and_gather_feat

from model.losses import FastFocalLoss, RegWeightedL1Loss, CostVolumeLoss1D, DiceLoss
from model.losses import BinRotLoss, WeightedBCELoss
from model.decode import generic_decode
from model.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr
from utils.debugger import Debugger
from utils.post_process import generic_post_process


class GenericLoss(torch.nn.Module):
  def __init__(self, opt,optimizer):
    super(GenericLoss, self).__init__()
    self.crit = FastFocalLoss(opt=opt)
    self.crit_reg = RegWeightedL1Loss()
    self.optimizer = optimizer

    if opt.loss_switch == 'DWA':
      self.avg_cost = np.zeros([opt.num_epochs + 1, 6], dtype=np.float32)
      self.lambda_weight = np.ones([6, opt.num_epochs + 1])
    if opt.loss_switch == 'Grad':
      self.n = 4
      self.number = 0
      # 关注前每个任务的前5个loss历史
      self.history_loss = np.ones([6, self.n], dtype=np.float32)
      # 对应6个任务的loss
      self.weight = np.ones([1, 6])

    if opt.loss_switch == 'DWA_change':
      self.number = 0
      self.avg_cost = np.ones([2, 6], dtype=np.float32)
      self.lambda_weight = np.ones([1, 6], dtype=np.float32)

    if 'rot' in opt.heads:
      self.crit_rot = BinRotLoss()
    if 'nuscenes_att' in opt.heads:
      self.crit_nuscenes_att = WeightedBCELoss()
    self.opt = opt
    if opt.trades:
      self.crit_cost_volume_1d = CostVolumeLoss1D()
    # if opt.seg:
    #   self.crit_mask = DiceLoss(opt.seg_feat_channel)

    # if 'seg_feat' in opt.heads:
    #   self.crit_mask = DiceLoss(opt.seg_feat_channel)
    if opt.training_seg:
      self.crit_mask = DiceLoss(opt.seg_feat_channel)

  def _sigmoid_output(self, output):
    if 'hm' in output:
      output['hm'] = _sigmoid(output['hm'])
    if 'hm_hp' in output:
      output['hm_hp'] = _sigmoid(output['hm_hp'])
    if 'dep' in output:
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
    return output

  def forward(self, outputs, batch,epoch):
    opt = self.opt
    losses = {}

    # # if opt.backbone_2Dhead_or_seg == 'False': # Only train seg
    #   losses['mask_loss'] = 0

    for head in opt.heads:
      if 'conv_weight' in head or 'seg_feat' in head:
        continue
      losses[head] = 0
    if opt.trades:
      losses['cost_volume'] = 0
    if opt.training_seg:
      losses['mask_loss'] = 0

    for s in range(opt.num_stacks):
      output = outputs[s]
      output = self._sigmoid_output(output)

      # 如果需要训练主干网络
      # if opt.backbone_2Dhead_or_seg == 'True':
      if opt.trades:
        losses['cost_volume'] += self.crit_cost_volume_1d(output['h_volume'], batch['h_hm_down'], batch['h_ind_down'], batch['mask_down'], batch['cat_down'])
        losses['cost_volume'] += self.crit_cost_volume_1d(output['w_volume'], batch['w_hm_down'], batch['w_ind_down'], batch['mask_down'], batch['cat_down'])
        for temporal_id in range(2, opt.clip_len):
          losses['cost_volume'] += self.crit_cost_volume_1d(output['h_volume_prev{}'.format(temporal_id)],
                                                            batch['h_hm_down_prev{}'.format(temporal_id)], batch['h_ind_down_prev{}'.format(temporal_id)],
                                                            batch['mask_down_prev{}'.format(temporal_id)], batch['cat_down_prev{}'.format(temporal_id)])
          losses['cost_volume'] += self.crit_cost_volume_1d(output['w_volume_prev{}'.format(temporal_id)],
                                                            batch['w_hm_down_prev{}'.format(temporal_id)], batch['w_ind_down_prev{}'.format(temporal_id)],
                                                            batch['mask_down_prev{}'.format(temporal_id)], batch['cat_down_prev{}'.format(temporal_id)])
        losses['cost_volume'] = losses['cost_volume']*1.0 / (opt.clip_len - 1)

      # if opt.seg:
      # 如果需要训练实例分割功能
      # if opt.backbone_2Dhead_or_seg == 'False':  # Only train seg
      if opt.training_seg:
        losses['mask_loss'] += self.crit_mask(output['seg_feat'], output['conv_weight'],
                                            batch['mask'], batch['ind'], batch['instance_mask'], batch['num_obj'])

      # else:  # Only train backbone and 2Dhead
      if 'hm' in output:
        losses['hm'] += self.crit(
          output['hm'], batch['hm'], batch['ind'],
          batch['mask'], batch['cat']) / opt.num_stacks

      regression_heads = [
        'reg', 'wh', 'tracking', 'ltrb', 'ltrb_amodal', 'hps',
        'dep', 'dim', 'amodel_offset', 'velocity']

      for head in regression_heads:
        if head in output:
          losses[head] += self.crit_reg(
            output[head], batch[head + '_mask'],
            batch['ind'], batch[head]) / opt.num_stacks

      if 'hm_hp' in output:
        losses['hm_hp'] += self.crit(
          output['hm_hp'], batch['hm_hp'], batch['hp_ind'],
          batch['hm_hp_mask'], batch['joint']) / opt.num_stacks
        if 'hp_offset' in output:
          losses['hp_offset'] += self.crit_reg(
            output['hp_offset'], batch['hp_offset_mask'],
            batch['hp_ind'], batch['hp_offset']) / opt.num_stacks

      if 'rot' in output:
        losses['rot'] += self.crit_rot(
          output['rot'], batch['rot_mask'], batch['ind'], batch['rotbin'],
          batch['rotres']) / opt.num_stacks

      if 'nuscenes_att' in output:
        losses['nuscenes_att'] += self.crit_nuscenes_att(
          output['nuscenes_att'], batch['nuscenes_att_mask'],
          batch['ind'], batch['nuscenes_att']) / opt.num_stacks

    losses['tot'] = 0

    if opt.loss_switch == 'None':
      # if opt.seg:
      # if opt.backbone_2Dhead_or_seg == 'False':  # Only train seg
      if opt.training_seg:
        losses['tot'] += 1.0 * losses['mask_loss']
      # else:
      for head in opt.heads:
        if 'conv_weight' in head or 'seg_feat' in head:
          continue
        losses['tot'] += opt.weights[head] * losses[head]
      if opt.trades:
        losses['tot'] += 0.5 * losses['cost_volume']

    elif opt.loss_switch == 'Uncertainty_Weighting':
      losses['tot'] = opt.awl(losses['hm'],losses['reg'],losses['wh'],losses['ltrb_amodal'],losses['cost_volume'],losses['mask_loss'])

    elif opt.loss_switch == 'DWA':
      if epoch == 1 or epoch == 2:
        self.lambda_weight[:, epoch] = 1.0
      else:
        w_1 = self.avg_cost[epoch - 1, 0] / self.avg_cost[epoch - 2, 0]
        w_2 = self.avg_cost[epoch - 1, 1] / self.avg_cost[epoch - 2, 1]
        w_3 = self.avg_cost[epoch - 1, 2] / self.avg_cost[epoch - 2, 2]
        w_4 = self.avg_cost[epoch - 1, 3] / self.avg_cost[epoch - 2, 3]
        w_5 = self.avg_cost[epoch - 1, 4] / self.avg_cost[epoch - 2, 4]
        w_6 = self.avg_cost[epoch - 1, 5] / self.avg_cost[epoch - 2, 5]


        self.lambda_weight[0, epoch] = 6 * np.exp(w_1 / 2) / (np.exp(w_1 / 2) + np.exp(w_2 / 2) + np.exp(w_3 / 2) + np.exp(w_4 / 2) + np.exp(w_5 / 2) + np.exp(w_6 / 2))
        self.lambda_weight[1, epoch] = 6 * np.exp(w_2 / 2) / (np.exp(w_1 / 2) + np.exp(w_2 / 2) + np.exp(w_3 / 2) + np.exp(w_4 / 2) + np.exp(w_5 / 2) + np.exp(w_6 / 2))
        self.lambda_weight[2, epoch] = 6 * np.exp(w_3 / 2) / (np.exp(w_1 / 2) + np.exp(w_2 / 2) + np.exp(w_3 / 2) + np.exp(w_4 / 2) + np.exp(w_5 / 2) + np.exp(w_6 / 2))
        self.lambda_weight[3, epoch] = 6 * np.exp(w_4 / 2) / (np.exp(w_1 / 2) + np.exp(w_2 / 2) + np.exp(w_3 / 2) + np.exp(w_4 / 2) + np.exp(w_5 / 2) + np.exp(w_6 / 2))
        self.lambda_weight[4, epoch] = 6 * np.exp(w_5 / 2) / (np.exp(w_1 / 2) + np.exp(w_2 / 2) + np.exp(w_3 / 2) + np.exp(w_4 / 2) + np.exp(w_5 / 2) + np.exp(w_6 / 2))
        self.lambda_weight[5, epoch] = 6 * np.exp(w_6 / 2) / (np.exp(w_1 / 2) + np.exp(w_2 / 2) + np.exp(w_3 / 2) + np.exp(w_4 / 2) + np.exp(w_5 / 2) + np.exp(w_6 / 2))
      i = 0
      for loss in losses:
        if loss != 'tot':
          losses['tot'] += self.lambda_weight[i, epoch] * losses[loss]
          self.avg_cost[epoch, i] = losses[loss] / opt.num_epochs
          i = i + 1

    elif opt.loss_switch == 'Grad':
      if self.number < self.n:
        # 计算前5次的loss
        for head in opt.heads:
          if 'conv_weight' in head or 'seg_feat' in head:
            continue
          losses['tot'] += opt.weights[head] * losses[head]
        if opt.training_seg:
          losses['tot'] += 1.0 * losses['mask_loss']
        if opt.trades:
          losses['tot'] += 0.5 * losses['cost_volume']

        i = 0
        for loss in losses:
          if loss != 'tot':
            self.history_loss[i,self.number] = losses[loss]
          i = i + 1
        self.number = self.number + 1
      else:
        i = 0
        for loss in losses:
          if loss != 'tot':
            # 计算对应的权重
            for number in range(self.n):
              self.weight[0, i] += abs(losses[loss] - self.history_loss[i,number]) / (self.n - number)

            if loss == 'wh':
              self.weight[0, i] = 0.1 / (self.weight[0, i] / self.n)
            elif loss == 'ltrb_amodal':
              self.weight[0, i] = 0.1 / (self.weight[0, i] / self.n)
            elif loss == 'cost_volume':
              self.weight[0, i] = 0.5 / (self.weight[0, i] / self.n)
            else:
              self.weight[0, i] = 1 / (self.weight[0, i] / self.n)
            losses['tot'] += self.weight[0, i] * losses[loss]

            # 更新历史loss
            for k in range(self.n - 1):
              self.history_loss[i, k] = self.history_loss[i, k + 1]
            self.history_loss[i, self.n - 1] = losses[loss]
            # self.history_loss[i, 0] = self.history_loss[i, 1]
            # self.history_loss[i, 1] = self.history_loss[i, 2]
            # self.history_loss[i, 2] = self.history_loss[i, 3]
            # self.history_loss[i, 3] = self.history_loss[i, 4]
            # self.history_loss[i, 4] = losses[loss]
            i = i + 1

    elif opt.loss_switch == 'DWA_change':
      if self.number < 2:
        # 计算前5次的loss
        for head in opt.heads:
          if 'conv_weight' in head or 'seg_feat' in head:
            continue
          losses['tot'] += opt.weights[head] * losses[head]
        if opt.training_seg:
          losses['tot'] += 1.0 * losses['mask_loss']
        if opt.trades:
          losses['tot'] += 0.5 * losses['cost_volume']

        i = 0
        for loss in losses:
          if loss != 'tot':
            self.avg_cost[0, i] = self.avg_cost[1, i]
            self.avg_cost[1, i] = losses[loss] / opt.num_epochs
          i = i + 1
        self.number = self.number + 1

      else:
        w_1 = self.avg_cost[1, 0] / self.avg_cost[0, 0]
        w_2 = self.avg_cost[1, 1] / self.avg_cost[0, 1]
        w_3 = self.avg_cost[1, 2] / self.avg_cost[0, 2]
        w_4 = self.avg_cost[1, 3] / self.avg_cost[0, 3]
        w_5 = self.avg_cost[1, 4] / self.avg_cost[0, 4]
        w_6 = self.avg_cost[1, 5] / self.avg_cost[0, 5]
        self.lambda_weight[0, 0] = 6 * np.exp(w_1 / 2) / (np.exp(w_1 / 2) + np.exp(w_2 / 2) + np.exp(w_3 / 2) + np.exp(w_4 / 2) + np.exp(w_5 / 2) + np.exp(w_6 / 2))
        self.lambda_weight[0, 1] = 6 * np.exp(w_2 / 2) / (np.exp(w_1 / 2) + np.exp(w_2 / 2) + np.exp(w_3 / 2) + np.exp(w_4 / 2) + np.exp(w_5 / 2) + np.exp(w_6 / 2))
        self.lambda_weight[0, 2] = 6 * np.exp(w_3 / 2) / (np.exp(w_1 / 2) + np.exp(w_2 / 2) + np.exp(w_3 / 2) + np.exp(w_4 / 2) + np.exp(w_5 / 2) + np.exp(w_6 / 2))
        self.lambda_weight[0, 3] = 6 * np.exp(w_4 / 2) / (np.exp(w_1 / 2) + np.exp(w_2 / 2) + np.exp(w_3 / 2) + np.exp(w_4 / 2) + np.exp(w_5 / 2) + np.exp(w_6 / 2))
        self.lambda_weight[0, 4] = 6 * np.exp(w_5 / 2) / (np.exp(w_1 / 2) + np.exp(w_2 / 2) + np.exp(w_3 / 2) + np.exp(w_4 / 2) + np.exp(w_5 / 2) + np.exp(w_6 / 2))
        self.lambda_weight[0, 5] = 6 * np.exp(w_6 / 2) / (np.exp(w_1 / 2) + np.exp(w_2 / 2) + np.exp(w_3 / 2) + np.exp(w_4 / 2) + np.exp(w_5 / 2) + np.exp(w_6 / 2))
        i = 0
        for loss in losses:
          if loss != 'tot':
            losses['tot'] += self.lambda_weight[0, i] * losses[loss]
            self.avg_cost[0, i] = self.avg_cost[1, i]
            self.avg_cost[1, i] = losses[loss] / opt.num_epochs
            i = i + 1

    return losses['tot'], losses

class ModleWithLoss(torch.nn.Module):
  def __init__(self, model, loss, opt):
    super(ModleWithLoss, self).__init__()
    self.model = model
    self.loss = loss
    self.opt = opt
  
  def forward(self,batch,epoch):
    pre_img = batch['pre_img_1'] if 'pre_img_1' in batch else None
    pre_hm = batch['pre_hm_1'] if 'pre_hm_1' in batch else None

    addtional_pre_imgs = []
    addtional_pre_hms = []
    for i in range(1, self.opt.clip_len):
      addtional_pre_imgs.append(batch['pre_img_{}'.format(i+1)])
      addtional_pre_hms.append(batch['pre_hm_{}'.format(i + 1)])

    outputs = self.model(batch['image'], pre_img, pre_hm, addtional_pre_imgs, addtional_pre_hms)
    loss, loss_stats = self.loss(outputs, batch,epoch)
    return outputs[-1], loss, loss_stats

class Trainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModleWithLoss(model, self.loss, opt)

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model_with_loss = DataParallel(
        self.model_with_loss, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.model_with_loss = self.model_with_loss.to(device)
    
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats \
                      if l == 'mask_loss' or l == 'tot' or opt.weights[l] > 0}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)

      for k in batch:
        if k != 'meta' and k != 'img_info':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)
      output, loss, loss_stats = model_with_loss(batch,epoch)
      loss = loss.mean()

      if phase == 'train':
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['image'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
        '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0: # If not using progress bar
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        bar.next()
      
      if opt.debug > 0:
        self.debug(batch, output, iter_id, dataset=data_loader.dataset)
      
      del output, loss, loss_stats
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results
  
  def _get_losses(self, opt):
    loss_order = ['hm', 'wh', 'reg', 'ltrb', 'hps', 'hm_hp', \
      'hp_offset', 'dep', 'dim', 'rot', 'amodel_offset', \
      'ltrb_amodal', 'tracking', 'nuscenes_att', 'velocity']
    loss_states = ['tot'] + [k for k in loss_order if k in opt.heads]
    loss_states = loss_states + ['cost_volume'] if opt.trades else loss_states
    # loss_states = loss_states + ['mask_loss'] if opt.seg else loss_states
    # if opt.backbone_2Dhead_or_seg == 'False':  # Only train seg
    if opt.training_seg:
      loss_states = loss_states + ['mask_loss']
    loss = GenericLoss(opt,self.optimizer)
    return loss_states, loss

  def debug(self, batch, output, iter_id, dataset):
    opt = self.opt
    if 'pre_hm' in batch:
      output.update({'pre_hm': batch['pre_hm']})
    dets = generic_decode(output, K=opt.K, opt=opt)
    for k in dets:
      dets[k] = dets[k].detach().cpu().numpy()
    dets_gt = batch['meta']['gt_det']
    for i in range(1):
      debugger = Debugger(opt=opt, dataset=dataset)
      img = batch['image'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * dataset.std + dataset.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')

      if 'pre_img' in batch:
        pre_img = batch['pre_img'][i].detach().cpu().numpy().transpose(1, 2, 0)
        pre_img = np.clip(((
          pre_img * dataset.std + dataset.mean) * 255), 0, 255).astype(np.uint8)
        debugger.add_img(pre_img, 'pre_img_pred')
        debugger.add_img(pre_img, 'pre_img_gt')
        if 'pre_hm' in batch:
          pre_hm = debugger.gen_colormap(
            batch['pre_hm'][i].detach().cpu().numpy())
          debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')

      debugger.add_img(img, img_id='out_pred')
      if 'ltrb_amodal' in opt.heads:
        debugger.add_img(img, img_id='out_pred_amodal')
        debugger.add_img(img, img_id='out_gt_amodal')

      # Predictions
      for k in range(len(dets['scores'][i])):
        if dets['scores'][i, k] > opt.vis_thresh:
          debugger.add_coco_bbox(
            dets['bboxes'][i, k] * opt.down_ratio, dets['clses'][i, k],
            dets['scores'][i, k], img_id='out_pred')

          if 'ltrb_amodal' in opt.heads:
            debugger.add_coco_bbox(
              dets['bboxes_amodal'][i, k] * opt.down_ratio, dets['clses'][i, k],
              dets['scores'][i, k], img_id='out_pred_amodal')

          if 'hps' in opt.heads and int(dets['clses'][i, k]) == 0:
            debugger.add_coco_hp(
              dets['hps'][i, k] * opt.down_ratio, img_id='out_pred')

          if 'tracking' in opt.heads:
            debugger.add_arrow(
              dets['cts'][i][k] * opt.down_ratio, 
              dets['tracking'][i][k] * opt.down_ratio, img_id='out_pred')
            debugger.add_arrow(
              dets['cts'][i][k] * opt.down_ratio, 
              dets['tracking'][i][k] * opt.down_ratio, img_id='pre_img_pred')

      # Ground truth
      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt['scores'][i])):
        if dets_gt['scores'][i][k] > opt.vis_thresh:
          debugger.add_coco_bbox(
            dets_gt['bboxes'][i][k] * opt.down_ratio, dets_gt['clses'][i][k],
            dets_gt['scores'][i][k], img_id='out_gt')

          if 'ltrb_amodal' in opt.heads:
            debugger.add_coco_bbox(
              dets_gt['bboxes_amodal'][i, k] * opt.down_ratio, 
              dets_gt['clses'][i, k],
              dets_gt['scores'][i, k], img_id='out_gt_amodal')

          if 'hps' in opt.heads and \
            (int(dets['clses'][i, k]) == 0):
            debugger.add_coco_hp(
              dets_gt['hps'][i][k] * opt.down_ratio, img_id='out_gt')

          if 'tracking' in opt.heads:
            debugger.add_arrow(
              dets_gt['cts'][i][k] * opt.down_ratio, 
              dets_gt['tracking'][i][k] * opt.down_ratio, img_id='out_gt')
            debugger.add_arrow(
              dets_gt['cts'][i][k] * opt.down_ratio, 
              dets_gt['tracking'][i][k] * opt.down_ratio, img_id='pre_img_gt')

      if 'hm_hp' in opt.heads:
        pred = debugger.gen_colormap_hp(
          output['hm_hp'][i].detach().cpu().numpy())
        gt = debugger.gen_colormap_hp(batch['hm_hp'][i].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hmhp')
        debugger.add_blend_img(img, gt, 'gt_hmhp')


      if 'rot' in opt.heads and 'dim' in opt.heads and 'dep' in opt.heads:
        dets_gt = {k: dets_gt[k].cpu().numpy() for k in dets_gt}
        calib = batch['meta']['calib'].detach().numpy() \
                if 'calib' in batch['meta'] else None
        det_pred = generic_post_process(opt, dets, 
          batch['meta']['c'].cpu().numpy(), batch['meta']['s'].cpu().numpy(),
          output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
          calib)
        det_gt = generic_post_process(opt, dets_gt, 
          batch['meta']['c'].cpu().numpy(), batch['meta']['s'].cpu().numpy(),
          output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
          calib)

        debugger.add_3d_detection(
          batch['meta']['img_path'][i], batch['meta']['flipped'][i],
          det_pred[i], calib[i],
          vis_thresh=opt.vis_thresh, img_id='add_pred')
        debugger.add_3d_detection(
          batch['meta']['img_path'][i], batch['meta']['flipped'][i], 
          det_gt[i], calib[i],
          vis_thresh=opt.vis_thresh, img_id='add_gt')
        debugger.add_bird_views(det_pred[i], det_gt[i], 
          vis_thresh=opt.vis_thresh, img_id='bird_pred_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)
