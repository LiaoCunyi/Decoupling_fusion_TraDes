from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .datasets.coco import COCO
from .datasets.kitti import KITTI
from .datasets.coco_hp import COCOHP
# from src.tools.eval_kitti_track.mot import MOT
from .datasets.nuscenes import nuScenes
from .datasets.crowdhuman import CrowdHuman
from .datasets.kitti_tracking import KITTITracking
from .datasets.youtube_vis import youtube_vis
from .datasets.custom_dataset import CustomDataset
from .datasets.kitti_seg import KITTISeg
from .datasets.kitti_tracking_seg import KITTITrackingSeg

dataset_factory = {
  'custom': CustomDataset,
  'coco': COCO,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  # 'mot': MOT,
  'nuscenes': nuScenes,
  'crowdhuman': CrowdHuman,
  'kitti_tracking': KITTITracking,
  'kitti_seg': KITTISeg,
  'youtube_vis': youtube_vis,
  'kitti_tracking_seg':KITTITrackingSeg
}


def get_dataset(dataset):
  return dataset_factory[dataset]
