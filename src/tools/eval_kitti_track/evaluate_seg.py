"""
Evaluation for COCO val2017:
python ./tools/coco_instance_evaluation.py \
    --gt-json-file COCO_GT_JSON \
    --dt-json-file COCO_DT_JSON \
    --iou-type boundary
"""
import argparse
import json
from coco import COCO
from cocoeval import COCOeval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json-file", default="/home/actl/data/liaocunyi/TraDeS-master/data/kitti_seg/annotations/tracking_val_half.json")
    parser.add_argument("--dt-json-file", default="/home/actl/data/liaocunyi/TraDeS-master/src/lib/../../exp/tracking/kitti_seg/kitti_result_val_half.json")
    # parser.add_argument("--gt-json-file",
    #                     default="/home/actl/data/liaocunyi/TraDeS-master/data/kitti_tracking/annotations/tracking_val_new.json")
    # parser.add_argument("--dt-json-file",
    #                     default="/home/actl/data/liaocunyi/TraDeS-master/src/lib/../../exp/tracking/kitti_merge_dla34_ECA_attentionmerge/kitti_result_val_new.json")


    # test_file_gt = "/home/actl/data/liaocunyi/TraDeS-master/data/kitti_seg/annotations/tracking_val_half.json"
    # test_file_dt = "/home/actl/data/liaocunyi/TraDeS-master/src/lib/../../exp/tracking/kitti_seg/kitti_result_val_half.json"
    # use_file_gt = "/home/actl/data/liaocunyi/TraDeS-master/data/kitti_tracking/annotations/tracking_val_new.json"
    # use_file_dt = "/home/actl/data/liaocunyi/TraDeS-master/src/lib/../../exp/tracking/kitti_merge_dla34_ECA_attentionmerge/kitti_result_val_new.json"
    #
    # ann_test_file_gt = json.load(open(test_file_gt))
    # ann_test_file_dt = json.load(open(test_file_dt))
    # ann_use_file_gt = json.load(open(use_file_gt))
    # ann_use_file_dt = json.load(open(use_file_dt))

    # parser.add_argument("--gt-json-file", default="/home/actl/data/liaocunyi/TraDeS-master/data/kitti_seg/annotations/tracking_val_half.json")
    # parser.add_argument("--dt-json-file", default="/home/actl/data/liaocunyi/TraDeS-master/src/lib/../../exp/tracking/kitti_seg/kitti_result_val_half.json")
    parser.add_argument("--iou-type", default="segm")
    parser.add_argument("--dilation-ratio", default="0.020", type=float)
    parser.add_argument("--lvis", action='store_true')
    args = parser.parse_args()
    print(args)

    annFile = args.gt_json_file
    resFile = args.dt_json_file
    dilation_ratio = args.dilation_ratio
    if args.iou_type == "boundary":
        get_boundary = True
    else:
        get_boundary = False

    cocoGt = COCO(annFile, get_boundary=get_boundary, dilation_ratio=dilation_ratio)
    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType=args.iou_type, dilation_ratio=dilation_ratio)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == '__main__':
    main()
