import json

cats = ['Pedestrian', 'Car','DontCare']
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}

cat_info = []
for i, cat in enumerate(cats):
    if cat != 'DontCare':
        cat_info.append({'name': cat, 'id': i + 1})
    else:
        cat_info.append({'name': cat, 'id': 10})

if __name__ == '__main__':
    ret = {'images': [], 'annotations': [], "categories": cat_info,
               'videos': []}
    resFile = "/home/actl/data/liaocunyi/TraDeS-master/exp/tracking/kitti_merge_dla34_ECA_attentionmerge/results_kitti_trackingval_half.json"
    # resFile = "/home/actl/data/liaocunyi/SearchTrack-master/exp/tracking,seg/mots_sch/save_results_kitti_motsval_new.json"
    test = "/home/actl/data/liaocunyi/TraDeS-master/data/kitti_tracking/annotations/tracking_val_half.json"
    ann_test = json.load(open(test))
    anns = json.load(open(resFile))

    ret['videos'] = ann_test['videos']
    ret['image'] = ann_test['images']

    for image_id in range(1,(len(anns) + 1)):
        for i in range(len(anns[str(image_id)])):
            ann = {'image_id': image_id,
                   'score':anns[str(image_id)][i]['score'],
                   'id': int(len(ret['annotations']) + 1),
                   'category_id': anns[str(image_id)][i]['class'],
                   'bbox': anns[str(image_id)][i]['bbox'],  # x1, y1, w, h
                   'segmentation': anns[str(image_id)][i]['pred_mask'],
                   # 'segmentation': anns[str(image_id)][i]['seg'],
                   'track_id': anns[str(image_id)][i]['tracking_id']}


            ret['annotations'].append(ann)
        # if anns[str(i)] != []:
        #     message = anns[str(i)]
        #     print('OK')

    # out_path = '/home/actl/data/liaocunyi/TraDeS-master/exp/tracking/kitti_merge_dla34_ECA_attentionmerge/kitti_result_val_new.json'
    # json.dump(ret, open(out_path, 'w'))





    out_path = '/home/actl/data/liaocunyi/TraDeS-master/exp/tracking/kitti_seg/kitti_result_val_half.json'
    json.dump(ret, open(out_path, 'w'))