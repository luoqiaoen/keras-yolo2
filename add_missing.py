from preprocessing import parse_annotation
import os
import json
import pandas as pd

manual_check = pd.read_csv("../large_dataset/whale_box/yolo_box/manual_check_train.csv")
df = pd.read_csv("../large_dataset/whale_box/yolo_box/train_selected.csv")
with open('config.json') as config_buffer:
    config = json.loads(config_buffer.read())

###############################
#   Parse the annotations
###############################

# parse annotations of the training set
train_imgs, train_labels = parse_annotation(config['train']['missed_annot_folder'],
                                            config['train']['missed_image_folder'],
                                            config['model']['labels'])

for i in range(len(train_imgs)):
    head, tail = os.path.split(train_imgs[i]['filename'])
    manual_check.at[manual_check.Image == tail, 'Xmin'] = train_imgs[i]['object'][0]['xmin']
    manual_check.at[manual_check.Image == tail, 'Ymin'] = train_imgs[i]['object'][0]['ymin']
    manual_check.at[manual_check.Image == tail, 'Xmax'] = train_imgs[i]['object'][0]['xmax']
    manual_check.at[manual_check.Image == tail, 'Ymax'] = train_imgs[i]['object'][0]['ymax']

manual_check_final = manual_check.dropna()
#manual_check_final.to_csv('manual_check_final.csv', encoding='utf-8', index=False)


merged = pd.concat([df,manual_check_final])
print(merged)
merged.to_csv('train_final.csv', encoding='utf-8', index=False)
