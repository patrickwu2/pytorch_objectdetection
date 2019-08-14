# built-in packages
import pdb
import cv2
import numpy as np
import pandas as pd
from PIL import Image
# torch
import torch
from torch.utils.data import Dataset, DataLoader
# albumentations
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Resize,
    Rotate,
    ToGray,
    HueSaturationValue,
    CenterCrop,
    RandomCrop,
    Crop,
    Compose
)
class CSVDataset(Dataset):
    def __init__(self, anno_file, class_file, train=True):

        # parse provided class file
        self.classes = pd.read_csv(class_file).set_index('class_name').to_dict()['class_id']

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # parse annotation file
        self.read_annotation(anno_file)

        self.image_names = list(self.image_data.keys())

    def read_annotation(self, anno_file):
        df = pd.read_csv(anno_file)
        self.image_data = {}
        df.apply(self.parse_anno_row, axis=1)

    def parse_anno_row(self,row):
        img_file, x1, y1, x2, y2, class_name = row
        if img_file not in self.image_data:
            self.image_data[img_file] = []
        anno = {'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2, 'class_name':class_name}
        self.image_data[img_file].append(anno)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # image
        img_fname = self.image_names[idx]
        img = cv2.imread(img_fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # bbox and category
        bboxes, category_id = self.load_annotation(self.image_names[idx])
        batch = {'image':img, 'bboxes':bboxes, 'category_id':category_id}

        return batch
    
    def load_annotation(self, key):
        anno_list = self.image_data[key]
        bboxes = []
        category_id = []

        for idx, a in enumerate(anno_list):
            bbox = [a['x1'], a['y1'], a['x2'], a['y2']]
            cat_id = self.classes[a['class_name']]
            bboxes.append(bbox)
            category_id.append(cat_id)

        return bboxes, category_id

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):
    x_min, y_min, x_max, y_max = [int(x) for x in bbox]
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    return img


def visualize(annotations, category_id_to_name, fname):
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
    #plt.figure(figsize=(12, 12))
    cv2.imwrite(fname, img)

def customed_collate_fn(batch):

    def get_aug(aug, min_area=0., min_visibility=0.):
        return Compose(aug, bbox_params={'format': 'pascal_voc', 'min_area': min_area, 'min_visibility': min_visibility, 'label_fields': ['category_id']})       

    aug = get_aug([Rotate(limit=20, p=0.3), ToGray(p=0.3), HueSaturationValue(20, 30, 20, p=0.2)])

    def _transform_fn_(one_batch):
        augmented = aug(**one_batch)
        augmented['image'] = augmented['image'].astype(np.float32) / 255.0
        augmented['image'] = torch.tensor(augmented['image']).type(torch.float32)
        return augmented

    category_id_to_name = {0:'signature', 1:'stamp', 2:'date'}
    batch = [_transform_fn_(one_batch) for one_batch in batch]
    return batch
    #visualize(one_batch, category_id_to_name, 'original.png')

if __name__ == "__main__":
    train_annotation = '/tmp2/patrickwu2/labeled_data/train_annotation.csv'
    class_list = '/tmp2/patrickwu2/labeled_data/class_list.csv'
    dataset = CSVDataset(train_annotation, class_list)
    
    loader = DataLoader(dataset, batch_size=4, collate_fn=customed_collate_fn)
    from tqdm import tqdm
    for batch in tqdm(loader):
        print (batch)
        pass
