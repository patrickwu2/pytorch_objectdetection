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
        import pickle
        pickle.dump(self.image_data, open("image_data.pkl", "wb"))

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
    def num_classes(self):
        return max(self.classes.values()) + 1

def customed_collate_fn(batch):
    def get_aug(aug, min_area=0., min_visibility=0.):
        return Compose(aug, bbox_params={'format': 'pascal_voc', 'min_area': min_area, 'min_visibility': min_visibility, 'label_fields': ['category_id']})       

    '''
    def find_new_size(img_size):
        min_side = 608
        max_side = 1024
        rows, cols, cns = img_size
        smallest_side = min(rows, cols)
        # scale
        scale = min_side / smallest_side

        return (int(round(rows*scale)), int(round(cols*scale)))
    '''

    def _transform_fn_(one_batch):
        # augmentation
        #w, h = find_new_size(one_batch['image'].shape)
        aug = get_aug([Rotate(limit=20, p=0.3), ToGray(p=0.3), HueSaturationValue(20, 30, 20, p=0.2), Resize(896,736)])
        augmented = aug(**one_batch)
        
        new_one_batch = {}
        # image
        new_one_batch['image'] = augmented['image'].astype(np.float32) / 255.0
        # annotation
        new_one_batch['annot'] = []
        for bbox, cat in zip(augmented['bboxes'], augmented['category_id']):
            bbox = [int(x) for x in bbox]
            anno = bbox + [cat]
            new_one_batch['annot'].append(anno)
        new_one_batch['annot'] = np.array(new_one_batch['annot'])
        pad_num = 15 - new_one_batch['annot'].shape[0]
        pad = np.ones((pad_num, 5)) * -1
        new_one_batch['annot'] = np.concatenate([new_one_batch['annot'], pad], axis=0)
        return new_one_batch
    batch = [_transform_fn_(one_batch) for one_batch in batch]
    values = {}
    values['img'] = torch.stack([torch.tensor(x['image']).type(torch.float32).permute(2, 0, 1) for x in batch], 0, out=None)
    
    values['annot'] = torch.stack([torch.tensor(x['annot']).type(torch.float32) for x in batch], 0, out=None)
    return values
    #visualize(one_batch, category_id_to_name, 'original.png')

if __name__ == "__main__":
    train_annotation = '/tmp2/patrickwu2/labeled_data/train_annotation.csv'
    class_list = '/tmp2/patrickwu2/labeled_data/class_list.csv'
    dataset = CSVDataset(train_annotation, class_list)
    exit()
    loader = DataLoader(dataset, batch_size=4, collate_fn=customed_collate_fn)
    from tqdm import tqdm
    for batch in tqdm(loader):
        print (batch)
        pass
