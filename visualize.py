import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from data_loader import CSVDataset, customed_collate_fn

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
        parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

        parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
        parser.add_argument('--coco_path', help='Path to COCO directory')
        parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
        parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

        parser.add_argument('--model', help='Path to model (.pt) file.')

        parser = parser.parse_args(args)

        dataset_val = CSVDataset(parser.csv_val, parser.csv_classes)

        dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=customed_collate_fn, batch_size=1)

        retinanet = torch.load(parser.model)

        use_gpu = True

        if use_gpu:
                retinanet = retinanet.cuda()

        retinanet.eval()


        def draw_caption(image, box, caption):

                b = np.array(box).astype(int)
                cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        for idx, data in enumerate(dataloader_val):
                
                with torch.no_grad():
                        st = time.time()
                        scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
                        print('Elapsed time: {}'.format(time.time()-st))
                        scores = scores.cpu().data.numpy()
                        idxs = np.where(scores>0.5)
                        img = np.array(255*data['img'][0, :, :, :]).copy()
                        #img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

                        img[img<0] = 0
                        img[img>255] = 255

                        img = np.transpose(img, (1, 2, 0))

                        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

                        for j in range(idxs[0].shape[0]):
                                bbox = transformed_anchors[idxs[0][j], :]
                                x1 = int(bbox[0])
                                y1 = int(bbox[1])
                                x2 = int(bbox[2])
                                y2 = int(bbox[3])
                                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                                draw_caption(img, (x1, y1, x2, y2), label_name)

                                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                                print(label_name)

                        cv2.imwrite('testing/img_{}.png'.format(idx), img)



if __name__ == '__main__':
 main()
