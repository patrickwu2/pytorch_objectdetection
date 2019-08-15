# built-in packages
import numpy as np
from tqdm import tqdm
import collections
# torch
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# torchvision
from torchvision import transforms

# from my module
from base_agent import BaseAgent
from data_loader import CSVDataset, customed_collate_fn
import model
from resnet import RetinaNet
import csv_eval

class RetinaNetAgent(BaseAgent):
    def __init__(self, config=None, file_manager=None):
        super(RetinaNetAgent, self).__init__(config, file_manager)
        # cuda / model setting
        self.use_cuda = True if len(self._config['device_ids']) >= 1 else False
        self.device_ids = self._config['device_ids']
        #self.model = RetinaNet(num_classes=3)
        self.model = getattr(model, self._config['model_name']) \
                (num_classes=3, pretrained=True)
        if self.use_cuda:
            self.model = self.model.to(self.device_ids[0])
        # optimizer / epoch
        self.optimizer = optim.Adam(self.model.parameters(), \
                                    lr=self._config['lr'])  
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)
        
        self._epoch = 0
        # load model if needed
        if self._config['load_model_path'] is not None:
            param_only = self._config['param_only']
            self.load_model(self._config['load_model_path'], param_only)
            if self.param_only is False: # load optimizer for re-training
                new_optimizer = optim.Adam(self.model.parameters(), 
                                            lr=self._config['lr'])  
                new_optimizer.load_state_dict(self.optimizer.state_dict())
                self.optimizer = new_optimizer
        print (self._config)
        print (self.model)

    def run(self):
        """Load data and start running model"""
        assert self._config['mode'] in ['train', 'test']
        dataset_name = self._config['dataset_name']

        if self._config['mode'] == 'train':
            print ("Mode : training")
            # load dataset
            dataset_train = CSVDataset(self._config['train_annotation'], \
                                        self._config['class_list'])
            # dataloader
            self.train_loader = DataLoader(dataset_train, num_workers=self._config['num_workers'], batch_size=self._config['batch_size'], shuffle=True, collate_fn=customed_collate_fn)

            # validation
            if self._config['validation'] is True:
                self.dataset_test = CSVDataset(self._config['test_annotation'], \
                                                self._config['class_list'])
                # dataloader
                self.test_loader = DataLoader(self.dataset_test, num_workers=self._config['num_workers'], batch_size=1, shuffle=False, collate_fn=customed_collate_fn)
            # file manager
            self.log_file = open("log.txt", "w")
            # train
            self.train()

    def test(self, validation=False):
        tqdm_loader = tqdm(self.test_loader, total=len(self.test_loader))
        self.change_model_state('eval')
            
        # array to save loss for testing data
        mAP = csv_eval.evaluate(self.dataset_test, self.test_loader, self.model, self.log_file)
                

    def train(self):
        print ('Start Training ...')
        # init log
        #self.test(validation=True)
        self.epoch_loss = []
        start_epoch = 0 if self._epoch == 0 else self._epoch + 1
        for self._epoch in range(start_epoch, self._config['epoches']):
            print (f'Start {self._epoch} epoch')
            # train
            self.train_one_epoch()
            # save log
            # self.save_log(print_msg=True)
            # save model
            if self._epoch > 5:
                self.test(validation=True)
            torch.save(self.model, 'saved/retinanet_{}.pt'.format(self._epoch))
            
    def train_one_epoch(self):
        tqdm_loader = tqdm(self.train_loader, total=len(self.train_loader))
        loss_hist = collections.deque(maxlen=500)
        self.change_model_state('train')
        self.model.freeze_bn()
        #self.model.training = True
        for step, batch in enumerate(tqdm_loader):
            loss, detailed_loss = self.feed_into_net(batch)
            self.update(loss)
            self.epoch_loss.append(loss.cpu().data.numpy()[0])
            loss_hist.append(loss.cpu().data.numpy()[0])

            regression_loss = detailed_loss['reg_loss'].cpu().data.numpy()[0]
            classification_loss = detailed_loss['class_loss'].cpu().data.numpy()[0]
            print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(self._epoch, step, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
            del loss
            del detailed_loss

    def update(self, loss):
        if bool(loss == 0):
            return
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        self.optimizer.step()


    def feed_into_net(self, batch, train=True):
        # load into GPU
        if self.use_cuda: 
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device_ids[0])
        # process batch data
        img = batch['img']
        annot = batch['annot']
        #cls_targets = batch['cls']
        #loc_targets = batch['bbox']
        #loc_preds, cls_preds = self.model(img)
        #print (loc_pred.size(), cls_pred.size())
        #from focal_loss import FocalLoss
        #self.focal_loss = FocalLoss()
        #loc_loss, cls_loss = self.focal_loss(loc_preds, loc_targets, cls_preds, cls_targets)
        
        class_loss, reg_loss = self.model((img, annot))
        
        #exit()
        # loss
        detailed_loss = {'class_loss':class_loss, 'reg_loss':reg_loss}
        loss = class_loss + reg_loss
        return loss, detailed_loss

    def change_model_state(self, state):
        if state == 'train':
            self.model.train()
        elif state == 'eval':
            self.model.eval()

if __name__ == "__main__":
    import config 
    agent = RetinaNetAgent(config.train_config)
    agent.run()

