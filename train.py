import pandas as pd
import numpy as np

import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from torch import nn 

#from torch.utils.tensorboard import SummaryWriter

import warnings

import torch
from sklearn.metrics import r2_score
from copy import deepcopy

from dataset_process import data, data_test
from model import GAT

warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
is_cuda = True

batch_size = 64
mask_edge = torch.from_numpy(np.stack([np.arange(batch_size),np.arange(batch_size)]))


model= GAT()


model.loss_func = nn.SmoothL1Loss()

model.optimizer = torch.optim.Adam(params = model.parameters(),lr = 0.001)#(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001)#

model.LR_scheduler = torch.optim.lr_scheduler.ExponentialLR(model.optimizer, gamma=0.99)
def r2(y_pred,y_true):
    y_pred_cls = y_pred.data
    if is_cuda:
        y_pred_cls = y_pred_cls.cpu()
        y_true = y_true.cpu()
    return r2_score(y_true,y_pred_cls)
model.metric_func = r2

model.metric_name = 'r2'

if is_cuda:
   model.cuda()
   mask_edge.cuda()


def train(model, features, edge, labels, batch_t, step:bool, pos):

    model.train()
    model.optimizer.zero_grad()
    predictions, embedding = model(features, edge, batch_t, pos)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)
    loss.backward()
    model.optimizer.step()
    if step:
        pass
    return loss.item(),metric.item()


@torch.no_grad()
def valid(model, features, edge, labels, batch_v, pos):

    model.eval()
    prediction, embedding = model(features, edge, batch_v, pos)
    loss = model.loss_func(prediction,labels)
    metric = model.metric_func(prediction,labels)
    return loss.item(),metric.item()

def ANN_train(model,epochs,dl_train,dl_test):
    metric_name = model.metric_name
    metirc_history = pd.DataFrame(columns = ['epoch','loss',metric_name,'test_loss','test_'+metric_name, 'val_loss','val_'+metric_name])
    print('training')
    for epoch in range(1,epochs+1):

        # 1.train-------------------------------------------
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        init_epochs = 0 
        lr_step = False
        if epoch % (0.1 * epochs-init_epochs) == 0 and epoch > init_epochs:
            lr_step = True
        for step,batch in enumerate(dl_train,1):
            features = batch.x.float()
            labels = batch.y
            edge = batch.edge_index.long()
            batch_t = batch.batch
            pos = batch.pos
            if is_cuda:
                features = features.cuda()
                labels = labels.cuda()
                edge = edge.cuda()
                batch_t = batch_t.cuda()

            loss,metric = train(model, features, edge, labels, batch_t, lr_step, pos)
            loss = loss
            metric = metric
            loss_sum += loss
            metric_sum += metric
        
        # 2.validation-------------------------------------------
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1
        for val_step, batch_v in enumerate(dl_val, 1):
            features_v = batch_v.x.float()
            labels_v = batch_v.y
            edge_v = batch_v.edge_index.long()
            batch_b = batch_v.batch
            pos_v = batch_v.pos
            if is_cuda:
                features_v = features_v.cuda()
                labels_v = labels_v.cuda()
                edge_v = edge_v.cuda()
                batch_b = batch_b.cuda()
            val_loss,val_metric = valid(model, features_v, edge_v, labels_v, batch_b, pos_v)
            val_loss_sum += val_loss
            val_metric_sum += val_metric            

        # 3.test-------------------------------------------
        test_loss_sum = 0.0
        test_metric_sum = 0.0
        test_step = 1
        for test_step, batch_t in enumerate(dl_test, 1):
            features_t = batch_t.x.float()
            labels_t = batch_t.y
            pos_t = batch_t.pos
            edge_t = batch_t.edge_index.long()
            batch_b = batch_t.batch
            if is_cuda:
                features_t = features_t.cuda()
                labels_t = labels_t.cuda()
                edge_t = edge_t.cuda()
                batch_b = batch_b.cuda()
            test_loss,test_metric = valid(model, features_t, edge_t, labels_t, batch_b, pos_t)
            test_loss_sum += test_loss
            test_metric_sum += test_metric

        # 4.record-------------------------------------------------
        info = (epoch, loss_sum/step, metric_sum/step, test_loss_sum/test_step, test_metric_sum/test_step, val_loss_sum/val_step, val_metric_sum/val_step)
        metirc_history.loc[epoch-1] = info

        if epoch % (0.1 * epochs) == 0:
            print(f'epoch = {round(info[0], 4)},loss = {round(info[1], 4)},{metric_name} = {round(info[2], 4)},val_loss = {round(info[3], 4)},val_{metric_name} = {round(info[4], 4)},test_loss = {round(info[5], 4)},test_{metric_name} = {round(info[6], 4)}')
            print(str(round(torch.cuda.memory_allocated()/1024/1024, 2)) + 'MB')
        
        if epoch == 1:
            best_param = deepcopy(model.state_dict())
            best_info = info

        else:
            if info[6] > best_info[6]:
                best_param = deepcopy(model.state_dict())
                best_info = info
        
    print('done!')
    print(f'epoch = {round(best_info[0], 4)},best_val_loss = {round(best_info[3], 4)},val_{metric_name} = {round(best_info[4], 4)},test_loss = {round(best_info[5], 4)},test_{metric_name} = {round(best_info[6], 4)}')
    
    val_loss = round(best_info[3], 4)
    val_R2 = round(best_info[4], 4)

    return metirc_history, best_param, val_loss, val_R2

fold_number = 8
val_loss_sum = 0.0
val_R2_sum = 0.0
epochs = 6000
data_size = len(data)
dl_test = DataLoader(data_test, batch_size=64) 
val_range = data_size/fold_number

for index in range(fold_number):
    
    dl_val = DataLoader(data[int(index * val_range):int((index + 1) * val_range)], batch_size= batch_size, shuffle=True)#
    dl_train = DataLoader(ConcatDataset([data[:int(index * val_range)], data[int((index + 1) * val_range):]]), batch_size= batch_size)
    historydf, model_param, model_loss, model_R2= ANN_train(model,epochs,dl_train,dl_test)
    val_loss_sum += model_loss
    val_R2_sum += model_R2

average_loss = val_loss_sum/fold_number
average_R2 = val_R2_sum/fold_number

if __name__ == '__main__':
    print(average_loss)
    print(average_R2)

