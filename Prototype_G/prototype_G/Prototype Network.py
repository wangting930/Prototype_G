#Prototype network framework
import torch

import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from GraphSAGE import GraphSage
import torch.nn.functional as F
import Data_processing
from Sampling import multihop_sampling

from collections import namedtuple
INPUT_DIM = 1360   # input dimension
HIDDEN_DIM = [256, 128]   # number of hidden unit nodes
NUM_NEIGHBORS_LIST = [6,6] #Note: The number of neighbours sampled needs to be the same as the number of layers in the network
BTACH_SIZE = 10    # Batch size
EPOCHS = 20
NUM_BATCH_PER_EPOCH = 20   # number of batches per epoch cycle
LEARNING_RATE = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Data = namedtuple('Data', ['x', 'y', 'adjacency_dict',
                           'train_mask', 'val_mask','test_mask'])

data = data.process_data()
x=data.x

train_index = np.where(data.train_mask)[0]#training Nodes/corresponding index
test_index = np.where(data.test_mask)[0]#test Nodes/corresponding index
random.shuffle(train_index)#knock out the index value of a node
random.shuffle(test_index)
train_label = data.y

model = GraphSage(input_dim=INPUT_DIM,hidden_dim=HIDDEN_DIM,
                 num_neighbors_list=NUM_NEIGHBORS_LIST).to(DEVICE)
print(model)


criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

def index_select(index,number):
    label_0 = []
    label_1 = []
    label_2 = []
    label_3 = []
    label_4 = []
    for m in index:
        if train_label[m] == 0:
            label_0.append(m)
        else:
            if train_label[m] == 1:
                label_1.append(m)
            else:
                if train_label[m] == 2:
                    label_2.append(m)
                else:
                    if train_label[m] == 3:
                        label_3.append(m)
                    else:
                        if train_label[m] == 4:
                            label_4.append(m)
    a = np.arange(BTACH_SIZE*5)
    for i in range(number):
        a[i]=np.random.choice(label_0, size=(1,))
        a[i+BTACH_SIZE]=np.random.choice(label_1, size=(1,))
        a[i+BTACH_SIZE*2]=np.random.choice(label_2, size=(1,))
        a[i +BTACH_SIZE*3] = np.random.choice(label_3, size=(1,))
        a[i + BTACH_SIZE*4] = np.random.choice(label_4, size=(1,))
    return a


def train():
    model.train()
    for e in range(EPOCHS):
        for batch in range(NUM_BATCH_PER_EPOCH):
            batch_src_index=index_select(train_index,10)
            batch_src_label = torch.from_numpy(train_label[batch_src_index]).long().to(DEVICE)  # Labels of the training set
            batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST,
                                                      data.adjacency_dict)  # Perform two levels of sampling neighbour nodes
            batch_sampling_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in batch_sampling_result]#Obtain the feature vectors corresponding to the sampled nodes
            batch_train_logits, all_hidden = model(batch_sampling_x)  # Model output to get features
            train_yuanxin = representation(batch_train_logits, batch_src_label)  # The prototype representation is calculated from the support set during training

            train_test_index=[]
            for i in train_index:
                if i not in batch_src_index:
                    train_test_index.append(i)
            batch_src_testindex = index_select(train_test_index,5)#Randomly select training nodes as test nodes
            batch_src_testlabel = torch.from_numpy(train_label[batch_src_testindex]).long().to(DEVICE)#Labells of the query set for the training process
            batch_trainsampling_result = multihop_sampling(batch_src_testindex, NUM_NEIGHBORS_LIST,data.adjacency_dict)
            train_x_tezheng = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in batch_trainsampling_result]#得到训练过程中测试集的特征向量
            train_logits= model(train_x_tezheng)
            test_disttrain = euclidean_dist(train_logits, train_yuanxin)#Calculate the distance from the point in the query set to the prototype during training
            log_p_y = F.log_softmax(-test_disttrain, dim=1)
            #loss = criterion(log_p_y, batch_src_testlabel)
            loss = F.nll_loss(log_p_y,batch_src_testlabel)#To adjust the model parameters
            optimizer.zero_grad()#Zeroing the gradient, i.e. making the derivative of loss with respect to weight zero
            loss.backward()  # Back propagation of the gradient of the calculation parameters
            optimizer.step()  # Gradient update
            print("Epoch {:03d} Batch {:03d} train_Loss: {:.4f}".format(e, batch, loss.item()))
        test(train_yuanxin)

def test(yuanxin):
    model.eval()
    with torch.no_grad():

        test_sampling_result = multihop_sampling(test_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
        test_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in test_sampling_result]
        test_logits,tezheng = model(test_x)
        test_dist=euclidean_dist(test_logits, yuanxin)#Get the distance from the test node to the prototype
        test_label = torch.from_numpy(data.y[test_index]).long().to(DEVICE)
        print(test_index)
        print(test_label)
        predict_y = test_dist.min(1)[1]#prediction labels, minimum values by row for prediction results
        print(predict_y)

        accuarcy = torch.eq(predict_y, test_label).float().mean().item() #Calculate the accuracy rate
        print("Test Accuracy: ", accuarcy)



def representation(feature,label):
    sum = torch.zeros(5,128)
    yuanxin = torch.zeros(5,128)
    for batch in range(BTACH_SIZE*5):
        if label[batch]==0:
            sum[0] = sum[0] + feature[batch]
        if label[batch]==1:
            sum[1] = sum[1] + feature[batch]
        if label[batch]==2:
            sum[2] = sum[2] + feature[batch]
        if label[batch]==3:
            sum[3] = sum[3] + feature[batch]
        if label[batch]==4:
            sum[4] = sum[4] + feature[batch]
    for i in range(5):
        yuanxin[i]=sum[i]/5
    return yuanxin

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


if __name__ == '__main__':
    train()

