import argparse
import imp
from turtle import down, forward
from attr import validate
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np 
from polyaxon import tracking

class Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,10)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

def create_dataloader(batch_size_input):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

    trainset = datasets.FashionMNIST("./pytorch/F_MNIST_data", download=True, train=True, transform=transform)
    testset = datasets.FashionMNIST("./pytorch/F_MNIST_data", download=True, train=False,transform=transform)
    train_loader = torch.utils.data.Dataloader(trainset, batch_size=batch_size_input)
    valid_loader = torch.utils.data.Dataloader(testset, batch_size=batch_size_input)
    return train_loader, valid_loader

def train_model(lr_input, epoch_input,train_dataloader, valid_dataloader, optim_input, loss_fn_input):
    model = Classifier()

    tracking.init()
    log_dir = tracking.get_tensorboard_path()
    writer = SummaryWriter(log_dir=log_dir)


    OPTIMIZERS = {
        'SGD':optim.SGD(model.parameters(), lr = lr_input),
        'Adam':optim.Adam(model.parameters(),lr=lr_input)
    }

    LOSS_FN = {
        'nll':nn.NLLLoss,
        'categorical':nn.CrossEntropyLoss
    }

    
    optimizer = OPTIMIZERS[optim_input]
    loss_fn = LOSS_FN[loss_fn_input]

    


    for e in range(epoch_input):
        running_loss = 0
        valid_loss = 0
        train_correct = 0
        valid_correct = 0
        model.train()
        for images, labels in train_dataloader:
            optimizer.zero_grad()
            log_ps = model(images)
            loss = loss_fn(log_ps,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*images.size(0)
            train_correct += (log_ps == labels).float().sum()

        
        for images,labels in valid_dataloader:
            model.eval()
            with torch.no_grad():
                log_ps = model(images)
                loss = loss_fn(log_ps,labels)
                valid_loss = loss.item()*images.size(0)
                valid_correct += (log_ps==labels).float().sum()

        running_loss = running_loss / len(train_dataloader.dataset)
        valid_loss = valid_loss / len(valid_dataloader.dataset)
        train_accuracy = 100 * train_correct / len(train_dataloader.dataset)
        test_accuracy = 100 * valid_correct / len(valid_dataloader.dataset)

        writer.add_scalar('Loss/Train', running_loss,e)
        writer.add_scalar('Loss/Test', valid_loss, e)
        writer.add_scalar('Accuracy/Train', train_accuracy,e)
        writer.add_scalar('Accuracy/Test', test_accuracy,e)

        tracking.log_metric(name='Loss/Train',value=running_loss,step=e)
        tracking.add_scalar(name='Loss/Test', value = valid_loss, step=e)
        tracking.add_scalar(name='Accuracy/Train', value= train_accuracy,step=e)
        tracking.add_scalar(name='Accuracy/Test', value =test_accuracy,step=e)
    
    asset_path = tracking.get_outputs_path('model.ckpt')
    torch.save(model.state_dict(), asset_path)
    tracking.log_artifact_ref(asset_path, framework="pytorch")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size',
        type=int,
        default=8
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.4
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=10
    )
    parser.add_argument(

        '--optim',
        type=str,
        default='SGD'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='nll'
    )

    args = parser.parse_args()
    
    
    train_loader, valid_loader = create_dataloader(args.batch_size)

    train_model(args.learning_rate,args.epoch,train_loader,valid_loader,
    args.optim,args.loss)

            

