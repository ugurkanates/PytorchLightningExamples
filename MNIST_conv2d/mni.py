import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.metrics import accuracy_score
from pytorch_lightning import Trainer
from argparse import ArgumentParser

import pytorch_lightning as pl

input_channel = 1
out_channel = 64
class MNISTModel(pl.LightningModule):

    def __init__(self,params):
        super(MNISTModel, self).__init__()
        # not the best model...
        self.params = params
        
        self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=10,kernel_size=3)
        self.pool1 = torch.nn.MaxPool2d(2)

        self.conv2 = torch.nn.Conv2d(in_channels=10,out_channels=20,kernel_size=3)
        self.pool2 = torch.nn.MaxPool2d(2)

        self.conv3 = torch.nn.Conv2d(in_channels=20,out_channels=30,kernel_size=3)
        self.dropout1 = torch.nn.Dropout2d(p=0.25) # probabilit 0.25

        self.fc3 = torch.nn.Linear(30*3*3,270)
        self.fc4 = torch.nn.Linear(270,26)

        self.softmax = torch.nn.LogSoftmax(dim=1)


    def forward(self, x):
        # called with self(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.dropout1(x)
        x = x.view(-1, 30 * 3 * 3) 
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.softmax(x)



    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        logits = F.log_softmax(y_hat, dim=1)
        preds = torch.topk(logits, dim=1, k=1)[1].view(-1)
        accuracy = accuracy_score(y.cpu(),  preds.cpu())

        loss = F.cross_entropy(y_hat, y)
        return {'val_loss': loss,
                'accuracy': torch.tensor(accuracy)}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        return self.validation_step(batch, batch_nb)


    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()

        logs = {'test_loss': avg_loss, 'test_acc': accuracy}
        return {'avg_val_loss': avg_loss,
                'progress_bar': logs,
                'log': logs}


    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adadelta(self.parameters(), lr=self.params.lr)


    def train_dataloader(self):
        # REQUIRED
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32,num_workers=4)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32,num_workers=4)

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=32,num_workers=4)

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--final_dim', type=int, default=128)
    args.add_argument('--lr', type=float, default=0.02)
    args.add_argument('--pretrained', type=bool, default=True)
    params = args.parse_args()
   
    model = MNISTModel.load_from_checkpoint(checkpoint_path=os.getcwd()+"/test.ckpt",params=params)

    trainer = Trainer(weights_save_path=os.getcwd(),gpus=1,max_epochs=1)
    #trainer.fit(model)
    trainer.test(model)
