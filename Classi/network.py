import nbimporter
import dataset_class as StreetSign
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch import nn
from torch.optim import SGD
from sklearn.metrics import accuracy_score
from os.path import join
import torch
import numpy as np
from PIL import Image
from os import path

import matplotlib.pyplot as plt

np.random.seed(1328)
torch.random.manual_seed(1328);

class AverageValueMeter():
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.sum = 0
        self.num = 0
        
    def add(self, value, num):
        self.sum += value*num
        self.num += num
        
    def value(self):
        try :
            return self.sum/ self.num
        except :
            return None
        
class LeNetColor(nn.Module):
    def __init__ (self):
        super(LeNetColor, self).__init__()
        #ridefiniamo iL modeLLo utilizzando i moduli sequentiaL.
        #ne definiamo due: un "feature extractor", che estrae Le feature maps
        #e un "classificatore" che implementa i LiveLLy FC
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 18, kernel_size=5), #Input: 3 x 32 x 32. Ouput: 18 x 28 x 28
            nn.MaxPool2d(2), #Input: 18 x 28 x 28. Output: 18 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(18, 28, kernel_size=5), #lnput 18 x 14 x 14. Output: 28 x le x le
            nn.MaxPool2d(2), #Input 28 x le x le. Output: 28 x 5 x 5
            nn.ReLU()
        )    
            
        self.classifier = nn.Sequential(
            nn.Linear(28 * 5 * 5, 360), #rnput: 28 * 5 * 5
            nn.ReLU(),
            nn.Linear(360, 252),
            nn.ReLU(),
            nn.Linear(252, 100)
        )
        
    def forward(self,x):
        #AppLichiamo Le diverse trasformazioni in cascata
        x = self.feature_extractor(x)
        x = self.classifier(x.view(x.shape[0],-1))
        return x
    
class MiniAlexNet(nn.Module):
    def __init__(self, input_channels=3, out_classes=100):
        super(MiniAlexNet, self).__init__()
        #ridefiniamo il modello utilizzando i moduli sequential.
        #ne definiamo due: un "feature extractor", che estrae le feature map.
        #e un "classificatore" che implementa i livelly FC
        self.feature_extractor = nn.Sequential(
            #Conv1
            nn.Conv2d(input_channels, 16, 5, padding=2), #Input: 3 x 32 x 32. Ouput: 16 x 32 x 32
            nn.MaxPool2d(2), #Input: 16 x 32 x 32. Output: 16 x 16 x 16
            nn.ReLU(),
            
            #Conv2
            nn.Conv2d(16, 32, 5, padding=2), #Input 16 x 16 x 16. Output: 32 x 16 x 16
            nn.MaxPool2d(2), #Input: 32 x 16 x 16. Output: 32 x 8 x 8
            nn.ReLU(),
            
            #Conv3
            nn.Conv2d(32, 64, 3, padding=1), #Input 32 x 8 x 8. Output: 64 x 8 x 8
            nn.ReLU(),
            
            #Conv4
            nn.Conv2d(64, 128, 3, padding=1), #Input 64 x 8 x 8. Output: 128 x 8 x 8
            nn.ReLU(),
            
            #Conv5
            nn.Conv2d(128, 256, 3, padding=1), #Input 128 x 8 x 8. Output: 256 x 8 x 8
            nn.MaxPool2d(2), #Input: 256 x 8 x 8. Output: 256 x 4 x 4
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            #FC6
            nn.Linear(4096, 2048), #Input: 256 * 4 * 4
            nn.ReLU(),
            
            #FC7
            nn.Linear(2048, 1024),
            nn.ReLU(),
            
            #FC8
            nn.Linear(1024, out_classes)
        )
        
        
    def forward(self,x):
        #Applichiamo le diverse trasformazioni in cascata
        x = self.feature_extractor(x)
        x = self.classifier(x.view(x.shape[0],-1))
        return x
    
def train_classifier(model, train_loader, test_loader, exp_name='experiment' ,
                     lr=0.01, epochs=100, momentum=0.99, logdir='logs'):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr, momentum=momentum)
    #meters
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    #writer
    writer = SummaryWriter(join(logdir, exp_name))
    #device
    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    #definiamo un dizionario contenente i Loader di training e test
    loader = {
        'train' : train_loader,
        'test' : test_loader
    }
    #iniziaLizziamo iL global step
    global_step = 0
    for e in range(epochs):
        #iteriamo tra due modalità: train e test
        for mode in ['train' , 'test']:
            loss_meter.reset(); acc_meter.reset()
            model.train() if mode == 'train' else model.eval()
            with torch.set_grad_enabled(mode=='train'): #abiLitiamo i gradienti SOLO in training
                for i, batch in enumerate(loader[mode]):
                    x = list(batch.values())[0].to(device).float()
                    y = list(batch.values())[1].to(device).long()  # Assicurati che y sia di tipo Long
                    y = y.squeeze()

                    #print(y)  # Stampa il tensore target y per il controllo
                    output = model(x)
                    
                    #aggiorniamo iL gLobaL_step
                    #conterrà iL numero di campioni visti durante iL training
                    n = x.shape[0] #numero di elementi nel batch
                    global_step += n
                    l = criterion(output, y)
                    
                    if mode=='train' :
                        l.backward()
                        optimizer.step( )
                        optimizer.zero_grad ( )
                    
                    acc = accuracy_score(y.to('cpu'),output.to('cpu').max(1)[1])
                    loss_meter.add(l.item(), n)
                    acc_meter.add(acc,n)
                    
                    #Loggiamo i risultati iterazione per iterazione SOLO durante iL training
                    if mode=='train' :
                        writer.add_scalar( ' loss/train ' ,loss_meter.value(), global_step=global_step)
                        writer.add_scalar( 'accuracy/train' ,acc_meter.value(), global_step=global_step)            
                #una voLta finita L 'epoca (sia nel caso di training che test, Loggiamo Le stime finali)
                writer.add_scalar( 'loss/' + mode, loss_meter.value(), global_step=global_step)
                writer.add_scalar( 'accuracy/' + mode, acc_meter.value(), global_step=global_step)
        #conserviamo i pesi del model Lo aLLa fine di un ciclo di training e test
        torch.save(model.state_dict(), 'modelli\%s-%d.pth'%(exp_name,e+1))
    return model

def test_classifier(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    predictions, labels = [], []
    for batch in loader:
        x = list(batch.values())[0].to(device).float()
        y = list(batch.values())[1].to(device).long()  
        y = y.squeeze()
        
        output = model(x)
        preds = output.to('cpu').max(1)[1].numpy()
        labs = y.to('cpu').numpy()
        predictions.extend(list(preds))
        labels.extend(list(labs))
    return np.array(predictions), np.array(labels)

