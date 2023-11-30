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

import python_file.network as network
import matplotlib.pyplot as plt

def train_classifier(model, train_loader, test_loader, exp_name='experiment' ,
                     lr=0.01, epochs=5, momentum=0.99, logdir='logs'):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr, momentum=momentum)
    #meters
    loss_meter = network.AverageValueMeter()
    acc_meter = network.AverageValueMeter()
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

