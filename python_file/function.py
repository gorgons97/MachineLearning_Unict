from torch import nn
from torch.optim import SGD
from sklearn.metrics import accuracy_score
from os import path
from os.path import join

import torch
import numpy as np
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt

import python_file.network as network

from python_file.dirPath import modelliDir, checkpointsDir, historiesDir

def train_classifier(model, train_loader, test_loader, exp_name='experiment' ,
                     lr=0.01, epochs=5, momentum=0.99, save_model=True, save_checkpoints=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr, momentum=momentum)
    #meters
    loss_meter = network.AverageValueMeter()
    acc_meter = network.AverageValueMeter()
    #device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    #definiamo un dizionario contenente i Loader di training e test
    loader = {'train' : train_loader, 'test' : test_loader }
    #definiamo un dizionario per salvare la history di training e test
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    #iniziaLizziamo iL global step e definiamo i nomi dei file per salvare i modelli e le history
    fileName = f'{exp_name}-{epochs}.pth'
    saveName = modelliDir / fileName
    save_path = historiesDir / f'{exp_name}-{epochs}.json'
    csv_path = historiesDir / f'{exp_name}-{epochs}.csv'
    for e in range(epochs):
        #iteriamo tra due modalità: train e test
        for mode in ['train' , 'test']:
            loss_meter.reset(); 
            acc_meter.reset()
            model.train() if mode == 'train' else model.eval()

            with torch.set_grad_enabled(mode=='train'): #abiLitiamo i gradienti SOLO in training
                for batch in loader[mode]:
                    x = list(batch.values())[0].to(device).float()
                    y = list(batch.values())[1].to(device).long()  
                    y = y.squeeze()

                    #print(y)  # Stampa il tensore target y per il controllo
                    output = model(x)
                    l = criterion(output, y)                    
                    
                    if mode=='train' :
                        optimizer.zero_grad ( )
                        l.backward()
                        optimizer.step( )

                    n = x.shape[0] #numero di elementi nel batch
                    
                    acc = accuracy_score(y.cpu().numpy(),output.max(1)[1].cpu().numpy())
                    loss_meter.add(l.item(), n)
                    acc_meter.add(acc,n)
            
            if save_model: 
                #Salviamo la history di ogni epoca in modo che possa essere visualizzata a posteriori
                epoch_loss = loss_meter.value()
                epoch_acc = acc_meter.value()

                if mode == 'train':
                    history["train_loss"].append(epoch_loss)
                    history["train_acc"].append(epoch_acc)
                else:
                    history["test_loss"].append(epoch_loss)
                    history["test_acc"].append(epoch_acc)

        #conserviamo i pesi del modello ad ogni epoca, in modo da poter visualizzare l'andamento della loss e dell'accuracy durante le epoche di addestramento
        if save_checkpoints:
            torch.save(model.state_dict(), checkpointsDir / f'{exp_name}-{e+1}.pth')
    
    if save_model:
        torch.save(model.state_dict(), saveName)  

        payload = {
            "exp_name": exp_name,
            "epochs": epochs,
            "history": history
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "test_loss", "train_acc", "test_acc"])
            for i in range(epochs):
                writer.writerow([
                    i + 1,
                    history["train_loss"][i],
                    history["test_loss"][i],
                    history["train_acc"][i],
                    history["test_acc"][i],
                ])

    return model

def test_classifier(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    probabilities, labels = [], []
    for batch in loader:
        x = list(batch.values())[0].to(device).float()
        y = list(batch.values())[1].to(device).long()  
        y = y.squeeze()
        
        output = model(x)
        probs = torch.softmax(output, dim=1).detach().cpu().numpy()
        labs = y.cpu().numpy()
        probabilities.extend(list(probs))
        labels.extend(list(labs))
    return np.array(probabilities), np.array(labels)

#Funzione per calcolare la curva di regressione, data una serie di predizioni e i corrispondenti valori reali Inutile per il caso di classificazione
def rec_curve(predictions, gt):
    assert predictions.shape == gt.shape
    # calcoliamo tutti gli errori mediante MAE
    errors = np.abs(np.array((predictions-gt)))
    
    # prendiamo i valori unici degli errori e ordiniamoli
    tolerances = sorted(np.unique(errors))
    correct= [] #lista delle "accuracy" relative a ogni soglia
    
    for t in tolerances:
        correct.append((errors<=t).mean()) # frazione di elementi "correttamente" regressi
    AUC = np.trapezoid(correct, tolerances) #area sotto la curva calcolata col metodo dei trapezi
    tot_area = np.max(tolerances)*1 # area totale
    AOC = tot_area - AUC
    # restituiamo le soglie, la frazione di campioni correttamente regressi e l'area sopra la curva
    return tolerances, correct, AOC

#Funzione per plottare la loss di training e test a partire da un file CSV generato durante il training. Utile per visualizzare l'andamento della loss durante le epoche di addestramento.
def plot_loss_from_csv(csv_path, title=None):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_loss"], label="train loss")
    plt.plot(df["epoch"], df["test_loss"], label="test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title or csv_path)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

#Funzione per plottare più curve di loss a partire da più file CSV. Utile per confrontare l'andamento della loss di diversi esperimenti o modelli.
def plot_multiple_losses(csv_files, labels=None):
    plt.figure(figsize=(10, 6))

    for i, csv_path in enumerate(csv_files):
        df = pd.read_csv(csv_path)
        label = labels[i] if labels else csv_path
        plt.plot(df["epoch"], df["test_loss"], label=f"{label} test loss")
        plt.plot(df["epoch"], df["train_loss"], linestyle="--", label=f"{label} train loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()