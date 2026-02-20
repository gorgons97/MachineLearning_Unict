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


class MiniAlexNetV2(nn.Module):
    def __init__(self, input_channels=3, out_classes=100):
        super(MiniAlexNetV2, self).__init__()
        
        # Definiamo il modello utilizzando moduli Sequential.
        # Abbiamo due parti principali: "feature extractor" e "classificatore".
        
        # Feature Extractor
        self.feature_extractor = nn.Sequential(
            # Conv1
            nn.Conv2d(input_channels, 16, 5, padding=2),
            # Input: 3 x 32 x 32. Output: 16 x 32 x 32
            nn.MaxPool2d(2),
            # Input: 16 x 32 x 32. Output: 16 x 16 x 16
            nn.ReLU(),
            
            nn.BatchNorm2d(16), 
            # Conv2
            nn.Conv2d(16, 32, 5, padding=2),
            # Input: 16 x 16 x 16. Output: 32 x 16 x 16
            nn.MaxPool2d(2),
            # Input: 32 x 16 x 16. Output: 32 x 8 x 8
            nn.ReLU(),
            
            nn.BatchNorm2d(32),
            # Conv3
            nn.Conv2d(32, 64, 3, padding=1),
            # Input: 32 x 8 x 8. Output: 64 x 8 x 8
            nn.ReLU(),
            
            nn.BatchNorm2d(64),
            # Conv4
            nn.Conv2d(64, 128, 3, padding=1),
            # Input: 64 x 8 x 8. Output: 128 x 8 x 8
            nn.ReLU(),
            
            nn.BatchNorm2d(128),            
            # Conv5
            nn.Conv2d(128, 256, 3, padding=1),
            # Input: 128 x 8 x 8. Output: 256 x 8 x 8
            nn.MaxPool2d(2),
            # Input: 256 x 8 x 8. Output: 256 x 4 x 4
            nn.ReLU()
        )
        
        # Classificatore
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # I layer di dropout vanno posizionati prima di FC6 e FC7
            
            nn.BatchNorm1d(4096),  
            # FC6
            nn.Linear(4096, 2048),
            # Input: 256 * 4 * 4
            nn.ReLU(),
            nn.Dropout(),
            
             nn.BatchNorm1d(2048),
            # FC7
            nn.Linear(2048, 1024),
            nn.ReLU(),
            
            nn.BatchNorm1d(1024),
            # FC8
            nn.Linear(1024, out_classes)
        )

    def forward(self, x):
        # Applichiamo le diverse trasformazioni in cascata
        x = self.feature_extractor(x)
        x = self.classifier(x.view(x.shape[0], -1))
        return x


class MiniAlexNetV3(nn.Module):
   def __init__(self, input_channels=3, out_classes=100):
      super(MiniAlexNetV3, self).__init__()
      #ridefiniamo il modello utilizzando i moduli sequential.
      #ne definiamo due: un "feature extractor", che estrae le feature maps
      #e un "classificatore" che implementa i livelly FC
      self.feature_extractor = nn.Sequential(
        #Conv1
        nn.Conv2d(input_channels, 16, 5, padding=2), #Input: 3 x 28 x 28. Ouput: 16 x 28 x 28
        nn.MaxPool2d(2), #Input: 16 x 28 x 28. Output: 16 x 14 x 14
        nn.ReLU(),
        
        #Conv2
        nn.Conv2d(16, 32, 5, padding=2), #Input 16 x 14 x 14. Output: 32 x 14 x 14
        nn.MaxPool2d(2), #Input: 32 x 14 x 14. Output: 32 x 7 x 7
        nn.ReLU(),
        
        #Conv3
        nn.Conv2d(32, 64, 3, padding=1), #Input 32 x 7 x 7. Output: 64 x 7 x 7
        nn.ReLU(),
        
        #Conv4
        nn.Conv2d(64, 128, 3, padding=1), #Input 64 x 7 x 7. Output: 128 x 7 x 7
        nn.ReLU(),
        
        #Conv5
        nn.Conv2d(128, 256, 3, padding=1), #Input 128 x 7 x 7. Output: 256 x 7 x 7
        nn.MaxPool2d(2), #Input: 256 x 7 x 7. Output: 256 x 3 x 3
        nn.ReLU()
      )
      
      self.classifier = nn.Sequential(
        nn.Dropout(), #i layer di dropout vanno posizionati prima di FC6 e FC7
        #FC6
        nn.Linear(2304, 2048), #Input: 256 * 3 * 3
        nn.ReLU(),
        
        nn.Dropout(),
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
   
class MiniAlexNetV4(nn.Module):
   def __init__(self, input_channels=3, out_classes=100):
      super(MiniAlexNetV4, self).__init__()
      #ridefiniamo il modello utilizzando i moduli sequential.
      #ne definiamo due: un "feature extractor", che estrae le feature maps
      #e un "classificatore" che implementa i livelly FC
      self.feature_extractor = nn.Sequential(
         #Conv1
         nn.Conv2d(input_channels, 16, 5, padding=2), #Input: 3 x 28 x 28. Ouput: 16 x 28 x 28
         nn.MaxPool2d(2), #Input: 16 x 28 x 28. Output: 16 x 14 x 14
         nn.ReLU(),
         
         #Conv2
         nn.BatchNorm2d(16), #dobbiamo passare come parametro il numero di mappe di feature in input
         nn.Conv2d(16, 32, 5, padding=2), #Input 16 x 14 x 14. Output: 32 x 14 x 14
         nn.MaxPool2d(2), #Input: 32 x 14 x 14. Output: 32 x 7 x 7
         nn.ReLU(),
         
         #Conv3
         nn.BatchNorm2d(32), #dobbiamo passare come parametro il numero di mappe di feature in input
         nn.Conv2d(32, 64, 3, padding=1), #Input 32 x 7 x 7. Output: 64 x 7 x 7
         nn.ReLU(),
         
         #Conv4
         nn.BatchNorm2d(64), #dobbiamo passare come parametro il numero di mappe di feature in input
         nn.Conv2d(64, 128, 3, padding=1), #Input 64 x 7 x 7. Output: 128 x 7 x 7
         nn.ReLU(),
         
         #Conv5
         nn.BatchNorm2d(128), #dobbiamo passare come parametro il numero di mappe di feature in input
         nn.Conv2d(128, 256, 3, padding=1), #Input 128 x 7 x 7. Output: 256 x 7 x 7
         nn.MaxPool2d(2), #Input: 256 x 7 x 7. Output: 256 x 3 x 3
         nn.ReLU()
      )
      
      self.classifier = nn.Sequential(
         nn.Dropout(), #i layer di dropout vanno posizionati prima di FC6 e FC7
         #FC6
         nn.BatchNorm1d(2304), #dobbiamo passare come parametro il numero di feature in input
         nn.Linear(2304, 2048), #Input: 256 * 3 * 3
         nn.ReLU(),
         
         nn.Dropout(),
         #FC7
         nn.BatchNorm1d(2048), #dobbiamo passare come parametro il numero di mappe di feature in input
         nn.Linear(2048, 1024),
         nn.ReLU(),
         
         #FC8
         nn.BatchNorm1d(1024), #dobbiamo passare come parametro il numero di mappe di feature in input
         nn.Linear(1024, out_classes)
      )

      
   def forward(self,x):
      #Applichiamo le diverse trasformazioni in cascata
      x = self.feature_extractor(x)
      x = self.classifier(x.view(x.shape[0],-1))
      return x