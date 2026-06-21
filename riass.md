# Riassunto del README.md

## Obiettivo del Progetto
L'obiettivo del progetto è analizzare e confrontare diverse reti neurali (AlexNet, MiniAlexNet, MiniAlexNetV2) con diverse combinazioni di iperparametri (Learning Rate e Momento) per determinare la configurazione ottimale per il training.

## Principali Risultati
- **MiniAlexNetV2** con **Learning Rate = 0.003** e **Momento = 0.6** ha mostrato prestazioni eccellenti, raggiungendo un'accuratezza del 100% dopo 1200 epoche.
- **AlexNet** e **MiniAlexNet** hanno mostrato prestazioni scadenti o pessime, probabilmente a causa della dimensione eccessiva del modello rispetto al dataset.
- **MiniAlexNetV2** con **Learning Rate = 0.0001** ha mostrato prestazioni pessime, probabilmente a causa di un learning rate troppo basso.

## Confronto tra 200 e 1200 Epoche
- Il modello **MiniAlexNetV2** ha raggiunto un'accuratezza del 100% già a 200 epoche, ma le ulteriori 1000 epoche hanno portato a un miglioramento marginale.
- Non è stato osservato overfitting, anche dopo 1200 epoche.

## Problemi con YOLO
- **Qualità dell'immagine**: Immagini reali hanno rumore, variazioni di luce, sfocatura.
- **Occlusioni**: Oggetti parzialmente nascosti o sovrapposti.
- **Scale diverse**: Oggetti di dimensioni molto variabili.
- **Background complesso**: Sfondi con pattern simili agli oggetti.
- **Pochi dati di addestramento**: Dataset limitato o non rappresentativo.
- **Epoche insufficienti**: 200 epoche potrebbero non essere sufficienti per YOLO.

## Soluzioni per YOLO
- Addestrare YOLO con un numero maggiore di epoche (ad esempio, 500 epoche).