# Riassunto del Progetto

## Obiettivi Principali
- Creare un notebook **Graphs** per generare grafici in modo centralizzato
- Ristrutturare le versioni di training per eseguirle in parallelo
- Confrontare tre modelli (AlexNet, MiniAlexNet, MiniAlexNetV2) con diverse combinazioni di iperparametri (LR e Momentum)
- Identificare la migliore configurazione per MiniAlexNetV2 e estendere il training a 1200 epoche
- Valutare eventuali riallineamenti finali per AlexNet e MiniAlexNet

## Risultati Principali
### Valutazione dei Modelli
| Modello           | LR    | Momentum | Accuratezza Test | Giudizio       |
|-------------------|-------|---------|------------------|----------------|
| MiniAlexNetV2     | 0.003 | 0.6     | ~1.00            | **PERFETTO**   |
| MiniAlexNetV2     | 0.003 | 0.9     | ~0.95            | **OTTIMO**     |
| MiniAlexNetV2     | 0.003 | 0.3     | ~0.90            | **OTTIMO**     |
| MiniAlexNet       | 0.001 | 0.6     | ~0.55            | **SCADENTE**   |
| AlexNet           | 0.001 | 0.6     | ~0.45            | **PESSIMO**    |

### Performance a 1200 Epoche
- **MiniAlexNetV2 (LR=0.003, Momentum=0.6)**:
  - Accuratezza: 100% (train/test)
  - Loss: ~0.0 (train/test)
  - AUC ROC: ~1.00 (perfetta discriminazione)
  - Nessun overfitting

## Analisi Reale (YOLO)
### Problemi Riscontrati
1. **Dataset limitato**: YOLO richiede migliaia di immagini per classe
2. **Iperparametri non ottimizzati**: LR, batch size, anchors
3. **Annotazioni imprecise**: bounding box non corrette
4. **Threshold di confidenza** non adatti

### Soluzioni Proposte
- Aumentare il numero di epoche a **500**
- Ottimizzare il dataset con **augmentation avanzata**
- Rivedere i **bounding box** e le **annotazioni**
- Personalizzare i **parametri di confidenza**

## Conclusione
- **MiniAlexNetV2** è il modello più performante con la configurazione LR=0.003 e Momentum=0.6
- YOLO richiede un'attenzione particolare al dataset e agli iperparametri
- Le 1200 epoche hanno confermato l'assenza di overfitting e la stabilità del modello