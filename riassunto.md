# Presentazione sintetica del progetto
## 1) Obiettivo
Confrontare tre modelli (**AlexNet**, **MiniAlexNet**, **MiniAlexNetV2**) e trovare la migliore combinazione di iperparametri (**Learning Rate**, **Momentum**) tramite analisi di accuratezza, loss e ROC.

## 2) Piano di lavoro
- Confronto iniziale con configurazione comune: **LR=0.001, momentum=0.6**.
- Tuning completo su **MiniAlexNetV2**.
- Selezione della combinazione migliore.
- Addestramento esteso fino a **1200 epoche**.

## 3) Risultati a 200 epoche
- **AlexNet**: performance basse, apprendimento insufficiente.
- **MiniAlexNet**: miglioramento lieve, ancora debole.
- **MiniAlexNetV2**: modello nettamente migliore.

### Migliore configurazione (200 epoche)
- **MiniAlexNetV2 + LR=0.003 + momentum=0.6**
- **Test accuracy ~0.95–1.00**
- Loss ridotta e ROC eccellente.

## 4) Estensione a 1200 epoche (best config)
Con **MiniAlexNetV2 (LR=0.003, momentum=0.6)**:
- Train accuracy ~1.00
- Test accuracy ~1.00
- Train/Test loss ~0
- AUC ROC media ~1.00
- Nessun overfitting evidente

## 5) Messaggio chiave
- Il miglioramento principale arriva entro le prime 200 epoche.
- Le epoche aggiuntive consolidano il risultato e riducono ulteriormente la loss.
- **MiniAlexNetV2 è la scelta migliore complessiva**.

## 6) Considerazioni per il mondo reale (YOLO)
Anche con classificazione quasi perfetta, il detection può degradare per:
- immagini rumorose/occluse;
- dataset non abbastanza ampio o bilanciato;
- annotazioni imperfette;
- soglie di confidenza e iperparametri non ottimizzati.

## 7) Azione consigliata
Per YOLO: aumentare il training (es. **500 epoche**), migliorare augmentation e qualità delle annotazioni.
