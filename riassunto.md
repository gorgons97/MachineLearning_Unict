# Riassunto del progetto
Il README descrive un lavoro di confronto tra tre reti neurali (**AlexNet**, **MiniAlexNet**, **MiniAlexNetV2**) e un tuning degli iperparametri (learning rate e momentum), con analisi su accuratezza, loss e curve ROC.

## Obiettivi operativi
- Confrontare inizialmente i tre modelli con una configurazione comune (LR=0.001, momentum=0.6).
- Eseguire il tuning completo di **MiniAlexNetV2** su più combinazioni di iperparametri.
- Selezionare la combinazione migliore e fare un training esteso fino a **1200 epoche**.
- Eventualmente riallenare gli altri modelli per un confronto finale “alla pari”.

## Esito del confronto (200 epoche)
- **AlexNet**: prestazioni molto basse, modello giudicato non adatto al dataset.
- **MiniAlexNet**: leggero miglioramento rispetto ad AlexNet, ma ancora insufficiente.
- **MiniAlexNetV2**: modello nettamente migliore, soprattutto con learning rate più alto.

Configurazione migliore a 200 epoche:
- **MiniAlexNetV2, LR=0.003, momentum=0.6**
- Accuratezza test ~0.95–1.00, loss bassa, ROC eccellente.

## Risultati estesi (1200 epoche)
Con **MiniAlexNetV2 (LR=0.003, momentum=0.6)**:
- Train e test accuracy arrivano a ~1.00.
- Train e test loss scendono quasi a zero.
- ROC media ~1.00.
- Nessun segnale evidente di overfitting.

Conclusione: il grosso del miglioramento avviene entro le prime 200 epoche; le epoche aggiuntive consolidano e rifiniscono il risultato.

## Nota sul mondo reale (YOLO)
Il README evidenzia che, anche con classificazione quasi perfetta, un detector come **YOLO** può fallire più spesso in scenari reali per:
- qualità immagini, occlusioni, variazioni di scala/sfondo;
- dataset e annotazioni non ottimali;
- iperparametri e soglie di confidenza non ben tarati.

Indicazione pratica finale: per YOLO è consigliato aumentare il training (es. **500 epoche** invece di 200) e curare meglio dati/annotazioni/augmentation.
