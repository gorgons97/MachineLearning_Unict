# Reti Neurali

## Cose da fare

Creare e riadattare il notebook **Graphs** per poter creare i grafici tutti in una volta e risistemare le versioni di training per poterle lanciare tutte insieme contemporaneamente

1. Confronto iniziale delle tre reti con un caso comune, per esempio lr = 0.001 e momentum = 0.6.
1. Tuning completo della sola MiniAlexNetV2 su tutte le combinazioni testate.
1. Scelta della combinazione migliore per MiniAlexNetV2 e training esteso a 1200 epoche.
1. Eventuale riallenamento finale di AlexNet e MiniAlexNet solo se vuoi un confronto conclusivo “alla pari” con gli iperparametri migliori trovati.

---

# Report di Valutazione Modelli e Iperparametri - Versione Finale

## 1. Introduzione

Sono stati analizzati tre modelli di rete neurale (AlexNet, MiniAlexNet, MiniAlexNetV2) con diverse combinazioni di iperparametri (Learning Rate e Momento) per un totale di 200 epoche. I dati provengono da grafici di accuratezza, loss e curve ROC.

---

## 2. Valutazione dei Modelli

### 2.1 AlexNet (LR=0.001, Momento=0.6)

- **Accuratezza Test**: ~0.40 - 0.45
- **Loss**: Train e test loss rimangono elevate
- **Curve ROC**: Prestazioni appena sopra il caso casuale (TPR ~0.5)
- **Giudizio**: **PESSIMO**. Il modello non riesce ad apprendere. Probabilmente troppo grande per il dataset.

### 2.2 MiniAlexNet (LR=0.001, Momento=0.6)

- **Accuratezza Test**: ~0.55
- **Loss**: In discesa ma convergenza lenta
- **Curve ROC**: TPR ~0.5, capacità discriminatoria limitata
- **Giudizio**: **SCADENTE**. Meglio di AlexNet ma insufficiente.

### 2.3 MiniAlexNetV2 - Analisi Dettagliata

#### Combinazione LR=0.0001 (TUTTI I MOMENTI: 0.3, 0.6, 0.9)

- **Accuratezza Test**: 0.20 - 0.40
- **Loss (Test)**: 1.2 - 1.7
- **Giudizio**: **PESSIMO**. Learning rate troppo basso, il modello non impara nulla. Momento più alto (0.9) dà solo un lievissimo miglioramento ma rimane inaccettabile.

#### Combinazione LR=0.001

| Momento | Accuratezza Test | Loss (Test) | Giudizio                                           |
| ------- | ---------------- | ----------- | -------------------------------------------------- |
| 0.3     | ~0.60 - 0.65     | 0.8 - 1.0   | **DISCRETO**. Impara ma lentamente.                |
| 0.6     | ~0.75 - 0.80     | 0.5 - 0.7   | **BUONO**. Buon bilanciamento.                     |
| 0.9     | ~0.70 - 0.75     | 0.5 - 0.8   | **BUONO**. Simile a 0.6 ma con lievi oscillazioni. |

#### Combinazione LR=0.003

| Momento | Accuratezza Test | Loss (Test)   | Giudizio                              |
| ------- | ---------------- | ------------- | ------------------------------------- |
| 0.3     | ~0.85 - 0.90     | 0.3 - 0.5     | **OTTIMO**. Rapida convergenza.       |
| **0.6** | **~0.95 - 1.00** | **0.1 - 0.3** | **ECCELLENTE**. MIGLIORE IN ASSOLUTO. |
| 0.9     | ~0.90 - 0.95     | 0.2 - 0.4     | **OTTIMO**. Leggermente meno stabile. |

---

## 3. Classifica Finale Completa

| Posizione | Modello       | Learning Rate | Momento     | Accuratezza Test | Giudizio       |
| --------- | ------------- | ------------- | ----------- | ---------------- | -------------- |
| **1**     | MiniAlexNetV2 | **0.003**     | **0.6**     | **~0.95 - 1.00** | **ECCELLENTE** |
| 2         | MiniAlexNetV2 | 0.003         | 0.9         | ~0.90 - 0.95     | OTTIMO         |
| 3         | MiniAlexNetV2 | 0.003         | 0.3         | ~0.85 - 0.90     | OTTIMO         |
| 4         | MiniAlexNetV2 | 0.001         | 0.6         | ~0.75 - 0.80     | BUONO          |
| 5         | MiniAlexNetV2 | 0.001         | 0.9         | ~0.70 - 0.75     | BUONO          |
| 6         | MiniAlexNetV2 | 0.001         | 0.3         | ~0.60 - 0.65     | DISCRETO       |
| 7         | MiniAlexNet   | 0.001         | 0.6         | ~0.55            | SCADENTE       |
| 8         | AlexNet       | 0.001         | 0.6         | ~0.45            | PESSIMO        |
| 9-11      | MiniAlexNetV2 | 0.0001        | 0.3/0.6/0.9 | ~0.20 - 0.40     | PESSIMO        |

---

# Report di Valutazione Modelli e Iperparametri - Risultati a 1200 Epoche

## 1. Introduzione

Sono stati analizzati tre modelli di rete neurale (AlexNet, MiniAlexNet, MiniAlexNetV2) con diverse combinazioni di iperparametri. Questo report si concentra sui risultati ottenuti da **MiniAlexNetV2 con LR=0.003 e Momento=0.6** addestrato per **1200 epoche**, confrontandoli con i risultati a 200 epoche.

---

## 2. Risultati a 1200 Epoche - MiniAlexNetV2 (LR=0.003, Momento=0.6)

### 2.1 Accuratezza

- **Train Accuracy**: ~1.00 (100%)
- **Test Accuracy**: ~1.00 (100%)
- **Osservazioni**: 
    - L'accuratezza raggiunge il 100% già intorno a 400-500 epoche.
    - Train e test accuracy sono praticamente sovrapposte, indicando un eccellente bilanciamento.
    - Nessun segno di overfitting, anche dopo 1200 epoche.

### 2.2 Loss

- **Train Loss**: ~0.0 (prossima a zero)
- **Test Loss**: ~0.0 (prossima a zero)
- **Osservazioni**:
    - La loss scende rapidamente nelle prime 200 epoche.
    - Si stabilizza a valori prossimi a zero già intorno a 400-500 epoche.
    - Train e test loss rimangono vicine, confermando l'assenza di overfitting.

### 2.3 Curva ROC

| Classe    | AUC (Area Under Curve) |
| --------- | ---------------------- |
| Classe 0  | ~1.00                  |
| Classe 1  | ~1.00                  |
| Classe 2  | ~1.00                  |
| **Media** | **~1.00**              |

- **Osservazioni**:
    - TPR (True Positive Rate) raggiunge 1.0 per tutte le classi già a FPR molto bassi (~0.02-0.04).
    - Prestazioni perfette: il modello discrimina tutte le classi con accuratezza del 100%.
    - La curva ROC è praticamente sovrapposta all'angolo superiore sinistro (perfezione).

---

## 3. Confronto: 200 Epoche vs 1200 Epoche

| Metrica            | 200 Epoche   | 1200 Epoche | Miglioramento           |
| ------------------ | ------------ | ----------- | ----------------------- |
| **Test Accuracy**  | ~0.95 - 1.00 | ~1.00       | +0-5%                   |
| **Train Accuracy** | ~0.95 - 1.00 | ~1.00       | +0-5%                   |
| **Test Loss**      | ~0.1 - 0.3   | ~0.0        | Riduzione significativa |
| **Train Loss**     | ~0.1 - 0.3   | ~0.0        | Riduzione significativa |
| **AUC ROC**        | ~0.99        | ~1.00       | Perfezione raggiunta    |
| **Overfitting**    | Nessuno      | Nessuno     | Confermato              |

### Osservazioni sul Confronto

1. **L'accuratezza a 200 epoche era già eccellente** (0.95-1.00). Le 1000 epoche aggiuntive hanno portato a:

   - Stabilizzazione e perfezionamento dell'accuratezza (da 0.95-0.98 a 1.00).
   - Riduzione della loss a valori prossimi a zero.
   - Conferma della stabilità del modello.
1. **Il modello non ha mostrato overfitting**:

   - Train e test accuracy sono rimaste allineate.
   - Train e test loss sono rimaste vicine.
   - La generalizzazione è perfetta.
1. **Rendimenti decrescenti**:

   - Il grosso del miglioramento si è ottenuto nelle prime 200 epoche.
   - Le successive 1000 epoche hanno portato solo a raffinamenti marginali.

---

## 4. Classifica Finale Aggiornata

| Posizione | Modello       | Learning Rate | Momento     | Epoche   | Accuratezza Test | Giudizio     |
| --------- | ------------- | ------------- | ----------- | -------- | ---------------- | ------------ |
| **1**     | MiniAlexNetV2 | **0.003**     | **0.6**     | **1200** | **~1.00**        | **PERFETTO** |
| 2         | MiniAlexNetV2 | 0.003         | 0.6         | 200      | ~0.95 - 1.00     | ECCELLENTE   |
| 3         | MiniAlexNetV2 | 0.003         | 0.9         | 200      | ~0.90 - 0.95     | OTTIMO       |
| 4         | MiniAlexNetV2 | 0.003         | 0.3         | 200      | ~0.85 - 0.90     | OTTIMO       |
| 5         | MiniAlexNetV2 | 0.001         | 0.6         | 200      | ~0.75 - 0.80     | BUONO        |
| 6         | MiniAlexNetV2 | 0.001         | 0.9         | 200      | ~0.70 - 0.75     | BUONO        |
| 7         | MiniAlexNetV2 | 0.001         | 0.3         | 200      | ~0.60 - 0.65     | DISCRETO     |
| 8         | MiniAlexNet   | 0.001         | 0.6         | 200      | ~0.55            | SCADENTE     |
| 9         | AlexNet       | 0.001         | 0.6         | 200      | ~0.45            | PESSIMO      |
| 10-12     | MiniAlexNetV2 | 0.0001        | 0.3/0.6/0.9 | 200      | ~0.20 - 0.40     | PESSIMO      |

---

# Analisi dei Risultati  sul Mondo Reale

## 1. Il Problema nel Mondo Reale: YOLO e Sistemi Imperfetti

### Analisi del Problema

Hai descritto una situazione classica in ambito computer vision:

1. **Modello di classificazione (MiniAlexNetV2)**: Funziona perfettamente (100%).
2. **Modello di rilevamento (YOLO)**: Non trova sempre tutti i dati.

### Perché YOLO Fallisce?

| Fattore                         | Descrizione                                                | Impatto    |
| ------------------------------- | ---------------------------------------------------------- | ---------- |
| **Qualità dell'immagine**       | Immagini reali hanno rumore, variazioni di luce, sfocatura | Alto       |
| **Occlusioni**                  | Oggetti parzialmente nascosti o sovrapposti                | Alto       |
| **Scale diverse**               | Oggetti di dimensioni molto variabili                      | Medio-Alto |
| **Background complesso**        | Sfondi con pattern simili agli oggetti                     | Medio      |
| **Pochi dati di addestramento** | Dataset limitato o non rappresentativo                     | Alto       |
| **200 epoche**                  | Potrebbero non essere sufficienti per YOLO                 | Medio      |

### Differenza tra i Due Modelli

| Aspetto                    | MiniAlexNetV2 (Classificazione) | YOLO (Rilevamento)            |
| -------------------------- | ------------------------------- | ----------------------------- |
| **Compito**                | Classificare un'immagine intera | Trovare e localizzare oggetti |
| **Difficoltà**             | Bassa (immagini già ritagliate) | Alta (scene complesse)        |
| **Epoche necessarie**      | 200-500                         | 300-1000+                     |
| **Sensibilità al dataset** | Media                           | Molto alta                    |
| **Precisione raggiunta**   | 100%                            | Variabile                     |

---

## 2. Perché YOLO Non Trova Tutti i Dati?

### Cause Principali

#### A. **Dataset di Addestramento**

- **Poche immagini**: YOLO ha bisogno di migliaia di immagini per classe.
- **Bilanciamento scarso**: Poche istanze di alcuni oggetti.
- **Augmentation insufficiente**: Non abbastanza variazioni (rotazione, scala, luminosità).

#### B. **Iperparametri**

- **Learning Rate**: 200 epoche potrebbero essere poche per LR basso.
- **Batch Size**: Troppo grande o troppo piccolo influisce sulla convergenza.
- **Ancore (Anchors)**: Non ottimizzate per il tuo dataset specifico.

#### C. **Qualità delle Annotazioni**

- **Bounding box imprecise**: Box troppo grandi o troppo piccole.
- **Oggetti mancanti**: Alcuni oggetti non sono stati annotati nel training set.

#### D. **Threshold di Confidenza**

- **Troppo alta**: Il modello trova solo oggetti con alta confidenza.
- **Troppo bassa**: Troppi falsi positivi.

---

## 3. Soluzioni Pratiche per YOLO

addestrare YOLO considerando un valore di epoche uguale a 500  # Invece di 200