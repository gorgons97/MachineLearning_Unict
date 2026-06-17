# Sintesi del README

## Obiettivi principali
- Riorganizzare il notebook **Graphs** per generare tutti i grafici in un'unica esecuzione.
- Unificare le versioni di training in modo da lanciare tutti gli esperimenti contemporaneamente.

## Piano di lavoro
1. **Confronto iniziale** delle tre reti (AlexNet, MiniAlexNet, MiniAlexNetV2) con iper‑parametri di base (LR=0.001, momentum=0.6).
2. **Tuning completo** di MiniAlexNetV2 su tutte le combinazioni testate.
3. **Selezione** della combinazione ottimale (LR=0.003, momentum=0.6) e training esteso a **1200 epoche**.
4. **Eventuale riallenamento** di AlexNet e MiniAlexNet per un confronto finale “alla pari”.

---

## Risultati chiave (rapido)
- **MiniAlexNetV2 (LR=0.003, momentum=0.6)** è la migliore: acc. test ~0.95‑1.00, loss molto bassa, AUC≈1.0.
- A 1200 epoche il modello raggiunge **100 % di accuratezza** sia in training che in test, senza over‑fitting.
- AlexNet e MiniAlexNet hanno performance scarse (accuracy ≤ 0.55).

---

## Prossimi passi
- Consolidare tutti i grafici in un unico notebook (`Graphs.ipynb`).
- Automatizzare il training multi‑configurazione.
- Valutare il modello di detection (YOLO) con dataset più grande e iper‑parametri adeguati.
