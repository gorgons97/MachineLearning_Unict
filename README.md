# Reti Neurali

## Cose da fare

Creare e riadattare il notebook **Graphs** per poter creare i grafici tutti in una volta e risistemare le versioni di training per poterle lanciare tutte insieme contemporaneamente

1. Confronto iniziale delle tre reti con un caso comune, per esempio lr = 0.001 e momentum = 0.6.
1. Tuning completo della sola MiniAlexNetV2 su tutte le combinazioni testate.
1. Scelta della combinazione migliore per MiniAlexNetV2 e training esteso a 1200 epoche.
1. Eventuale riallenamento finale di AlexNet e MiniAlexNet solo se vuoi un confronto conclusivo “alla pari” con gli iperparametri migliori trovati.

## Analisi delle configurazioni

Dall’analisi congiunta delle curve di accuracy, delle curve di loss e della ROC curve, la configurazione che risulta complessivamente migliore è quella con **learning rate = 0.003** e **momentum = 0.9**, poiché mostra le prestazioni più elevate in termini di accuratezza di test e la migliore capacità discriminante tra le classi.

### Classifica dalla migliore alla peggiore

1. **Learning rate = 0.003, Momentum = 0.9**  
   È la configurazione migliore in assoluto. La test accuracy raggiunge i valori più alti tra le configurazioni analizzate, e la ROC curve mostra le AUC migliori, pari a 0.660 per la classe 0, 0.808 per la classe 1 e 0.761 per la classe 2.
1. **Learning rate = 0.003, Momentum = 0.6**  
   Mostra prestazioni elevate e abbastanza stabili. La test accuracy è leggermente inferiore rispetto alla configurazione precedente, ma resta tra le più alte e con un andamento complessivamente buono.
1. **Learning rate = 0.003, Momentum = 0.3**  
   È una configurazione valida, ma leggermente meno efficace delle due precedenti. L’accuracy di test cresce in modo regolare, ma si ferma su valori un po’ più bassi.
1. **Learning rate = 0.001, Momentum = 0.9**  
   Offre un buon comportamento generale, con una test accuracy discreta e curve abbastanza stabili. Tuttavia, risulta inferiore rispetto alle configurazioni con learning rate 0.003.
1. **Learning rate = 0.001, Momentum = 0.6**  
   Configurazione intermedia, con prestazioni accettabili ma non ottimali. Le curve mostrano un apprendimento corretto, ma una capacità di generalizzazione inferiore rispetto ai casi migliori.
1. **Learning rate = 0.001, Momentum = 0.3**  
   L’apprendimento è più lento e la test accuracy finale è più contenuta. Il modello converge, ma con risultati inferiori rispetto ai momentum più alti.
1. **Learning rate = 0.0001, Momentum = 0.9**  
   La rete apprende in modo lento e raggiunge prestazioni modeste. La test accuracy cresce, ma resta sensibilmente più bassa rispetto alle configurazioni con learning rate maggiore.
1. **Learning rate = 0.0001, Momentum = 0.6**  
   Configurazione poco efficace, con apprendimento limitato e prestazioni finali modeste. Risulta inferiore rispetto ai casi con learning rate 0.001 e 0.003.
1. **Learning rate = 0.0001, Momentum = 0.3**  
   È una delle peggiori configurazioni analizzate. La test accuracy resta molto bassa e la loss di test tende a peggiorare, indicando scarsa capacità di generalizzazione.
1. **Configurazione MiniAlexNetV2 a 1200 epoche**  
   Pur mostrando un miglioramento progressivo dell’accuracy di training, la test accuracy resta moderata e la loss di test cresce nel tempo, evidenziando un marcato fenomeno di overfitting. La ROC curve finale resta comunque discreta, ma la configurazione non è competitiva rispetto ai migliori setup a 200 epoche.

## Considerazioni finali

Nel complesso, i grafici mostrano che valori di **learning rate più alti**, in particolare **0.003**, producono i risultati migliori, soprattutto se combinati con un **momentum elevato** . Al contrario, valori di learning rate troppo bassi, come **0.0001**, tendono a rallentare eccessivamente l’apprendimento e a limitare le prestazioni del modello.