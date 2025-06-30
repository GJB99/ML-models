# 02_Gated_Recurrent_Unit

The Gated Recurrent Unit (GRU) is a newer generation of Recurrent Neural Networks and is pretty similar to an LSTM. GRU gets rid of the cell state and uses the hidden state to transfer information.

### The Core Idea:

A GRU has two gates:
1.  **Update Gate**: This gate acts similar to the forget and input gate of an LSTM. It decides what information to throw away and what new information to add.
2.  **Reset Gate**: This gate is used to decide how much of the past information to forget.

### GRU vs LSTM:

-   **Simpler Architecture**: GRUs have fewer parameters than LSTMs, as they lack an output gate.
-   **Faster Training**: With fewer parameters, GRUs can be slightly faster to train.
-   **Performance**: Their performance is generally comparable to LSTMs, and the choice between them often depends on the specific dataset and task. There is no clear winner. 