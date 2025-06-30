# 05_Recurrent_Neural_Networks

Recurrent Neural Networks (RNNs) are a class of neural networks that are powerful for modeling sequence data such as time series or natural language. They are called "recurrent" because they perform the same task for every element of a sequence, with the output being depended on the previous computations.

### How they work:

RNNs have a "memory" which captures information about what has been calculated so far. They have loops in them, allowing information to persist.

### The Vanishing/Exploding Gradient Problem:

Standard RNNs suffer from the vanishing gradient problem, which means they have difficulty learning long-range dependencies. They can also suffer from the exploding gradient problem.

### Advanced RNN Architectures:

To overcome these issues, more advanced architectures were developed:

-   **Long Short-Term Memory (LSTM)**: A special kind of RNN, capable of learning long-term dependencies. LSTMs have a more complex cell structure with input, output, and forget gates, allowing them to selectively remember or forget information.
-   **Gated Recurrent Unit (GRU)**: A newer and slightly simpler alternative to the LSTM. It combines the forget and input gates into a single "update gate" and also merges the cell state and hidden state. 