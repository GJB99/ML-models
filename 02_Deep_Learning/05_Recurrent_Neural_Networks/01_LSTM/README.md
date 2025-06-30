# 01_Long_Short-Term_Memory

Long Short-Term Memory (LSTM) networks are a special kind of RNN, capable of learning long-term dependencies. They were introduced to solve the vanishing gradient problem of traditional RNNs.

### The Core Idea: Gates

The key to LSTMs is the **cell state**, a horizontal line running down the top of the diagram. The cell state is like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It's very easy for information to just flow along it unchanged.

LSTMs have the ability to remove or add information to the cell state, carefully regulated by structures called **gates**. Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation.

An LSTM has three of these gates to protect and control the cell state:
1.  **Forget Gate**: Decides what information we're going to throw away from the cell state.
2.  **Input Gate**: Decides what new information we're going to store in the cell state.
3.  **Output Gate**: Decides what we're going to output. This output will be based on our cell state, but will be a filtered version. 