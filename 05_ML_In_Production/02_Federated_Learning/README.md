# 02_Federated_Learning

Federated Learning is a machine learning setting where many clients (e.g., mobile phones, hospitals) collaboratively train a model under the coordination of a central server, while keeping the training data decentralized and private.

### Why Federated Learning?

It addresses fundamental issues of data privacy, security, and access rights. In many scenarios, data is sensitive and cannot be moved to a central server for training (e.g., medical records, financial data). Federated learning allows for models to be trained on this data without the data ever leaving the client device.

### How it works:

A typical federated learning process involves these steps:
1.  **Initialization**: The central server initializes a global model.
2.  **Distribution**: The server sends the current global model to a subset of clients.
3.  **Client Training**: Each selected client computes an update to the model by training it locally on its own data.
4.  **Aggregation**: The clients send their model updates (not their data) back to the central server.
5.  **Global Model Update**: The server aggregates the updates from the clients (e.g., by averaging them) to produce an improved global model.
6.  **Repeat**: The process is repeated for several rounds until the global model converges. 