# client.py
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model import HybridNet

# Create a small synthetic dataset per client
def make_local_data(seed=0, n_samples=200, n_features=8, n_classes=2):
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=6, n_redundant=0, n_classes=n_classes,
                               random_state=seed)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))
    return train_ds, test_ds

# Flower client
class FLClient(fl.client.NumPyClient):
    def _init_(self, model, trainloader, testloader, device):
        self.model = model.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self):
        # return model parameters as a list of numpy arrays
        return [val.cpu().detach().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # set model params from list of numpy arrays
        params_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v) for k, v in params_dict.items()})

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # one epoch for demo
        for epoch in range(1):
            for X, y in self.trainloader:
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                optimizer.step()
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        total = 0
        loss_total = 0.0
        with torch.no_grad():
            for X, y in self.testloader:
                X = X.to(self.device)
                y = y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss_total += float(loss.item()) * X.size(0)
                preds = outputs.argmax(dim=1)
                correct += int((preds == y).sum().item())
                total += X.size(0)
        return float(loss_total / total), total, {"accuracy": correct / total}

if _name_ == "_main_":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, default=0, help="client id (seed)")
    parser.add_argument("--samples", type=int, default=200, help="samples per client")
    args = parser.parse_args()

    device = torch.device("cpu")
    train_ds, test_ds = make_local_data(seed=args.cid, n_samples=args.samples, n_features=8)
    trainloader = DataLoader(train_ds, batch_size=16, shuffle=True)
    testloader  = DataLoader(test_ds, batch_size=32)

    model = HybridNet(input_dim=8, n_qubits=4, q_layers=1, hidden=32, n_classes=2)

    client = FLClient(model, trainloader, testloader, device)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)