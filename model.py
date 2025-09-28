# model.py
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp

# Quantum device (simulator)
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# A small variational circuit: angle encoding -> variational layers -> measurement expectations
def variational_circuit(inputs, weights):
    # inputs: length == n_qubits
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    # weights shape: (layers, n_qubits, 3) or flattened — we will flatten and reshape inside
    l = weights.shape[0]
    idx = 0
    for layer in range(l):
        for q in range(n_qubits):
            qml.RX(weights[layer, q, 0], wires=q)
            qml.RY(weights[layer, q, 1], wires=q)
            qml.RZ(weights[layer, q, 2], wires=q)
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q+1])
    # return expectation values (one per qubit)
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# Wrap as a QNode that returns expectation vector
@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnode(inputs, weights):
    return variational_circuit(inputs, weights)

# Torch Module that uses qnode
class QuantumLayer(nn.Module):
    def _init_(self, n_qubits=4, n_layers=1):
        super()._init_()
        # weights param with shape (n_layers, n_qubits, 3)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        weight_shape = (n_layers, n_qubits, 3)
        # torch Parameter initialized small random
        self.weights = nn.Parameter(0.01 * torch.randn(*weight_shape))

    def forward(self, x):
        # x shape: (batch, features). We'll reduce features to n_qubits via a linear map
        batch = x.shape[0]
        # Map to n_qubits three-angle inputs. Simple approach: use a linear layer without bias
        # Note: do this on CPU/tensor device matching the qnode
        # Reduce feature dim to n_qubits using a linear transform
        # For simplicity, assume x already has at least n_qubits features
        if x.shape[1] < self.n_qubits:
            # pad with zeros
            pad = torch.zeros((batch, self.n_qubits - x.shape[1]), device=x.device)
            x_in = torch.cat([x, pad], dim=1)
        else:
            x_in = x[:, : self.n_qubits]

        # Normalize inputs to [-pi, pi]
        x_in = torch.tanh(x_in) * 3.1415

        outs = []
        for i in range(batch):
            # convert to numpy (qml will accept torch if using interface torch; ensure dtype float32)
            single = qnode(x_in[i], self.weights)
            outs.append(single)
        out = torch.stack(outs)
        return out  # shape (batch, n_qubits)


# Hybrid neural network: small classical front-end -> quantum layer -> classical classifier
class HybridNet(nn.Module):
    def _init_(self, input_dim=8, n_qubits=4, q_layers=1, hidden=32, n_classes=2):
        super()._init_()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.relu = nn.ReLU()
        self.q_layer = QuantumLayer(n_qubits=n_qubits, n_layers=q_layers)
        # classical head that maps quantum outputs to class logits
        self.fc2 = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.q_layer(x)     # returns (batch, n_qubits)
        x = self.fc2(x)
        return x