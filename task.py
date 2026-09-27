import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
#from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cpu")


# ----------------------------
# 1. Neural Network Definition
# ----------------------------
class HeartModel(nn.Module):
    def __init__(self, input_dim):
        super(HeartModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# 2. Data Loading
# ----------------------------
def load_data(client_id=0, inject_noise=False):
    df = pd.read_csv("heart.csv")

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Convert target to binary (0 and 1)
    y = (y.astype(float) > 0).astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Simulate hospital split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42 + client_id
    )

    if inject_noise:
        noise = np.random.normal(0, 1, X_train.shape)
        X_train = X_train + noise

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train, y_train, X_test, y_test


# ----------------------------
# 3. Train Function
# ----------------------------
def train(model, X_train, y_train, epochs=2):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    losses = []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


# ----------------------------
# 4. Test Function
# ----------------------------
def test(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        preds = (outputs > 0.5).float()
        correct = (preds == y_test).sum().item()
        accuracy = correct / len(y_test)
    return accuracy


# ----------------------------
# 5. Weight Helpers
# ----------------------------
def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, weights):
    state_dict = model.state_dict()
    for key, val in zip(state_dict.keys(), weights):
        state_dict[key] = torch.tensor(val)
    model.load_state_dict(state_dict)