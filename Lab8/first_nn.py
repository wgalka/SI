import torch
import torch.nn as nn

import matplotlib.pyplot as plt

# Definicja sieci neuronowej
class ExampleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ExampleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.relu(out)
        return out

# Parametry sieci
input_size = 10
hidden_size = 20
output_size = 1

# Przykładowe dane
X = torch.randn(100, input_size)
y = torch.randint(0, 2, (100, output_size)).float()

# Inicjalizacja modelu
model = ExampleNN(input_size, hidden_size, output_size)

# Definicja funkcji straty i optymalizatora
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

losses = []

# Trening modelu
for epoch in range(1000):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass i aktualizacja wag
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Dodanie aktualnego spadku błędu do listy
    losses.append(loss.item())

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Wykres błędu sieci neuronowej w danej epoce
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()