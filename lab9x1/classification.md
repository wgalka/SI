# Sieci neuronowe w zadaniach identyfikacji i klasyfikacji 

## Problem regresji
Regresja to jedno z podstawowych zagadnień w uczeniu maszynowym, które zajmuje się przewidywaniem wartości ciągłych na podstawie danych wejściowych. W problemie regresji, celem jest znalezienie zależności pomiędzy zmiennymi niezależnymi (cechami) a zmienną zależną (wartością przewidywaną). Przykładem może być przewidywanie ceny domu na podstawie jego cech takich jak powierzchnia, liczba pokoi, lokalizacja itp.

### Cechy problemu regresji:

1. Zmienna zależna jest ciągła.
2. Cel to znalezienie funkcji, która najlepiej odwzorowuje zależności między cechami a wartością przewidywaną.
3. Do oceny jakości modelu stosuje się metryki takie jak błąd średniokwadratowy (MSE) lub współczynnik determinacji ($$R^2 score$$).

## Problem klasyfikacji

Klasyfikacja jest innym ważnym zadaniem w uczeniu maszynowym, które polega na przypisywaniu obiektów do określonych klas na podstawie ich cech. Celem klasyfikacji jest znalezienie modelu, który potrafi rozróżniać różne klasy na podstawie danych treningowych, aby później przewidywać klasy nowych obiektów. Przykładem może być klasyfikacja e-maili jako spam lub nie-spam na podstawie ich treści i nagłówka.

### Cechy problemu klasyfikacji:

1. Zmienna zależna jest dyskretna i składa się z ograniczonego zbioru klas.
2. Cel to znalezienie modelu, który efektywnie rozdziela obiekty na różne klasy na podstawie ich cech.
3. Do oceny jakości modelu stosuje się metryki takie jak dokładność, krzywa ROC, czy macierz pomyłek.

## Różnice między regresją a klasyfikacją

- Typ zmiennej zależnej: W regresji zmienna zależna jest ciągła, podczas gdy w klasyfikacji jest dyskretna.
- Cel: W regresji celem jest przewidywanie wartości ciągłych, natomiast w klasyfikacji celem jest przypisanie obiektów do określonych klas.
- Metryki oceny: W regresji stosuje się metryki takie jak błąd średniokwadratowy (MSE) lub współczynnik determinacji ($$R^2 score$$ ), podczas gdy w klasyfikacji używane są metryki takie jak dokładność, krzywa ROC czy macierz pomyłek.

W obu przypadkach, celem jest znalezienie modelu, który dobrze odwzorowuje zależności między cechami a wartością przewidywaną lub między cechami a klasami, jednakże techniki i metody stosowane w przypadku regresji różnią się od tych stosowanych w przypadku klasyfikacji.


## Macierz pomyłek

Macierz pomyłek, znana również jako tabela kontyngencji, to narzędzie używane do wizualizacji wyników klasyfikacji binarnej lub wieloklasowej. Jest to tabela, która pokazuje liczbę poprawnie i niepoprawnie sklasyfikowanych próbek przez model klasyfikacyjny w odniesieniu do rzeczywistych klas.

### Struktura macierzy pomyłek:
W przypadku klasyfikacji binarnej (dwuklasowej), macierz pomyłek składa się z 2x2 komórek:


|                    | Klasyfikacja pozytywna | Klasyfikacja negatywna |
|--------------------|------------------------|------------------------|
| Rzeczywista pozytywna | True Positive (TP)   | False Negative (FN)    |
| Rzeczywista negatywna | False Positive (FP)  | True Negative (TN)     |


- **True Positive (TP):** Liczba obserwacji, które zostały poprawnie sklasyfikowane jako pozytywne.
- **False Negative (FN):** Liczba obserwacji, które zostały błędnie sklasyfikowane jako negatywne, podczas gdy powinny być sklasyfikowane jako pozytywne.
- **False Positive (FP):** Liczba obserwacji, które zostały błędnie sklasyfikowane jako pozytywne, podczas gdy powinny być sklasyfikowane jako negatywne.
- **True Negative (TN):** Liczba obserwacji, które zostały poprawnie sklasyfikowane jako negatywne.


**W przypadku klasyfikacji wieloklasowej, macierz pomyłek będzie miała wymiary NxN, gdzie N to liczba klas.**

Macierz pomyłek pozwala na ocenę wydajności modelu klasyfikacyjnego poprzez obliczanie różnych metryk, takich jak:

- **Dokładność (Accuracy):** Stosunek poprawnie sklasyfikowanych próbek do ogólnej liczby próbek.

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

- **Czułość (Sensitivity):** Stosunek poprawnie sklasyfikowanych pozytywnych próbek do ogólnej liczby rzeczywistych pozytywnych próbek.

$$
\text{Sensitivity} = \frac{TP}{TP + FN}
$$

- **Specyficzność (Specificity):** Stosunek poprawnie sklasyfikowanych negatywnych próbek do ogólnej liczby rzeczywistych negatywnych próbek.

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

- **Precyzja (Precision):** Stosunek poprawnie sklasyfikowanych pozytywnych próbek do ogólnej liczby pozytywnych klasyfikacji.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

- **F1 Score:** Średnia harmoniczna między precyzją i czułością.

$$
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Sensitivity}}{\text{Precision} + \text{Sensitivity}}
$$


## Tworzenie macierzy pomyłek w jezyku python z wykorzystaniem `scikit-learn`

```python
from sklearn.metrics import confusion_matrix

# Przykładowe etykiety rzeczywiste i przewidziane przez model
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 1, 1, 0, 1, 0]

# Obliczanie macierzy pomyłek
cm = confusion_matrix(y_true, y_pred)

print(f"Macierz pomyłek: {cm}")
```

## Obliczanie miar jakości klasyfikacji w jezyku pytohn z wykorzystaniem scikit-learn

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Przykładowe etykiety rzeczywiste i przewidziane przez model
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 1, 1, 0, 1, 0]

# Dokładność (Accuracy)
accuracy = accuracy_score(y_true, y_pred)
print("Dokładność (Accuracy):", accuracy)

# Precyzja (Precision)
precision = precision_score(y_true, y_pred)
print("Precyzja (Precision):", precision)

# Czułość (Recall)
recall = recall_score(y_true, y_pred)
print("Czułość (Recall):", recall)

# F1-score
f1 = f1_score(y_true, y_pred)
print("F1-score:", f1)
```

## Zadania

Zadaniem jest stworzenie modelu klasyfikacyjnego opartego na sieci neuronowej, który będzie przewidywał, czy nowotwór piersi jest złośliwy czy też nie, na podstawie cech opisujących komórki rakowe.

### Dane:
Wykorzystaj zbiór danych [Breast Cancer Wisconsin Original](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original).

### Przygotowanie danych:

- Wczytaj dane z pliku CSV.
- Przeprowadź czyszczenie danych, usuń brakujące wartości (jeśli istnieją).
- Podziel dane na zbiór treningowy i testowy.

### Budowa modelu:

- Stwórz sieć neuronową wykorzystując bibliotekę TensorFlow lub PyTorch.
- Zdefiniuj warstwy sieci, funkcje aktywacji i inne parametry.
- Skonfiguruj proces uczenia, wybierz odpowiednią funkcję kosztu i optymalizator.

### Uczenie modelu:

- Wytrenuj model na zbiorze treningowym.
- Monitoruj wskaźniki wydajności, takie jak dokładność, funkcja kosztu itp.

### Ocena modelu:

- Przetestuj model na zbiorze testowym.
- Oblicz dokładność predykcji i inne metryki jakości klasyfikacji.
- Zweryfikuj wyniki, analizując macierz pomyłek.

### Optymalizacja modelu:

- Ewentualnie dostosuj architekturę sieci, hiperparametry lub techniki regularyzacji, aby poprawić wydajność modelu.
- Ponownie przetestuj model i porównaj wyniki.

### Podsumowanie:

- Przedstaw wyniki eksperymentów w sposób czytelny i zrozumiały.
- Wnioski dotyczące wydajności modelu i ewentualnych kierunków dalszych badań.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Przykładowe dane - zastąp je danymi z Breast Cancer Wisconsin Original
X = np.random.randn(1000, 10)  # 1000 próbek, 10 cech
y = np.random.randint(0, 2, size=1000)  # Klasy: 0 lub 1

# Przykładowa definicja modelu sieci neuronowej
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Ustawienia hiperparametrów
input_size = 10
hidden_size = 64
output_size = 2
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Inne funkcje straty https://pytorch.org/docs/stable/nn.html#loss-functions
criterion = nn.CrossEntropyLoss()
# Inne algorytmy optymalizacyjne https://pytorch.org/docs/stable/optim.html
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# K-krotna walidacja krzyżowa
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Przygotowanie danych do tensorów
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Inicjalizacja modelu
    model = NeuralNetwork(input_size, hidden_size, output_size)
    
    # Trenowanie modelu
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Testowanie modelu
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = accuracy_score(y_test, predicted.numpy())
        accuracies.append(accuracy)

# Obliczenie średniej dokładności
mean_accuracy = np.mean(accuracies)
print(f'Średnia dokładność: {mean_accuracy:.4f}')
```
