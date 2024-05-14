# Aproksymacja sztuczną siecią neuronową

## Budowa sieci neuronowej w Pytorch

https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html

Zacznijmy od przykładu sieci neuronowej w `Pytorch` składającej się z 1 warstwy wejściowej, 1 warstwy ukrytej ukrytej i warstwy wyjściowej. Warstwa ukryta będzie mieć funkcję sigmoidalną aktywacji natomiast wyjściowa Relu.

```Python
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
```

W tym przykładzie:

- Definiujemy klasę OneLayerNN, która dziedziczy po `nn.Module` i reprezentuje naszą sieć neuronową składającą się z warstwy wejściowej, jednej warstwy ukrytej i warstwy wyjściowej. Warstwa ukryta ma rozmiar hidden_size i używa funkcji aktywacji sigmoidalnej (`nn.Sigmoid()`). Na końcu mamy warstwę wyjściową o rozmiarze output_size, która używa funkcji aktywacji ReLU (`nn.ReLU()`).
- W konstruktorze (metoda `__init__()`) definiujemy warstwy sieci oraz funkcje przejścia.
    - fc1 - jest to nasza pierwsza warstwa sieci. Funkcja `nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)` wykonuje mnożenie wejścia przez wagi, sumowanie oraz dodaje bias w poszczeólnych neuronoach wyjściowych. Jako `in_features` podajemy rozmiar warstwy wejściowej (ile danych wejściowych przyjmie sieć), jako `out_features` definiujemy ilość neuronów (ile wyjść będzie miała warstwa ukryta).
    - sigmoid - gdy będziemy mieć wynik (sumę pomnożenia wag i wejścia neuronów + bias) do każdego neuronu będzimey aplikować funkcję sigmoidalną - `nn.Sigmoid()`.
    - fc2 - jest to warstwa wyjściowa. Tak jak w pierwszej warstwie do obliczenia  input * W.T + b używamy funkcji `nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)` gdzie jako `in_features` podajemy ilość wyjść (neuronów) z `fc1` a jako `out_features` ile neuronów ma być w warstwie wyjściowej.
    - relu - funkcja aktywacji w warstwie wyjściowej - `nn.ReLU()`
- funkcja `forward()` symuluje przejścia pomiędzy warstwami. Kolejno, wektor wejściowy trafia do `fc1`, wynik `fc1` trafia do funkcji `sigmoid`, z funkcji `sigmoid` przekazujemy dane do `fc2` i finalnie wynik `fc2` przekazujemy do funkcji aktywacji `relu` co daje nam finalny output sieci neuronowej.
- Następnie przeprowadzamy trening modelu przez 100 epok, obliczając stratę i aktualizując wagi za pomocą algorytmu SGD. Również za pomocą gotowych funkcji.


## Miary jakości modelu w problemie regresji

W problemie regresji celem jest przewidywanie wartości ciągłych zmiennych, na przykład prognozowanie ceny domu na podstawie jego cech, przewidywanie temperatury na podstawie danych meteorologicznych, czy prognozowanie sprzedaży produktu na podstawie danych historycznych.

Popularne miary służące do oceny jakości modelu regresyjnego:

### Mean Squared Error (MSE)
Jest to jedna z najczęściej stosowanych miar do oceny wydajności modeli regresji. MSE oblicza średnią kwadratów różnicy między wartościami przewidywanymi przez model a rzeczywistymi wartościami etykiet. Im niższa wartość MSE, tym lepiej model dopasowuje się do danych.

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

### Mean Absolute Error (MAE)

MAE oblicza średnią wartość bezwzględnej różnicy między wartościami przewidywanymi przez model a rzeczywistymi wartościami etykiet. Jest mniej wrażliwy na wartości odstające niż MSE, ponieważ nie podnosi błędów do kwadratu.

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

### Root Mean Squared Error (RMSE):

RMSE jest podobny do MSE, ale zamiast obliczać średnią kwadratów różnic, oblicza pierwiastek kwadratowy z MSE. Oznacza to, że RMSE mierzy średnią różnicę między wartościami przewidywanymi przez model a rzeczywistymi wartościami etykiet, ale zwraca wynik w tych samych jednostkach co oryginalne dane, co ułatwia interpretację. Im niższa wartość RMSE, tym lepiej model dopasowuje się do danych.


$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

### $R^2$ Score

$R^2$ Score, znany również jako współczynnik determinacji, mierzy, jak dobrze model regresji dopasowuje się do danych. Przyjmuje wartość między 0 a 1, gdzie 1 oznacza idealne dopasowanie modelu do danych, a 0 oznacza brak dopasowania. R^2 Score jest szczególnie przydatny do porównywania różnych modeli regresji.

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$



Gdzie:
- $n$ liczba próbek w zbiorze danych,
- $y_i$ prawdziwa decyzja dla i tej próbki,
- $\hat{y}_i$ wartość przewidziana przez model dla itej próbki,
- $\bar{y}$ średnia wartość decyzji w zbiorze danych

***

W bibliotece `scikit-learn` znajdują się gotowe implementacje powyższych funkcji.

https://scikit-learn.org/stable/

- [MSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)

- [RMSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html)

- [MAE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)

- [R^2 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)


Poniżej przykładowy kod który trenuje sieć neuronową aż osiągnie zadany błąd MSE:

```Python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Przygotowanie danych
x = np.arange(0, 4*np.pi, 0.2)  # lub np.arange(0, 4*np.pi, 0.1) dla większej gęstości punktów
y = np.sin(x)
x_tensor = torch.FloatTensor(x.reshape(-1, 1))
y_tensor = torch.FloatTensor(y.reshape(-1, 1))

# Podział na zbiór treningowy i testowy
split_idx = int(0.8 * len(x))
x_train, x_test = x_tensor[:split_idx], x_tensor[split_idx:]
y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]

# 2. Zdefiniowanie modelu sieci neuronowej
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()  # Funkcja aktywacji w warstwie ukrytej

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. Trenowanie modelu
def train_model(hidden_size, learning_rate, num_epochs, mse_threshold):
    model = NeuralNet(1, hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
            
        # Warunek przerwania treningu, jeśli błąd spadnie poniżej progu
        if loss.item() <= mse_threshold:
            print(f'MSE Threshold reached at epoch {epoch+1}. Stopping training.')
            break

    return model

# Parametry
hidden_size = 10
learning_rate = 0.01
num_epochs = 1000
mse_threshold = 0.01  # Próg błędu MSE

# Trenowanie modelu
model = train_model(hidden_size, learning_rate, num_epochs, mse_threshold)

# Ocena modelu
model.eval()
with torch.no_grad():
    outputs = model(x_test)
    mse = nn.MSELoss()(outputs, y_test)
print(f'Mean Squared Error (Test): {mse.item()}')

# Wykres
plt.plot(x_test.numpy(), y_test.numpy(), label='True')
plt.plot(x_test.numpy(), outputs.numpy(), label='Predicted')
plt.legend()
plt.show()
```

### Zadanie 1: Interpolacja funkcji \( y = \sin(x) \) za pomocą sieci neuronowej

**Opis zadania:**
Celem tego zadania jest projektowanie sieci neuronowej do interpolacji funkcji sinusoidalnej \( y = \sin(x) \), przy użyciu punktów (x, y) w postaci danych uczących. Przeprowadzimy badania mające na celu zrozumienie wpływu różnych czynników na jakość interpolacji, w tym liczby punktów w zbiorze uczącym, liczby neuronów w warstwie wejściowej oraz wartości funkcji celu (kryterium zatrzymania trenowania). Do oceny jakości interpolacji użyjemy błędu średniokwadratowego.

**Badane czynniki:**
1. Liczba punktów w zbiorze uczącym:
   - \( x = 0:0.2:4\pi \)
   - \( x = 0:0.1:4\pi \)

2. Liczba neuronów w warstwie ukrytej:
   - 3, 5, 7, 10, 30, 50, 100

3. Wartości funkcji celu (kryterium zatrzymania trenowania):
   - 0.1
   - 0.01
   - 0.001
   - 0.0001

## Zadania
### Zadanie 1: Interpolacja funkcji $ y = \sin(x) $ za pomocą sieci neuronowej

**Opis zadania:**
Celem tego zadania jest projektowanie sieci neuronowej do interpolacji funkcji sinusoidalnej $ y = \sin(x) $, przy użyciu punktów (x, y) w postaci danych uczących. Przeprowadzimy badania mające na celu zrozumienie wpływu różnych czynników na jakość interpolacji, w tym liczby punktów w zbiorze uczącym, liczby neuronów w warstwie wejściowej oraz wartości funkcji celu (kryterium zatrzymania trenowania). Do oceny jakości interpolacji użyjemy błędu średniokwadratowego.

**Badane czynniki:**
1. Liczba punktów w zbiorze uczącym:
   - $ x = 0:0.2:4\pi $
   - $ x = 0:0.1:4\pi $
2. Liczba neuronów w warstwie ukrytej:
   - 3, 5, 7, 10, 30, 50, 100

3. Wartości funkcji celu (kryterium zatrzymania trenowania):
   - 0.1
   - 0.01
   - 0.001
   - 0.0001

**Metoda oceny jakości interpolacji:**
Do oceny jakości interpolacji użyjemy błędu średniokwadratowego (MSE).

**Kroki do wykonania:**
1. Przygotowanie danych: Generacja punktów (x, y) na podstawie funkcji $ y = \sin(x) $.
2. Definiowanie modelu sieci neuronowej: Stworzenie modelu z jedną warstwą ukrytą.
3. Trenowanie modelu: Trenowanie modelu dla różnych wartości liczby neuronów w warstwie ukrytej i wartości funkcji celu. W przypadku, gdy wartość funkcji celu spadnie poniżej ustalonego progu, trenowanie zostanie przerwane.
4. Ocena modelu: Ocena jakości interpolacji na zbiorze testowym.
5. Analiza wyników: Przedstawienie wyników na wykresach i wnioski dotyczące wpływu badanych czynników na jakość interpolacji.

**Wyniki:**
1. Tabelaryczne zestawienie testów.
2. Opisane wykresy przedstawiające punkty uczące i przebieg funkcji dla każdego przypadku.
3. Wnioski dotyczące wpływu:
   - Liczby neuronów w warstwie wejściowej.
   - Wartości funkcji celu.
   - Liczby punktów w zbiorze uczącym na jakość interpolacji wykonanej przez sieć neuronową.

