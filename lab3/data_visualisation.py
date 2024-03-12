import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

X = pd.read_csv("lab3/xor.data").to_numpy()
print(X)
y = pd.read_csv("lab3/xor.labels").to_numpy().ravel()
print(y)
colors = ["r", "g", "b"]




# Utwórz nowy wykres
fig, ax = plt.subplots()

# Narysuj punkty na linii decyzyjnej

# Narysuj punkty danych, każda klasa oznacozna innym kolorem
for index, label in enumerate(np.unique(y)):
    print("label",label)
    data = X[y == label]
    ax.scatter(data[:,0], data[:,1], color = colors[index], label=label)

# Dodaj etykiety osi
ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')

# Dodaj tytuł
ax.set_title('Decision Boundary')

# Dodaj legendę
ax.legend()

# Oznaczneia osi X i Y
ax.set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

# Ustawienie limitu dla osi X i Y
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)

# Wyśietlenie siatki wykresu
plt.grid()
plt.title("XOR GATE")

plt.savefig("lab3/xor.svg")
# Wyświetlenie wykresu
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Parametry danych
num_samples_per_class = 100
noise_factor = 0.1

# Generowanie punktów dla pierwszej klasy
class1_x = np.random.normal(2, 1, num_samples_per_class)
class1_y = np.random.normal(2, 1, num_samples_per_class)

# Generowanie punktów dla drugiej klasy
class2_x = np.random.normal(5, 1, num_samples_per_class)
class2_y = np.random.normal(5, 1, num_samples_per_class)

# Tworzenie macierzy cech i etykiet
X = np.vstack([np.hstack([class1_x, class2_x]), np.hstack([class1_y, class2_y])]).T
y = np.hstack([np.zeros(num_samples_per_class), np.ones(num_samples_per_class)])

# Dodanie szumu do danych
X += np.random.normal(0, noise_factor, X.shape)

# Wykres danych
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Generated Data')
plt.show()