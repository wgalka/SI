### Brakujące dane

#### Brakujące dane mogą wystąpić z różnych powodów, takich jak:

1. Błędy ludzkie w zbieraniu danych: np. respondent nie udzielił odpowiedzi na pytanie w ankiecie.
2. Błędy w systemie zbierania danych: np. awaria sprzętu lub błąd oprogramowania mogą prowadzić do utraty danych.
3. Naturalna zmienność danych: niektóre cechy mogą być rzadkie lub niemożliwe do zarejestrowania w określonych warunkach.

#### Skutki brakujących danych:

1. Zmniejszona dokładność modeli: Brakujące dane mogą prowadzić do błędnych wniosków i obniżać skuteczność modeli.
2. Zniekształcenie wyników analiz: Brakujące dane mogą prowadzić do fałszywych wniosków podczas analizy danych.
3. Zwiększone ryzyko błędów: Analizując dane z brakującymi wartościami, istnieje większe ryzyko popełnienia błędów wynikających z niepełnych informacji.

#### Rozwiązania problemu brakujących danych obejmują:

1. Usunięcie obserwacji z brakującymi danymi: Może być stosowane, gdy brakujące dane są niewielkie lub gdy usuwanie ich nie wpływa znacząco na wyniki analizy.
2. Uzupełnienie brakujących danych: Można to zrobić, na przykład, poprzez uzupełnienie brakujących wartości średnią, medianą lub wartością modalną danej cechy, bądź stosując bardziej zaawansowane metody imputacji danych.
3. Uwzględnienie brakujących danych w modelach: Niektóre modele potrafią radzić sobie z brakującymi danymi bez potrzeby ich uzupełniania, na przykład drzewa decyzyjne.

### Skalowanie

## Zadania

Poniżej znajduje się funkcja usuwająca losowo zadaną wartość procentową komórek z macierzy. 

```python
def remove_random_cells(dataframe, percent):
    # Validate the percentage
    if percent <= 0 or percent >= 100:
        raise ValueError("Percentage must be greater than 0 and less than 100.")
    
    # Calculate the number of cells to remove
    num_cells = dataframe.size
    num_cells_to_remove = int((percent / 100) * num_cells)
    
    # Generate random row and column indices
    num_rows, num_cols = dataframe.shape
    random_row_indices = np.random.choice(num_rows, size=num_cells_to_remove, replace=True)
    random_col_indices = np.random.choice(num_cols, size=num_cells_to_remove, replace=True)
    
    # Set the selected cells to NaN
    dataframe.iloc[random_row_indices, random_col_indices] = np.nan
    
    return dataframe
```

Przygoruj 3 wersje zbioru danych iris (https://archive.ics.uci.edu/dataset/53/iris):
- bez usuwania wartości
- Zbiór gdzie usunięto 5% wartości
- Zbiór gdzie usunięto 10% wartości
- Zbiór gdzie usunięto 20% wartości

Nstępnie zaproponuj model sieci neuronowej który zostannie zwalidowany 10 krotną kroswalidacją na przygotowanych zbiorach danych. Rezultaty z każdego eksperymentu zbierz w arkuszu excel. W zbiorach gdzie usunięto wartości uzupełnij je na 4 rózne wybrane sposoby uzupełniania brakujących wartości (np. średnia, mediana, wartość zero, minimum). Sumarycznie ma powstać 13 zbiorów danych.

!UWAGA wyliczając średnią do uzupenienia wartości należy wyliczyć ją tylko na zbiorze treningowym.

Porównaj wyniki wskazując która metoda uzupełniania brakujących wartości sprawdzi się najlepiej.


## Preprocessing

Załóż konto na platformie kaggle.com a następnie zapoznaj się ze zbiorem danych: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer

data = pd.read_csv("train.csv", index_col="Id")
train_data = data.iloc[:, :-1]
y = data.iloc[:, -1]


class Preprocesser:

    # w polach klasy będziemy przechowywać modele i dane np średnie np. używane do uzupełnienia braków wartości
    def __init__(self):
        self.columns_with_nans_5 = []

    def fit(self, data):
        print("Nazwa kolumnt \t\t % brakujących wartości")
        for column_name in data.columns[:-1]:  # iterujemy po kolumnach oprócz klasy decyzyjnej
            nan_ratio = data[column_name].isna().sum() / data[column_name].__len__()
            print(f"{column_name} \t\t\t{nan_ratio}")
            if nan_ratio > 0.05: self.columns_with_nans_5.append(column_name)

    def transform(self, data):
        data = data.drop(columns=self.columns_with_nans_5)
        return data


prep = Preprocesser()
prep.fit(train_data)

train_prepared = prep.transform(train_data)
print("dane przed usunięcem kolumn gdzie ilość brakujacych wartości przekracza 5%:", train_data.shape)
print("dane po usunięciu:", train_prepared.shape)

test_data = pd.read_csv("test.csv", index_col="Id")
test_prepared = prep.transform(test_data)
print("dane przed usunięcem kolumn gdzie ilość brakujacych wartości przekracza 5%:", test_data.shape)
print("dane po usunięciu:", test_prepared.shape)

```
