# Python - podstawowe operacje

W kontekście Sztucznej Inteligencji, macierze są fundamentalnym narzędziem do przechowywania i przetwarzania danych. Są strukturami danych składającymi się z wierszy i kolumn, które umożliwiają organizację danych w postaci tabelarycznej. W kontekście języka python macierz będą tworzyć listy zawarte w liście "opakowującej".

<div class="codeblock-label">pure python</div>

```python
# Macierz o wymiarach 4 wiersze 3 kolumny
matrix = [[1, 2, 3],
    [5, 3, 6],
    [4, 5, 6],
    [7, 8, 9]]
```

Gdy zachodzi potrzeba zastosowania większej liczby wymiarów niż posiada macierz stosuje się Tensory które są rozszerzeniem pojęcia macierzy, gdzie macierz to specjalny przypadek tensora dwuwymiarowego. W przetwarzaniu obrazów, obrazy cyfrowe są często reprezentowane jako tensory, gdzie dwa pierwsze wymiary odpowiadają za wysokość, szerokość a kolejne za kanały kolorów (np. czerwony, zielony, niebieski).

Oto przykładowa trójwymiarowa macierz reprezentująca niewielki obraz RGB o rozmiarze 3x3 pikseli:

<div class="codeblock-label">pure python</div>

```python
image = [
    [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    [[255, 255, 0], [255, 255, 255], [0, 255, 255]],
    [[128, 128, 128], [0, 0, 0], [255, 128, 128]]
]
```

Wskaźniki indeksujące w macierzach i tensorach są sposobem określania konkretnych elementów w tych strukturach danych. Pozwalają one na odwoływanie się do określonych wartości poprzez podanie odpowiednich indeksów, które określają położenie elementu w macierzy lub tensorze.

<div class="codeblock-label">pure python</div>

```python
# Macierz A
A = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Tensor T
T = [
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
]

# Odwołanie do elementów macierzy A i tensora T za pomocą wskaźników indeksujących
element_A = A[1][2]  # Odwołanie do elementu w drugim wierszu i trzeciej kolumnie macierzy A
element_T = T[0][1][0]  # Odwołanie do elementu w pierwszym wymiarze, drugim wierszu i pierwszej kolumnie tensora T
```

