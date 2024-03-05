## Python - wykresy

## Zagadnienia

1. Wczytywanie danych w bibliotece Pandas i Numpy.

2. Radzenie sobie z brakującymi warotściami, zamiana brakujących wartości na dane liczbowe lub łańcuchy znaków.

3. Tworzenie prostych wykresów różnego typu, stylowanie wykresów.

4. Tworzenie kilku wykresów na jednym obrazku.

### Wczytanie danych

W zależności od charakteru, przeznaczenia oraz potrzeb użytkowników istnieje wiele formatów przechowywania danych, które zostały zaprojektowane tak, aby zapewnić skuteczną organizację, szybki dostęp i wydajne przetwarzanie. Najczęściej dane są przechowywane w plikach tekstowych o określonym schemacie np.:

- CSV
- XML
- JSON
- EXCEL

W bibliotece Numpy dostępne są funkcje przetwarzające pliki tekstowe na macierze:

<https://numpy.org/doc/stable/reference/routines.io.html#text-files>

oraz umożliwiające zapis macierzy numpy do pliku binarnego co umożliwia poźniej szybkie wczytanie danych:

<https://numpy.org/doc/stable/reference/routines.io.html#text-files>

O wiele więcej i prostszych w obsłudze funkcji do odczytu popularnych formatów przechowywania danych ma biblioteka Pandas:

<https://pandas.pydata.org/pandas-docs/stable/reference/io.html>

#### Zadanie

Następnie utwórz projekt `Lab2` zawierający skrypt `main.py`. Następnie pobierz i dołącz do projektu (utwórz folder datasets gdzie wkleisz pobrane pliki) dwa zbiory danych:

<https://archive.ics.uci.edu/dataset/27/credit+approval>

<https://archive.ics.uci.edu/dataset/53/iris>

Wczytaj pliki najpierw za pomocą funkcji dostępnych w bibliotece `numpy` a następnie `pandas`. Wskaż problemy występujące podczas korzystania z bibliotek. Jeden ze zbiorów danych zawiera brakujące wartości. Jak z brakami radzą sobie obydwie biblioteki?

Znajdz ile występuje braków danych w każdej z kolumn. <https://numpy.org/doc/stable/reference/generated/numpy.isnan.html>


### Wykresy

Jedną z najpopularniejszych opcji tworzenia wykresów jest biblioteka [`Matplotlib`](https://matplotlib.org/stable/).
Zawiera najpopularniejsze typy wykresów 2D i 3D które można dostosować do własnych potrzeb:

<iframe class="container-lg" src="https://matplotlib.org/stable/plot_types/index.html"> </iframe>

#### Tworzenie wykresu - podstawy

Biblioteka `matplotlib` posiada [dwa interfejsy](https://matplotlib.org/stable/users/explain/figure/api_interfaces.html#api-interfaces) wykorzystywane do tworzenia wykresów:

1. Explict "Axes":
    - Zalety:
        - Większa kontrola: Interfejs "Axes" daje większą kontrolę nad tworzeniem i dostosowywaniem wykresów, ponieważ umożliwia bezpośrednią interakcję z obiektami figury i osi.
        - Zwięzłość kodu: Kod korzystający z interfejsu "Axes" często jest bardziej czytelny i zwięzły, ponieważ bezpośrednie odwołania do metod obiektów figury i osi pozwalają jasno określić, co się dzieje.
        - Elastyczność: Interfejs "Axes" jest bardziej elastyczny i umożliwia bardziej zaawansowane modyfikacje wykresów, takie jak dodawanie dodatkowych elementów (np. tekstu, linii, punktów) lub manipulowanie atrybutami tych elementów.
    - Wady:
        - Więcej kodu: Czasami potrzebujesz więcej linii kodu, aby stworzyć wykres przy użyciu interfejsu "Axes", ponieważ musisz wywołać więcej metod na obiektach figury i osi.
        - Początkowy próg uczenia: Dla początkujących użytkowników Matplotlib interfejs "Axes" może wymagać więcej czasu na naukę, ponieważ wymaga zrozumienia struktury obiektów figury i osi.
2. Implict "pyplot":
    - Zalety:
        - Łatwość użycia: Interfejs "pyplot" jest łatwiejszy w użyciu dla początkujących użytkowników, ponieważ nie wymaga bezpośredniej interakcji z obiektami figury i osi. Użytkownik może po prostu użyć funkcji plt do tworzenia wykresów i dodawania do nich elementów.
        - Szybkość prototypowania: Dzięki automatycznemu śledzeniu ostatniej figury i osi, interfejs "pyplot" pozwala szybko prototypować wykresy, ponieważ nie trzeba za każdym razem odwoływać się do obiektów figury i osi.
    - Wady:
        - Mniej elastyczność: Interfejs "pyplot" może być mniej elastyczny niż interfejs "Axes", ponieważ niektóre zaawansowane operacje mogą być trudniejsze do wykonania.
        - Mniej kontrola: Dla bardziej zaawansowanych scenariuszy wizualizacji, gdzie potrzebujesz większej kontroli nad wykresami, interfejs "pyplot" może być mniej odpowiedni.


Na początku programu należy zaimportować moduł `matplotlib.pyplot`, który umożliwia tworzenie wykresów i manipulację nimi. Zwykle importuje się go z aliasem `plt`, co ułatwia korzystanie z jego funkcji.

```python
import matplotlib.pyplot as plt
```

Następnie należy przygotować dane, które chcemy przedstawić na wykresie. Może to być jedna lub więcej list, tablice NumPy lub dane wczytane z pliku.

```python
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
```

Utwórz obiekt [`Figure`](https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure) oraz [`Axes`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes) za pomocą [`subplots()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)

```python
fig, ax = plt.subplots()
```


Wybierz rodzaj wykresu, który chcesz stworzyć, na przykład wykres liniowy, punktowy, słupkowy itp., i użyj odpowiedniej funkcji do jego stworzenia.

```python
ax.plot(x, y)  # Wykres liniowy
ax.scatter(x, y)  # Wykres punktowy
ax.bar(x, y)  # Wykres słupkowy
```

Dodaj etykiety osi, tytuł wykresu, legendę i inne elementy, aby wykres był czytelny i przejrzysty.

```python
ax.set_xlabel('Oś X')
ax.set_ylabel('Oś Y')
ax.set_title('Tytuł wykresu')
ax.legend(['Dane 1'])
```

Aby wyświetlić wykres, użyj funkcji `show()`.

```python
plt.show()
```

### Zadania

Przeanalizuj przykłady ze strony:

<https://matplotlib.org/stable/users/explain/quick_start.html>

1. Utwórz wykres rozproszenia (scatterplot) dla długości działek kielicha i płatków dla wszystkich trzech gatunków irysów:
    - Umieść długość działek kielicha na osi x, a długość płatków na osi y.
    - Użyj różnych kolorów lub markerów dla każdego gatunku irysów.

2. Histogram dla długości działek kielicha dla każdego gatunku irysów:
    - Stwórz trzy oddzielne histogramy, każdy dla jednego gatunku irysów.
    - Osie x powinny reprezentować długość działek kielicha, a osie y liczbę wystąpień.

3. Wykres pudełkowy dla szerokości i długości płatków oraz liści dla każdego gatunku irysów:
    - Stwórz trzy wykresy, każdy dla innego gatunku irysów.
    - Osie x powinny reprezentować nazwy gatunków, a osie y dane na temat danego gatunku np. szerokość płatków.
    - podpisz każde z pudełek odpowiednią etykietą np. sepal-width

4. Porównanie średnich szerokości płatków dla różnych gatunków irysów:
    - Stwórz wykres słupkowy przedstawiający średnie szerokości płatków dla każdego gatunku irysów.
    - Osie x powinny reprezentować nazwy gatunków, a osie y średnie szerokości płatków.

5. Wykres kołowy przedstawiający procentowy udział każdego gatunku irysów w zbiorze danych:
    - Stwórz wykres kołowy, gdzie każdy segment będzie reprezentować procentowy udział jednego z trzech gatunków irysów w całym zbiorze danych.

6. Wykres wieloliniowy dla wszystkich trzech gatunków irysów, przedstawiający zmianę długości płatków w zależności od indeksu danych:
    - Stwórz wykres z trzema liniami, gdzie na osi x będzie indeks danych, a na osi y wartości długości płatków dla każdego gatunku irysów.