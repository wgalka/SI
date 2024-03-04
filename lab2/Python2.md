## Python - wykresy

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

O wiele więcej funkcji do odczytu popularnych formatów przechowywania danych ma biblioteka Pandas:

<https://pandas.pydata.org/pandas-docs/stable/reference/io.html>

#### Zadanie

Następnie utwórz projekt `Lab2` zawierający skrypt `main.py`. Następnie pobierz i dołącz do projektu (utwórz folder datasets gdzie wkleisz pobrane pliki) dwa zbiory danych:

<https://archive.ics.uci.edu/dataset/27/credit+approval>

<https://archive.ics.uci.edu/dataset/53/iris>

Wczytaj pliki najpierw za pomocą funkcji dostępnych w bibliotece `numpy` a następnie `pandas`. Wskaż problemy występujące podczas korzystania z bibliotek. Jeden ze zbiorów danych zawiera brakujące wartości. Jak z brakami radzą sobie obydwie biblioteki?

Znajdz ile występuje braków danych w każdej z kolumn osobna. <https://numpy.org/doc/stable/reference/generated/numpy.isnan.html>


### Wykresy

Jedną z najpopularniejszych opcji tworzenia wykresów jest biblioteka [`Matplotlib`](https://matplotlib.org/stable/).

<iframe src="https://matplotlib.org/stable/plot_types/index.html"> </iframe>



