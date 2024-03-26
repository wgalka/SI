## Algorytm wstecznej propagacji błędu

## Zagadnienia

### Sieci jednokierunkowe

 Sieci jednokierunkowe wielowarstwowe, cieszą się dużym zainteresowaniem ze względu na ich prostą strukturę i łatwość w uczeniu. Sieci te są zbliżone do budowy mózgu, który również posiada strukturę warstwową w dużej cześci jednokierunkową.

W sieciach jednokierunkowych występują uporządkowane warstwy neuronów, w tym warstwa wejściowa i wyjściowa. Neurony w każdej warstwie wykonują tę samą funkcję, ale mogą mieć różne funkcje przejścia. Połączenia między neuronami występują tylko pomiędzy sąsiednimi warstwami - połączenia każdy z każdym. Sygnały przesyłane są od warstwy wejściowej przez warstwy ukryte (jeśli występują) do warstwy wyjściowej.

Neurony w warstwie wejściowej mają jedno wejście i prostą funkcję przejścia, która wykonuje wstępną obróbkę sygnału. Neurony w warstwach ukrytych i wyjściowej przetwarzają informacje decyzyjne, a odpowiedź jest udzielana przez neurony w warstwie wyjściowej.

Dobór odpowiednich funkcji aktywacji dla neuronów jest ważnym zagadnieniem. W warstwach ukrytych często używa się funkcji sigmoidalnych, takich jak sinus hiperboliczny lub tangens hiperboliczny. W warstwie wyjściowej funkcja aktywacji zależy od oczekiwanych wartości odpowiedzi. Często stosuje się funkcje sigmoidalne, ale czasami konieczne jest zastosowanie neuronów z liniową funkcją przejścia.

Jest to istotne, aby uniknąć zastosowania samych funkcji liniowych we wszystkich warstwach, ponieważ sieć wielowarstwowa z samymi funkcjami liniowymi może zostać zastąpiona siecią jednowarstwową.

Poniżej znajduje się piaskownica na której można sprawdzić powyższe twierdzenie:

<iframe class="container-lg" src="https://playground.tensorflow.org/"> </ifame>

### Uczenie sieci

Uczenie sieci jednokierunkowych może odbywać się w trybie nadzorowanym lub nienadzorowanym. W trybie nadzorowanym konieczna jest znajomość oczekiwanych odpowiedzi neuronów w warstwie wyjściowej, co stanowi wyzwanie, gdyż dla warstw ukrytych te odpowiedzi nie są znane. Przez wiele lat ograniczenie to uniemożliwiało efektywne uczenie sieci wielowarstwowych. Jednakże opracowanie metody wstecznej propagacji błędu (backpropagation), która pozwala matematycznie wyznaczyć błąd popełniany przez neurony w warstwach ukrytych na podstawie błędu warstwy wyjściowej, umożliwiło skuteczne wykorzystanie reguł uczenia nadzorowanego do treningu sieci wielowarstwowych. Dzięki tej metodzie możliwe jest dostosowywanie wag neuronów w warstwach ukrytych, co jest kluczowe dla efektywnego uczenia się sieci. Metoda wstecznej propagacji błędu jest obecnie powszechnie stosowana w uczeniu sieci wielowarstwowych.

### Algorytm wstecznej propagacji

Algorytm wstecznej propagacji błędu można zapisać następująco:  

1. Wygeneruj losowo wektory wag.
2. Podaj wybrany wzorzec na wejście sieci.  
3. Wyznacz odpowiedzi wszystkich neuronów wyjściowych sieci: 

$$
y_k^{w y j}=\mathbf{f}\left(\sum_{j=1}^l w_{k j}^{w y j} y_j^{w y j-1}\right)
$$

4. Oblicz błędy wszystkich neuronów warstwy wyjściowej:

$$
\delta_k^{w y j}=z_k-y_k^{w y j}
$$

5. Oblicz błędy w warstwach ukrytych (pamiętając, że, aby wyznaczyć błąd w warstwie h - 1, konieczna jest znajomość błędu w warstwie po niej następującej - h):

$$
\delta_j^{h-1}=\frac{d \mathrm{f}\left(u_j^{h-1}\right)}{d u_j^{h-1}} \sum_{k=1}^l \delta_k^h w_{k j}^h
$$

6. Zmodyfikuj wagi wg zależności:

$$
W_{ji}^{h-1}=W_{ji}^{h-1}+\eta\delta_j^{h-1}y_i^{h-1}
$$

7. Jeżeli wartość funkcji celu jest zbyt duża wróć do punktu 2.

Jednym z głównych wyzwań związanych z zastosowaniem tej metody jest optymalizacja parametrów procesu uczenia, w tym zwłaszcza wielkości współczynnika uczenia (learning rate). **Niewłaściwy dobór learning rate może prowadzić do problemów, takich jak zbyt wolne uczenie się sieci lub jego zbyt szybkie rozbieganie**.

**Niestety, metoda wstecznej propagacji błędu charakteryzuje się często długim czasem uczenia**. Proces ten może być czasochłonny, zwłaszcza w przypadku dużych i złożonych sieci neuronowych, co stanowi dodatkowe wyzwanie w praktycznym zastosowaniu tej techniki.

**Wybór optymalnych parametrów uczenia, takich jak learning rate, jest procesem eksperymentalnym, który wymaga prób i błędów**. Często konieczne jest dostosowywanie tych parametrów podczas procesu uczenia, aby osiągnąć najlepsze wyniki.


### Dobór struktury i danych w uczeniu sieci neuronowych

- Dobór danych uczących:
    - Reprezentatywny ciąg uczący jest kluczowy dla efektywnego uczenia sieci neuronowej.
    - Istnieje wiele metod oceny generalizacji sieci, takich jak średnia liczba alternatywnych generalizacji zbioru treningowego.
    - Dobór odpowiednich danych pozwala sieci na efektywne generalizowanie problemu.

- Liczba warstw ukrytych:
    - Najczęściej stosowane są sieci z jedną lub dwiema warstwami ukrytymi.
    - Wybór liczby warstw ukrytych zależy od złożoności problemu i wydajności procesu uczenia.
    - Reguły matematyczne, takie jak twierdzenie Kołmogorowa, mogą pomóc w określeniu minimalnej liczby neuronów potrzebnych do rozwiązania problemu.

- Rozmiary warstw sieci:
    - Liczba neuronów w warstwach wejściowej i wyjściowej jest łatwa do ustalenia.
    - Dobór liczby neuronów w warstwie ukrytej zależy od wielu czynników, takich jak złożoność problemu i możliwości sieci w uczeniu się istotnych cech.
    - Istnieją różne metody przybliżone określenia niezbędnej liczby neuronów w warstwie ukrytej.

- Czas uczenia:
    - Proces uczenia sieci może być długotrwały i wymaga odpowiedniego dostosowania parametrów uczenia.
    - Istnieje wiele metod oceny skuteczności uczenia, które pozwalają określić optymalny moment zakończenia procesu uczenia.
    - Przeprowadzanie eksperymentów i analiza różnych konfiguracji sieci są kluczowe dla znalezienia optymalnych ustawień dla konkretnego zadania.


## Zadania

### Zadanie 1:

1. Opracować skrypt tworzący jednokierunkową sieć neuronową.
2. Sieć ma składać się z 2 neuronów w warstwie ukrytej i 1 neuronu liniowego w warstwie wyjściowej.
3. Użyć sigmoidalnej funkcji przejścia w warstwie ukrytej.
4. Ustalić parametry uczenia: 1000 epok, współczynnik uczenia 0.1, maksymalna wartość funkcji celu 0.00001.
5. Uczyć sieć na zbiorze danych reguł bramki XOR: P=[1 1 0 0 ;1 0 1 0], T=[0 1 1 0].
6. Zasymulować pracę nauczonej sieci.
7. Skomentować uzyskane rezultaty.

### Zadanie 2:

1. Badać wpływ liczby neuronów w warstwie ukrytej (trzy różne wartości).
2. Badać wpływ wartości współczynnika uczenia (trzy różne wartości, różniące się o jeden rząd).
3. Wykonać symulacje dla dwóch kombinacji funkcji przejścia neuronów (np.: sigmoidalna+liniowa, tangensoidalna+liniowa itp.).

### Zadanie 3:

1. Opracować skrypt tworzący jednokierunkową sieć neuronową do rozpoznawania symboli x, o, +, -.
2. Symbole zapisane są na macierzy 3x3.
3. Zakodować identyfikowane symbole binarnie, liniowo oraz kodem 1 z N (stworzyć odrębną sieć dla każdego kodowania).
4. Przeprowadzić symulacje działania nauczonej sieci.
5. Zbadać skuteczność w rozpoznawaniu zniekształconych symboli.
6. Skomentować uzyskane rezultaty.

![alt text](zad3.svg)



