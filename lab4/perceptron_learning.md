## Uczenie perceptronu

## Zagadnienia

### Uczenie nadzorowane
- W uczeniu nadzorowanym, model uczący jest trenowany na podstawie danych treningowych, które składają się z par wejście-wyjście oraz odpowiadających im etykiet (odpowiedzi).
- Celem jest nauczenie modelu przewidywania odpowiedzi dla nowych danych na podstawie wcześniej widzianych przykładów.
- Przykłady algorytmów uczenia nadzorowanego to: regresja liniowa, regresja logistyczna, drzewa decyzyjne, sieci neuronowe, itp.
- Przykładowe zadania obejmują klasyfikację (np. rozpoznawanie obrazów, diagnoza medyczna) oraz regresję (np. prognozowanie cen, szacowanie wartości).

### Uczenie nie nadzorowane
- W uczeniu nienadzorowanym, model uczący jest trenowany na danych treningowych, które nie posiadają etykiet lub mają tylko częściowe etykiety.
- Celem jest wykrycie struktury lub wzorców w danych bez konieczności posiadania wcześniejszej wiedzy na temat etykiet danych.
- Przykłady algorytmów uczenia nienadzorowanego to: klastrowanie, redukcja wymiarowości, itp.
- Przykładowe zadania obejmują grupowanie (np. segmentacja klientów, analiza rynku), redukcję wymiarowości (np. analiza skupień).

<details>
<summary>TLDR</summary>
Uczenie nadzorowane wymaga etykietowanych danych treningowych, podczas gdy uczenie nienadzorowane może być stosowane do danych bez etykiet lub z częściowymi etykietami.
</details>


### Reguła Hebba

Ogólna reguła uczenia neuronu, zwana także regułą Hebba, opiera się na zasadzie modyfikacji wag połączeń między neuronami w zależności od aktywności tych neuronów. 

Ogólna reguła uczenia neuronu mówi, że:
- Jeśli dwa neurony są aktywowane jednocześnie, to waga połączenia między nimi zostaje wzmocniona.
- Jeśli dwa neurony są aktywowane naprzemiennie (jeden jest aktywowany, gdy drugi jest nieaktywny), to waga połączenia między nimi zostaje osłabiona.
- Jeśli neurony są nieaktywne, waga połączenia pozostaje niezmieniona.

Ogólnie rzecz biorąc, reguła Hebbiana opiera się na zasadzie, że neurony, które są aktywowane jednocześnie, prawdopodobnie mają związek funkcjonalny i warto zwiększyć siłę połączenia między nimi, aby ułatwić przetwarzanie informacji. Z drugiej strony, neurony, które są aktywowane w odwrotnych sekwencjach, prawdopodobnie nie są ze sobą związane i warto osłabić ich połączenie.

Formalnie, reguła Hebba może być wyrażona w postaci równania wag:

$$
\Delta w_{ij} = \eta \cdot x_i \cdot y_j
$$

$$
\Delta w_{ij} -\text{ to zmiana wagi i-tego wejścia} 
$$
  
$$
η - \text{to współczynnik uczenia (learning rate)}
$$


$$
x_i - \text{to aktywacja
neuronu wejściowego i}
$$

$$
y_j - \text{to aktywacja neuronu j}
$$


Uczenie perceptronu jest zazwyczaj przeprowadzane w sposób nadzorowany, gdzie każda próbka treningowa ma przypisaną etykietę klasy. (poprzednie laboratorium).

Jednakże, jeśli chcemy wykorzystać perceptron do uczenia nienadzorowanego, możemy dostosować procedurę trenowania, aby sieć "uczyła się" struktury danych, ale bez korzystania z etykiet klas. Jednym z podejść do uczenia nienadzorowanego perceptronu jest zastosowanie reguły Hebba.

### Reguły uczenia

Ogólna reguła uczenia

$$
\Delta \boldsymbol{w}=c * r(\boldsymbol{w}, \boldsymbol{x}, d) * \boldsymbol{x}
$$

gdzie: 
- $\Delta w$ - zmiana wektora wag
- c - współczynnik uczenia
- $r$ - sygnał uczący który może składać się z:
    - wektora wag (w), 
    - wektora wejściowego (x) 
    - oczekiwanej odpowiedzi (d)


#### Reguła Hebba

Metoda uczenia - nienadzorowana

**Sygnał uczący**

$$
r=y=f\left(\boldsymbol{w}^{\boldsymbol{t}} \boldsymbol{x}\right) 
$$

**Korekta wag**

$$
\Delta \boldsymbol{w}=c y \boldsymbol{x}=c f\left(\boldsymbol{w}^t \boldsymbol{x}\right) \boldsymbol{x}
$$

#### Reguła Perceptronowa

Metoda uczenia - nadzorowana

**Sygnał uczący**

$$
r=d-y 
$$

**Korekta wag**

$$
\Delta \boldsymbol{w}=c\left(d-f\left(\boldsymbol{w}^t \boldsymbol{x}\right)\right) \boldsymbol{x}
$$


### Funkcja celu

Funkcja celu, znana również jako funkcja kosztu lub funkcja straty, jest to funkcja, która mierzy, jak dobrze model przewiduje rzeczywiste wartości na podstawie danych wejściowych. Jest to kluczowa koncepcja w uczeniu maszynowym, ponieważ podczas trenowania modelu próbujemy minimalizować wartość tej funkcji, aby uzyskać jak najlepsze dopasowanie do danych treningowych.

Poniżej znajduje się kilka popularnych funkcji celu, które są stosowane w różnych problemach uczenia maszynowego:

- Klasyfikacja binarna:
    - Funkcja entropii krzyżowej (cross-entropy loss)
    - Funkcja kosztu binarnej entropii krzyżowej (binary cross-entropy loss)
    - Funkcja kosztu logistycznej regresji (logistic regression loss)

- Klasyfikacja wieloklasowa:
    - Entropia krzyżowa (categorical cross-entropy loss)
    - Logarytmiczna funkcja straty (log loss)
    - Softmax cross-entropy loss

- Regresja:
    - Błąd średniokwadratowy (mean squared error loss)
    - Średni błąd bezwzględny (mean absolute error loss)
    - Funkcja straty Hubera



## Zadania

### Zadanie 1
Opracować program do uczenia reguł bramki NAND pojedynczego dwuwejściowego sztucznego neuronu z unipolarną funkcją przejścia. Uczenie zrealizować w oparciu o regułę Hebba a następnie perceptronową.

- Proces uczenia wymaga określenia funkcji celu oraz podawania kolejnych wektorów uczących wraz z odpowiadającymi im odpowiedziami w kolejności losowej.

### Zadanie 2
Sprawdzić działanie nauczonego w zadaniu 1 neuronu – opracować program, który w oparciu o wzór opisujący działanie neuronu wyznaczy odpowiedzi wszystkich możliwych wektorów wejściowych. Porównać uzyskane wyniki z oczekiwanymi odpowiedziami.

### Zadanie 3
Zbadać wpływ wartości współczynnika uczenia na przebieg procesu uczenia i efektywności działania sieci. W tym celu dla trzech różnych wartości współczynnika uczenia (różniących się o rząd wielkości) wykonać po 10 prób uczenia wyniki zebrać w tabelce i skomentować.

| Lp. | Ilość powtórzeń uczenia                       | 
|-----|------------------------------------------------|
|     | \( c = x \)                                   | 
|     | \( c = 100x \)                                | 
|     | \( c = 0.01x \)                               | 
|-----|------------------------------------------------|
| 1.  |                                                |
| 2.  |                                                |
| ... |                                                |
| 10. |                                                |
|-----|------------------------------------------------|
| Wartość średnia |                                  |
|-----|------------------------------------------------|
| Mediana         |                                  |
|-----|------------------------------------------------|


### Zadanie 4
Zbadać wpływ błędu końcowego (założonej wartości funkcji celu) na przebieg procesu uczenia i efektywności działania sieci. W tym celu dla trzech różnych wartości funkcji celu wykonać po 5 prób uczenia wyniki zebrać w tabelce i skomentować. Wzory funkcji celu znaleźć w internecie.











<!-- https://ii.uni.wroc.pl/~aba/teach/NN/w6pca.pdf -->