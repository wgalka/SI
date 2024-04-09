


## Zadania

1. Zaimplementować sieć neuronową w ten sposób aby aktualizacja wag mogła odbywać się według klasycznego algorytmu wstecznej propagacji błędu lub algorytmu wstecznej propagacji błędu z momentum.

2. Stworzyć generator który jako parametr przyjmie zbiór danych a następnie podzieli go na `n` cześci. Dane powinny być podzielone w sposób statyfikowany (każda z `n` cześci powinna zawierać taką samą ilość obiektów z każdej klasy decyzyjnej). Funkcja będzie zwracać zbiór testowy i uczący w iteracjach od 0 no `n` w ten sposób że w 0 iteracji zerowa część danych posłuży jako zbiór testowy natomiast pozostałe części jako zbiór uczący. W kolejnej iteracji, 1 część danych zostanie użyta jako zbiór testowy natomiast reszta jako zbiór uczący. 

3. Utworzyć dwie sieci neuronowe o tej samej strukturze natomist różniące się metodą wstecznej propagacji błędu (sieci powinny mieć ustalone te same wagi początkowe, liczbę epok itp.). Przetestować działanie algorytmu wykorzystując metodę stratyfikwoanej walidacji krzyżowej (procedura z zadania 2). Dane podzielić na 10 części. Proces testu w każdej z iteracji wygląda następująco:
    - Uczymy sieć neuronową na zbiorze treningowym
    - Zbieramy decyzje sieci neuronowej podając jej wejśiach dane ze zbioru testowego
    - Porównujemy otrzymane rezultaty z prawdziwymi decyzjami obliczając stosunek poprawnie przydzielonych decyzji przez model do ogólnej liczby testowanych obiektów (accuracy).
Dane zebrać w macierzy i zaprezentować za pomocą wykresu.
Z procesu uczenia zebrać również informacje jak malał błąd w każdej z epok. Ważny jest tutaj błąd końcowy aby można było porównać która z metod wstecznej propagacji błędu daje lepsze rezultaty.