## BP z moemntum

Metoda wstecznej propagacji z momentum to modyfikacja standardowej techniki propagacji wstecznej, która wykorzystuje dodatkowy parametr nazywany powszechnie "momentum". Jest to popularna technika optymalizacji używana podczas trenowania sieci neuronowych, która pomaga przyspieszyć proces uczenia i zmniejszyć ryzyko utknięcia w lokalnym minimum funkcji kosztu.

Główną ideą metody wstecznej propagacji z momentum jest wykorzystanie historycznych aktualizacji wag do przyspieszenia aktualizacji wag podczas kolejnych iteracji. W standardowej propagacji wstecznej, aktualizacja wag na podstawie gradientów odbywa się bez uwzględnienia poprzednich kroków, co może powodować oscylacje wzdłuż dolin funkcji kosztu oraz spowalniać proces uczenia, szczególnie w przypadku płaskich lub mało stromych obszarów funkcji kosztu.

W metodzie zastosowania momentum, podczas aktualizacji wag, uwzględniane są również poprzednie zmiany wag. Jest to realizowane poprzez dodanie do aktualizacji wag pewnego współczynnika momentum, który jest proporcjonalny do poprzedniej aktualizacji oraz aktualnego gradientu. W ten sposób, jeśli aktualna aktualizacja wag ma zbliżoną kierunek do poprzedniej, to współczynnik momentum pozwoli przyspieszyć proces, redukując oscylacje i umożliwiając płynniejsze zbieganie do minimum funkcji kosztu.

W praktyce, dobór optymalnej wartości współczynnika momentum oraz współczynnika uczenia jest kluczowy dla efektywności metody wstecznej propagacji z momentum. Zbyt wysoki współczynnik momentum może prowadzić do przeskakiwania po minimum globalnym, podczas gdy zbyt niski może spowodować wolniejsze zbieganie. Dlatego też, dobór tych parametrów często wymaga eksperymentów i dostosowywania w zależności od konkretnego problemu i struktury sieci.

## Krosswalidacja

Podział na zbiór treningowy i testowy to jedna z podstawowych technik walidacji modelu, która jest wykorzystywana w połączeniu z metodami kroswalidacji. Ta technika polega na podziale dostępnych danych na dwa niezależne zbiory: zbiór treningowy, na którym trenowany jest model, oraz zbiór testowy, który służy do oceny wydajności modelu.

Podział ten jest zazwyczaj wykonywany w następujących proporcjach:

- Zbiór treningowy: większy podzbiór danych, na którym model jest uczony. Zawiera zazwyczaj od 60% do 80% dostępnych danych.
- Zbiór testowy: mniejszy podzbiór danych, który służy do oceny wydajności modelu. Zawiera zazwyczaj od 20% do 40% dostępnych danych.

Podział na zbiór treningowy i testowy ma na celu zapewnienie niezależnej oceny modelu. Dzięki temu model jest trenowany na jednym zbiorze danych, a jego wydajność jest oceniana na drugim, który nie był używany w procesie trenowania. Pozwala to na obiektywne określenie, jak dobrze model generalizuje się na nowe, nieznane dane.

W praktyce, podział na zbiór treningowy i testowy może być wykonany losowo, biorąc pod uwagę, że próbki w obu zbiorach są reprezentatywne dla całego zbioru danych. Jednakże, aby zapewnić wiarygodność oceny wydajności modelu, ważne jest, aby zachować tę samą proporcję klas (w przypadku problemów klasyfikacyjnych) w obu zbiorach. W przypadku problemów regresji, należy także zadbać o zachowanie rozkładu wartości docelowych.

Podział na zbiór treningowy i testowy jest często wykonywany raz, na początku eksperymentu. Następnie na zbiorze treningowym stosuje się metodę kroswalidacji, aby dostarczyć modelowi wielu różnych prób danych treningowych, które mogą poprawić jego ogólną zdolność do generalizacji. Na końcu, ostateczna ocena wydajności modelu jest dokonywana na zbiorze testowym, który był wyłączony od początku analizy.

Udoskonaleniem podziału na zbiór testowy i treningowy aby uzyskać jeszcze lepszą generalizację są metody kroswalidacji. Pozwalają one na dokładną ocenę wydajności modelu poprzez podział dostępnych danych na zbiór treningowy i testowy w sposób powtarzalny i obiektywny.

Istnieje kilka różnych metod kroswalidacji, z których najczęściej stosowanymi są:

- K-krotna kroswalidacja (k-fold cross-validation):
    - W tej metodzie dane dzielone są na k równych części (k-fold), gdzie każda część pełni rolę zbioru testowego dokładnie raz, a pozostałe części są używane jako zbiór treningowy. Następnie model trenowany jest k razy, każdorazowo na innych danych treningowych, a wyniki są uśredniane.

- K-krotna kroswalidacja stratyfikowana (stratified k-fold cross-validation):

    - Jest to rozszerzenie k-krotnej kroswalidacji, w którym zachowuje się proporcje klas w każdej części podziału, co jest szczególnie ważne w przypadku niezrównoważonych zbiorów danych, gdzie jedna klasa może być znacznie liczniejsza od innych.

- Walidacja krzyżowa z jednym wyłączonym (leave-one-out cross-validation, LOOCV):

    - Jest to skrajna forma k-krotnej kroswalidacji, gdzie k jest równe liczbie próbek w zbiorze danych. Dla każdej iteracji jedna próbka jest wyłączana jako zbiór testowy, a pozostałe próbki służą jako zbiór treningowy.

Wszystkie te metody mają na celu zapewnienie obiektywnej i rzetelnej oceny wydajności modelu oraz zmniejszenie ryzyka nadmiernego dopasowania (overfitting) poprzez ocenę na danych, które nie były używane podczas procesu trenowania. Wybór konkretnej metody kroswalidacji zależy od charakterystyki danych oraz konkretnego problemu, który chcemy rozwiązać.


## Zadania

1. Zaimplementować sieć neuronową w ten sposób aby aktualizacja wag mogła odbywać się według klasycznego algorytmu wstecznej propagacji błędu lub algorytmu wstecznej propagacji błędu z momentum.

2. Stworzyć generator który jako parametr przyjmie zbiór danych a następnie podzieli go na `n` cześci. Dane powinny być podzielone w sposób statyfikowany (każda z `n` cześci powinna zawierać taką samą ilość obiektów z każdej klasy decyzyjnej). Funkcja będzie zwracać zbiór testowy i uczący w iteracjach od 0 no `n` w ten sposób że w 0 iteracji zerowa część danych posłuży jako zbiór testowy natomiast pozostałe części jako zbiór uczący. W kolejnej iteracji, 1 część danych zostanie użyta jako zbiór testowy natomiast reszta jako zbiór uczący. 

3. Utworzyć dwie sieci neuronowe o tej samej strukturze natomist różniące się metodą wstecznej propagacji błędu (sieci powinny mieć ustalone te same wagi początkowe, liczbę epok itp.). Przetestować działanie algorytmu wykorzystując metodę stratyfikwoanej walidacji krzyżowej (procedura z zadania 2). Dane podzielić na 10 części. Proces testu w każdej z iteracji wygląda następująco:
    - Uczymy sieć neuronową na zbiorze treningowym
    - Zbieramy decyzje sieci neuronowej podając jej wejśiach dane ze zbioru testowego
    - Porównujemy otrzymane rezultaty z prawdziwymi decyzjami obliczając stosunek poprawnie przydzielonych decyzji przez model do ogólnej liczby testowanych obiektów (accuracy).
Dane zebrać w macierzy i zaprezentować za pomocą wykresu.
Z procesu uczenia zebrać również informacje jak malał błąd w każdej z epok. Ważny jest tutaj błąd końcowy aby można było porównać która z metod wstecznej propagacji błędu daje lepsze rezultaty.