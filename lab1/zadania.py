import numpy as np
# zadanie 1
eksperymenty = np.random.rand(5, 10)
srednia_pomiaru = np.mean(eksperymenty, axis=0)
srednia_eksperymentu = np.mean(eksperymenty, axis=1)
najlepszy_eksperyment = np.argmax(np.min(eksperymenty, axis=1))
najlepszy_pomiar = np.argmax(np.mean(eksperymenty, axis=0))
std_dev = np.std(eksperymenty)
# Obliczanie odchylenia standardowego dla każdego eksperymentu
odchylenia_standardowe = np.std(eksperymenty, axis=1)

# Wyznaczanie indeksów eksperymentów z wartościami pomiarów powyżej odchylenia standardowego
indeksy_przekraczajace = np.where(eksperymenty > odchylenia_standardowe[:, None])

print("Indeksy eksperymentów z wartościami pomiarów powyżej odchylenia standardowego:")
print(indeksy_przekraczajace)