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
print(A[1][2])  # Odwołanie do elementu w drugim wierszu i trzeciej kolumnie macierzy A
print(T[0][1][0])  # Odwołanie do elementu w pierwszym wymiarze, drugim wierszu i pierwszej kolumnie tensora T