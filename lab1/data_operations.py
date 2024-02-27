import numpy as np
import pandas as pd
# Przykładowe dane dotyczące irysów z dodanymi rodzajami
# Każdy wiersz reprezentuje jeden kwiat irysa, a kolumny reprezentują różne cechy irysów oraz rodzaj irysa

np_iris = np.array([
    [5.1, 3.5, 1.4, 0.2, 'Iris-setosa'],      # Pierwszy kwiat irysa
    [4.9, 3.0, 1.4, 0.2, 'Iris-setosa'],      # Drugi kwiat irysa
    [7.0, 3.2, 4.7, 1.4, 'Iris-versicolor'],  # Szósty kwiat irysa
    [6.4, 3.2, 4.5, 1.5, 'Iris-versicolor'], 
    [6.3, 3.3, 6.0, 2.5, 'Iris-virginica'],   # Jedenasty kwiat irysa
    [5.8, 2.7, 5.1, 1.9, 'Iris-virginica'],   # Dwunasty kwiat irysa
   # Piętnasty kwiat irysa
], dtype=object)

print(np_iris[:,1]+np_iris[:,2])

pd_iris = pd.DataFrame(np_iris,
                   columns=[ "Sepal Length"," Sepal Width","Petal Length" , "Petal Width" , "Species" ])

classes = np_iris[:,4].copy()
classes[1] = "xd"
print(np_iris)



