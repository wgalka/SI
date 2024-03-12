import numpy as np
import matplotlib.pyplot as plt
# OR gate dataset
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
np.savetxt('lab3/or.data', X_or, delimiter=',', header='x,y', comments='')
y_or = np.array([0, 1, 1, 1])
np.savetxt('lab3/or.labels',  y_or, delimiter=',', header='output', comments='')

# XOR gate dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
np.savetxt('lab3/xor.data', X_xor, delimiter=',', header='x,y', comments='')
y_xor = np.array([0, 1, 1, 0])
np.savetxt('lab3/xor.labels',  y_xor, delimiter=',', header='output', comments='')

def binary_activation(sumation_result):
    if sumation_result > 0:
        return 1
    else:
        return 0

class Perceptron:
    def __init__(self,input_size, activation_function, weights=None, bias=None, learning_rate= 0.1) -> None:
        self.input_size = input_size
        if weights is None:
            self.weights = np.random.rand(input_size)
        if bias is None:
            self.bias = np.random.rand()
        self.learning_rate = learning_rate
        self.activation_function = activation_function

    def forward(self, inputs):
        # Sumowanie wejść z odpowiednimi wagami
        total_input = np.dot(self.weights, inputs) + self.bias
        # Zastosowanie funkcji aktywacji (np. sigmoidalnej)
        prediction = self.activation_function(total_input)
        return prediction
    
    def update_weights_and_bias(self, error, inputs):
        self.weights += self.learning_rate * error * inputs
        self.bias += self.learning_rate * error

    
    def __str__(self) -> str:
        return f"Perceptron({self.input_size}, {self.activation_function}, {self.weights}, {self.bias}, {self.learning_rate})"
    


a = Perceptron(2,binary_activation)
print(a)

def train(perceptron, epochs, X, y):
    for _ in range(epochs):
            epoch_error = 0
            for inputs, label in zip(X, y):
                prediction = perceptron.forward(inputs)
                error = label - prediction
                perceptron.update_weights_and_bias(error, inputs)
                epoch_error += abs(error)
            print(_, epoch_error/len(X))
train(a, 100, X_or,y_or)
print(a)


def draw_decision_boundary(perceptron, X, y):
    colors = ["r", "g", "b"]
    # Wagi perceptronu
    w1, w2 = perceptron.weights
    # Bias
    b = perceptron.bias

    # Punkty na linii decyzyjnej
    x_ax = np.linspace(-1, 1, 10)
    y_ax = (-w1 * x_ax - b) / w2

    # Utwórz nowy wykres
    fig, ax = plt.subplots()

    # Narysuj punkty na linii decyzyjnej
    ax.plot(x_ax, y_ax, '-r', label='Decision Boundary')
    # Narysuj punkty danych, każda klasa oznacozna innym kolorem
    for index, label in enumerate(np.unique(y)):
        print("label",label)
        data = X[y == label]
        ax.scatter(data[:,0], data[:,1], color = colors[index], label=label)

    # Dodaj etykiety osi
    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')

    # Dodaj tytuł
    ax.set_title('Decision Boundary')

    # Dodaj legendę
    ax.legend()
   
    # Oznaczneia osi X i Y
    ax.set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

    # Ustawienie limitu dla osi X i Y
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    # Wyśietlenie siatki wykresu
    plt.grid()

    # Wyświetlenie wykresu
    plt.show()

draw_decision_boundary(a, X_or,y_or)