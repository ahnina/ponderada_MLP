import numpy as np
import math

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Inicializando pesos e bias
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros(output_size)
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def _mse(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        return mse

    def forward_pass(self, X):
        # Passagem para a camada escondida
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self._sigmoid(self.hidden_input)

        # Passagem para a camada de saída
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self._sigmoid(self.final_input)
        
        return self.final_output

    def backpropagation(self, X, y):
        # Cálculo do erro na saída
        output_error = y - self.final_output
        output_delta = output_error * self._sigmoid_derivative(self.final_output)

        # Cálculo do erro na camada escondida
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self._sigmoid_derivative(self.hidden_output)

        # Atualização dos pesos e biases da camada de saída
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * self.learning_rate

        # Atualização dos pesos e biases da camada escondida
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * self.learning_rate

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            self.forward_pass(X)
            self.backpropagation(X, y)

            # Calcula e imprime o MSE na última época
            if epoch == 9999:
                mse = self._mse(y, self.final_output)
                print(f"Ao treinar com 10000 épocas o MSE final é de aproximadamente: {mse}")

    def predict(self, X):
        return self.forward_pass(X)

# Dados de entrada para a porta XOR
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Saída esperada para a porta XOR
y = np.array([[0], 
              [1], 
              [1], 
              [0]])

# Definindo e treinando a MLP
mlp = MLP(input_size=2, hidden_size=2, output_size=1)
mlp.train(X, y)

# Testando a MLP treinada
predictions = mlp.predict(X)
print("Predictions after training:")
print(predictions)
