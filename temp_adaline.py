
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. IMPLEMENTAÇÃO DA ADALINE
# ==========================================

class Adaline:

    def __init__(self, taxa_aprendizado=0.01, epocas=8000):

        # Ela controla o quanto os pesos e o bias serão modificados a cada atualização.
        self.taxa_aprendizado = taxa_aprendizado

        # Quantidade de vezes que a rede repetirá o treinamento.
        self.epocas = epocas

        # Peso e bias começam em zero.
        self.peso = 0.0
        self.bias = 0.0

        # Armazena o erro de cada época.
        self.historico_erro = []

    # ==========================================
    # 2. TREINAMENTO DA REDE
    # ==========================================

    def treinar(self, X, y):

        # Inicializa peso e bias.
        self.peso = 0.0
        self.bias = 0.0

        self.historico_erro = []

        # Repete o treinamento durante as épocas.
        for epoca in range(self.epocas):

            # Calcula a saída da rede.
            saida = X * self.peso + self.bias

            # Calcula o erro entre o valor esperado
            # e o valor produzido pela ADALINE.
            # O erro é a diferença entre o resultado esperado e o resultado produzido pela rede
            erro = y - saida

            # Calcula o erro quadrático médio (MSE).
            mse = np.mean(erro ** 2)

            # Guardamos o valor do erro em uma lista para posteriormente construir o gráfico de aprendizagem.
            self.historico_erro.append(mse)

            # Atualiza o peso pela regra Delta (LMS).
            # A ideia é ajustar o peso para reduzir o erro das próximas previsões.
            self.peso += (
                self.taxa_aprendizado *
                np.mean(X * erro)
            )

            # Atualiza o bias.
            # O bias também é atualizado utilizando o erro médio.
            self.bias += (
                self.taxa_aprendizado *
                np.mean(erro)
            )

        return self

    # ==========================================
    # 3. PREVISÃO
    # ==========================================

    def prever(self, celsius):

        # Dividimos Celsius por 100 para facilitar
        # o treinamento com valores de entrada menores.
        entrada = celsius / 100

        # Calcula a saída da ADALINE.
        fahrenheit = self.peso * entrada + self.bias

        return fahrenheit


# ==========================================
# 4. DADOS DE TREINAMENTO
# ==========================================

# Exemplos fornecidos para a rede aprender.
# A entrada foi dividida por 100 para facilitar
# o ajuste dos pesos durante o treinamento.

X = np.array([0, 1], dtype=float)

# Valores de Fahrenheit esperados.
y = np.array([32, 212], dtype=float)


# ==========================================
# 5. CRIAÇÃO E TREINAMENTO DA REDE
# ==========================================

modelo = Adaline(
    taxa_aprendizado=0.01,
    epocas=8000     
)

# Treina a rede com os exemplos fornecidos.
modelo.treinar(X, y)


# ==========================================
# 6. RESULTADOS DO APRENDIZADO
# ==========================================

# Como a entrada foi dividida por 100,
# precisamos dividir o peso aprendido por 100
# para recuperar o coeficiente em Celsius.

peso_original = modelo.peso / 100

print("========== RESULTADO DO TREINAMENTO ==========")

print(f"Peso aprendido: {peso_original:.4f}")
print(f"Bias aprendido: {modelo.bias:.4f}")

print("\nFórmula aprendida pela ADALINE:")

print(
    f"F = {peso_original:.4f} * C + {modelo.bias:.4f}"
)


# ==========================================
# 7. TESTE COM ALGUMAS TEMPERATURAS
# ==========================================

temperaturas = [0, 25, 50, 75, 100]

print("\n========== CONVERSÕES ==========")

for celsius in temperaturas:

    resultado = modelo.prever(celsius)

    print(
        f"{celsius}°C = {resultado:.2f}°F"
    )


# ==========================================
# 8. PREVISÃO DE UMA NOVA TEMPERATURA
# ==========================================

nova_temperatura = 30

resultado = modelo.prever(nova_temperatura)

print("\n========== NOVA PREVISÃO ==========")

print(f"Temperatura informada: {nova_temperatura}°C")
print(f"Temperatura prevista: {resultado:.2f}°F")



# ==========================================
# 9. GRÁFICOS DO PROJETO
# ==========================================

# ------------------------------------------
# GRÁFICO 1: EVOLUÇÃO DO APRENDIZADO
# ------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    modelo.historico_erro,
    linewidth=2,
    label="Erro durante o treinamento"
)

plt.title("Evolução do aprendizado da ADALINE")
plt.xlabel("Épocas")
plt.ylabel("Erro quadrático médio (MSE)")

# Escala logarítmica para visualizar melhor
# a redução do erro.
plt.yscale("log")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()


# ------------------------------------------
# GRÁFICO 2: COMPARAÇÃO DAS FÓRMULAS
# ------------------------------------------

# Cria várias temperaturas entre 0 e 100.
temperaturas = np.linspace(0, 100, 100)

# Calcula os valores pela fórmula matemática original.
fahrenheit_real = 1.8 * temperaturas + 32

# Calcula os valores aprendidos pela ADALINE.
fahrenheit_adaline = modelo.prever(temperaturas)

plt.figure(figsize=(9, 5))

# Desenha a fórmula matemática original.
plt.plot(
    temperaturas,
    fahrenheit_real,
    linewidth=2,
    label="Fórmula original"
)

# Desenha a fórmula aprendida pela rede.
plt.plot(
    temperaturas,
    fahrenheit_adaline,
    linestyle="--",
    linewidth=2,
    label="ADALINE"
)

# Mostra os exemplos utilizados no treinamento.
plt.scatter(
    [0, 100],
    [32, 212],
    s=70,
    label="Dados de treinamento",
    zorder=3
)

plt.title("Comparação: fórmula original x ADALINE")
plt.xlabel("Temperatura em Celsius (°C)")
plt.ylabel("Temperatura em Fahrenheit (°F)")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()