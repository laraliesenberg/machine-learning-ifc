import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# ==========================================
# 1. IMPLEMENTAÇÃO DA ADALINE (VÁRIAS ENTRADAS)
# ==========================================

class Adaline:

    def __init__(self, taxa_aprendizado=0.01, epocas=200):

        # Controla o quanto os pesos e o bias são modificados a cada atualização.
        self.taxa_aprendizado = taxa_aprendizado

        # Quantidade de vezes que a rede repetirá o treinamento.
        self.epocas = epocas

        # Agora temos UM PESO POR CARACTERÍSTICA (vetor), e não mais um único peso.
        self.pesos = None
        self.bias = 0.0

        # Armazena o erro de cada época.
        self.historico_erro = []

    # ==========================================
    # 2. TREINAMENTO DA REDE
    # ==========================================

    def treinar(self, X, y):

        # X tem formato (amostras, características).
        # Inicializa os pesos e o bias em zero.
        self.pesos = np.zeros(X.shape[1])
        self.bias = 0.0

        self.historico_erro = []

        for epoca in range(self.epocas):

            # Saída linear da rede para TODAS as amostras de uma vez:
            # saida = X · pesos + bias
            saida = X @ self.pesos + self.bias

            # Erro = esperado - produzido.
            erro = y - saida

            # Erro quadrático médio (MSE).
            mse = np.mean(erro ** 2)
            self.historico_erro.append(mse)

            # Regra Delta (LMS) para cada peso:
            # cada peso é ajustado pela média de (entrada daquele peso * erro).
            self.pesos += self.taxa_aprendizado * (X.T @ erro) / len(y)

            # Atualiza o bias com o erro médio.
            self.bias += self.taxa_aprendizado * np.mean(erro)

        return self

    # ==========================================
    # 3. PREVISÃO
    # ==========================================

    def saida_linear(self, X):
        # Saída contínua da ADALINE (antes de decidir a classe).
        return X @ self.pesos + self.bias

    def prever(self, X):
        # Se a saída for >= 0, classe +1; caso contrário, classe -1.
        return np.where(self.saida_linear(X) >= 0, 1, -1)


# ==========================================
# 4. DADOS DE TREINAMENTO (DATASET IRIS)
# ==========================================

iris = load_iris()

# Vamos usar só 2 classes (setosa x versicolor) e 2 características,
# para conseguir desenhar a fronteira de decisão em um gráfico 2D.
#   característica 0 = comprimento da sépala (cm)
#   característica 2 = comprimento da pétala (cm)
nomes_caracteristicas = ["Comprimento da sépala (cm)", "Comprimento da pétala (cm)"]

mascara = iris.target < 2                      # Separa só as flores com espécie 0 (setosa) e 1 (versicolor)
X = iris.data[mascara][:, [0, 2]]

# A ADALINE trabalha melhor com alvos -1 e +1. Cria a “resposta certa” de cada flor: -1 para setosa e +1 para versicolor.
y = np.where(iris.target[mascara] == 0, -1, 1).astype(float)


# ==========================================
# 5. PREPARAÇÃO DOS DADOS
# ==========================================

# Separa em treino (80%) e teste (20%), embaralhando com semente fixa.
rng = np.random.default_rng(42)
indices = rng.permutation(len(y))

corte = int(0.8 * len(y))
X_treino, y_treino = X[indices[:corte]], y[indices[:corte]]
X_teste, y_teste = X[indices[corte:]], y[indices[corte:]]

# Padronização: média 0 e desvio 1 (usando SOMENTE o treino).
# Faz o mesmo papel do "dividir por 100" do projeto anterior:
# deixa as entradas em escala pequena para o treinamento convergir bem.
media = X_treino.mean(axis=0)
desvio = X_treino.std(axis=0)

X_treino_pad = (X_treino - media) / desvio
X_teste_pad = (X_teste - media) / desvio


# ==========================================
# 6. CRIAÇÃO E TREINAMENTO DA REDE
# ==========================================

modelo = Adaline(
    taxa_aprendizado=0.01,
    epocas=200
)

modelo.treinar(X_treino_pad, y_treino)


# ==========================================
# 7. RESULTADOS DO APRENDIZADO
# ==========================================

print("========== RESULTADO DO TREINAMENTO ==========")

for nome, peso in zip(nomes_caracteristicas, modelo.pesos):
    print(f"Peso de '{nome}': {peso:.4f}")

print(f"Bias aprendido: {modelo.bias:.4f}")
print(f"MSE inicial: {modelo.historico_erro[0]:.4f}")
print(f"MSE final:   {modelo.historico_erro[-1]:.4f}")

acuracia_treino = np.mean(modelo.prever(X_treino_pad) == y_treino)
acuracia_teste = np.mean(modelo.prever(X_teste_pad) == y_teste)

print("\n========== DESEMPENHO ==========")
print(f"Acurácia no treino: {acuracia_treino * 100:.1f}%")
print(f"Acurácia no teste:  {acuracia_teste * 100:.1f}%")


# ==========================================
# 8. PREVISÃO DE UMA NOVA FLOR
# ==========================================

nova_flor = np.array([[5.0, 1.5]])   # sépala 5.0 cm, pétala 1.5 cm
nova_flor_pad = (nova_flor - media) / desvio

classe = modelo.prever(nova_flor_pad)[0]
nome_classe = "setosa" if classe == -1 else "versicolor"

print("\n========== NOVA PREVISÃO ==========")
print(f"Sépala: {nova_flor[0][0]} cm | Pétala: {nova_flor[0][1]} cm")
print(f"Saída linear da ADALINE: {modelo.saida_linear(nova_flor_pad)[0]:.4f}")
print(f"Classe prevista: {nome_classe}")


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
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()


# ------------------------------------------
# GRÁFICO 2: FRONTEIRA DE DECISÃO
# ------------------------------------------

# A fronteira é onde a saída linear é zero:
#   w1*x1 + w2*x2 + bias = 0
# Para desenhá-la, criamos uma grade de pontos e pintamos
# cada região com a classe que a rede prevê.

x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

grade_x1, grade_x2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min, x2_max, 300)
)

grade = np.c_[grade_x1.ravel(), grade_x2.ravel()]
grade_pad = (grade - media) / desvio
regioes = modelo.prever(grade_pad).reshape(grade_x1.shape)

plt.figure(figsize=(9, 5))

plt.contourf(grade_x1, grade_x2, regioes, alpha=0.2, levels=[-2, 0, 2])

plt.scatter(
    X_treino[y_treino == -1, 0], X_treino[y_treino == -1, 1],
    label="Setosa (treino)", s=50, edgecolors="k"
)
plt.scatter(
    X_treino[y_treino == 1, 0], X_treino[y_treino == 1, 1],
    label="Versicolor (treino)", s=50, edgecolors="k"
)
plt.scatter(
    X_teste[:, 0], X_teste[:, 1],
    marker="*", s=150, c="gold", edgecolors="k",
    label="Dados de teste", zorder=3
)

plt.title("Fronteira de decisão aprendida pela ADALINE")
plt.xlabel(nomes_caracteristicas[0])
plt.ylabel(nomes_caracteristicas[1])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()


iris = load_iris()
print(iris.feature_names)   # nome das 4 colunas
print(iris.data[:10])       # medidas das 10 primeiras flores
print(iris.target[:10])     # espécie de cada uma (0, 1 ou 2)

iris = load_iris(as_frame=True)
iris.frame.to_csv("iris.csv", index=False)