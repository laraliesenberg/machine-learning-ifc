# ADALINE do zero: dois mini-projetos

Dois programas em Python que implementam a **ADALINE** (*ADAptive LInear NEuron*) do zero, usando apenas NumPy. O objetivo é entender o processo de aprendizado da rede passo a passo, sem esconder a matemática atrás de uma biblioteca.

| Projeto | Arquivo | Tipo de problema |
|---|---|---|
| 1. Conversão de temperatura | `temp_adaline.py` | Regressão |
| 2. Classificação de flores (Iris) | `adaline_iris.py` | Classificação |

---

## O que é a ADALINE

A ADALINE é um neurônio artificial simples. Ela calcula uma **soma ponderada das entradas** mais um bias:

```
saída = peso1 * entrada1 + peso2 * entrada2 + ... + bias
```

O treinamento usa a **regra Delta (LMS)**. A cada época:

1. A rede calcula a saída para todas as amostras.
2. Calcula o erro: `erro = esperado - produzido`.
3. Mede o erro quadrático médio (MSE).
4. Ajusta pesos e bias na direção que reduz o erro:

```
peso += taxa_aprendizado * média(entrada * erro)
bias += taxa_aprendizado * média(erro)
```

Esse método é um caso simples de **gradiente descendente**. A diferença para o Perceptron clássico é que a ADALINE aprende a partir da saída **linear** (antes de qualquer decisão), e não da classe final.

---

## Projeto 1: Conversão Celsius → Fahrenheit

**Ideia:** ensinar a rede a descobrir a fórmula `F = 1,8 · C + 32` a partir de apenas dois exemplos.

- **Dados:** 0 °C → 32 °F e 100 °C → 212 °F.
- **Entrada:** 1 valor (Celsius), dividido por 100 para ficar em escala pequena.
- **Saída:** 1 valor contínuo (Fahrenheit).
- **Parâmetros:** taxa de aprendizado 0,01 e 7000 épocas.
- **Resultado:** o peso aprendido, dividido por 100, converge para `1,8`, e o bias converge para `32`.
- **Gráficos:** curva de MSE (escala logarítmica) e comparação entre a fórmula original e a reta da ADALINE.

É um problema ideal para aprender, porque a resposta correta é conhecida e a rede consegue chegar a ela de forma exata.

---

## Projeto 2: Classificação de flores (Iris)

**Ideia:** separar duas espécies de flor, **setosa** e **versicolor**, a partir de medidas reais.

### O dataset

O Iris é um dataset clássico, com **150 flores** de 3 espécies (setosa, versicolor e virginica). Cada flor tem 4 medidas, em cm: comprimento e largura da **sépala** (a folha verde que protege o botão) e comprimento e largura da **pétala** (a parte colorida da flor). Ele vem embutido no scikit-learn (`load_iris`).

Neste projeto:

- Usamos só **setosa e versicolor**, o que dá 100 flores.
- Usamos só **2 medidas**: comprimento da sépala e comprimento da pétala. Assim dá para desenhar o resultado em um gráfico 2D.
- A setosa tem pétalas bem menores que a versicolor, então as duas classes são **linearmente separáveis**: uma reta consegue separá-las.

### Como funciona

- **Alvos:** -1 para setosa e +1 para versicolor.
- **Pesos:** um por medida (um vetor, não mais um número único).
- **Padronização:** média 0 e desvio 1, calculados só com os dados de treino.
- **Treino e teste:** 80% das flores para treinar e 20% para testar, com embaralhamento (semente fixa 42).
- **Decisão:** se a saída linear for `>= 0`, a classe é versicolor; caso contrário, é setosa.
- **Gráficos:** curva de MSE e **fronteira de decisão**, a reta onde `w1·x1 + w2·x2 + bias = 0`.

### Resultados obtidos

| Métrica | Valor |
|---|---|
| Peso da sépala (padronizado) | 0,2390 |
| Peso da pétala (padronizado) | 0,6614 |
| Bias | 0,0650 |
| MSE inicial → final | 1,0000 → 0,1311 |
| Acurácia no treino | 97,5% |
| Acurácia no teste | 100% |

O peso da pétala é bem maior que o da sépala. Isso indica que, para separar essas duas espécies, o comprimento da pétala é a medida mais informativa.

---

## Diferenças entre os dois projetos

| | Temperatura | Iris |
|---|---|---|
| Tipo de problema | Regressão (prever um número) | Classificação (prever uma classe) |
| Número de entradas | 1 | 2 |
| Pesos | 1 número | 1 vetor (um por medida) |
| Cálculo da saída | `X * peso + bias` | `X @ pesos + bias` |
| Dados | 2 pontos de uma fórmula exata | 100 medições reais, com variação natural |
| Alvos | 32 e 212 | -1 e +1 |
| Saída final | O próprio valor da rede | Sinal da saída (`>= 0` ou `< 0`) |
| MSE final | Praticamente 0 | Estaciona em ~0,13 |
| Ajuste de escala | `celsius / 100` | Padronização (média 0, desvio 1) |
| Como avaliar | Compara com a fórmula `1,8·C + 32` | Acurácia em treino e teste |
| Gráfico principal | Reta da rede sobre a reta da fórmula | Fronteira de decisão |

### Os pontos mais importantes

1. **A regra de aprendizado é a mesma.** O que muda é só a quantidade de entradas e o que se faz com a saída.
2. **Regressão x classificação.** No primeiro projeto, a saída da rede é a resposta. No segundo, a rede produz um número e uma regra de decisão o transforma em classe. O treino continua minimizando o MSE da saída linear, e não o número de acertos.
3. **O MSE não chega a zero no Iris.** Os alvos são -1 e +1 e a saída é uma reta, então sempre sobra algum erro, mesmo com todas as flores classificadas corretamente. Por isso 97,5% de acerto convive com um MSE de 0,13.
4. **Generalização.** No projeto de temperatura, a rede é conferida contra uma fórmula conhecida. No Iris, o conjunto de teste mostra como ela se comporta com flores que nunca viu.

---

## Extensão: várias unidades de temperatura

O arquivo `adaline_temperaturas.py` estende o Projeto 1: em vez de só Fahrenheit, a rede aprende **Fahrenheit, Kelvin, Rankine e Réaumur** ao mesmo tempo. Cada unidade é um neurônio ADALINE independente, com seu próprio peso e bias. Como todas essas conversões são retas, a rede aprende as quatro fórmulas com exatidão.

---

## Como executar

Instale as dependências:

```
python -m pip install numpy matplotlib scikit-learn
```

Execute cada projeto:

```
python temp_adaline.py
python adaline_iris.py
python adaline_temperaturas.py
```

O `scikit-learn` só é necessário para o Projeto 2, para carregar o dataset Iris.

Observação: o `plt.show()` fica no final de cada script e **bloqueia a execução** enquanto as janelas dos gráficos estiverem abertas. Feche os gráficos para o programa terminar.

---

## Limitações da ADALINE

- Só aprende **relações lineares**. Se as classes não puderem ser separadas por uma reta (como setosa x virginica quando se usam outras medidas, ou versicolor x virginica), o erro estaciona em um valor alto.
- Para problemas não lineares, é preciso redes com camadas ocultas (como o Perceptron multicamadas).
- É sensível à escala das entradas, por isso o `celsius / 100` no primeiro projeto e a padronização no segundo.

## Ideias para continuar

- Trocar as medidas do Iris (por exemplo, usar as larguras) e comparar os resultados.
- Tentar versicolor x virginica e observar o MSE estacionar em um valor alto.
- Variar a taxa de aprendizado e o número de épocas e ver o efeito na curva de erro.
- Fazer uma versão de regressão com outro dataset, como o de diabetes do scikit-learn.
