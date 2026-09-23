# deteccao-fraudes-transacoes
Este é um caderno de estudo do curso Bootcamp Bradesco da Dio



# Detecção de Fraude 

Este notebook é um roteiro de **análise antifraude** usando o dataset clássico
[creditcard.csv](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv)
(transações de cartão de crédito europeias, com colunas `V1`...`V28` geradas por PCA,
mais `Time`, `Amount` e o rótulo `Class` — onde `1` = fraude e `0` = transação normal).

Vamos cobrir, na ordem em que um analista faria isso no mundo real:

1. Carregar os dados e entender o **desbalanceamento de classes**
2. **Encontrar padrões** que diferenciam fraude de transação normal
3. Criar novas variáveis (**Feature Engineering**)
4. Preparar o **balanceamento de dados** 
5. Treinar uma **Regressão Logística**
6. Avaliar o modelo com as métricas certas (accuracy é enganosa aqui!)

> Rode este notebook no Google Colab ou Jupyter local com internet — ele baixa o
> CSV (~150MB) diretamente da URL do TensorFlow.

## 0. Instalar/Importar bibliotecas

Se faltar alguma lib, rode `!pip install imbalanced-learn` numa célula.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)

# imbalanced-learn é uma extensão do scikit-learn feita justamente
# para lidar com datasets desbalanceados (como fraude, doenças raras, churn, etc.)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

sns.set_style("whitegrid")
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
```

## 1. Carregando o dataset

Nada de mágico aqui: `pandas` lê o CSV direto da URL.

```python
URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(URL)

print("Formato do dataset:", df.shape)
df.head()
```

```python
df.info()
df.describe().T
```

## 2. O problema da classificação desbalanceada

Essa é a característica **mais importante** de qualquer dataset de fraude: fraudes são
raras. Se você treinar um modelo ingênuo que sempre prevê "não é fraude", ele já acerta
mais de 99% das vezes — e ainda assim é um modelo **inútil**, porque nunca pega uma
fraude sequer.

Vamos medir esse desbalanceamento.

```python
contagem = df["Class"].value_counts()
percentual = df["Class"].value_counts(normalize=True) * 100

resumo = pd.DataFrame({"contagem": contagem, "percentual (%)": percentual})
print(resumo)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(x="Class", data=df, ax=axes[0])
axes[0].set_title("Contagem absoluta (escala normal)")
axes[0].set_xticklabels(["Normal (0)", "Fraude (1)"])

sns.countplot(x="Class", data=df, ax=axes[1])
axes[1].set_yscale("log")
axes[1].set_title("Mesma contagem, escala log\n(para conseguir enxergar as fraudes)")
axes[1].set_xticklabels(["Normal (0)", "Fraude (1)"])
plt.tight_layout()
plt.show()
```

**O que isso significa na prática:**

- Se as fraudes são ~0,17% do dataset, um modelo "preguiçoso" atinge ~99,83% de
  *accuracy* sem aprender nada útil.
- Por isso, **nunca avaliamos modelo antifraude só por accuracy**. Vamos usar
  `precision`, `recall`, `F1`, `AUC-ROC` e principalmente `AUC-PR` (Precision-Recall),
  que é a métrica preferida quando a classe positiva é rara.
- E, na hora de treinar, precisamos **balancear** os dados ou avisar o algoritmo que
  as classes têm pesos diferentes (`class_weight`).

## 3. Encontrando padrões — análise exploratória

Agora o trabalho de detetive: quais variáveis realmente diferenciam fraude de
transação normal? Vamos olhar `Amount` (valor da transação), `Time` (segundos desde a
primeira transação do dataset) e a correlação de cada `V1`...`V28` com `Class`.

```python
# Estatísticas de Amount separadas por classe
print("Transações normais - Amount:")
print(df.loc[df["Class"] == 0, "Amount"].describe())
print("\nTransações fraudulentas - Amount:")
print(df.loc[df["Class"] == 1, "Amount"].describe())
```

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.boxplot(x="Class", y="Amount", data=df, ax=axes[0], showfliers=False)
axes[0].set_title("Distribuição do valor da transação por classe")
axes[0].set_xticklabels(["Normal", "Fraude"])

# Amount tem cauda longa (poucas transações com valores altíssimos), então
# olhamos em escala log para comparar melhor as distribuições
sns.histplot(df, x="Amount", hue="Class", bins=50, log_scale=(True, False),
             stat="density", common_norm=False, ax=axes[1])
axes[1].set_title("Densidade do valor (escala log) por classe")
plt.tight_layout()
plt.show()
```

```python
# Padrão temporal: fraudes acontecem mais em certos horários?
df["Hora_do_dia"] = (df["Time"] // 3600) % 24  # ver Feature Engineering na seção 4

fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df, x="Hora_do_dia", hue="Class", bins=24, stat="density",
             common_norm=False, multiple="dodge", ax=ax)
ax.set_title("Distribuição por hora do dia — Normal vs Fraude (densidade)")
plt.show()
```

```python
# Quais variáveis V1..V28 mais se correlacionam (em módulo) com a Class?
correlacoes = df.corr(numeric_only=True)["Class"].drop("Class")
top_correlacoes = correlacoes.abs().sort_values(ascending=False).head(10)

print("Top 10 variáveis mais correlacionadas com fraude:")
print(top_correlacoes)

fig, ax = plt.subplots(figsize=(8, 5))
correlacoes.loc[top_correlacoes.index].sort_values().plot(kind="barh", ax=ax)
ax.set_title("Correlação de cada variável com Class (sinal importa)")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.show()
```

**Como ler isso como analista:**

- Se uma variável `Vx` tem correlação forte (positiva ou negativa) com `Class`,
  ela separa bem fraude de não-fraude — é uma boa candidata para o modelo prestar
  atenção nela.
- Padrões de horário (`Hora_do_dia`) e de valor (`Amount`) ajudam a construir
  **regras de negócio** complementares ao modelo (ex.: alertas extras à noite,
  ou para valores fora do padrão).

## 4. Feature Engineering

O dataset já vem com `V1`...`V28` (resultado de PCA, então não têm significado direto de
negócio), mas conseguimos criar variáveis novas que ajudam o modelo a enxergar padrões
que ele não veria nas variáveis originais:

- **Hora_do_dia**: já criamos acima, a partir de `Time` (segundos totais).
- **Amount_log**: `Amount` tem distribuição bem assimétrica (poucos valores enormes);
  aplicar log comprime essa escala e ajuda modelos lineares como a Regressão Logística.
- **Amount_scaled** e **Time_scaled**: `V1`...`V28` já estão numa escala parecida
  (resultado do PCA), mas `Amount` e `Time` estão em escalas bem diferentes — é
  importante padronizar antes de usar num modelo linear.
- **Periodo_do_dia**: transformar a hora numérica em categorias (madrugada, manhã,
  tarde, noite) pode capturar padrões que não são lineares em relação à hora.

```python
df["Amount_log"] = np.log1p(df["Amount"])  # log1p evita erro com Amount = 0

def periodo_do_dia(hora):
    if hora < 6:
        return "madrugada"
    elif hora < 12:
        return "manha"
    elif hora < 18:
        return "tarde"
    else:
        return "noite"

df["Periodo_do_dia"] = df["Hora_do_dia"].apply(periodo_do_dia)
df = pd.get_dummies(df, columns=["Periodo_do_dia"], drop_first=True)

df[["Time", "Amount", "Amount_log", "Hora_do_dia"]].head()
```

> Dica: é fácil viajar demais no Feature Engineering. Sempre valide se a nova
> variável realmente melhora a métrica do modelo (comparando com/sem ela) — variável
> a mais também pode trazer ruído.

## 5. Preparando os dados para o modelo

Separamos `X` (features) de `y` (rótulo) e fazemos o split treino/teste. Um detalhe
**crucial** com dados desbalanceados: usar `stratify=y` para garantir que treino e
teste tenham a mesma proporção de fraudes.

```python
colunas_para_remover = ["Class", "Time"]  # Time bruto sai, já extraímos Hora_do_dia
X = df.drop(columns=colunas_para_remover)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Treino:", X_train.shape, "| % fraude:", y_train.mean() * 100)
print("Teste :", X_test.shape, "| % fraude:", y_test.mean() * 100)

# Padronizar Amount, Amount_log e Hora_do_dia (V1..V28 já vêm normalizadas do PCA)
colunas_para_escalar = ["Amount", "Amount_log", "Hora_do_dia"]
scaler = StandardScaler()
X_train[colunas_para_escalar] = scaler.fit_transform(X_train[colunas_para_escalar])
X_test[colunas_para_escalar] = scaler.transform(X_test[colunas_para_escalar])
```

## 6. Balanceamento de dados

Existem três abordagens comuns — vamos comparar todas:

1. **`class_weight="balanced"`**: não mexe nos dados, apenas diz ao algoritmo para
   "punir mais" os erros na classe rara durante o treino. Simples e um ótimo primeiro
   passo.
2. **Undersampling**: remove aleatoriamente transações normais até equilibrar as
   classes. Rápido, mas descarta informação (perde dados "bons").
3. **Oversampling com SMOTE**: cria exemplos sintéticos de fraude interpolando entre
   fraudes reais vizinhas, em vez de só duplicar. Geralmente funciona melhor que
   duplicar puro.

**Importante**: o balanceamento só deve ser aplicado no **treino**. O conjunto de
teste tem que continuar com a proporção real do mundo, senão a avaliação fica
mentirosa.

```python
print("Distribuição original no treino:", y_train.value_counts().to_dict())

# --- Undersampling ---
rus = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
print("Depois do undersampling:      ", y_train_under.value_counts().to_dict())

# --- Oversampling com SMOTE ---
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("Depois do SMOTE:              ", y_train_smote.value_counts().to_dict())
```

## 7. Regressão Logística

A Regressão Logística é um ótimo ponto de partida em antifraude: é rápida, fácil de
interpretar (os coeficientes mostram o peso de cada variável) e serve de linha de base
antes de tentar modelos mais complexos (Random Forest, XGBoost etc.).

Vamos treinar 3 versões e comparar:

- **(A)** dados originais + `class_weight="balanced"`
- **(B)** dados com undersampling
- **(C)** dados com SMOTE

```python
def treinar_e_avaliar(X_tr, y_tr, X_te, y_te, nome):
    modelo = LogisticRegression(max_iter=1000, class_weight=None)
    modelo.fit(X_tr, y_tr)

    y_pred = modelo.predict(X_te)
    y_proba = modelo.predict_proba(X_te)[:, 1]

    print(f"\n===== {nome} =====")
    print(classification_report(y_te, y_pred, target_names=["Normal", "Fraude"], digits=3))
    print("AUC-ROC :", round(roc_auc_score(y_te, y_proba), 4))
    print("AUC-PR  :", round(average_precision_score(y_te, y_proba), 4))

    return modelo, y_pred, y_proba

# (A) class_weight="balanced", sem reamostrar os dados
modelo_a = LogisticRegression(max_iter=1000, class_weight="balanced")
modelo_a.fit(X_train, y_train)
y_pred_a = modelo_a.predict(X_test)
y_proba_a = modelo_a.predict_proba(X_test)[:, 1]
print("===== (A) class_weight='balanced' =====")
print(classification_report(y_test, y_pred_a, target_names=["Normal", "Fraude"], digits=3))
print("AUC-ROC :", round(roc_auc_score(y_test, y_proba_a), 4))
print("AUC-PR  :", round(average_precision_score(y_test, y_proba_a), 4))

# (B) Undersampling
modelo_b, y_pred_b, y_proba_b = treinar_e_avaliar(
    X_train_under, y_train_under, X_test, y_test, "(B) Undersampling"
)

# (C) SMOTE
modelo_c, y_pred_c, y_proba_c = treinar_e_avaliar(
    X_train_smote, y_train_smote, X_test, y_test, "(C) SMOTE"
)
```

## 8. Avaliando corretamente: por que accuracy engana

`precision`, `recall` e `F1` da classe **Fraude** são o que realmente importa:

- **Recall (sensibilidade)**: de todas as fraudes reais, quantas o modelo pegou?
  Baixo recall = muita fraude passando despercebida.
- **Precision**: de tudo que o modelo marcou como fraude, quanto realmente era fraude?
  Baixa precision = muitos clientes bons sendo bloqueados/incomodados à toa.
- Em antifraude normalmente há um trade-off: aumentar recall (pegar mais fraude)
  costuma reduzir precision (mais falsos alarmes) — a escolha do ponto de corte
  depende do custo de negócio de cada erro.

Vamos visualizar a matriz de confusão e a curva Precision-Recall do melhor modelo.

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, y_pred, titulo in zip(
    axes, [y_pred_a, y_pred_b, y_pred_c],
    ["(A) class_weight", "(B) Undersampling", "(C) SMOTE"]
):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Fraude"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(titulo)

plt.tight_layout()
plt.show()
```

```python
fig, ax = plt.subplots(figsize=(7, 5))

for y_proba, nome in zip(
    [y_proba_a, y_proba_b, y_proba_c],
    ["class_weight", "Undersampling", "SMOTE"]
):
    precisao, revocacao, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    ax.plot(revocacao, precisao, label=f"{nome} (AUC-PR={ap:.3f})")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Curva Precision-Recall — comparação das estratégias de balanceamento")
ax.legend()
plt.show()
```

## 9. Coeficientes do modelo — o que ele "aprendeu"

Como a Regressão Logística é linear, dá pra olhar direto quais variáveis mais pesam
na decisão (coeficiente positivo empurra para "fraude", negativo empurra para
"normal").

```python
coeficientes = pd.Series(modelo_c.coef_[0], index=X_train.columns)
top_coef = coeficientes.reindex(coeficientes.abs().sort_values(ascending=False).index).head(15)

fig, ax = plt.subplots(figsize=(8, 6))
top_coef.sort_values().plot(kind="barh", ax=ax)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Top 15 variáveis por peso no modelo (SMOTE)")
plt.tight_layout()
plt.show()
```

## 10. Conclusão e próximos passos

Resumo do fluxo de trabalho de um analista antifraude:

1. **Sempre** meça o desbalanceamento antes de treinar qualquer coisa.
2. Investigue padrões (valor, horário, variáveis mais correlacionadas) antes de
   sair treinando modelo — isso também gera insights para regras de negócio.
3. Crie features que exponham esses padrões de forma mais direta para o modelo.
4. Trate o desbalanceamento (`class_weight`, undersampling, SMOTE) e **sempre**
   avalie com `precision`, `recall`, `F1` e `AUC-PR` — nunca só accuracy.
5. Regressão Logística é uma ótima baseline interpretável. Depois, vale comparar
   com Random Forest, Gradient Boosting (XGBoost/LightGBM) e ver se ganham
   AUC-PR de forma relevante — geralmente ganham, mas perdem interpretabilidade.
6. Na prática, o ponto de corte (threshold) de decisão deve ser escolhido junto do
   time de negócio, pesando custo de falso positivo (cliente incomodado) vs
   falso negativo (fraude não detectada).
