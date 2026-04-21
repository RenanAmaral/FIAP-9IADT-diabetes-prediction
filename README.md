# Predição de Diabetes com Dados NHANES 2021-2023

Projeto de Machine Learning para predição de diabetes utilizando dados populacionais do **NHANES (2021-2023)**.

---

## 🔗 Link do Repositório

https://github.com/RenanAmaral/FIAP-9IADT-diabetes-prediction

---

## 👥 Autores

- Renan Ribeiro do Amaral — RM: 370618
- Marcos Tadeu Carmona — RM: 370618  

---

## 🎯 Objetivo

Desenvolver um modelo de Machine Learning capaz de classificar indivíduos adultos como **diabéticos ou não diabéticos**, a partir de dados estruturados, sem utilizar diretamente critérios diagnósticos como features.

---

## 📊 Fonte dos Dados

NHANES (CDC):  
https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023

---

## 🧠 Discussão da Análise Exploratória

A análise exploratória evidenciou:

- Presença de **valores ausentes** em diferentes variáveis  
- **Desbalanceamento entre classes**, com predominância de não diabéticos  
- **Correlação entre variáveis antropométricas**, como IMC, peso e circunferência  
- Distribuições assimétricas em variáveis laboratoriais  

Esses fatores impactam diretamente o desempenho dos modelos e orientaram as decisões de pré-processamento.

---

## ⚙️ Estratégias de Pré-processamento

Foram aplicadas as seguintes técnicas:

- Tratamento de valores ausentes com imputação  
- Padronização das variáveis numéricas (**StandardScaler**)  
- Criação da variável target com base em critérios clínicos  
- Remoção de colunas que causariam **data leakage**  
- Uso de **Pipeline** para garantir consistência no fluxo  
- Separação treino/teste com **estratificação**  

---

## 🤖 Modelos Utilizados e Justificativa

Foram avaliados quatro modelos:

- **Logistic Regression** → baseline interpretável e robusto  
- **Random Forest** → captura relações não lineares  
- **SVM Linear** → bom desempenho em espaços de alta dimensão  
- **KNN** → modelo baseado em distância para comparação  

A escolha buscou comparar abordagens lineares e não lineares em dados tabulares.

---

## 📈 Resultados Obtidos

### Métricas avaliadas:
- Accuracy  
- Recall  
- F1-score  
- AUC-ROC  

### Principais resultados:

- **Logistic Regression**
  - Melhor recall (~0.73)
  - Melhor equilíbrio geral (F1)

- **Random Forest**
  - Melhor AUC (~0.82)
  - Baixo recall (~0.10)

- **SVM**
  - Alta precisão
  - Recall limitado

- **KNN**
  - Pior desempenho geral

---

## 📊 Análises e Interpretação dos Resultados

- A **Logistic Regression** foi o modelo mais adequado para o problema, devido à sua maior capacidade de identificar a classe positiva  
- O **Random Forest**, apesar da alta AUC, mostrou-se conservador no threshold padrão  
- O **desbalanceamento entre classes** impactou fortemente os modelos, especialmente o KNN  
- A escolha da métrica **recall** foi fundamental para avaliar corretamente o problema  

Gráficos gerados:
- Curva ROC  
- Matriz de confusão  
- Feature importance  
- Validação cruzada  

---

## ⚙️ Execução do Projeto

🐳 Execução com Docker
docker build -t fiap-diabetes-prediction .
docker run -p 8888:8888 fiap-diabetes-prediction

### 🔹 Execução Local

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook

