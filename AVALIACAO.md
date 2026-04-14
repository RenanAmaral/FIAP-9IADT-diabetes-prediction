# Avaliação Comparativa dos Modelos

## Contexto

Foram treinados 4 modelos de classificação binária (diabético vs. não diabético) usando dados do NHANES 2021-2023, com **8.150 adultos** (84% não diabéticos, 16% diabéticos). A avaliação foi realizada sobre o conjunto de teste (**1.630 amostras**, split estratificado 80/20).

O desbalanceamento da classe positiva (16%) é um fator importante que influencia diretamente o desempenho dos modelos.

---

## Explicação das Métricas

### Accuracy (Acurácia)

Percentual de previsões corretas sobre o total:

$$Accuracy = \frac{VP + VN}{VP + VN + FP + FN}$$

**Limitação**: Em datasets desbalanceados, um modelo que prevê "Não Diabético" para todos atinge ~84% de acurácia sem aprender nada. Por isso, não deve ser a métrica principal.

### Precision (Precisão)

Dos que o modelo classificou como diabéticos, quantos realmente são:

$$Precision = \frac{VP}{VP + FP}$$

**Interpretação clínica**: Precision alta = poucos alarmes falsos. Se um paciente for classificado como diabético, há alta confiança de que realmente é.

### Recall (Sensibilidade / Revocação)

Dos diabéticos reais, quantos o modelo conseguiu identificar:

$$Recall = \frac{VP}{VP + FN}$$

**Interpretação clínica**: Recall alta = poucos diabéticos passam despercebidos. **Métrica mais crítica neste contexto** — deixar de diagnosticar diabetes (falso negativo) tem consequências graves para a saúde do paciente.

### F1-Score

Média harmônica entre Precision e Recall. Útil quando ambas são importantes:

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**Interpretação**: Penaliza desbalanceamento entre precision e recall. Um modelo com precision=0.90 e recall=0.02 terá F1 muito baixo (~0.04), expondo que ele quase não detecta a classe positiva.

### AUC-ROC (Area Under the ROC Curve)

Área sob a curva ROC, que plota a taxa de verdadeiros positivos vs. falsos positivos em diferentes limiares de decisão:

- **1.0** = modelo perfeito
- **0.5** = equivalente a jogar uma moeda
- **< 0.5** = pior que aleatório

**Vantagem**: Avalia a capacidade discriminativa do modelo independentemente do threshold de decisão (0.5 por padrão). É a métrica mais robusta para comparar modelos em datasets desbalanceados.

---

## Resultados

| Modelo | Accuracy | Precision (Diab.) | Recall (Diab.) | F1 (Diab.) | AUC-ROC |
|---|---|---|---|---|---|
| **Logistic Regression** | 70,49% | 0,32 | **0,73** | **0,44** | 0,8001 |
| **Random Forest** | **83,93%** | **0,49** | 0,10 | 0,16 | **0,8216** |
| **SVM Linear** | 84,54% | 0,56 | 0,16 | 0,25 | 0,7997 |
| **KNN (k=29)** | 83,87% | 0,42 | 0,02 | 0,04 | 0,7598 |

---

## Análise por Modelo

### Logistic Regression — Melhor modelo para uso clínico

- **Recall de 73%**: identifica quase 3 de cada 4 diabéticos reais — muito superior aos demais
- **AUC-ROC de 0,80**: boa capacidade discriminativa
- **Accuracy de 70%**: a mais baixa, porém isso é um reflexo direto da estratégia `class_weight='balanced'`, que sacrifica acurácia global para aumentar a detecção da classe minoritária
- **Precision de 32%**: a cada 3 alertas de diabetes, 1 é correto — o trade-off pela alta sensibilidade
- **F1 de 0,44**: o melhor equilíbrio entre precision e recall entre todos os modelos

**Por que se destaca**: Em contexto clínico, é preferível ter falsos positivos (que serão descartados com exames adicionais) a ter falsos negativos (diabéticos não diagnosticados que podem sofrer complicações graves).

### Random Forest — Melhor AUC-ROC, mas recall insuficiente

- **AUC-ROC de 0,82**: a mais alta entre todos, indicando boa capacidade de separação probabilística
- **Recall de apenas 10%**: apesar do `class_weight='balanced'`, o modelo ainda é conservador — 90% dos diabéticos passam despercebidos
- **Precision de 49%**: quando alerta, acerta quase metade das vezes
- **Paradoxo AUC alta + recall baixo**: o modelo tem boa separação probabilística (as probabilidades discriminam bem), mas o threshold padrão de 0,5 é muito alto. Ajustar o threshold poderia melhorar significativamente o recall

### SVM Linear — Desempenho intermediário

- **Recall de 16%**: melhor que RF e KNN, mas muito inferior ao LR
- **AUC-ROC de 0,80**: praticamente igual à Logistic Regression
- **Precision de 56%**: a mais alta entre todos — quando alerta, geralmente está certo
- **Limitação**: modelo linear com margem máxima que, neste dataset, não encontrou uma separação eficiente para a classe minoritária mesmo com `class_weight='balanced'`

### KNN (k=29) — Pior desempenho geral

- **Recall de 2%**: praticamente não detecta diabéticos (identifica apenas ~5 dos ~261 do teste)
- **AUC-ROC de 0,76**: a mais baixa, indicando capacidade discriminativa limitada
- **F1 de 0,04**: o pior entre todos os modelos
- **Por que falhou**: KNN é sensível à "maldição da dimensionalidade" — com 25 features, a noção de "distância" perde significado. Além disso, o desbalanceamento 84/16 faz a maioria dos vizinhos serem não diabéticos, dominando a votação mesmo com `weights='distance'`

---

## Ranking Final

| Posição | Modelo | Justificativa |
|---|---|---|
| 1º | **Logistic Regression** | Melhor recall (73%), melhor F1, AUC competitiva. Ideal para triagem clínica |
| 2º | **Random Forest** | Melhor AUC (0,82), potencial de melhoria com ajuste de threshold |
| 3º | **SVM Linear** | Melhor precision, mas recall baixo limita utilidade clínica |
| 4º | **KNN (k=29)** | Recall quase nulo, inadequado para este problema |

---

## Oportunidades de Melhoria

### 1. Ajuste de Threshold (Random Forest)

O Random Forest tem a melhor AUC-ROC (0,82) mas recall de apenas 10%. Isso significa que as probabilidades previstas são discriminativas, mas o limiar padrão de 0,5 é conservador demais. Reduzir o threshold para ~0,3 pode elevar o recall significativamente, com trade-off controlado na precision.

```python
# Em vez de predict() com threshold fixo de 0.5:
threshold = 0.30
y_pred_rf_ajustado = (y_prob_rf >= threshold).astype(int)
```

### 2. Técnicas de Balanceamento (SMOTE)

O `class_weight='balanced'` ajusta os pesos da loss, mas não altera a distribuição dos dados. Técnicas de oversampling como **SMOTE** (Synthetic Minority Oversampling Technique) criam exemplos sintéticos da classe minoritária, potencialmente melhorando o aprendizado dos modelos baseados em árvore e distância.

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

smote = SMOTE(random_state=42)
# Integrar ao pipeline via imblearn
```

### 3. Feature Selection / Redução de Dimensionalidade

O KNN sofre com a maldição da dimensionalidade (25 features). Opções:
- **PCA**: reduzir para 10-15 componentes antes do KNN
- **Feature importance do Random Forest**: selecionar apenas as top 10-15 features mais relevantes
- **Remoção de features multicolineares**: cintura, peso, quadril e IMC são altamente correlacionados — manter apenas 1-2 dessas medidas pode ajudar

### 4. Modelos Ensemble Avançados

- **XGBoost / LightGBM**: gradient boosting geralmente supera Random Forest em dados tabulares, com melhor tratamento nativo de desbalanceamento via `scale_pos_weight`
- **Stacking**: combinar as previsões do LR (bom recall) com RF (bom AUC) em um meta-learner

### 5. Otimização de Hiperparâmetros (GridSearchCV / RandomizedSearchCV)

Os modelos foram treinados com hiperparâmetros padrão ou mínimos. Um grid search otimizando para `recall` ou `f1` poderia encontrar combinações melhores:

- **Random Forest**: `max_depth`, `min_samples_leaf`, `n_estimators`, `max_features`
- **Logistic Regression**: `C` (regularização), `penalty` (l1/l2)
- **SVM Linear**: `C`

### 6. Curva Precision-Recall

Complementar a análise com a **curva Precision-Recall** (PR curve), que é mais informativa que a ROC em datasets desbalanceados. A área sob a curva PR (AP — Average Precision) penaliza mais os modelos que falham na classe minoritária.

### 7. Validação Cruzada na Avaliação Final

A avaliação atual usa um único split fixo de teste. Adotar **cross-validation estratificado de 5 ou 10 folds** na avaliação final daria uma estimativa mais robusta do desempenho, com intervalos de confiança para cada métrica.
