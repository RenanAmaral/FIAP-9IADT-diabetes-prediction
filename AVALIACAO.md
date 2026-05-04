# Avaliação Comparativa dos Modelos

## 1. Contexto

Foram treinados quatro modelos de classificação binária para predição de diabetes (**diabético vs. não diabético**) usando dados do **NHANES 2021-2023**.

O problema apresenta **desbalanceamento de classes**, com predominância da classe negativa, o que torna inadequado avaliar desempenho apenas por acurácia. Em contexto clínico, a principal preocupação é evitar **falsos negativos**, ou seja, pacientes diabéticos não identificados pelo sistema.

O objetivo deste projeto não é substituir o diagnóstico médico, mas atuar como ferramenta de **triagem** e **apoio à decisão clínica**.

---

## 2. Métricas Utilizadas

### Accuracy

Representa a proporção total de previsões corretas:

\[
Accuracy = \frac{VP + VN}{VP + VN + FP + FN}
\]

Apesar de útil, essa métrica pode ser enganosa em datasets desbalanceados, pois um modelo pode obter alta acurácia apenas prevendo a classe majoritária.

### Precision

Indica, dentre os indivíduos classificados como diabéticos, quantos realmente eram diabéticos:

\[
Precision = \frac{VP}{VP + FP}
\]

Clinicamente, uma precision mais alta significa menos alarmes falsos.

### Recall

Indica, dentre os diabéticos reais, quantos foram corretamente identificados:

\[
Recall = \frac{VP}{VP + FN}
\]

Essa é a métrica mais importante neste projeto. Em triagem clínica, é mais grave não identificar um paciente com diabetes do que gerar um falso positivo.

### F1-score

É a média harmônica entre precision e recall:

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

Essa métrica é útil quando há necessidade de equilíbrio entre sensibilidade e precisão.

### AUC-ROC

A AUC-ROC mede a capacidade do modelo de separar as duas classes em diferentes thresholds.

- **1.0** = separação perfeita
- **0.5** = equivalente ao acaso

É uma métrica robusta para comparação entre classificadores, principalmente em cenários de desbalanceamento.

---

## 3. Modelos Avaliados

Foram comparados quatro algoritmos:

- Logistic Regression
- Random Forest
- SVM Linear
- KNN

Todos foram avaliados com:
- conjunto de teste
- validação cruzada estratificada
- análise das métricas de classificação
- comparação crítica com foco em contexto clínico

---

## 4. Resultado Comparativo

| Modelo | Accuracy | Precision (Diab.) | Recall (Diab.) | F1 (Diab.) | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 70,49% | 0,32 | **0,73** | **0,44** | 0,8001 |
| Random Forest | **83,93%** | 0,49 | 0,10 | 0,16 | **0,8216** |
| SVM Linear | 84,54% | **0,56** | 0,16 | 0,25 | 0,7997 |
| KNN | 83,87% | 0,42 | 0,02 | 0,04 | 0,7598 |

---

## 5. Análise Crítica por Modelo

### Logistic Regression

Foi o modelo mais adequado para o problema proposto.

Pontos fortes:
- melhor recall entre todos os modelos
- melhor F1-score
- AUC-ROC competitiva
- comportamento mais alinhado com triagem clínica

Ponto de atenção:
- acurácia inferior aos demais modelos

Interpretação:
A menor acurácia é consequência direta da priorização da classe minoritária, especialmente com o uso de `class_weight='balanced'`. Em um problema clínico, essa troca é aceitável e desejável, porque reduz falsos negativos.

### Random Forest

Apresentou a melhor AUC-ROC, o que indica boa capacidade de discriminar as classes.

Pontos fortes:
- maior AUC-ROC
- bom potencial para interpretação por feature importance

Pontos fracos:
- recall muito baixo para uso clínico direto
- grande parte dos diabéticos reais não foi detectada no threshold padrão

Interpretação:
O modelo parece separar bem as probabilidades, mas o threshold padrão de decisão é conservador demais. Isso sugere potencial de melhoria com ajuste de limiar.

### SVM Linear

Apresentou desempenho intermediário.

Pontos fortes:
- maior precision
- AUC próxima à da Logistic Regression

Pontos fracos:
- recall ainda insuficiente para triagem clínica

Interpretação:
Embora seja mais conservador e erre menos nos positivos previstos, deixa passar muitos diabéticos reais.

### KNN

Foi o modelo com pior desempenho geral.

Pontos fracos:
- recall praticamente nulo
- pior F1-score
- pior AUC-ROC

Interpretação:
O KNN mostrou baixa capacidade de generalização neste problema, provavelmente devido ao desbalanceamento da classe positiva e à alta dimensionalidade das features.

---

## 6. Ranking Final

| Posição | Modelo | Justificativa |
|---|---|---|
| 1º | Logistic Regression | Melhor recall, melhor F1 e maior utilidade para triagem |
| 2º | Random Forest | Melhor AUC-ROC e bom potencial após ajuste de threshold |
| 3º | SVM Linear | Boa precisão, mas recall limitado |
| 4º | KNN | Desempenho insuficiente para a classe positiva |

---

## 7. Interpretação do Modelo

Como etapa de interpretabilidade, foi utilizada **feature importance** no modelo Random Forest para identificar as variáveis mais relevantes para a classificação.

Essa análise é importante porque:
- aumenta a transparência do modelo
- ajuda a verificar coerência clínica
- permite discutir se o modelo está aprendendo padrões plausíveis

Esse ponto é essencial em aplicações médicas, em que modelos puramente “caixa-preta” tendem a ser menos aceitos sem explicação adequada.

---

## 8. Aderência ao Contexto Clínico

O modelo escolhido para melhor desempenho prático foi a **Logistic Regression**, mesmo não tendo a maior acurácia.

Justificativa:
- em triagem clínica, o custo de um falso negativo é mais alto que o de um falso positivo
- pacientes classificados incorretamente como diabéticos ainda podem ser avaliados posteriormente
- pacientes diabéticos não identificados podem permanecer sem tratamento ou acompanhamento

Portanto, o recall foi priorizado como principal critério de decisão.

---

## 9. Limitações do Projeto

- Dados observacionais e populacionais
- Desbalanceamento entre as classes
- Avaliação feita sobre uma única base
- Ausência de validação externa

---

## 10. Melhorias Futuras

1. Ajuste de Threshold de Decisão (Random Forest):
Apesar de apresentar a maior AUC-ROC (~0,82), o Random Forest possui recall baixo (~10%), indicando que o threshold padrão de 0,5 é conservador; reduzir o limiar (ex.: ~0,3) pode aumentar significativamente o recall com impacto controlado na precision.
    
    ```python
        threshold = 0.30
        y_pred_rf_ajustado = (y_prob_rf >= threshold).astype(int)
    ```
2. Técnicas de Balanceamento (SMOTE):
O uso de class_weight='balanced' ajusta os pesos da função de perda, mas não altera a distribuição dos dados; técnicas como SMOTE podem gerar amostras sintéticas da classe minoritária, melhorando o aprendizado dos modelos.
    ```python
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
    ```
3. Seleção de Variáveis e Redução de Dimensionalidade: 
A alta dimensionalidade impacta modelos como KNN; estratégias como PCA, seleção via feature importance e remoção de variáveis altamente correlacionadas (ex.: IMC, peso, cintura) podem melhorar o desempenho.
4. Modelos Ensemble Avançados:
Modelos como XGBoost e LightGBM tendem a apresentar melhor desempenho em dados tabulares e lidam melhor com desbalanceamento; técnicas de stacking também podem combinar o alto recall da Logistic Regression com a alta AUC do Random Forest.
5. Otimização de Hiperparâmetros:
Os modelos foram utilizados com configurações padrão; a aplicação de GridSearchCV ou RandomizedSearchCV, com foco em recall ou F1-score, pode resultar em melhorias relevantes de desempenho.
6. Curva Precision-Recall:
A inclusão da curva Precision-Recall complementa a análise, sendo mais adequada para datasets desbalanceados, pois penaliza mais os erros na classe minoritária.
---

## 11. Conclusão

O projeto demonstrou a viabilidade da aplicação de técnicas de Machine Learning na predição de diabetes a partir de dados médicos estruturados.

Entre os modelos avaliados, a Logistic Regression apresentou o melhor desempenho para o problema proposto, destacando-se pelo maior equilíbrio entre as métricas analisadas, especialmente pela alta sensibilidade na detecção da classe positiva.

Os resultados obtidos evidenciam a importância da escolha adequada das métricas em cenários com desbalanceamento entre classes, bem como o impacto direto dessa escolha na interpretação e utilização dos modelos.
