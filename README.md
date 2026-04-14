# Predição de Diabetes com Dados NHANES 2021-2023

Projeto de Machine Learning para predição de diabetes usando dados populacionais do **NHANES (National Health and Nutrition Examination Survey)** — ciclo 2021-2023, coletados pelo CDC (Centers for Disease Control and Prevention).

## Objetivo

Classificar indivíduos adultos como **diabéticos ou não diabéticos** a partir de variáveis demográficas, antropométricas, laboratoriais e de estilo de vida, sem utilizar diretamente os critérios diagnósticos clínicos como features.

## Dados

Os dados são públicos e disponíveis em: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023

| Dataset | Arquivo | Conteúdo |
|---|---|---|
| Demográfico | `DEMO_L.xpt` | Idade, sexo, raça, escolaridade, renda |
| Antropométrico | `BMX_L.xpt` | IMC, cintura, peso, altura, quadril, braço |
| Pressão arterial | `BPXO_L.xpt` | 3 medições de pressão sistólica e diastólica |
| Diabetes (questionário) | `DIQ_L.xpt` | Diagnóstico médico de diabetes |
| Hemoglobina glicada | `GHB_L.xpt` | HbA1c (%) |
| Glicemia de jejum | `GLU_L.xpt` | Glicose plasmática (mg/dL) |
| Colesterol | `TCHOL_L.xpt` | Colesterol total |
| Condições médicas | `MCQ_L.xpt` | Artrite, AVC, infarto, doenças cardíacas |
| Atividade física | `PAQ_L.xpt` | Minutos de atividade física/sedentarismo |
| Tabagismo | `SMQ_L.xpt` | Histórico e status de fumante |

## Variável Target

A variável `DIABETES` é criada combinando **3 critérios diagnósticos da ADA (American Diabetes Association)**:

- **Critério A:** Diagnóstico médico reportado (`DIQ010 == 1`)
- **Critério B:** HbA1c ≥ 6,5% (`LBXGH >= 6.5`)
- **Critério C:** Glicemia de jejum ≥ 126 mg/dL (`LBXGLU >= 126`)

As colunas dos critérios são removidas antes do treinamento para evitar *data leakage*.

## Pipeline do Notebook

O notebook `src/modelo.ipynb` executa as seguintes etapas:

1. **Carregamento e JOIN** dos 10 datasets via `SEQN`
2. **Criação do target** com os 3 critérios ADA + filtro adultos ≥ 18 anos
3. **Limpeza** de códigos especiais NHANES (recusa/não sabe → `NaN`)
4. **Análise de missings** por coluna
5. **Feature engineering**: média das pressões, preenchimento do SMQ040
6. **Matriz de correlação** com nomes legíveis em português
7. **Pré-processamento**: imputação (mediana/moda) + StandardScaler via Pipeline
8. **Split estratificado** 80/20
9. **KNN** — Curva de Validação do K (cross-val 5-fold) + treinamento com K ótimo
10. **SVM Linear** — LinearSVC + CalibratedClassifierCV
11. **Logistic Regression** — com `class_weight='balanced'`
12. **Random Forest** — 200 estimadores com `class_weight='balanced'`
13. **Avaliação comparativa** — Classification Report + AUC-ROC

## Resultados (base de teste)

| Modelo | AUC-ROC | Recall (Diabético) |
|---|---|---|
| Logistic Regression | ~0.80 | ~73% |
| Random Forest | ~0.82 | ~10% |
| SVM Linear | — | — |
| KNN | ~0.74 | ~6% |

> O **Recall** é a métrica prioritária neste contexto clínico: é pior deixar de identificar um diabético (falso negativo) do que classificar erroneamente um não-diabético (falso positivo).

## Estrutura do Projeto

```
FIAP-9IADT-diabetes-prediction/
├── datasets/           # Arquivos .xpt do NHANES (não versionados)
├── src/
│   └── modelo.ipynb    # Notebook principal
├── requirements.txt
└── README.md
```

## Configuração do Ambiente

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Tecnologias

- Python 3.12
- pandas, numpy
- scikit-learn (Pipeline, ColumnTransformer, KNN, SVM, LR, RF)
- matplotlib, seaborn
