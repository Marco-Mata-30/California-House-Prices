# California-House-Prices

## Projeto de Machine Learning – Predição de Preços de casas na Califórnia

A base "California Housing Prices", apresentada no livro Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow, contém informações sobre o mercado imobiliário na Califórnia, com o objetivo principal de prever o valor mediano das casas em diferentes regiões.
Este projeto foi desenvolvido utilizando técnicas de **Machine Learning supervisionado** para prever a **probabilidade de sobrevivência dos passageiros**, com base em atributos como:

- Coordenada geográfica de longitude da localização;

- Coordenada geográfica de latitude da localização;

- Idade mediana das construções residenciais na área;

- Total de cômodos por área;

- Total de quartos por área;

- População total da área;

- Número de domicílios na área;

- Renda mediana da área, em dezenas de milhares de dólares;

- Valor mediano dos imóveis na área (em dólares);

- Proximidade com o oceano;

O pipeline completo inclui:

- **Análise exploratória de dados (EDA)**  
- **Pré-processamento e limpeza dos dados**  
- **Seleção de features relevantes**  
- **Treinamento de modelos supervisionados**
- **Avaliação de desempenho com métricas apropriadas**

---

## Como instalar e executar

### Clone o repositório:

```bash
git clone https://github.com/Marco-Mata-30/California-House-Prices.git
```

Em seguida, execute o código Python após instalar os pré-requisitos.

## Pré-requisitos

Certifique-se de ter o **Python** instalado e, em seguida, instale as bibliotecas necessárias:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
```

Você pode instalar as bibliotecas com:

```bash
pip install pandas numpy seaborn matplotlib plotly scikit-learn
```
## Resultado
O modelo final de Regressão Random Forest MAE de 31.852,70 e R² igual a 0,8217.
https://www.kaggle.com/code/marcomata/california-house-predict

## Autor

Marco Antonio Silva da Mata - MM
