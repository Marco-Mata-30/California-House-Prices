import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io

st.title('Regressão - Base de dados de casas na Califórnia (1990)')
st.markdown('Como prever o preço de uma casa usando modelos de *Machine Learning*?')
st.markdown('---')
st.markdown('''Esse projeto foi desenvolvido como inspiração do livro "*Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*" de Aurélien Géron. Ainda nos primeiros capítulos do livro, o autor apresenta uma introdução ao Machine Learning e aos conceitos de regressão.
    O objetivo é prever o preço das casas em regiões da Califórnia usando modelos de regressão. A base de dados utilizada é a "California Housing Prices" do **Kaggle**, que contém informações sobre casas na Califórnia em 1990.''')

st.markdown('---')

house_db = pd.read_csv("housing.csv")

st.subheader('EDA (*Exploratory Data Analysis*):')

st.markdown('''**Segue a descrição da base de dados:**
1. **longitude** — Coordenada geográfica de longitude da localização;

2. **latitude** — Coordenada geográfica de latitude da localização;

3. **housing_median_age** — Idade mediana das construções residenciais na área;

4. **total_rooms** — Total de cômodos por área.;

5. **total_bedrooms** — Total de quartos por área;

6. **population** — População total da área;

7. **households** — Número de domicílios na área;

8. **median_income** — Renda mediana da área, em dezenas de milhares de dólares;

9. **median_house_value** — Valor mediano dos imóveis na área (em dólares);

**⚠️ Observação importante:** essa variável possui um valor máximo truncado em $500.000,00. Isso significa que imóveis com valores superiores foram limitados a esse teto, o que pode introduzir um viés nos modelos de regressão e limitar sua capacidade de prever valores muito altos com precisão.

10 **ocean_proximity** — Proximidade com o oceano, uma variável categórica que indica se a área está:

* *NEAR BAY*: Próximo à baía;

* *<1H OCEAN*: Menos de 1 hora do oceano;

* *INLAND*: Interior, longe do oceano;

* *NEAR OCEAN*: Próximo ao oceano;

* *ISLAND*: Em uma ilha.''')

st.dataframe(house_db.head(10))

st.markdown('Logo no início da nossa exploração da base de dados, podemos observar que ela possui alguns valores ausentes, especialmente na coluna "**total_bedrooms**". Isto pode impactar a análise e os modelos de regressão que iremos construir. Portanto, é importante lidar com esses valores ausentes antes de prosseguir com a modelagem.')

buffer = io.StringIO()
house_db.info(buf=buffer)
s = buffer.getvalue()
st.code(s, language='text')

st.markdown('''Para lidar com os valores ausentes, iremos usar o método de preenchimento a partir da Regressão Polinomial de grau 2. 
Esse método é eficaz para preencher valores ausentes em colunas numéricas, como "**total_bedrooms**", utilizando outras colunas como preditores. A ideia é treinar um modelo de regressão com as colunas disponíveis e prever os valores ausentes com base nesse modelo.
A escolha desse método é justificada por conta de sua eficiência em lidar com dados numéricos e sua capacidade de capturar relações entre as variáveis, o que pode resultar em previsões mais precisas do que métodos mais simples, como a média ou mediana.

Uma prova de sua eficiência e precisão são os resultados obtidos a partir das avaliações de desempenho do modelo, que mostram que o método de regressão é capaz de prever os valores ausentes com uma precisão razoável, minimizando o impacto dos dados ausentes na análise geral da base de dados:
''')

st.markdown('''```
MAE: 25.22
MSE: 1685.34
RMSE: 41.05
R²: 0.9906
```''')

house_db_regressao = pd.read_csv("housing_regr.csv")

st.markdown('''Após o preenchimento dos valores ausentes, podemos prosseguir com a análise exploratória e a construção dos modelos de regressão.

Nossa primeira análise será a montagem de um **mapa da Califórnia**, onde cada ponto representa uma casa, e a cor do ponto indica o valor mediano da casa, e o tamanho do ponto indica a população da área. Isso nos permitirá visualizar a distribuição dos preços das casas na região e identificar possíveis padrões geográficos.''')

# Criando a figura e o gráfico
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    house_db_regressao['longitude'],
    house_db_regressao['latitude'],
    c=house_db_regressao['median_house_value'],
    cmap='viridis',
    alpha=0.5,
    s=house_db_regressao['population'] / 100
)

# Configurações do gráfico
cbar = plt.colorbar(scatter, ax=ax, label='Valor Mediano das Casas')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Mapa de Calor dos Valores das Casas na Califórnia')

# Exibindo no Streamlit
st.pyplot(fig)
st.markdown('''Nessa primeira análise, podemos observar que casas localizadas próximas ao oceano tendem a ter valores medianos mais altos, enquanto casas localizadas no interior tendem a ter valores medianos mais baixos. Ao mesmo tempo, a população da área também parece influenciar o valor mediano das casas, com áreas mais populosas apresentando valores medianos mais altos.''')

st.markdown('''Seguindo para nossa próxima análise iremos montar **gráficos de dispersão** para analisar a relação entre as variáveis numéricas da base de dados.
Esses gráficos nos ajudarão a identificar correlações entre as variáveis e a entender melhor como elas se relacionam com o valor mediano das casas.
Abaixo estão os gráficos de dispersão para as variáveis 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population': ''')

# Criando a matriz de dispersão com Plotly
grafico = px.scatter_matrix(
    house_db_regressao,
    dimensions=['housing_median_age', 'total_rooms', 'total_bedrooms', 'population'],
    color='median_house_value',
    title='Matriz de Dispersão - Casas na Califórnia',
    height=700
)

grafico.update_traces(diagonal_visible=False)  # esconde gráficos na diagonal

st.plotly_chart(grafico, use_container_width=True)

st.markdown('''Uma análise importante a ser feita é a distribuição dos valores mediano das casas.
Para isso, iremos utilizar o **histograma** e o **boxplot**. O histograma nos permitirá visualizar a distribuição dos valores, enquanto o boxplot nos ajudará a identificar possíveis outliers e a dispersão dos dados.
Abaixo estão os gráficos de histograma 'median_house_value': ''')

# Criando a figura
fig, ax = plt.subplots()
sns.histplot(house_db_regressao['median_house_value'], kde=True, ax=ax)

# Título e rótulos (opcional)
ax.set_title('Distribuição do Valor Mediano das Casas')
ax.set_xlabel('Valor Mediano')
ax.set_ylabel('Frequência')

# Exibindo no Streamlit
st.pyplot(fig)
st.text('''Note que ocorre uma concentração de valores em torno de $500.000,00. Isso acontece porque a variável "median_house_value" possui um valor máximo truncado em $500.000,00. Ou seja, imóveis com valores superiores foram limitados a esse teto, o que pode introduzir um viés nos modelos de regressão e limitar sua capacidade de prever valores muito altos com precisão.''')
# Criando a figura
fig, ax = plt.subplots(figsize=(10, 6))  # Tamanho opcional
sns.boxplot(x='ocean_proximity', y='median_house_value', data=house_db_regressao, ax=ax)

# Título e rótulos
ax.set_title('Valor Mediano das Casas por Proximidade ao Oceano')
ax.set_xlabel('Proximidade ao Oceano')
ax.set_ylabel('Valor Mediano das Casas')

# Exibindo no Streamlit
st.pyplot(fig)
st.markdown('''O BoxPlot apresenta a seguinte visão: O eixo X representa a proximidade com o oceano, enquanto o eixo Y representa o valor mediano das casas.''')
st.markdown('''**Interpretação do BOXPLOT de acordo com a proximidade com o oceano**:
* NEAR BAY e NEAR OCEAN

Valores medianos mais altos, indicando que estar perto da baía ou do oceano eleva o preço das casas.

Também têm uma faixa de preços bem distribuída.

* <1H OCEAN

Tem uma mediana menor que as anteriores.

Alguns outliers indicam casas de valor elevado mesmo não estando tão perto do mar.

* INLAND

Menores preços medianos disparado.

Distribuição concentrada em valores mais baixos.

Muitos outliers para cima, mas o grosso das casas é barato.

* ISLAND

Curiosamente, tem um preço mediano bem alto, com pouca variabilidade.

Pode indicar poucas observações, pois o box é estreito e com poucos pontos.''')

st.markdown('''Por fim, iremos analisar a correlação entre as variáveis numéricas da base de dados.
Para isso, iremos utilizar o mapa de calor (heatmap) do Seaborn. O mapa de calor nos permitirá visualizar a correlação entre as variáveis e identificar quais delas estão mais relacionadas com o valor mediano das casas.''')
st.image('img/corr.png', caption='Gráfico gerado a partir de código Python', use_container_width=True)
st.markdown('''A partir do mapa de calor, podemos observar que a variável da mediana salarial da região ('*median_income*') é a que possui a maior correlação positiva com o valor mediano das casas.''')
st.markdown('---')
st.subheader('MÃOS-À-OBRA - Prevendo o Valor Mediano das Casas')
st.markdown('''Entre os modelos de regressão que iremos utilizar estão:
1. Regressão Múltipla;

2. Regressão Polinomial;

3. Regressão de Ridge;

4. Regressão Lasso;

5. ElasticNet;

6. Regressão de Árvore de Decisão;

7. Regressão Random Forest;

8. Regressão SVM (SVR);

9. Regressão de Redes Neurais.
''')
st.markdown('''Após os testes e avaliações de desempenho dos modelos, o modelo de Regressão Random Forest se destacou como o mais eficaz para prever o valor mediano das casas na Califórnia, apresentando os seguintes resultados:
#### 👑Regressão Random Forest👑:
* Tempo de treinamento: 29.48 segundos
* MAE: 31.852,70
* MSE: 2.429.746.971,64
* RMSE: 49.292,46
* R²: 0,8217

Para alcançarmos esses resultados, foram utilizados 200 estimadores (árvores) no modelo de Regressão Random Forest. Esse número foi escolhido com base em testes de desempenho, onde foi observado que aumentar o número de árvores além de 200 não resultou em melhorias significativas na precisão do modelo, mas aumentou o tempo de treinamento. Portanto, 200 árvores foram consideradas um bom compromisso entre desempenho e eficiência computacional.

Além disso, o modelo foi configurado com um Random State de 42 para garantir a reprodutibilidade dos resultados. Isso significa que, ao executar o código várias vezes, os resultados permanecerão consistentes, permitindo uma comparação justa entre diferentes execuções e modelos.

Para reduzirmos o Overfitting, foi utilizado o parâmetro "max_depth" com valor de 20. Isso limita a profundidade máxima das árvores, evitando que o modelo se torne excessivamente complexo e se ajuste demais aos dados de treinamento, o que poderia prejudicar sua capacidade de generalização para novos dados, e min samples leaf com valor de 3. Isso significa que cada folha da árvore deve conter pelo menos 3 amostras, o que ajuda a evitar que o modelo se ajuste demais aos dados de treinamento e melhora sua capacidade de generalização.

O seguinte gráfico faz uma comparação entre os valores reais e os valores previstos pelo modelo de Regressão Random Forest. O eixo X representa os valores reais, enquanto o eixo Y representa os valores previstos pelo modelo. A linha diagonal representa a linha de perfeição, onde os valores previstos seriam iguais aos valores reais. Quanto mais próximo os pontos estiverem dessa linha, melhor será o desempenho do modelo.''')

st.image('img/Final.png', caption='Gráfico gerado a partir de código Python', use_container_width=True)

st.markdown('''---''')
st.subheader('Conclusão')
st.markdown('''Finalizamos mais um projeto de Machine Learning, desta vez focado em um problema de regressão, com o objetivo de prever os valores medianos dos preços de casas na Califórnia na década de 1990. O modelo alcançou um desempenho razoável, com um MAE (Erro Absoluto Médio) de aproximadamente 31.852,70.

Apesar do bom desempenho em termos absolutos, é importante ressaltar que, em um cenário real de negócios — como uma empresa que dependa da previsão precisa dos preços das casas —, esse nível de erro pode ser considerado elevado. Em situações comerciais, um R² próximo de 0.95 ou superior seria mais desejável para minimizar riscos e perdas financeiras.

Portanto, embora o modelo desenvolvido seja adequado para fins acadêmicos ou exploratórios, ele ainda carece de aprimoramentos e ajustes para aplicações em ambiente de produção ou decisões críticas de negócios.''')

st.markdown('''---''')
st.subheader('Referências')
st.markdown('''
1. Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media.
2. Kaggle. (n.d.). California Housing Prices Dataset. Retrieved from https://www.kaggle.com/datasets/camnugent/california-housing-prices
3. Scikit-learn Documentation. (n.d.). Retrieved from https://scikit-learn.org/1.6/api/sklearn.html          
''')