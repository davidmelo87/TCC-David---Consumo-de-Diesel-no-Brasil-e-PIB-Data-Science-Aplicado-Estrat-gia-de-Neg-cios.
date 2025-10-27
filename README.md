Repositório criado para armazenar os códigos do Trabalho de Conclusão de Curso do MBA de Data Science e Analytics da USP & ESALQ

**Aluno:** David Melo Jeronimo

**Professor:** Marcos Júnior Ribeiro

# Consumo de Diesel no Brasil e PIB: Data Science Aplicado à Estratégia de Negócios.

# Resumo:
O presente trabalho analisa a relação entre variáveis macroeconômicas e setoriais e o consumo de Diesel B no Brasil, com ênfase em como indicadores como Produto Interno Bruto (PIB), produção agrícola, taxa de câmbio e preços internacionais do petróleo influenciam a demanda nacional. A pesquisa adota técnicas de estatística aplicada e econometria, com destaque para a regressão múltipla e procedimentos de diagnóstico e validação dos pressupostos. Foram utilizados dados oficiais da ANP, IBGE, Banco Central do Brasil e outras fontes institucionais, cobrindo o período de 2017 a 2024. Os resultados demonstram que o consumo de Diesel B está fortemente associado à atividade econômica e ao transporte rodoviário de cargas, além de apresentar sazonalidade vinculada ao ciclo agrícola. O estudo contribui ao fornecer uma ferramenta analítica para compreender a dinâmica do mercado de combustíveis e apoiar a formulação de estratégias no setor energético.

**Palavras-chave:** consumo de diesel; inferência estatística; regressão múltipla; macroeconomia; previsão de demanda.

## Metodologia
- DataLoader & Data Wrangling
- Análise Descritiva
- Análise Temporal e Decomposição da Série Temporal
- Modelo base: **Regressão Regressão Múltipla Stepwise + Lags**
- Validação dos pressupostos: independência, homocedasticidade, normalidade e autocorrelação
- Seleção automatizada do melhor modelo com métricas de desempenho
- Análise de importância dos preditores (variáveis explicativas)

## Principais Tecnologias
- Python 3.x  
- Pandas, Numpy  
- Scikit-learn  
- Statsmodels  
- Matplotlib

## Resultados
O modelo Stepwise + Lags apresentou **melhor equilíbrio entre desempenho e robustez estatística**, permitindo interpretações confiáveis e reprodutíveis.

## ⚠️ Observações
Os dados utilizados não estão incluídos neste repositório por questões de confidencialidade.  
O código está preparado para ingestão, transformação e modelagem de séries temporais de demanda de Diesel B.

## Licença
Este projeto é distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para mais informações.

**Organização:** USP Esalq - MBA em Data Science & Analytics  
**Ano:** 2025

