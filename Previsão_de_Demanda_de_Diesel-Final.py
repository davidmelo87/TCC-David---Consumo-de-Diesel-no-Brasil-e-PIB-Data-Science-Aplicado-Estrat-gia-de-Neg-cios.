# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:35:58 2024

@author: david.jeronimo

Consumo de Diesel no Brasil e PIB: Data Science Aplicado à Estratégia de Negócios.
Pipeline completo para análise e previsão mensal do consumo de Diesel B.
"""

#%% 1 INSTALANDO OS PACOTES

!pip install ipeadatapy
!pip install python-docx
!pip install EIA_python
!pip install requests pandas
!pip install xgboost
!pip install prophet
!pip install python-bcb
!pip install yfinance
!pip install geopandas
!pip install mplcursors
!pip install geobr
!pip install tensorflow
!pip install scikit-learn
!pip install statsmodels
!pip install openpyxl
!pip install shap

#%% SEÇÃO 2 IMPORTANTANDO OS PACOTES

# Bibliotecas básicas e manipulação de dados
import pandas as pd
import numpy as np
import os
import io
from io import BytesIO
import zipfile
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Visualização de dados
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import cm
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec

# Geoprocessamento
import geopandas as gpd
from geobr import read_state
from matplotlib.patches import Patch

# Análise estatística e modelagem
from scipy import stats
from scipy.stats import linregress, shapiro, anderson, skew, kurtosis, gaussian_kde, probplot
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_arch, het_breuschpagan
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Séries temporais e dados financeiros
from prophet import Prophet
import yfinance as yf
import ipeadatapy as ip

# Manipulação de documentos
from docx import Document

# Redes neurais
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# Configurações de estilo
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
sns.set_palette("Blues")

#%% SEÇÃO 3 CARREGAMENTO E TRATAMENTO DE DADOS

import pandas as pd
import requests
from io import BytesIO
import io
import ipeadatapy as ip
from datetime import datetime

class DataLoader:
    """Classe para carregamento e tratamento de dados de múltiplas fontes"""

    def __init__(self):
        self.data_sources = {}
        self.processed_data = {}
    
    def load_anp_data(self, file_path='Liquidos_Vendas_Atual.csv'):
        """Carrega e processa dados da ANP - Vendas de Combustíveis"""
        print("Carregando dados da ANP...")
        
        try:
            # Carregar o arquivo com encoding alternativo
            bd_anp = pd.read_csv(file_path, sep=';', encoding='latin-1')
            
            # Processo de Data Wrangling dos dados da ANP
            mapa_segmento = {
                'CONSUMIDOR FINAL': 'B2B',
                'BANDEIRA BRANCA': 'BANDEIRA BRANCA',
                'BANDEIRADO': 'BANDEIRADO',
                'TRR': 'TRR',
                'POSTO DE COMBUSTÍVEIS - BANDEIRA BRANCA': 'BANDEIRA BRANCA',
                'POSTO DE COMBUSTÍVEIS - BANDEIRADO': 'BANDEIRADO',
                'TRRNI': 'TRR'}
            
            # Criar a nova coluna "Segmento" com base no mapeamento
            bd_anp['Segmento'] = bd_anp['Mercado Destinatário'].map(mapa_segmento)
            
            # Substituir vírgulas por pontos e remover espaços em branco
            bd_anp['Quantidade de Produto (mil m³)'] = bd_anp['Quantidade de Produto (mil m³)'].str.replace(',', '.').str.strip()
            
            # Garantir que a coluna esteja no formato numérico
            bd_anp['Quantidade de Produto (mil m³)'] = pd.to_numeric(bd_anp['Quantidade de Produto (mil m³)'], errors='coerce')
            
            # Converter de mil m³ para litros (1 mil m³ = 1 bilhão de litros)
            bd_anp['Volume em litros'] = bd_anp['Quantidade de Produto (mil m³)'] * 1_000_000
            
            # Converter a coluna 'Volume em litros' para números inteiros
            bd_anp['Volume em litros'] = bd_anp['Volume em litros'].round().astype('int64')
            
            # Criar a nova coluna "Data" combinando Ano e Mês
            bd_anp['Data'] = pd.to_datetime(bd_anp['Ano'].astype(str) + '-' + bd_anp['Mês'].astype(str) + '-01')
            
            # Criar coluna com o nome do mês
            bd_anp['Nome do Mês'] = bd_anp['Data'].dt.month_name(locale='pt_BR')
            
            # Criar coluna com o número do trimestre
            bd_anp['Trimestre'] = bd_anp['Data'].dt.quarter
            
            # Criar coluna com o semestre
            bd_anp['Semestre'] = bd_anp['Data'].dt.month.map(lambda x: 1 if x <= 6 else 2)
            
            # Criar coluna concatenando o Mês e Ano
            bd_anp['Mês/Ano'] = bd_anp['Nome do Mês'] + '/' + bd_anp['Ano'].astype(str)
            
            # Ajustar nomes dos estados
            bd_anp['UF Destino'] = bd_anp['UF Destino'].replace({
                'AC': 'Acre', 'AL': 'Alagoas', 'AM': 'Amazonas', 'AP': 'Amapá', 
                'BA': 'Bahia', 'CE': 'Ceará', 'DF': 'Distrito Federal', 'ES': 'Espírito Santo', 
                'GO': 'Goiás', 'MA': 'Maranhão', 'MT': 'Mato Grosso', 'MS': 'Mato Grosso do Sul', 
                'MG': 'Minas Gerais', 'PA': 'Pará', 'PB': 'Paraíba', 'PR': 'Paraná', 
                'PE': 'Pernambuco', 'PI': 'Piauí', 'RJ': 'Rio de Janeiro', 'RN': 'Rio Grande do Norte',
                'RS': 'Rio Grande do Sul', 'RO': 'Rondônia', 'RR': 'Roraima', 'SC': 'Santa Catarina',
                'SP': 'São Paulo', 'SE': 'Sergipe', 'TO': 'Tocantins'})
            
            # Definir a coluna 'Data' como índice
            bd_anp.set_index('Data', inplace=True)
            
            # Ordenar os dados pelo índice temporal
            bd_anp.sort_index(inplace=True)
            
            self.data_sources['anp'] = bd_anp
            print(f"Dados ANP carregados: {bd_anp.shape[0]} linhas, {bd_anp.shape[1]} colunas")
            return bd_anp
            
        except Exception as e:
            print(f"Erro ao carregar dados da ANP: {e}")
            return None

    def load_ppi_data(self, file_path='Histórico PPI.xlsx'):
        """Carrega dados do Preço de Paridade de Importação (PPI)"""
        print("Carregando dados do PPI...")
        
        try:
            # Carregar arquivo do PPI
            dados_ppi = pd.read_excel(file_path)
            
            # REMOVER coluna de gasolina se existir
            colunas_para_manter = [col for col in dados_ppi.columns 
                                  if 'GASOLINA' not in col.upper()]
            dados_ppi = dados_ppi[colunas_para_manter]
            
            # Processamento dos dados do PPI
            dados_ppi['Data'] = pd.to_datetime(dados_ppi['Data'], format='%Y-%m-%d')
            dados_ppi.set_index('Data', inplace=True)
            
            # Nome do mês em português
            dados_ppi['Mês'] = dados_ppi.index.month_name(locale='pt_BR')
            
            # Número do trimestre
            dados_ppi['Trimestre'] = dados_ppi.index.quarter
            
            # Número do semestre
            dados_ppi['Semestre'] = (dados_ppi['Trimestre'] + 1) // 2
            
            # Ano
            dados_ppi['Ano'] = dados_ppi.index.year.astype(str)
            
            # Mês e Ano concatenados
            dados_ppi['Mês/Ano'] = dados_ppi['Mês'] + "/" + dados_ppi['Ano']
            
            # Reorganizar as colunas
            colunas_novas_ppi = ['Ano', 'Mês', 'Mês/Ano', 'Trimestre', 'Semestre']
            colunas_restantes_ppi = [col for col in dados_ppi.columns if col not in colunas_novas_ppi]
            dados_ppi = dados_ppi[colunas_novas_ppi + colunas_restantes_ppi]
            
            # Criar dados mensais do PPI
            dados_ppi['Primeiro_Dia_Mês'] = pd.to_datetime(dados_ppi.index).to_period('M').to_timestamp()
            
            # Agrupar por mês
            dados_ppi_m = (
                dados_ppi
                .groupby(['Ano', 'Mês', 'Mês/Ano', 'Trimestre', 'Semestre', 'Primeiro_Dia_Mês'])
                .mean(numeric_only=True)
                .reset_index()
            )
            
            # Renomear a coluna 'Primeiro_Dia_Mês' para 'Data'
            dados_ppi_m.rename(columns={'Primeiro_Dia_Mês': 'Data'}, inplace=True)
            dados_ppi_m.set_index('Data', inplace=True)
            
            self.data_sources['ppi'] = dados_ppi_m
            print(f"Dados PPI carregados: {dados_ppi_m.shape[0]} linhas")
            print(f"Colunas PPI disponíveis: {dados_ppi_m.columns.tolist()}")
            return dados_ppi_m
            
        except Exception as e:
            print(f"Erro ao carregar dados do PPI: {e}")
            return None

    def load_fuel_prices(self):
        """Carrega dados de preços de combustíveis da ANP"""
        print("Carregando dados de preços de combustíveis...")
        
        # URL do arquivo
        url = "https://www.gov.br/anp/pt-br/assuntos/precos-e-defesa-da-concorrencia/precos/precos-revenda-e-de-distribuicao-combustiveis/shlp/semanal/semanal-estados-desde-2013.xlsx"
        
        try:
            # Fazendo o download do arquivo
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Lendo o arquivo Excel em memória
            excel_data = pd.ExcelFile(BytesIO(response.content))
            sheet_name = excel_data.sheet_names[0]
            df = pd.read_excel(excel_data, sheet_name=sheet_name, skiprows=17)
            
            # Verificando se a coluna "DATA INICIAL" existe antes de renomear
            if "DATA INICIAL" not in df.columns:
                raise ValueError("A coluna 'DATA INICIAL' não foi encontrada no arquivo.")
            
            # Renomeando e ajustando o índice
            df.rename(columns={"DATA INICIAL": "Data"}, inplace=True)
            df.set_index("Data", inplace=True)
            
            # Garantindo que o índice seja datetime
            df.index = pd.to_datetime(df.index, errors='coerce')
            
            # Adicionando as colunas solicitadas
            df["Ano"] = df.index.year
            df["Mês"] = df.index.month_name(locale="pt_BR.utf8")
            df["Trimestre"] = df.index.quarter
            df["Semestre"] = df.index.month.map(lambda x: 1 if x <= 6 else 2)
            df["Mês/Ano"] = df["Mês"] + "/" + df["Ano"].astype(str)
            
            # Reorganizando as colunas
            novas_colunas = ["DATA FINAL", "Ano", "Mês", "Mês/Ano", "Trimestre", "Semestre"]
            demais_colunas = [col for col in df.columns if col not in novas_colunas]
            df = df[novas_colunas + demais_colunas]
            
            # Criar dataframe consolidado
            dados_prc_b = df
            
            # Criar dados mensais
            dados_prc_b['Primeiro_Dia_Mês'] = pd.to_datetime(dados_prc_b.index).to_period('M').to_timestamp()
            
            # Agrupar por mês
            dados_prc_mensal = (
                dados_prc_b
                .groupby(['Ano', 'Mês', 'Mês/Ano', 'Trimestre', 'Semestre', 'Primeiro_Dia_Mês'])
                .mean(numeric_only=True)
                .reset_index()
            )
            
            # Renomear a coluna 'Primeiro_Dia_Mês' para 'Data'
            dados_prc_mensal.rename(columns={'Primeiro_Dia_Mês': 'Data'}, inplace=True)
            dados_prc_mensal.set_index('Data', inplace=True)
            
            self.data_sources['fuel_prices'] = dados_prc_mensal
            print(f"Dados de preços carregados: {dados_prc_mensal.shape[0]} linhas")
            print(f"Colunas disponíveis nos dados de preços: {dados_prc_mensal.columns.tolist()}")
            return dados_prc_mensal
            
        except Exception as e:
            print(f"Erro ao carregar dados de preços: {e}")
            return None

    def load_economic_data(self):
        """Carrega dados econômicos de várias fontes"""
        print("Carregando dados econômicos...")
        
        try:
            # Dados do PIB
            codigo_serie_pib = 'BM12_PIB12'
            dados_pib = ip.timeseries(codigo_serie_pib)
            dados_pib.index.name = 'Data'
            dados_pib = dados_pib.rename(columns={'VALUE (R$)': 'PIB (R$)'})
            
            # Dados do Dólar Ptax
            codigo_serie_dolar = 'BM12_ERC12'
            dados_dolar = ip.timeseries(codigo_serie_dolar)
            dados_dolar.index.name = 'Data'
            dados_dolar = dados_dolar.rename(columns={'VALUE (R$)': 'Dólar Ptax'})
            
            # Dados de licenciamento de veículos
            codigo_serie_lcv = 'ANFAVE12_LICVETOT12'
            dados_lcv = ip.timeseries(codigo_serie_lcv)
            dados_lcv.index.name = 'Data'
            dados_lcv = dados_lcv.rename(columns={'VALUE (Unidade)': 'Licenciamento de veiculos'})
            
            # Dados de vendas de veículos
            codigo_serie_vtv = 'ANFAVE12_LICVEN12'
            dados_vtv = ip.timeseries(codigo_serie_vtv)
            dados_vtv.index.name = 'Data'
            dados_vtv = dados_vtv.rename(columns={'VALUE (-)': 'Vendas de veiculos'})
            
            # Dados de taxa de desocupação
            codigo_serie_txd = 'PNADC12_TDESOC12'
            dados_txd = ip.timeseries(codigo_serie_txd)
            dados_txd.index.name = 'Data'
            dados_txd = dados_txd.rename(columns={'VALUE ((%))': 'Taxa de desemprego (%)'})
            
            # Dados de renda média
            codigo_serie_renda = 'PNADC12_RTE12'
            dados_renda = ip.timeseries(codigo_serie_renda)
            dados_renda.index.name = 'Data'
            dados_renda = dados_renda.rename(columns={'VALUE (R$)': 'Renda Média (R$)'})
            
            # Dados da Selic Copom
            codigo_serie_selic_copom = 'BM366_TJOVER366'
            dados_selic_copom = ip.timeseries(codigo_serie_selic_copom)
            dados_selic_copom.index.name = 'Data'
            dados_selic_copom = dados_selic_copom.rename(columns={'VALUE ((% a.a.))': 'Selic Copom (% a.a)'})
            
            # Dados da Selic
            codigo_serie_selic = 'BM12_TJOVER12'
            dados_selic = ip.timeseries(codigo_serie_selic)
            dados_selic.index.name = 'Data'
            dados_selic = dados_selic.rename(columns={'VALUE ((% a.m.))': 'Selic (% a.m)'})
            
            # Dados do IPCA
            codigo_serie_ipca = 'PRECOS12_IPCAG12'
            dados_ipca = ip.timeseries(codigo_serie_ipca)
            dados_ipca.index.name = 'Data'
            dados_ipca = dados_ipca.rename(columns={'VALUE ((% a.m.))': 'IPCA (% a.m)'})
            
            # Dados do IBC-Br
            codigo_serie_ibcbr = 'SGS12_IBCBRDESSAZ12'
            dados_ibcbr = ip.timeseries(codigo_serie_ibcbr)
            dados_ibcbr.index.name = 'Data'
            dados_ibcbr = dados_ibcbr.rename(columns={'VALUE (-)': 'IBC-Br'})
            
            # Dados de inadimplência
            codigo_serie_inad = 'CNC12_PEICT12'
            dados_inad = ip.timeseries(codigo_serie_inad)
            dados_inad.index.name = 'Data'
            dados_inad = dados_inad.rename(columns={'VALUE ((%))': 'Taxa de inadimplência (%)'})
            
            # Dados de produção industrial
            codigo_serie_pdi = 'PAN12_QIIGG12'
            dados_pdi = ip.timeseries(codigo_serie_pdi)
            dados_pdi.index.name = 'Data'
            dados_pdi = dados_pdi.rename(columns={'VALUE ((% a.a.))': 'Produção Industrial (% a.a)'})
            
            # Dados de confiança do empresário
            codigo_serie_cde = 'CNI12_ICEIGER12'
            dados_cde = ip.timeseries(codigo_serie_cde)
            dados_cde.index.name = 'Data'
            dados_cde = dados_cde.rename(columns={'VALUE (-)': 'Índice de Confiança'})
            
            # Combinar todos os dados econômicos
            indicadores_economicos = dados_pib[['PIB (R$)']]
            economic_data_sources = [
                dados_dolar[['Dólar Ptax']],
                dados_renda[['Renda Média (R$)']],
                dados_txd[['Taxa de desemprego (%)']],
                dados_lcv[['Licenciamento de veiculos']],
                dados_vtv[['Vendas de veiculos']],
                dados_selic_copom[['Selic Copom (% a.a)']],
                dados_selic[['Selic (% a.m)']],
                dados_ipca[['IPCA (% a.m)']],
                dados_ibcbr[['IBC-Br']],
                dados_inad[['Taxa de inadimplência (%)']],
                dados_pdi[['Produção Industrial (% a.a)']],
                dados_cde[['Índice de Confiança']]
            ]
            
            for data_source in economic_data_sources:
                indicadores_economicos = indicadores_economicos.merge(
                    data_source, on='Data', how='left'
                )
            
            # Adicionar colunas temporais
            indicadores_economicos.index = pd.to_datetime(indicadores_economicos.index)
            indicadores_economicos['Mês'] = indicadores_economicos.index.month_name(locale='pt_BR')
            indicadores_economicos['Trimestre'] = indicadores_economicos.index.quarter
            indicadores_economicos['Semestre'] = (indicadores_economicos['Trimestre'] + 1) // 2
            indicadores_economicos['Ano'] = indicadores_economicos.index.year.astype(str)
            indicadores_economicos['Mês/Ano'] = indicadores_economicos['Mês'] + "/" + indicadores_economicos['Ano']
            
            # Preencher valores nulos
            indicadores_economicos.fillna(method='ffill', inplace=True)
            
            self.data_sources['economic'] = indicadores_economicos
            print(f"Dados econômicos carregados: {indicadores_economicos.shape[0]} linhas")
            return indicadores_economicos
            
        except Exception as e:
            print(f"Erro ao carregar dados econômicos: {e}")
            return None

    def load_agricultural_data(self):
        """Carrega dados agrícolas do SIDRA/IBGE"""
        print("Carregando dados agrícolas do IBGE...")
        
        try:
            # URL do arquivo
            url = "https://sidra.ibge.gov.br/geratabela?format=xlsx&name=tabela6588.xlsx&terr=N&rank=-&query=t/6588/n1/all/n3/all/v/35/p/all/c48/39428%2039429%2039430%2039431%2039432%2039433%2039434%2039435%2039436%2039437%2039438%2039439%2039440%2039441%2039442%2039443%2039444%2039445%2039446%2039447%2039448%2039449%2039450%2039451%2039452%2039453%2039454%2039455%2039456%2039457%2039458%2039459%2039460%2039461%2039462%2039463%2039464%2039465%2039467%2039468%2039469%2039470%2039471%2040527/l/v,t,p%2Bc48"
            
            # Realiza o download do arquivo
            response = requests.get(url)
            
            if response.status_code == 200:
                # Lê o conteúdo do arquivo diretamente na memória
                file_content = io.BytesIO(response.content)
                
                # Lê o arquivo Excel em um DataFrame ignorando as primeiras três linhas
                dados_agro = pd.read_excel(file_content, engine="openpyxl", skiprows=3)
                
                # Remove a linha 'Brasil e Unidade da Federação' que está mesclada
                dados_agro = dados_agro[dados_agro.iloc[:, 0] != 'Brasil e Unidade da Federação']
                
                # Remove a última linha que contém a fonte
                dados_agro = dados_agro[:-1]
                
                # Corrige espaços extras na coluna A e converte para datetime
                dados_agro.iloc[:, 0] = dados_agro.iloc[:, 0].str.strip()
                
                # Força os meses para estarem em português
                meses_em_portugues = {
                    "janeiro": "01", "fevereiro": "02", "março": "03", "abril": "04",
                    "maio": "05", "junho": "06", "julho": "07", "agosto": "08",
                    "setembro": "09", "outubro": "10", "novembro": "11", "dezembro": "12"
                }
                
                # Substitui os meses por números e converte para datetime
                for mes, numero in meses_em_portugues.items():
                    dados_agro.iloc[:, 0] = dados_agro.iloc[:, 0].str.replace(mes, numero, case=False, regex=True)
                
                dados_agro.iloc[:, 0] = pd.to_datetime(dados_agro.iloc[:, 0], format="%m %Y", errors='coerce')
                
                # Renomeia a coluna A para 'Data'
                dados_agro.rename(columns={dados_agro.columns[0]: 'Data'}, inplace=True)
                
                # Remove a coluna B
                dados_agro = dados_agro.drop(dados_agro.columns[1], axis=1)
                
                # Transformar a coluna 'Data' em índice
                dados_agro['Data'] = pd.to_datetime(dados_agro['Data'], format='%Y-%m-%d')
                dados_agro.set_index('Data', inplace=True)
                
                # Nome do mês em português
                dados_agro['Mês'] = dados_agro.index.month_name(locale='pt_BR')
                
                # Número do trimestre
                dados_agro['Trimestre'] = dados_agro.index.quarter
                
                # Número do semestre
                dados_agro['Semestre'] = (dados_agro['Trimestre'] + 1) // 2
                
                # Ano
                dados_agro['Ano'] = dados_agro.index.year.astype(str)
                
                # Mês e Ano concatenados
                dados_agro['Mês/Ano'] = dados_agro['Mês'] + "/" + dados_agro['Ano']
                
                # Reorganizar as colunas
                colunas_novas = ['Ano', 'Mês', 'Mês/Ano', 'Trimestre', 'Semestre']
                colunas_restantes = [col for col in dados_agro.columns if col not in colunas_novas]
                dados_agro = dados_agro[colunas_novas + colunas_restantes]
                
                self.data_sources['agricultural'] = dados_agro
                print(f"Dados agrícolas carregados: {dados_agro.shape[0]} linhas")
                print(f"Colunas agrícolas disponíveis: {dados_agro.columns.tolist()}")
                return dados_agro
            else:
                print(f"Erro no download: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Erro ao carregar dados agrícolas: {e}")
            return None

    def load_road_traffic_data(self):
        """Carrega dados de tráfego rodoviário da ABCR"""
        print("Carregando dados de tráfego rodoviário...")
        
        try:
            # URL do arquivo
            url = "https://melhoresrodovias.org.br/wp-content/uploads/2025/01/abcr_0125.xlsx"
            
            # Fazer o download da planilha
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Ler a planilha na memória
            with BytesIO(response.content) as file:
                dados_rodovias = pd.read_excel(file, sheet_name='(C) Original', skiprows=2)
            
            # Excluir as colunas sem dados
            dados_rodovias.dropna(axis=1, how='all', inplace=True)
            
            # Excluir todas as colunas com 'TOTAL' no nome
            dados_rodovias = dados_rodovias.loc[:, ~dados_rodovias.columns.str.contains('TOTAL', case=False)]
            
            # Renomear as colunas conforme solicitado
            colunas_renomeadas = {
                'Unnamed: 0': 'Data',
                'LEVES': 'Fluxo Leves BR',
                'PESADOS': 'Fluxo Pesados BR',
                'LEVES.1': 'Fluxo Leves SP',
                'PESADOS.1': 'Fluxo Pesados SP',
                'LEVES.2': 'Fluxo Leves PR',
                'PESADOS.2': 'Fluxo Pesados PR',
                'LEVES.3': 'Fluxo Leves RJ',
                'PESADOS.3': 'Fluxo Pesados RJ'
            }
            dados_rodovias.rename(columns=colunas_renomeadas, inplace=True)
            
            # Transformar a coluna 'Data' em índice
            dados_rodovias['Data'] = pd.to_datetime(dados_rodovias['Data'], format='%Y-%m-%d')
            dados_rodovias.set_index('Data', inplace=True)
            
            # Nome do mês em português
            dados_rodovias['Mês'] = dados_rodovias.index.month_name(locale='pt_BR')
            
            # Número do trimestre
            dados_rodovias['Trimestre'] = dados_rodovias.index.quarter
            
            # Número do semestre
            dados_rodovias['Semestre'] = (dados_rodovias['Trimestre'] + 1) // 2
            
            # Ano
            dados_rodovias['Ano'] = dados_rodovias.index.year.astype(str)
            
            # Mês e Ano concatenados
            dados_rodovias['Mês/Ano'] = dados_rodovias['Mês'] + "/" + dados_rodovias['Ano']
            
            # Reorganizar as colunas
            colunas_novas_dados_rodovias = ['Ano', 'Mês', 'Mês/Ano', 'Trimestre', 'Semestre']
            colunas_restantes_dados_rodovias = [col for col in dados_rodovias.columns if col not in colunas_novas_dados_rodovias]
            dados_rodovias = dados_rodovias[colunas_novas_dados_rodovias + colunas_restantes_dados_rodovias]
            
            self.data_sources['road_traffic'] = dados_rodovias
            print(f"Dados de tráfego rodoviário carregados: {dados_rodovias.shape[0]} linhas")
            return dados_rodovias
            
        except Exception as e:
            print(f"Erro ao carregar dados de tráfego rodoviário: {e}")
            return None

    def load_vehicle_fuel_data(self):
        """Carrega dados de veículos por tipo de combustível da ANFAVEA"""
        print("Carregando dados de veículos por combustível...")
        
        try:
            dados_vbyfuel = pd.read_excel('Veiculos por combustiveis.xlsx')
            
            # Transformar a coluna 'Data' em índice
            dados_vbyfuel['Data'] = pd.to_datetime(dados_vbyfuel['Data'], format='%Y-%m-%d')
            dados_vbyfuel.set_index('Data', inplace=True)
            
            # Número do trimestre
            dados_vbyfuel['Trimestre'] = dados_vbyfuel.index.quarter
            
            # Número do semestre
            dados_vbyfuel['Semestre'] = (dados_vbyfuel['Trimestre'] + 1) // 2
            
            # Mês e Ano concatenados
            dados_vbyfuel['Mês/Ano'] = dados_vbyfuel['Mês'] + "/" + dados_vbyfuel['Ano'].astype(str)
            
            # Reorganizar as colunas
            colunas_novas_dados_vbyfuel = ['Ano', 'Mês', 'Mês/Ano', 'Trimestre', 'Semestre']
            colunas_restantes_dados_vbyfuel = [col for col in dados_vbyfuel.columns if col not in colunas_novas_dados_vbyfuel]
            dados_vbyfuel = dados_vbyfuel[colunas_novas_dados_vbyfuel + colunas_restantes_dados_vbyfuel]
            
            self.data_sources['vehicle_fuel'] = dados_vbyfuel
            print(f"Dados de veículos por combustível carregados: {dados_vbyfuel.shape[0]} linhas")
            print(f"Colunas de veículos disponíveis: {dados_vbyfuel.columns.tolist()}")
            return dados_vbyfuel
            
        except Exception as e:
            print(f"Erro ao carregar dados de veículos por combustível: {e}")
            return None

    def load_oil_prices_data(self, file_path='Cotações_Futuro.xlsx'):
        """Carrega dados de preços de petróleo (Brent, WTI, Heating Oil)"""
        print("Carregando dados de preços de petróleo...")
        
        try:
            # Importar o arquivo Excel e definir 'Data' como índice
            Cotações_Futuro = pd.read_excel(file_path, parse_dates=['Data'], index_col='Data')
            
            # Verificar se as colunas necessárias existem no arquivo
            colunas_necessarias = ['Brent', 'WTI', 'Heating Oil']
            colunas_disponiveis = [col for col in colunas_necessarias if col in Cotações_Futuro.columns]
            
            if len(colunas_disponiveis) < len(colunas_necessarias):
                colunas_faltantes = set(colunas_necessarias) - set(colunas_disponiveis)
                print(f"Atenção: As seguintes colunas não foram encontradas no arquivo: {', '.join(colunas_faltantes)}")
            
            # Manter apenas as colunas disponíveis e necessárias
            Cotações_Futuro = Cotações_Futuro[colunas_disponiveis]
            
            # Ordenar o DataFrame pelo índice (Data)
            Cotações_Futuro = Cotações_Futuro.sort_index()
            
            # Nome do mês em português
            Cotações_Futuro['Mês'] = Cotações_Futuro.index.strftime('%B').str.title()
            
            # Número do trimestre
            Cotações_Futuro['Trimestre'] = Cotações_Futuro.index.quarter
            
            # Número do semestre
            Cotações_Futuro['Semestre'] = (Cotações_Futuro['Trimestre'] + 1) // 2
            
            # Ano como string
            Cotações_Futuro['Ano'] = Cotações_Futuro.index.year.astype(str)
            
            # Mês e Ano concatenados
            Cotações_Futuro['Mês/Ano'] = Cotações_Futuro['Mês'] + "/" + Cotações_Futuro['Ano']
            
            # Reorganizar as colunas
            colunas_novas = ['Ano', 'Mês', 'Mês/Ano', 'Trimestre', 'Semestre']
            colunas_restantes = [col for col in Cotações_Futuro.columns if col not in colunas_novas]
            Cotações_Futuro = Cotações_Futuro[colunas_novas + colunas_restantes]
            
            # Preencher valores nulos
            Cotações_Futuro.fillna(method='ffill', inplace=True)
            
            # Criar dados mensais
            Cotações_Futuro['Primeiro_Dia_Mês'] = pd.to_datetime(Cotações_Futuro.index).to_period('M').to_timestamp()
            
            # Agrupar por mês
            dados_oil_prices = (
                Cotações_Futuro
                .groupby(['Ano', 'Mês', 'Mês/Ano', 'Trimestre', 'Semestre', 'Primeiro_Dia_Mês'])
                .mean(numeric_only=True)
                .reset_index()
            )
            
            # Renomear a coluna 'Primeiro_Dia_Mês' para 'Data'
            dados_oil_prices.rename(columns={'Primeiro_Dia_Mês': 'Data'}, inplace=True)
            dados_oil_prices.set_index('Data', inplace=True)
            
            self.data_sources['oil_prices'] = dados_oil_prices
            print(f"Dados de preços de petróleo carregados: {dados_oil_prices.shape[0]} linhas")
            return dados_oil_prices
            
        except Exception as e:
            print(f"Erro ao carregar dados de preços de petróleo: {e}")
            return None

    def create_consolidated_data(self):
        """Consolida todos os dados em um único DataFrame para modelagem"""
        print("Consolidando dados para modelagem...")
        
        # Verificar se todos os dados necessários foram carregados
        if 'anp' not in self.data_sources:
            self.load_anp_data()
        
        if 'economic' not in self.data_sources:
            self.load_economic_data()
        
        if 'fuel_prices' not in self.data_sources:
            self.load_fuel_prices()
        
        if 'ppi' not in self.data_sources:
            self.load_ppi_data()
        
        if 'agricultural' not in self.data_sources:
            self.load_agricultural_data()
        
        if 'road_traffic' not in self.data_sources:
            self.load_road_traffic_data()
        
        if 'vehicle_fuel' not in self.data_sources:
            self.load_vehicle_fuel_data()
        
        if 'oil_prices' not in self.data_sources:
            self.load_oil_prices_data()
        
        # Criar DataFrame consolidado para o consumo de Diesel no Brasil
        bd_anp = self.data_sources['anp']
        Diesel_Brasil = (
            bd_anp[bd_anp['Nome do Produto'] == 'Diesel B']
            .groupby(['Data', 'Ano', 'Mês', 'Mês/Ano', 'Trimestre', 'Semestre'])
            .agg({
                'Quantidade de Produto (mil m³)': 'sum',
                'Volume em litros': 'sum'
            })
            .reset_index()
            .rename(columns={
                'Quantidade de Produto (mil m³)': 'Quantidade_Diesel_B (mil m³)',
                'Volume em litros': 'Diesel B (Volume em litros)'
            })
        )
        
        # Adicionar coluna Data se necessário
        if 'Data' not in Diesel_Brasil.columns:
            Diesel_Brasil['Data'] = pd.to_datetime(Diesel_Brasil[['Ano', 'Mês']].assign(DIA=1))
        
        # Combinar com dados econômicos
        indicadores_economicos = self.data_sources['economic'].reset_index()
        modelo_diesel = Diesel_Brasil.merge(
            indicadores_economicos, 
            on='Data', 
            how='left',
            suffixes=('', '_econ')
        )
        
        # Combinar com dados de preços - VERIFICAR COLUNAS DISPONÍVEIS
        if 'fuel_prices' in self.data_sources:
            dados_prc_mensal = self.data_sources['fuel_prices'].reset_index()
            
            # Verificar quais colunas de preço de diesel estão disponíveis
            colunas_diesel = [col for col in dados_prc_mensal.columns 
                             if 'DIESEL' in col.upper() or 'OLEO' in col.upper() or 'S10' in col.upper()]
            
            print(f"Colunas de preço de diesel disponíveis: {colunas_diesel}")
            
            if colunas_diesel:
                # Usar a primeira coluna de diesel encontrada
                coluna_preco_diesel = colunas_diesel[0]
                modelo_diesel = modelo_diesel.merge(
                    dados_prc_mensal[['Data', coluna_preco_diesel]],
                    on='Data',
                    how='left'
                )
                modelo_diesel.rename(columns={coluna_preco_diesel: 'Preço de Bomba Diesel'}, inplace=True)
            else:
                print("Aviso: Nenhuma coluna de preço de diesel encontrada")
        
        # Combinar com dados do PPI
        if 'ppi' in self.data_sources:
            dados_ppi_m = self.data_sources['ppi'].reset_index()
            
            # Verificar quais colunas do PPI estão disponíveis (apenas diesel)
            colunas_ppi_diesel = [col for col in dados_ppi_m.columns 
                                 if 'DIESEL' in col.upper() and 'GASOLINA' not in col.upper()]
            
            print(f"Colunas PPI diesel disponíveis: {colunas_ppi_diesel}")
            
            if colunas_ppi_diesel:
                modelo_diesel = modelo_diesel.merge(
                    dados_ppi_m[['Data'] + colunas_ppi_diesel],
                    on='Data',
                    how='left'
                )
        
        # Combinar com dados agrícolas
        if 'agricultural' in self.data_sources:
            dados_agro = self.data_sources['agricultural'].reset_index()
            if 'Brasil' in dados_agro.columns:
                modelo_diesel = modelo_diesel.merge(
                    dados_agro[['Data', 'Brasil']],
                    on='Data',
                    how='left'
                )
                modelo_diesel.rename(columns={'Brasil': 'Produção Agrícola'}, inplace=True)
        
        # Combinar com dados de tráfego rodoviário
        if 'road_traffic' in self.data_sources:
            dados_rodovias = self.data_sources['road_traffic'].reset_index()
            if 'Fluxo Pesados BR' in dados_rodovias.columns:
                modelo_diesel = modelo_diesel.merge(
                    dados_rodovias[['Data', 'Fluxo Pesados BR']],
                    on='Data',
                    how='left'
                )
        
        # Combinar com dados de veículos por combustível
        if 'vehicle_fuel' in self.data_sources:
            dados_vbyfuel = self.data_sources['vehicle_fuel'].reset_index()
            colunas_veiculos = dados_vbyfuel.columns.tolist()
            
            # Procurar coluna relacionada a diesel
            colunas_diesel_veic = [col for col in colunas_veiculos 
                                  if 'DIESEL' in col.upper() or 'LICENCIAMENTO' in col.upper()]
            
            if colunas_diesel_veic:
                coluna_diesel = colunas_diesel_veic[0]
                modelo_diesel = modelo_diesel.merge(
                    dados_vbyfuel[['Data', coluna_diesel]],
                    on='Data',
                    how='left'
                )
                modelo_diesel.rename(columns={coluna_diesel: 'Licenciamento Diesel'}, inplace=True)
        
        # Combinar com dados de preços de petróleo
        if 'oil_prices' in self.data_sources:
            dados_oil = self.data_sources['oil_prices'].reset_index()
            colunas_oil = ['Brent', 'WTI', 'Heating Oil']
            colunas_disponiveis = [col for col in colunas_oil if col in dados_oil.columns]
            
            if colunas_disponiveis:
                modelo_diesel = modelo_diesel.merge(
                    dados_oil[['Data'] + colunas_disponiveis],
                    on='Data',
                    how='left'
                )
                
                # ADICIONAR AS TRANSFORMAÇÕES SOLICITADAS
                if all(col in modelo_diesel.columns for col in ['Brent', 'WTI', 'Heating Oil', 'Dólar Ptax']):
                    # Conversão de Heating Oil de USD/gal para R$/L
                    modelo_diesel['Heating Oil R$/L'] = (modelo_diesel['Heating Oil'] / 3.78541) * modelo_diesel['Dólar Ptax']
                    
                    # Brent e WTI em R$ por litro
                    modelo_diesel['Brent em R$ por litro'] = (modelo_diesel['Brent'] / 158.987) * modelo_diesel['Dólar Ptax']
                    modelo_diesel['WTI em R$ por litro'] = (modelo_diesel['WTI'] / 158.987) * modelo_diesel['Dólar Ptax']
                    
                    # Crack-spreads
                    modelo_diesel['Crack-spread WTI-HO'] = (modelo_diesel['Heating Oil'] * 42) - modelo_diesel['WTI']
                    modelo_diesel['Crack-spread Brent-HO'] = (modelo_diesel['Heating Oil'] * 42) - modelo_diesel['Brent']
        
        # Remover colunas duplicadas
        cols_to_keep = [col for col in modelo_diesel.columns if not col.endswith('_econ')]
        modelo_diesel = modelo_diesel[cols_to_keep]
        
        # Preencher valores nulos
        modelo_diesel.fillna(method='ffill', inplace=True)
        
        # Garantir que a coluna Data está no formato correto
        modelo_diesel['Data'] = pd.to_datetime(modelo_diesel['Data'])
        
        # Ordenar por data
        modelo_diesel = modelo_diesel.sort_values('Data')
        
        self.processed_data['modelo_diesel'] = modelo_diesel
        print(f"DataFrame consolidado criado: {modelo_diesel.shape[0]} linhas, {modelo_diesel.shape[1]} colunas")
        
        # Mostrar informações sobre as colunas disponíveis
        print("\nColunas disponíveis no modelo consolidado:")
        for col in modelo_diesel.columns:
            print(f"- {col}")
        
        return modelo_diesel

# Instanciar e carregar dados
loader = DataLoader()
dados_consolidados = loader.create_consolidated_data()

# Mostrar primeiras linhas
print("\nPrimeiras linhas do DataFrame consolidado:")
print(dados_consolidados.head())
print(f"\nShape final: {dados_consolidados.shape}")


dados_consolidados.info()

#%% SEÇÃO 4 ANÁLISE EXPLORATÓRIA E ESTATÍSTICA

print("=" * 80)
print("ANÁLISE EXPLORATÓRIA E ESTATÍSTICA COMPLETA")
print("=" * 80)

# --------------------------------------------------------
# IMPORTS
# --------------------------------------------------------
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, anderson, skew, kurtosis
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------
# HELPERS DE FORMATAÇÃO
# --------------------------------------------------------
def _to_bi(x):  # litros -> bilhões de litros
    return x / 1e9

def _to_mi(x):  # litros -> milhões de litros
    return x / 1e6

def _fmt_bi(x):  # 60.4 bi
    return f"{x:,.2f} bi".replace(",", "X").replace(".", ",").replace("X", ".")

def _fmt_mi(x):  # 5.036 mi
    return f"{x:,.0f} mi".replace(",", "X").replace(".", ",").replace("X", ".")

def print_linha(char="─", n=80):
    print(char * n)

def print_tabela_bonita(df: pd.DataFrame, titulo: str = ""):
    """Mostra DataFrame ‘describe’ com colunas fixas e números formatados."""
    if titulo:
        print(titulo); print_linha()
    # Ajusta nomes
    df = df.rename(columns={"count": "n", "mean": "média", "std": "desv.padr.",
                            "min": "min", "25%": "p25", "50%": "mediana",
                            "75%": "p75", "max": "max"})
    cols = ["n", "média", "desv.padr.", "min", "p25", "mediana", "p75", "max"]
    df = df[cols]

    # Formatação linha-a-linha
    col_widths = {"var": 30, "n": 6, "média": 14, "desv.padr.": 14,
                  "min": 14, "p25": 14, "mediana": 14, "p75": 14, "max": 14}

    header = (f"{'Variável':<{col_widths['var']}}"
              f"{'n':>{col_widths['n']}} "
              f"{'média':>{col_widths['média']}} "
              f"{'desv.p.':>{col_widths['desv.padr.']}} "
              f"{'min':>{col_widths['min']}} "
              f"{'p25':>{col_widths['p25']}} "
              f"{'mediana':>{col_widths['mediana']}} "
              f"{'p75':>{col_widths['p75']}} "
              f"{'max':>{col_widths['max']}}")
    print(header); print_linha()

    for var, row in df.iterrows():
        # Diesel B em bi L; demais no formato padrão (com separador)
        def fmt_val(v, diesel=False):
            if pd.isna(v): return "-"
            if diesel:  # bilhões
                return f"{_to_bi(v):>13.2f}".replace(".", ",")
            else:
                # usa notação comum com milhar e vírgula brasileira
                s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                # limita largura
                return f"{s:>13}"

        is_diesel = ("Diesel B" in var)
        linha = (f"{var:<{col_widths['var']}}"
                 f"{int(row['n']):>{col_widths['n']}} "
                 f"{fmt_val(row['média'], is_diesel)} "
                 f"{fmt_val(row['desv.padr.'], is_diesel)} "
                 f"{fmt_val(row['min'], is_diesel)} "
                 f"{fmt_val(row['p25'], is_diesel)} "
                 f"{fmt_val(row['mediana'], is_diesel)} "
                 f"{fmt_val(row['p75'], is_diesel)} "
                 f"{fmt_val(row['max'], is_diesel)}")
        print(linha)
    print_linha()

# --------------------------------------------------------
# CLASSE
# --------------------------------------------------------
class AnaliseExploratoriaOtimizada:
    """Análise exploratória, estatística e geração de insights automáticos (console)."""

    def __init__(self, dados: pd.DataFrame):
        self.dados = dados.copy()
        colunas_remover = ['Data', 'Ano', 'Mês', 'Mês/Ano', 'Trimestre', 'Semestre', 'Quantidade_Diesel_B (mil m³)']
        self.dados_analise = dados.drop(columns=[c for c in colunas_remover if c in dados.columns])
        self.dados_diesel = _to_mi(dados['Diesel B (Volume em litros)'])  # milhões/mês
        self.resultados_estatisticos = {}
        self.insights_textuais = []
        self.analise_realizada = False

    # --------------------------------------------------------
    # 1. INFORMAÇÕES GERAIS
    # --------------------------------------------------------
    def analise_inicial(self):
        print(f"Período: {self.dados['Data'].min():%Y-%m} a {self.dados['Data'].max():%Y-%m}")
        print(f"Total de observações: {len(self.dados)}")
        print(f"Variáveis disponíveis: {len(self.dados_analise.columns)}")

        numericas = self.dados_analise.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categoricas = self.dados_analise.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"\nVariáveis numéricas ({len(numericas)}): {', '.join(numericas)}")
        if categoricas:
            print(f"Variáveis categóricas ({len(categoricas)}): {', '.join(categoricas)}")

    # --------------------------------------------------------
    # 2. VALORES AUSENTES
    # --------------------------------------------------------
    def analise_valores_ausentes(self):
        faltantes = self.dados_analise.isnull().sum()
        faltantes = faltantes[faltantes > 0]
        if faltantes.empty:
            print("Não há valores ausentes.")
        else:
            print("Valores ausentes por variável:")
            print(faltantes.sort_values(ascending=False))

    # --------------------------------------------------------
    # 4. ESTATÍSTICAS DESCRITIVAS (com Diesel em bi L)
    # --------------------------------------------------------
    def estatisticas_descritivas(self):
        variaveis = [
            'Diesel B (Volume em litros)', 'PIB (R$)', 'Dólar Ptax',
            'Renda Média (R$)', 'Taxa de desemprego (%)',
            'Brent', 'WTI', 'Heating Oil',
            'Fluxo Pesados BR', 'Produção Agrícola'
        ]
        variaveis = [v for v in variaveis if v in self.dados_analise.columns]
        desc = self.dados_analise[variaveis].describe().T

        # Converte Diesel B para bi L na TABELA de saída (sem alterar dados originais)
        if 'Diesel B (Volume em litros)' in desc.index:
            idx = 'Diesel B (Volume em litros)'
            cols_num = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
            desc.loc[idx, cols_num] = desc.loc[idx, cols_num].apply(_to_bi)

        print_tabela_bonita(desc, "ESTATÍSTICAS DESCRITIVAS DETALHADAS")

    def analise_consumo_diesel(self):
        col = 'Diesel B (Volume em litros)'
        if col in self.dados_analise.columns:
            consumo = self.dados_analise[col]
            print("\nANÁLISE DO CONSUMO DE DIESEL (mensal):")
            print(f"• Média:   {_fmt_mi(_to_mi(consumo.mean()))}/mês")
            print(f"• Mediana: {_fmt_mi(_to_mi(consumo.median()))}/mês")
            print(f"• Desvio:  {_fmt_mi(_to_mi(consumo.std()))}/mês")
            print(f"• Amplitude: {_fmt_mi(_to_mi(consumo.max()-consumo.min()))}")
            print(f"• CV: {(consumo.std()/consumo.mean()*100):.1f}%")
            print(f"• Assimetria: {consumo.skew():.3f} | Curtose: {consumo.kurtosis():.3f}")

    # --------------------------------------------------------
    # 3. DISTRIBUIÇÃO E NORMALIDADE
    # --------------------------------------------------------
    def analise_distribuicao_avancada(self):
        dados = self.dados_diesel.dropna()  # milhões
        estat = {
            'media': dados.mean(), 'mediana': dados.median(),
            'moda': stats.mode(dados, keepdims=True)[0][0],
            'desvio': dados.std(), 'cv': (dados.std()/dados.mean())*100,
            'assimetria': skew(dados), 'curtose': kurtosis(dados, fisher=False)
        }
        _, p_shap = shapiro(dados)
        andr = anderson(dados)

        print("ANÁLISE DE DISTRIBUIÇÃO (mi L/mês)")
        print(f"Média {estat['media']:.0f} | Mediana {estat['mediana']:.0f} | Moda {estat['moda']:.0f}")
        print(f"Desvio {estat['desvio']:.0f} | CV {estat['cv']:.1f}%")
        print(f"Assimetria {estat['assimetria']:.3f} | Curtose {estat['curtose']:.3f}")
        print(f"Normalidade: Shapiro p={p_shap:.4f} | Anderson A²={andr.statistic:.2f}")

        self.resultados_estatisticos.update({
            "cv_pct": float(estat['cv']),
            "shapiro_p": float(p_shap),
            "anderson_A2": float(andr.statistic)
        })

    # --------------------------------------------------------
    # 5. OUTLIERS (IQR)
    # --------------------------------------------------------
    def analise_outliers_completa(self):
        variaveis = ['Diesel B (Volume em litros)', 'PIB (R$)', 'Dólar Ptax', 'Brent']
        for var in variaveis:
            if var in self.dados_analise.columns:
                data = self.dados_analise[var].dropna()
                q1, q3 = data.quantile(0.25), data.quantile(0.75)
                iqr = q3 - q1
                n_out = ((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)).sum()
                print(f"{var}: {n_out} outliers")

    # --------------------------------------------------------
    # 6. TEMPORAL (E CAGR)
    # --------------------------------------------------------
    def analise_serie_temporal(self):
        df = self.dados.copy()
        df['Ano_int'] = df['Data'].dt.year

        consumo_medio_ano_bi = df.groupby('Ano_int')['Diesel B (Volume em litros)'].mean().pipe(_to_bi)
        print("EVOLUÇÃO ANUAL DO CONSUMO (bi L/mês):")
        print(consumo_medio_ano_bi.round(2))

        total_anual = df.groupby('Ano_int')['Diesel B (Volume em litros)'].sum()
        a0, aF = int(total_anual.index.min()), int(total_anual.index.max())
        n = aF - a0
        cagr = (total_anual.loc[aF] / total_anual.loc[a0])**(1/n) - 1 if n > 0 else np.nan
        cres_total = (total_anual.loc[aF]/total_anual.loc[a0]-1)*100
        print(f"\nCrescimento total {a0}-{aF}: {cres_total:.1f}% | CAGR: {cagr:.2%}")

        self.resultados_estatisticos.update({"cagr": float(cagr), "crescimento_total_pct": float(cres_total)})

    # --------------------------------------------------------
    # 7. SAZONALIDADE
    # --------------------------------------------------------
    def analise_sazonalidade(self):
        df = self.dados.copy()
        df['Mes'] = df['Data'].dt.month
        media_mes_bi = df.groupby('Mes')['Diesel B (Volume em litros)'].mean().pipe(_to_bi)

        print("SAZONALIDADE MENSAL (bi L):")
        print(media_mes_bi.round(2))

        pico, vale = int(media_mes_bi.idxmax()), int(media_mes_bi.idxmin())
        amp_rel = (media_mes_bi.max()-media_mes_bi.min())/media_mes_bi.mean()*100
        print(f"\nPico: mês {pico} | Vale: mês {vale} | Variação intra-anual: {amp_rel:.1f}% da média")

        self.resultados_estatisticos.update({"sazonal_amp_rel_pct": float(amp_rel)})

    # --------------------------------------------------------
    # 8. MÉTRICAS AVANÇADAS
    # --------------------------------------------------------
    def calcular_metricas_avancadas(self):
        serie = self.dados.set_index('Data')['Diesel B (Volume em litros)']
        mi = _to_mi(serie)  # milhões

        # Volatilidade média rolling 12m
        vol12 = mi.rolling(12).std().mean()

        # YoY (taxa anual): compara mês com mesmo mês do ano anterior
        yoy = (mi.pct_change(12)*100).dropna()
        yoy_med, yoy_min, yoy_max = yoy.mean(), yoy.min(), yoy.max()

        # Drawdown (em relação ao máximo histórico de mi)
        running_max = mi.cummax()
        drawdown = (mi / running_max - 1.0) * 100
        dd_min = drawdown.min()  # valor negativo (pior queda)

        # Índice de estabilidade (média / desvio)
        estabilidade = mi.mean() / mi.std()

        # Autocorrelação lag 12 (sazonal)
        acf12 = acf(mi, nlags=12)[12]

        print("MÉTRICAS AVANÇADAS (Diesel B):")
        print(f"• Volatilidade média 12m: {vol12:.0f} mi L")
        print(f"• YoY: média {yoy_med:.1f}% | min {yoy_min:.1f}% | max {yoy_max:.1f}%")
        print(f"• Drawdown máximo: {dd_min:.1f}%")
        print(f"• Índice de estabilidade (μ/σ): {estabilidade:.2f}")
        print(f"• Autocorrelação lag 12: {acf12:.2f}")

        self.resultados_estatisticos.update({
            "vol12_mi": float(vol12),
            "yoy_media_pct": float(yoy_med),
            "drawdown_min_pct": float(dd_min),
            "estabilidade_mu_sigma": float(estabilidade),
            "acf_lag12": float(acf12)
        })

    # --------------------------------------------------------
    # 9. CORRELAÇÕES
    # --------------------------------------------------------
    def analise_correlacoes(self):
        target = 'Diesel B (Volume em litros)'
        corr = self.dados_analise.corr(numeric_only=True)[target].drop(target).sort_values(ascending=False)
        print("CORRELAÇÕES COM O CONSUMO DE DIESEL (TOP 10):")
        print(corr.head(10).round(3))

        top_pos = corr.head(3)
        top_neg = corr.tail(3)
        self.insights_textuais.append(
            "Correlação — positivos: " + ", ".join([f"{k} ({v:.2f})" for k, v in top_pos.items()]) +
            " | negativos: " + ", ".join([f"{k} ({v:.2f})" for k, v in top_neg.items()]) + "."
        )

    # --------------------------------------------------------
    # 10. ROBUSTAS (Tendência/Decomposição/Drivers)
    # --------------------------------------------------------
    def analise_tendencia_avancada(self):
        serie = self.dados.set_index('Data')['Diesel B (Volume em litros)']
        cres = ((serie.iloc[-1]/serie.iloc[0]) - 1) * 100
        print(f"Crescimento total no período: {cres:.1f}%")

    def decomposicao_temporal(self):
        serie = self.dados.set_index('Data')['Diesel B (Volume em litros)']
        if len(serie) >= 24:
            seasonal_decompose(serie, model='multiplicative', period=12)
            print("Decomposição temporal realizada (tendência, sazonalidade, resíduo).")

    def drivers_consumo(self):
        yname = 'Diesel B (Volume em litros)'
        if yname not in self.dados_analise.columns:
            print("Variável alvo ausente para drivers."); return
        X = self.dados_analise.drop(columns=[yname]).fillna(method="ffill").fillna(method="bfill")
        y = self.dados_analise[yname].copy()
        scaler = StandardScaler(); Xs = scaler.fit_transform(X)
        modelo = LinearRegression().fit(Xs, y)
        coef = pd.Series(modelo.coef_, index=X.columns).sort_values(key=np.abs, ascending=False)
        print("Principais drivers (coeficientes padronizados — top 5):")
        print(coef.head(5).round(2))
        self.resultados_estatisticos["top_driver"] = coef.index[0] if not coef.empty else None

    # --------------------------------------------------------
    # 11. ESTACIONARIEDADE / AUTOCORRELAÇÃO
    # --------------------------------------------------------
    def analise_estacionariedade(self):
        serie = self.dados.set_index('Data')['Diesel B (Volume em litros)'].dropna()
        adf_stat, adf_p, *_ = adfuller(serie)
        kpss_stat, kpss_p, *_ = kpss(serie, regression='c', nlags="auto")
        print("ADF: estat {:.4f}, p={:.4f} {}".format(adf_stat, adf_p, "→ não estacionária" if adf_p>=0.05 else "→ estacionária"))
        print("KPSS: estat {:.4f}, p={:.4f} {}".format(kpss_stat, kpss_p, "→ não estacionária" if kpss_p<0.05 else "→ estacionária"))
        self.resultados_estatisticos.update({"adf_p": float(adf_p), "kpss_p": float(kpss_p)})

    def analise_autocorrelacao(self):
        serie = self.dados.set_index('Data')['Diesel B (Volume em litros)'].dropna()
        lags = 12
        print("ACF (0–12):", acf(serie, nlags=lags).round(3))
        print("PACF (0–12):", pacf(serie, nlags=lags).round(3))

    # --------------------------------------------------------
    # 12. INSIGHTS
    # --------------------------------------------------------
    def gerar_insights_textuais(self):
        if self.insights_textuais:
            print("\nINSIGHTS AUTOMÁTICOS:")
            for i, ins in enumerate(self.insights_textuais, 1):
                print(f"{i}. {ins}")

    # --------------------------------------------------------
    # ORQUESTRAÇÃO NA ORDEM SOLICITADA
    # --------------------------------------------------------
    def analise_completa(self):
        print("\n1. INFORMAÇÕES GERAIS DO DATASET"); self.analise_inicial()
        print("\n2. ANÁLISE DE VALORES AUSENTES");    self.analise_valores_ausentes()
        print("\n4. ESTATÍSTICAS DESCRITIVAS DETALHADAS"); self.estatisticas_descritivas(); self.analise_consumo_diesel()
        print("\n3. ANÁLISE DE DISTRIBUIÇÃO E NORMALIDADE"); self.analise_distribuicao_avancada()
        print("\n5. ANÁLISE DE OUTLIERS");            self.analise_outliers_completa()
        print("\n6. ANÁLISE TEMPORAL DO CONSUMO");    self.analise_serie_temporal()
        print("\n7. ANÁLISE DE SAZONALIDADE");        self.analise_sazonalidade()
        print("\n8. MÉTRICAS AVANÇADAS");             self.calcular_metricas_avancadas()
        print("\n9. ANÁLISE DE CORRELAÇÕES");         self.analise_correlacoes()
        print("\n10. ANÁLISES ROBUSTAS DE CONSUMO E TENDÊNCIAS"); self.analise_tendencia_avancada(); self.decomposicao_temporal(); self.drivers_consumo()
        print("\n11. ESTACIONARIEDADE E AUTOCORRELAÇÃO"); self.analise_estacionariedade(); self.analise_autocorrelacao()
        print("\n12. INSIGHTS AUTOMÁTICOS");          self.gerar_insights_textuais()

# --------------------------------------------------------
# EXECUÇÃO
# --------------------------------------------------------
print("Iniciando análise exploratória completa...")
analise = AnaliseExploratoriaOtimizada(dados_consolidados)
analise.analise_completa()

print("\n" + "=" * 80)
print("ANÁLISE EXPLORATÓRIA E ESTATÍSTICA CONCLUÍDA")
print("=" * 80)

#%% SEÇÃO 5 VISUALIZAÇÕES E GRÁFICOS

print("=" * 80)
print("VISUALIZAÇÕES E GRÁFICOS COMPLETOS (Seção 5)")
print("=" * 80)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde, probplot
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# Configurações visuais
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# ---------- utilitários ----------
def formatar_volume(x, pos=None):
    """Formatter para eixos: mostra B para bilhões e M para milhões"""
    try:
        x = float(x)
    except Exception:
        return x
    if abs(x) >= 1e9:
        return f'{x/1e9:.1f}B'
    elif abs(x) >= 1e6:
        return f'{x/1e6:.0f}M'
    else:
        return f'{x:.0f}'

formatter = FuncFormatter(formatar_volume)

def garantir_data(df):
    """Garante a existência da coluna Data no formato datetime (inplace sobre cópia)"""
    d = df.copy()
    if 'Data' not in d.columns:
        if {'Ano', 'Mês'}.issubset(d.columns):
            # Mês pode ser nome ou número; tentar converter
            try:
                # Se Mês for texto com nomes, tenta mapear
                if d['Mês'].dtype == object:
                    # tenta parse seguro: se for 'Jan'/'Janeiro' etc, mapear por número
                    meses_map = {
                        'jan':1,'fev':2,'mar':3,'abr':4,'mai':5,'jun':6,
                        'jul':7,'ago':8,'set':9,'out':10,'nov':11,'dez':12
                    }
                    def mes_to_num(m):
                        if pd.isna(m): return np.nan
                        try:
                            mv = int(m)
                            return mv
                        except:
                            mm = str(m).strip().lower()[:3]
                            return meses_map.get(mm, np.nan)
                    d['Mês_Num'] = d['Mês'].apply(mes_to_num)
                else:
                    d['Mês_Num'] = d['Mês'].astype(int)
                d['Data'] = pd.to_datetime(d['Ano'].astype(str) + '-' + d['Mês_Num'].astype(int).astype(str).str.zfill(2) + '-01')
                d.drop(columns=['Mês_Num'], inplace=True, errors='ignore')
            except Exception:
                # fallback: apenas Ano-01-01
                d['Data'] = pd.to_datetime(d['Ano'].astype(str) + '-01-01')
        else:
            raise ValueError("Para criar 'Data' o DataFrame precisa ter colunas 'Ano' e 'Mês'.")
    else:
        d['Data'] = pd.to_datetime(d['Data'])
    return d

# Usar uma cópia local para evitar modificar o original
_dados = garantir_data(dados_consolidados)

# =====================================================================================
# 1. DISTRIBUIÇÃO (HISTOGRAMA, BOXPLOT, QQ-PLOT)
# =====================================================================================
def plot_distribuicao_completa(df=_dados):
    """Gera visualização completa da distribuição do consumo de diesel"""
    dados = df['Diesel B (Volume em litros)'] / 1e6  # Converter para milhões de litros
    
    # Calcular estatísticas
    stats_dict = {
        'media': dados.mean(),
        'mediana': dados.median(),
        'moda': stats.mode(dados, keepdims=True)[0][0],
        'desvio_padrao': dados.std(),
        'q1': dados.quantile(0.25),
        'q3': dados.quantile(0.75)
    }
    
    # Teste de normalidade
    stat_shapiro, p_shapiro = shapiro(dados.dropna())
    
    # Criar figura com 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(25, 10))
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
    
    # --------------------------
    # GRÁFICO 1: HISTOGRAMA MELHORADO
    # --------------------------
    ax1 = axes[0]
    n, bins, patches = ax1.hist(dados, bins=30, alpha=0.7, color='steelblue', 
                               edgecolor='white', linewidth=1.5, density=False)
    
    # Adicionar linha de densidade KDE
    density = gaussian_kde(dados)
    x = np.linspace(dados.min(), dados.max(), 300)
    y = density(x) * len(dados) * (bins[1] - bins[0])
    ax1.plot(x, y, color='darkred', linewidth=2.5, alpha=0.8, label='Curva de Densidade')
    
    # Linhas verticais para estatísticas
    ax1.axvline(stats_dict['media'], color='black', linestyle='--', linewidth=3, alpha=0.9, label=f'Média: {stats_dict["media"]:.1f}M')
    ax1.axvline(stats_dict['mediana'], color='darkblue', linestyle='--', linewidth=3, alpha=0.9, label=f'Mediana: {stats_dict["mediana"]:.1f}M')
    ax1.axvline(stats_dict['moda'], color='#FFE66D', linestyle='-', linewidth=3, alpha=0.9, label=f'Moda: {stats_dict["moda"]:.1f}M')
    
    # Sombreamento para destacar as áreas
    ax1.axvspan(stats_dict['media'] - stats_dict['desvio_padrao'], stats_dict['media'] + stats_dict['desvio_padrao'], 
               alpha=0.1, color='#FF6B6B', label='±1 Desvio Padrão')
    ax1.axvspan(stats_dict['q1'], stats_dict['q3'], alpha=0.1, color='#4ECDC4', label='Intervalo Interquartil (IQR)')
    
    ax1.set_title('Distribuição do Consumo de Diesel B\nAnálise Estatística Completa', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Consumo Mensal (milhões de litros)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequência', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # --------------------------
    # GRÁFICO 2: BOXPLOT
    # --------------------------
    ax2 = axes[1]
    boxplot = ax2.boxplot(dados, vert=False, patch_artist=True)
    
    # Personalizar cores do boxplot
    boxplot['boxes'][0].set_facecolor('lightsteelblue')
    boxplot['boxes'][0].set_alpha(0.7)
    boxplot['whiskers'][0].set_color('black')
    boxplot['whiskers'][1].set_color('black')
    boxplot['caps'][0].set_color('black')
    boxplot['caps'][1].set_color('black')
    boxplot['medians'][0].set_color('darkblue')
    boxplot['medians'][0].set_linewidth(2)
    
    ax2.axvline(stats_dict['media'], color='black', linestyle='--', alpha=0.7, label=f'Média: {stats_dict["media"]:.1f}M')
    ax2.axvline(stats_dict['mediana'], color='darkblue', linestyle='--', alpha=0.7, label=f'Mediana: {stats_dict["mediana"]:.1f}M')
    
    ax2.set_title('Boxplot - Identificação de Outliers', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Consumo (milhões de litros)', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # --------------------------
    # GRÁFICO 3: QQ-PLOT
    # --------------------------
    ax3 = axes[2]
    probplot(dados, dist="norm", plot=ax3)
    
    # Personalizar o QQ-plot
    line = ax3.lines[1]
    line.set_color('darkred')
    line.set_linewidth(2)
    line.set_alpha(0.8)
    
    scatter = ax3.lines[0]
    scatter.set_color('steelblue')
    scatter.set_marker('o')
    scatter.set_markersize(4)
    scatter.set_alpha(0.7)
    
    ax3.set_title('Q-Q Plot - Teste de Normalidade', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xlabel('Quantis Teóricos', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Quantis Amostrais', fontsize=11, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.3)
    
    # Adicionar informações sobre normalidade
    normality_text = (f'Teste Shapiro-Wilk:\n'
                      f'p-valor = {p_shapiro:.4f}\n'
                      f'{"Normal" if p_shapiro > 0.05 else "Não-normal"} (α=0.05)')
    ax3.text(0.02, 0.98, normality_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', bbox=bbox_props)
    
    plt.tight_layout()
    plt.show()

print("2) Distribuição completa (histograma, boxplot, QQ-plot)")
plot_distribuicao_completa()

# =====================================================================================
# 2. GRÁFICO DE BARRAS - CONSUMO ANUAL (com YoY, média, destaque e CAGR)
# =====================================================================================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def _fmt_bi(x):
    return f"{x/1e9:.1f} bi"

def criar_grafico_barras_anual(df=_dados):
    # agrega
    anual = (df.groupby(df["Data"].dt.year)["Diesel B (Volume em litros)"]
               .sum()
               .rename("Volume")
               .reset_index()
               .rename(columns={"Data": "Ano"}))
    anual["YoY"] = anual["Volume"].pct_change()

    # estatísticas
    media = anual["Volume"].mean()
    ano_min = anual.loc[anual["Volume"].idxmin(), "Ano"]
    ano_max = anual.loc[anual["Volume"].idxmax(), "Ano"]
    ano_ini, vol_ini = anual.iloc[0]["Ano"], anual.iloc[0]["Volume"]
    ano_fim, vol_fim = anual.iloc[-1]["Ano"], anual.iloc[-1]["Volume"]
    n_anos = (ano_fim - ano_ini)
    cagr = (vol_fim / vol_ini) ** (1 / n_anos) - 1 if n_anos > 0 else np.nan

    # paleta com destaques
    cores = ["#8FB8D8"] * len(anual)                # azul claro
    cores[anual[anual["Ano"] == ano_min].index[0]] = "#9aa6b2"  # cinza para mínimo
    cores[-1] = "#2F4858"                           # azul escuro para último ano

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=anual, x="Ano", y="Volume", palette=cores, edgecolor="black")

    # rótulos do valor em bilhões
    for p, v in zip(ax.patches, anual["Volume"]):
        h = p.get_height()
        ax.annotate(_fmt_bi(v), (p.get_x() + p.get_width()/2., h),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 6),
                    textcoords="offset points")

    # rótulos de variação YoY
    for i, (x, yoy) in enumerate(zip(ax.get_xticks(), anual["YoY"])):
        if pd.notna(yoy):
            ax.annotate(f"{yoy:+.1%}", (x, anual.loc[i, "Volume"]),
                        ha="center", va="bottom", fontsize=8, color="#555", xytext=(0, 22),
                        textcoords="offset points")

    # linha da média do período
    ax.axhline(media, ls="--", lw=1, color="#999")
    ax.text(ax.get_xticks()[0]-0.4, media*1.005, f"Média {_fmt_bi(media)}",
            fontsize=9, color="#666")

    # callout do CAGR (com anos inteiros e texto revisado)
    ax.text(
        0.02, 0.92,
        f"Taxa de Crescimento entre {int(ano_ini)}–{int(ano_fim)} = {cagr:.1%} ao ano",
        transform=ax.transAxes,
        fontsize=10,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc")
    )

    # ajustes visuais
    ax.yaxis.set_major_formatter(lambda v, pos: _fmt_bi(v))
    ax.set_xlabel("Ano")
    ax.set_ylabel("Volume")
    ax.set_title("Consumo anual de Diesel B (2017–2024)")
    ax.grid(axis="y", alpha=0.25)
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    plt.show()

print("2) Barras anuais")
criar_grafico_barras_anual()

# =====================================================================================
# 3. EVOLUÇÃO TEMPORAL (SÉRIE MENSAL) - média móvel + tendência + min/máx anotados
# =====================================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def criar_evolucao_temporal(df=_dados, rolling_months=3):
    # Série mensal agregada
    ts = (df.groupby('Data')['Diesel B (Volume em litros)']
            .sum()
            .reset_index()
            .sort_values('Data'))
    
    # Conversões de escala
    ts['Volume_M'] = ts['Diesel B (Volume em litros)'] / 1e6   # milhões
    ts['Volume_B'] = ts['Diesel B (Volume em litros)'] / 1e9   # bilhões
    ts['MM'] = ts['Volume_M'].rolling(window=rolling_months, min_periods=1).mean()

    # Posição de mínimo e máximo (em bilhões p/ legenda e rótulos)
    idx_min = ts['Volume_B'].idxmin()
    idx_max = ts['Volume_B'].idxmax()
    d_min, vmin_B = ts.loc[idx_min, 'Data'], ts.loc[idx_min, 'Volume_B']
    d_max, vmax_B = ts.loc[idx_max, 'Data'], ts.loc[idx_max, 'Volume_B']

    # Linha de tendência linear (sobre a escala em milhões)
    x_num = mdates.date2num(ts['Data'])                     # converter datas para números
    coef = np.polyfit(x_num, ts['Volume_M'], deg=1)         # [slope, intercept]
    trend_M = np.polyval(coef, x_num)
    # R² opcional (bom para slide)
    ss_res = np.sum((ts['Volume_M'] - trend_M)**2)
    ss_tot = np.sum((ts['Volume_M'] - ts['Volume_M'].mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ts['Data'], ts['Volume_M'], lw=2.2, label='Consumo mensal', color='navy')
    ax.plot(ts['Data'], ts['MM'], lw=1.8, ls='--', label=f'Média móvel ({rolling_months}m)', color='steelblue')
    ax.plot(ts['Data'], trend_M, lw=2.0, ls='-.', color='gray',
            label=f'Tendência linear (R²={r2:.2f})')

    # Marcadores min/máx com anotações
    ax.scatter([d_min, d_max],
               [vmin_B*1e3, vmax_B*1e3],  # voltar para milhões no eixo
               color=['firebrick','seagreen'], s=70, zorder=5)
    ax.annotate(f"mín: {vmin_B:.1f} bi",
                (d_min, vmin_B*1e3),
                xytext=(0, 12), textcoords='offset points',
                ha='center', color='firebrick', fontsize=10, weight='bold')
    ax.annotate(f"máx: {vmax_B:.1f} bi",
                (d_max, vmax_B*1e3),
                xytext=(0, 12), textcoords='offset points',
                ha='center', color='seagreen', fontsize=10, weight='bold')

    # Formatação
    ax.set_title('Evolução mensal do consumo de Diesel B', pad=12, fontsize=14)
    ax.set_ylabel('Milhões de litros')
    ax.grid(alpha=0.25, linestyle=':')
    ax.margins(x=0.01)

    # Datas legíveis no eixo x
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1,4,7,10)))

    # Legenda compacta com volumes agregados
    legenda = (f"Min: {vmin_B:.1f} bi ({d_min:%b/%Y})  |  "
               f"Máx: {vmax_B:.1f} bi ({d_max:%b/%Y})")
    ax.legend(title=legenda, loc='upper left', frameon=False)

    plt.tight_layout()
    plt.show()
print("3) Evolução temporal (mensal)")
criar_evolucao_temporal()

# =====================================================================================
# 4. GRÁFICO DOS ÚLTIMOS 5 ANOS (comparação mensal com faixa min-max e média)
# =====================================================================================
def plot_consumo_diesel_5_anos(df=_dados):
    try:
        # Preparar dados
        dados = df.copy()
        if 'Data' not in dados.columns:
            dados['Data'] = pd.to_datetime(dados['Ano'].astype(str) + '-' + dados['Mês'].astype(str) + '-01')
        dados['Ano'] = dados['Data'].dt.year
        dados['Mês_Num'] = dados['Data'].dt.month

        # Filtrar últimos 5 anos
        anos_recente = sorted(dados['Ano'].unique())[-5:]
        diesel_data = dados[dados['Ano'].isin(anos_recente)]
        diesel_aggregated = diesel_data.groupby(['Ano', 'Mês_Num'])['Diesel B (Volume em litros)'].sum().reset_index()

        # Pivotar dados
        historico = diesel_aggregated.pivot(index='Mês_Num', columns='Ano', values='Diesel B (Volume em litros)')
        media_historico = historico.mean(axis=1)
        min_max = diesel_aggregated.groupby('Mês_Num')['Diesel B (Volume em litros)'].agg(['min', 'max']).reset_index()

        # Encontrar valor máximo para posicionamento dos rótulos
        max_valor = min_max['max'].max()

        # Plotar
        plt.figure(figsize=(14, 8))
        plt.fill_between(
            min_max['Mês_Num'], min_max['min'], min_max['max'],
            color='lightgray', alpha=0.3, label='Mín-Máx (5 anos)'
        )

        cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
        anos_ordenados = sorted(historico.columns)
        for i, year in enumerate(anos_ordenados):
            is_recente = year == max(historico.columns)
            style = 'dotted' if not is_recente else 'solid'
            color = 'red' if is_recente else cores[i % len(cores)]
            linewidth = 3 if is_recente else 2
            plt.plot(
                historico.index, historico[year],
                label=str(year), linestyle=style, color=color,
                linewidth=linewidth, alpha=1.0 if is_recente else 0.6,
                marker='o' if is_recente else None, markersize=5 if is_recente else 0
            )

        plt.plot(
            media_historico.index, media_historico,
            color='black', linestyle='--', linewidth=2.5,
            label='Média (Últimos 5 anos)'
        )

        # Adicionar rótulos do ano mais recente
        ano_recente = max(historico.columns)
        if ano_recente in historico.columns:
            for mes in historico.index:
                valor = historico.loc[mes, ano_recente]
                if not pd.isna(valor):
                    valor_formatado = formatar_volume(valor, None)
                    plt.text(
                        mes, valor + (max_valor * 0.02), valor_formatado,
                        fontsize=10, ha='center', va='bottom', color='black',
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9)
                    )

        plt.title('Consumo de Diesel B - Volume Total Brasil\n(Últimos 5 anos)', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Volume em bilhões de litros', fontsize=12, fontweight='bold')
        plt.xlabel('Mês', fontsize=12, fontweight='bold')

        meses_pt = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        plt.xticks(range(1, 13), meses_pt, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.grid(axis='x', linestyle=':', alpha=0.3)

        y_min = min_max['min'].min() * 0.95
        y_max = min_max['max'].max() * 1.08
        plt.ylim(y_min, y_max)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=4, fontsize=11, frameon=False)

        total_ultimo_ano = diesel_aggregated[diesel_aggregated['Ano'] == ano_recente]['Diesel B (Volume em litros)'].sum()
        media_5_anos = media_historico.sum()

        plt.figtext(
            0.5, 0.04,
            f"Total {ano_recente}: {formatar_volume(total_ultimo_ano, None)} | "
            f"Média 5 anos: {formatar_volume(media_5_anos, None)}",
            ha='center', fontsize=11, style='italic', fontweight='bold'
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        plt.show()

    except Exception as e:
        print(f"Erro ao gerar gráfico: {e}")

print("4) Gráfico dos últimos 5 anos")
plot_consumo_diesel_5_anos()

# =====================================================================================
# 5. ANÁLISE SAZONAL (média mensal com desvio)
# =====================================================================================
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.pyplot as plt

def plot_analise_sazonal(df=_dados, ultimos_anos=None):
    tmp = df.copy()
    if ultimos_anos:
        ano_max = tmp['Data'].dt.year.max()
        tmp = tmp[tmp['Data'].dt.year >= (ano_max - ultimos_anos + 1)]

    tmp['Mes'] = tmp['Data'].dt.month
    media  = tmp.groupby('Mes')['Diesel B (Volume em litros)'].mean() / 1e9
    desvio = tmp.groupby('Mes')['Diesel B (Volume em litros)'].std()   / 1e9
    meses = np.arange(1, 13)

    fmt_bi = FuncFormatter(lambda x, pos: f"{x:.1f} bi")

    # cores
    azul_linha = "#0D47A1"
    azul_banda = "#90CAF9"
    cor_pico   = "#FFC107"
    cor_vale   = "#D32F2F"
    cor_texto  = "black"

    fig, ax = plt.subplots(figsize=(12, 6))

    # banda (±1 desvio) — azul
    ax.fill_between(meses, (media - desvio), (media + desvio),
                    color=azul_banda, alpha=0.35, label="± 1 desvio")

    # linha média — azul
    ax.plot(meses, media.values, marker='o', linewidth=2.5,
            color=azul_linha, label="Média mensal (2017–2024)")

    # rótulos — pretos
    for m, v in zip(meses, media.values):
        ax.text(m, v + 0.03, f"{v:.1f}", ha="center", va="bottom",
                fontsize=9, color=cor_texto)

    # pico e vale
    mes_pico = int(media.idxmax()); y_pico = float(media.loc[mes_pico])
    mes_vale = int(media.idxmin()); y_vale = float(media.loc[mes_vale])

    ax.scatter([mes_pico], [y_pico], s=90, color=cor_pico, edgecolor="black", zorder=4)
    ax.scatter([mes_vale], [y_vale], s=90, color=cor_vale, edgecolor="black", zorder=4)

    ax.annotate(f"Pico: {y_pico:.1f} bi",
                xy=(mes_pico, y_pico), xytext=(mes_pico, y_pico + 0.28),
                ha="center", color=cor_texto, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=cor_texto, lw=1.1))

    ax.annotate(f"Vale: {y_vale:.1f} bi",
                xy=(mes_vale, y_vale), xytext=(mes_vale, y_vale - 0.35),
                ha="center", va="top", color=cor_texto, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=cor_texto, lw=1.1))

    # eixos e título
    ax.set_xticks(meses)
    ax.set_xticklabels(['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'])
    ax.set_ylabel("Bilhões de litros", color=cor_texto)
    ax.yaxis.set_major_formatter(fmt_bi)

    periodo = f"{tmp['Data'].dt.year.min()}–{tmp['Data'].dt.year.max()}"
    media_anual = (tmp['Diesel B (Volume em litros)'].sum()
                   / tmp['Data'].dt.year.nunique()) / 1e9

    plt.title(
        f"Padrão Sazonal — Média Mensal (com desvio)\nPeríodo: {periodo} • Média anual: {media_anual:.1f} bi L/ano",
        color=cor_texto, fontsize=13, loc="center"
    )

    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plt.show()
print("5) Sazonalidade (gráfico)")
plot_analise_sazonal()


# =====================================================================================
# 6. MAPA DE CALOR (CORRELAÇÃO) - apenas heatmap visual
# =====================================================================================
def analise_correlacao_heatmap(df=_dados):
    # Seleciona apenas numéricas (exceto colunas temporais)
    exclude = ['Ano','Mês','Trimestre','Semestre','Quantidade_Diesel_B (mil m³)']
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    # Garante que Diesel B esteja presente
    if 'Diesel B (Volume em litros)' not in num_cols:
        num_cols.append('Diesel B (Volume em litros)')
    corr_df = df[num_cols].corr()
    # Queremos heatmap das correlações *com* Diesel B (coluna)
    corr_with_diesel = corr_df[['Diesel B (Volume em litros)']].sort_values(by='Diesel B (Volume em litros)', ascending=False)

    plt.figure(figsize=(6, max(6, len(corr_with_diesel)*0.25)))
    sns.heatmap(corr_with_diesel, annot=True, cmap='coolwarm', center=0, cbar=True, fmt='.3f', linewidths=.5)
    plt.title('Correlação das variáveis com Diesel B (visual)')
    plt.tight_layout()
    plt.show()

print("6) Heatmap de correlação (visual)")
analise_correlacao_heatmap()

# =====================================================================================
# 7. EVOLUÇÃO TEMPORAL COMPARATIVA (series normalizadas)
# =====================================================================================
def plot_evolucao_comparativa(df=_dados, variaveis=None):
    if variaveis is None:
        variaveis = ['Diesel B (Volume em litros)', 'PIB (R$)', 'Dólar Ptax', 'IBC-Br']
    vars_avail = [v for v in variaveis if v in df.columns]
    if len(vars_avail) < 2:
        print("Variáveis insuficientes para comparativo.")
        return
    n = len(vars_avail)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5*n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, var in zip(axes, vars_avail):
        series = df.set_index('Data')[var].astype(float)
        s_norm = (series - series.min()) / (series.max() - series.min())
        ax.plot(s_norm.index, s_norm.values, lw=1.8, color='darkblue')
        ax.set_title(var)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

print("7) Evolução comparativa (normalizada)")
plot_evolucao_comparativa()

# =====================================================================================
# 8. DISTRIBUIÇÃO POR TRIMESTRE (boxplot)
# =====================================================================================
def plot_analise_trimestral(df=_dados):
    if 'Data' not in df.columns:
        return
    tmp = df.copy()
    tmp['Ano'] = tmp['Data'].dt.year
    tmp['Trimestre'] = tmp['Data'].dt.quarter
    trimestral = tmp.groupby(['Ano','Trimestre'])['Diesel B (Volume em litros)'].sum().reset_index()
    plt.figure(figsize=(10,6))
    sns.boxplot(x='Trimestre', y='Diesel B (Volume em litros)', data=trimestral, palette='Blues')
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.title('Distribuição do Consumo por Trimestre')
    plt.tight_layout()
    plt.show()

print("8) Boxplot por trimestre")
plot_analise_trimestral()

# =====================================================================================
# 9. TENDÊNCIA DE CRESCIMENTO (barras + linha de crescimento %)
# =====================================================================================
def plot_tendencia_crescimento(df=_dados):
    annual = df.groupby(df['Data'].dt.year)['Diesel B (Volume em litros)'].sum().reset_index(name='Volume')
    annual['Crescimento_%'] = annual['Volume'].pct_change() * 100
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8), gridspec_kw={'height_ratios':[2,1]})
    ax1.bar(annual['Data'].astype(int).astype(str), annual['Volume']/1e9, color='steelblue')
    ax1.set_ylabel('Bilhões de litros')
    ax1.set_title('Consumo Anual')
    for p in ax1.patches:
        h = p.get_height()
        ax1.annotate(f'{h:.1f}B', (p.get_x()+p.get_width()/2., h), ha='center', va='bottom', fontsize=9)
    ax2.plot(annual['Data'].astype(int).astype(str), annual['Crescimento_%'], marker='o', color='green')
    ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Crescimento %')
    plt.tight_layout()
    plt.show()

print("9) Tendência de crescimento anual")
plot_tendencia_crescimento()

# =====================================================================================
# 10. GRÁFICOS COMPARATIVOS (Diesel vs outras variáveis) - múltiplas janelas
# =====================================================================================
def plot_comparativo_diesel_vs_variaveis(df=_dados, anos=5, max_vars=8):
    # garante Data
    d = df.copy()
    ultimo_ano = d['Data'].dt.year.max()
    anos_filtrar = list(range(ultimo_ano - anos + 1, ultimo_ano + 1))
    df_filtrado = d[d['Data'].dt.year.isin(anos_filtrar)]
    # seleciona variáveis numéricas exceto temporais e alvo
    exclude = ['Data','Ano','Mês','Trimestre','Semestre','Diesel B (Volume em litros)','Quantidade_Diesel_B (mil m³)']
    cand = [c for c in df_filtrado.select_dtypes(include=[np.number]).columns if c not in exclude]
    cand = cand[:max_vars]
    for var in cand:
        fig, ax1 = plt.subplots(figsize=(14,6))
        ax1.plot(df_filtrado['Data'], df_filtrado['Diesel B (Volume em litros)']/1e9, color='darkblue', lw=2.2, label='Diesel B (B)')
        ax1.set_ylabel('Diesel B (B)')
        ax2 = ax1.twinx()
        ax2.plot(df_filtrado['Data'], df_filtrado[var], color='red', lw=1.8, linestyle='--', label=var)
        ax2.set_ylabel(var)
        plt.title(f'Diesel B vs {var} (últimos {anos} anos)')
        # legenda combinada
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines+lines2, labels+labels2, loc='upper center', bbox_to_anchor=(0.5,-0.15), ncol=2, frameon=False)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.18)
        plt.show()

print("10) Comparativos Diesel vs variáveis econômicas")
plot_comparativo_diesel_vs_variaveis(anos=5)
plot_comparativo_diesel_vs_variaveis(anos=3)
plot_comparativo_diesel_vs_variaveis(anos=1)

# =====================================================================================
# 12. DECOMPOSIÇÃO SAZONAL (trend, seasonal, residual)
# =====================================================================================
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.ticker as mticker

def plot_decomposicao_sazonal(df=_dados):
    serie = df.set_index('Data')['Diesel B (Volume em litros)']
    decomposicao = seasonal_decompose(serie, model='multiplicative', period=12)

    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('Decomposição Sazonal do Consumo de Diesel B', fontsize=16, fontweight='bold', y=0.97)

    componentes = [
        ('Série Original', serie),
        ('Tendência', decomposicao.trend),
        ('Sazonalidade', decomposicao.seasonal),
        ('Resíduos', decomposicao.resid)
    ]

    for ax, (titulo, dados) in zip(axes, componentes):
        ax.plot(dados, color='#003366', linewidth=2)
        ax.set_ylabel(titulo, fontsize=11, fontweight='bold', labelpad=10)
        ax.grid(alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Formatação dos valores do eixo Y em bilhões
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e9:.1f} bi'))

    axes[-1].set_xlabel('Ano', fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

print("12) Decomposição sazonal")
plot_decomposicao_sazonal()

# =====================================================================================
# 13. HEATMAP SAZONAL (Ano x Mês)
# =====================================================================================
def plot_heatmap_sazonal(df=_dados):
    serie = df.set_index('Data')['Diesel B (Volume em litros)']
    decomposicao = seasonal_decompose(serie, model='multiplicative', period=12)
    df_decomp = pd.DataFrame({
        'Volume_Original': serie,
        'Tendencia': decomposicao.trend,
        'Sazonalidade': decomposicao.seasonal,
        'Resíduo': decomposicao.resid
    }).dropna()

    pivot_table = df_decomp.pivot_table(values='Sazonalidade',
                                        index=df_decomp.index.year,
                                        columns=df_decomp.index.month,
                                        aggfunc='mean')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='RdYlGn_r', center=1.0, annot=True, fmt='.3f')
    plt.title('Heatmap Sazonal - Consumo de Diesel B', fontsize=16, fontweight='bold')
    plt.xlabel('Mês')
    plt.ylabel('Ano')
    plt.tight_layout()
    plt.show()

print("13) Heatmap sazonal")
plot_heatmap_sazonal()

# =====================================================================================
# 14. ESTATÍSTICAS ROLLING (média, std, extremos, quantis, autocorrelação)
# =====================================================================================
def plot_estatisticas_rolling(df=_dados):
    serie = df.set_index('Data')['Diesel B (Volume em litros)']
    decomposicao = seasonal_decompose(serie, model='multiplicative', period=12)
    df_decomp = pd.DataFrame({
        'Volume_Original': serie,
        'Tendencia': decomposicao.trend,
        'Sazonalidade': decomposicao.seasonal,
        'Resíduo': decomposicao.resid
    }).dropna()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Média e Desvio Padrão (12 meses)
    axes[0, 0].plot(df_decomp['Volume_Original'].rolling(window=12).mean(), label='Média 12M', color='blue')
    axes[0, 0].plot(df_decomp['Volume_Original'].rolling(window=12).std(), label='Desvio Padrão 12M', color='red')
    axes[0, 0].set_title('Estatísticas Rolling (12 meses)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Mínimo e Máximo (12 meses)
    axes[0, 1].plot(df_decomp['Volume_Original'].rolling(window=12).min(), label='Mínimo 12M', color='green')
    axes[0, 1].plot(df_decomp['Volume_Original'].rolling(window=12).max(), label='Máximo 12M', color='orange')
    axes[0, 1].set_title('Extremos Rolling (12 meses)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Quartis (12 meses)
    axes[1, 0].plot(df_decomp['Volume_Original'].rolling(window=12).quantile(0.25), label='Q1 12M', color='purple')
    axes[1, 0].plot(df_decomp['Volume_Original'].rolling(window=12).quantile(0.75), label='Q3 12M', color='brown')
    axes[1, 0].set_title('Quantis Rolling (12 meses)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Autocorrelação (lags 1–12)
    autocorr = [df_decomp['Volume_Original'].autocorr(lag=lag) for lag in range(1, 13)]
    axes[1, 1].bar(range(1, 13), autocorr, color='teal')
    axes[1, 1].set_title('Autocorrelação Rolling (lags 1-12)')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('Autocorrelação')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

print("14) Estatísticas rolling")
plot_estatisticas_rolling()


# =====================================================================================
# 15. PLOTS ACF E PACF DOS RESÍDUOS
# =====================================================================================
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

def plot_acf_pacf_residuos(df=_dados, nlags=24):
    serie = df.set_index('Data')['Diesel B (Volume em litros)']
    decomposicao = seasonal_decompose(serie, model='multiplicative', period=12)
    df_decomp = pd.DataFrame({
        'Volume_Original': serie,
        'Tendencia': decomposicao.trend,
        'Sazonalidade': decomposicao.seasonal,
        'Resíduo': decomposicao.resid
    }).dropna()
    
    residuos = df_decomp['Resíduo'].dropna()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # ACF
    acf_values = acf(residuos, nlags=nlags, fft=True)
    ax1.stem(range(len(acf_values)), acf_values, basefmt=" ", linefmt='darkblue', markerfmt='o')
    conf = 1.96 / np.sqrt(len(residuos))
    ax1.axhline(y=conf, linestyle='--', color='red', alpha=0.7)
    ax1.axhline(y=-conf, linestyle='--', color='red', alpha=0.7)
    ax1.set_title('ACF - Resíduos', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('ACF')
    ax1.grid(alpha=0.3)
    
    # PACF
    pacf_values = pacf(residuos, nlags=nlags)
    ax2.stem(range(len(pacf_values)), pacf_values, basefmt=" ", linefmt='darkblue', markerfmt='o')
    ax2.axhline(y=conf, linestyle='--', color='red', alpha=0.7)
    ax2.axhline(y=-conf, linestyle='--', color='red', alpha=0.7)
    ax2.set_title('PACF - Resíduos', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('PACF')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print("15) ACF/PACF dos resíduos")
plot_acf_pacf_residuos()

# =====================================================================================
# 16. ANÁLISE DE DISPERSÃO COM IC 95% (layout otimizado)
# =====================================================================================
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analise_dispersao_otimizada(df=_dados):
    diesel_col = 'Diesel B (Volume em litros)'

    # Excluir colunas temporais irrelevantes
    exclude_cols = ['Ano', 'Mês', 'Trimestre', 'Semestre', 'Quantidade_Diesel_B (mil m³)']
    numeric_cols_all = df.select_dtypes(include=[np.number]).columns
    numeric_cols_filtered = [col for col in numeric_cols_all if col not in exclude_cols]

    modelo_diesel_clean = df[numeric_cols_filtered].dropna()

    # Selecionar apenas variáveis numéricas (exceto a dependente)
    numeric_cols = [col for col in modelo_diesel_clean.columns 
                    if col != diesel_col and modelo_diesel_clean[col].nunique() > 1]

    # Função para criar cada gráfico
    def create_scatterplot(var, x, y):
        fig = plt.figure(figsize=(12, 6.5))
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4, 1], hspace=0.1)
        ax = fig.add_subplot(gs[0])
        eq_ax = fig.add_subplot(gs[1])
        eq_ax.axis('off')

        # Dispersão
        sns.scatterplot(x=x, y=y, color='black', s=50, alpha=0.7, ax=ax)

        # Regressão com IC 95% (statsmodels)
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()

        x_pred = np.linspace(x.min(), x.max(), 100)
        X_pred = sm.add_constant(x_pred)
        predictions = model.get_prediction(X_pred)
        frame = predictions.summary_frame(alpha=0.05)

        # Linha de regressão
        ax.plot(x_pred, frame['mean'], color='darkblue', linewidth=2, label='Regressão Linear')

        # Banda de confiança
        ax.fill_between(x_pred, frame['mean_ci_lower'], frame['mean_ci_upper'],
                        color='blue', alpha=0.2, label='IC 95%')

        # Métricas
        slope = model.params[1]
        intercept = model.params[0]
        r2 = model.rsquared
        p_value = model.pvalues[1]

        def format_coef(coef):
            abs_coef = abs(coef)
            if abs_coef >= 1e9:
                return f"{coef/1e9:.2f}B"
            elif abs_coef >= 1e6:
                return f"{coef/1e6:.2f}M"
            elif abs_coef >= 1e3:
                return f"{coef/1e3:.2f}K"
            return f"{coef:.4f}"

        eq_text = (f"y = {format_coef(slope)}x "
                   f"{'+' if intercept >= 0 else '-'} {format_coef(abs(intercept))} | "
                   f"R² = {r2:.2f} | p = {p_value:.4f}")

        # Configurações visuais
        ax.set_title(f'Relação entre {diesel_col} e {var}', fontsize=13, pad=10)
        ax.set_xlabel(var, fontsize=11, labelpad=10)
        ax.set_ylabel(diesel_col, fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.4)

        # Texto da equação
        eq_ax.text(0.5, 0.3, eq_text, ha='center', va='center',
                   fontsize=10, bbox=dict(facecolor='white', alpha=0.9))

        plt.tight_layout(pad=2.0)
        plt.show()

    # Loop pelas variáveis independentes
    for var in numeric_cols:
        try:
            create_scatterplot(var, modelo_diesel_clean[var], modelo_diesel_clean[diesel_col])
        except Exception as e:
            print(f"Erro em {var}: {str(e)}")

print("16) Gráficos de Dispersão")
analise_dispersao_otimizada()

# =====================================================================================
# 17.  TOP 5 COEFICIENTES PADRONIZADOS (REGRESSÃO LINEAR)
# =====================================================================================
def plot_top5_coeficientes_padronizados(df=_dados, alvo='Diesel B (Volume em litros)'):
    """
    Ajusta uma regressão linear com variáveis padronizadas (z-score) e
    plota os 5 coeficientes com maior |magnitude| (coeficientes padronizados).

    Obs.: padroniza X e y -> coeficientes comparáveis entre variáveis.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression

    if alvo not in df.columns:
        print(f"Coluna alvo '{alvo}' não encontrada.")
        return

    # --- seleção de variáveis numéricas (exclui temporais e duplicatas do alvo)
    exclude = {'Data','Ano','Mês','Trimestre','Semestre','Quantidade_Diesel_B (mil m³)'}
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    # garante presença do alvo
    if alvo not in num_cols:
        num_cols.append(alvo)

    # base limpa (remove linhas com NaN apenas nas colunas usadas)
    base = df[num_cols].dropna().copy()

    # separa X/y e remove colunas constantes de X
    X = base.drop(columns=[alvo]).astype(float)
    y = base[alvo].astype(float)
    variaveis_validas = [c for c in X.columns if X[c].nunique() > 1]
    X = X[variaveis_validas]

    if X.shape[1] == 0:
        print("Sem variáveis explicativas válidas após limpeza.")
        return

    # --- padronização (z-score) de X e y
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    Xz = scaler_X.fit_transform(X)
    yz = scaler_y.fit_transform(y.values.reshape(-1,1)).ravel()

    # --- regressão linear
    modelo = LinearRegression().fit(Xz, yz)
    coefs = pd.Series(modelo.coef_, index=X.columns)

    # Top 5 por |coef|
    top5 = coefs.reindex(coefs.abs().sort_values(ascending=False).head(5).index)

    # --- gráfico (tema azul, barras horizontais)
    plt.figure(figsize=(10, 5.2))
    ax = sns.barplot(
        x=top5.values,
        y=top5.index,
        orient='h',
        palette=['#0D47A1' if v >= 0 else '#64B5F6' for v in top5.values],
        edgecolor="black"
    )

    # rótulos de valor ao final da barra
    for v, ytick in zip(top5.values, ax.get_yticks()):
        ax.text(
            x=v + (0.03 if v >= 0 else -0.03),
            y=ytick,
            s=f"{v:+.2f}",
            va='center',
            ha='left' if v >= 0 else 'right',
            fontsize=10,
            color='black'
        )

    ax.set_title('Top 5 coeficientes padronizados (regressão linear)', fontsize=13, pad=10)
    ax.set_xlabel('Coeficiente padronizado (β*)')
    ax.set_ylabel('Variável')
    ax.grid(axis='x', linestyle=':', alpha=0.35)
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    plt.show()

    # impressão opcional da tabela (útil no console)
    print("\nTop 5 coeficientes padronizados (β*):")
    print(top5.sort_values(key=np.abs, ascending=False).round(3))

print("17) Top 5 coeficientes padronizados (regressão)")
plot_top5_coeficientes_padronizados()

# =====================================================================================
# 18. FINAL DA SEÇÃO
# =====================================================================================
print("\n" + "=" * 80)
print("VISUALIZAÇÕES DA SEÇÃO 5 CONCLUÍDAS")
print("=" * 80)

#%% SEÇÃO 6 PRÉ-PROCESSAMENTO E MODELAGEM

print("=" * 80)
print("SEÇÃO 6 - PRÉ-PROCESSAMENTO E MODELAGEM")
print("=" * 80)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# =============================
# CONFIGURAÇÕES
# =============================

# Variável alvo em bilhões de litros
TARGET_VAR = "Diesel B (Volume em litros)"

# Usar dados consolidados como base
df = dados_consolidados.copy()

# =============================
# 1. LIMPEZA E PREPARAÇÃO
# =============================

# Colunas que devem ser removidas do modelo
colunas_descartar = [
    "Data",                       # índice temporal
    "Ano", "Mês", "Mês/Ano",      # variáveis temporais diretas
    "Trimestre", "Semestre",      # variáveis temporais diretas
    "Quantidade_Diesel_B (mil m³)" # mesma informação da variável alvo em outra escala
]

df = df.drop(columns=[c for c in colunas_descartar if c in df.columns])

# Remover linhas sem target
df = df.dropna(subset=[TARGET_VAR])

# =============================
# 2. DEFINIÇÃO DE FEATURES
# =============================

features = [col for col in df.columns if col != TARGET_VAR]
print(f"Features consideradas ({len(features)}): {features}")

X = df[features]
y = df[TARGET_VAR]

# =============================
# 3. SPLIT TREINO/TESTE
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # séries temporais → não embaralhar
)

# =============================
# 4. ESCALONAMENTO
# =============================

# Scaler para features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scaler para target
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# =============================
# 5. SALVAR SCALERS
# =============================

joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

print("Scalers salvos: scaler_X.pkl e scaler_y.pkl")

# =============================
# 6. RESULTADO FINAL DA SEÇÃO
# =============================

print("\nResumo da preparação:")
print(f"- Observações totais: {len(df)}")
print(f"- Conjunto de treino: {len(X_train)}")
print(f"- Conjunto de teste: {len(X_test)}")
print(f"- Features finais utilizadas: {features}")

# Retornar objetos principais para uso nas próximas seções
resultados_preprocessamento = {
    "X_train": X_train_scaled,
    "X_test": X_test_scaled,
    "y_train": y_train_scaled,
    "y_test": y_test_scaled,
    "features": features,
    "scaler_X": scaler_X,
    "scaler_y": scaler_y,
    "df_original": df
}

# =============================
# FINAL DA SEÇÃO
# =============================

#%% SEÇÃO 7 RESULTADOS DO MODELO

print("=" * 80)
print("SEÇÃO 7 - RESULTADOS DO MODELO")
print("=" * 80)

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ---------------------------
# MAPE com proteção a zero
# ---------------------------
def mean_absolute_percentage_error(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

# ---------------------------
# Carregar scalers salvos (Seção 6)
# ---------------------------
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# ---------------------------
# Recuperar dados da Seção 6 (compatível com 2 convenções de nome)
# ---------------------------
def _get(key_a, key_b=None):
    if key_a in resultados_preprocessamento:
        return resultados_preprocessamento[key_a]
    if key_b and key_b in resultados_preprocessamento:
        return resultados_preprocessamento[key_b]
    raise KeyError

try:
    # tenta pegar arrays escalados; se não existirem, pega os não escalados
    X_train_scaled = _get("X_train_scaled", "X_train")
    X_test_scaled  = _get("X_test_scaled",  "X_test")
    y_train_scaled = _get("y_train_scaled", "y_train")
    y_test_scaled  = _get("y_test_scaled",  "y_test")
    features = resultados_preprocessamento["features"]

    # se pegamos os NÃO escalados, os nomes acima ainda funcionam,
    # mas podem conter NaN — tratamos abaixo com um reparo automático.
    X_train_raw = resultados_preprocessamento.get("X_train_original")
    X_test_raw  = resultados_preprocessamento.get("X_test_original")
    y_train_raw = resultados_preprocessamento.get("y_train_original")
    y_test_raw  = resultados_preprocessamento.get("y_test_original")

except Exception:
    print("Aviso: resultados_preprocessamento incompleto. Recriando splits e escalas com imputação...")
    df = dados_consolidados.copy()

    target = "Diesel B (Volume em litros)"
    drop_cols = ["Data","Ano","Mês","Mês/Ano","Trimestre","Semestre","Quantidade_Diesel_B (mil m³)", target]
    features = [c for c in df.columns if c not in drop_cols]

    X = df[features].copy()
    y = df[target].copy()

    # split temporal (sem embaralhar)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # imputação + escala do X
    imputer_X = SimpleImputer(strategy="median")
    X_train_imp = imputer_X.fit_transform(X_train_raw)
    X_test_imp  = imputer_X.transform(X_test_raw)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_imp)
    X_test_scaled  = scaler_X.transform(X_test_imp)

    # escala do y
    y_train_scaled = scaler_y.fit_transform(y_train_raw.values.reshape(-1,1)).ravel()
    y_test_scaled  = scaler_y.transform(y_test_raw.values.reshape(-1,1)).ravel()

# ---------------------------
# Reparo automático se ainda houver NaN (ex.: carregou arrays "raw")
# ---------------------------
def _needs_fix(arr):
    return isinstance(arr, np.ndarray) and np.isnan(arr).any()

if _needs_fix(X_train_scaled) or _needs_fix(X_test_scaled):
    print("Reparo: detectado NaN em X_*; aplicando imputação (mediana) + reescala...")
    # Precisamos dos X/y não escalados para refazer o pipeline
    try:
        # tenta recuperar DataFrames brutos do dicionário
        X_train_raw = resultados_preprocessamento.get("X_train_original", None)
        X_test_raw  = resultados_preprocessamento.get("X_test_original", None)
        y_train_raw = resultados_preprocessamento.get("y_train_original", None)
        y_test_raw  = resultados_preprocessamento.get("y_test_original", None)
    except Exception:
        X_train_raw = X_test_raw = y_train_raw = y_test_raw = None

    if X_train_raw is None or X_test_raw is None:
        # reconstrói a partir do dataset, garantindo mesmíssima divisão temporal
        df = dados_consolidados.copy()
        target = "Diesel B (Volume em litros)"
        drop_cols = ["Data","Ano","Mês","Mês/Ano","Trimestre","Semestre","Quantidade_Diesel_B (mil m³)", target]
        features = [c for c in df.columns if c not in drop_cols]
        X = df[features].copy()
        y = df[target].copy()
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

    # imputação + escala
    imputer_X = SimpleImputer(strategy="median")
    X_train_imp = imputer_X.fit_transform(X_train_raw)
    X_test_imp  = imputer_X.transform(X_test_raw)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_imp)
    X_test_scaled  = scaler_X.transform(X_test_imp)

    y_train_scaled = scaler_y.fit_transform(np.asarray(y_train_raw).reshape(-1,1)).ravel()
    y_test_scaled  = scaler_y.transform(np.asarray(y_test_raw).reshape(-1,1)).ravel()

# ---------------------------
# Treinamento
# ---------------------------
modelo = LinearRegression()
modelo.fit(X_train_scaled, y_train_scaled)

# ---------------------------
# Previsões (desescalar p/ bilhões de litros)
# ---------------------------
y_pred_scaled = modelo.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel() / 1e9
y_test_real = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel() / 1e9

# ---------------------------
# Métricas
# ---------------------------
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
mae  = mean_absolute_error(y_test_real, y_pred)
mape = mean_absolute_percentage_error(y_test_real, y_pred)
r2   = r2_score(y_test_real, y_pred)

print("\nMétricas no conjunto de teste:")
print(f"RMSE: {rmse:.2f} bilhões de litros")
print(f"MAE : {mae:.2f} bilhões de litros")
print(f"MAPE: {mape:.2f}%")
print(f"R²  : {r2:.4f}")

print("\n--- Interpretação Didática ---")
print(f"• RMSE {rmse:.2f} bi indica o erro típico quando há desvios maiores.")
print(f"• MAE {mae:.2f} bi resume o erro médio absoluto.")
print(f"• MAPE {mape:.2f}% mostra o erro percentual médio relativo ao observado.")
print(f"• R² {r2:.2f} ⇒ ~{r2*100:.1f}% da variância explicada.")
print("--- Fim da Interpretação ---\n")

# ---------------------------
# Coeficientes (escala padronizada)
# ---------------------------
coeficientes = pd.DataFrame({
    "Variável": features,
    "Coeficiente (escala normalizada)": np.asarray(modelo.coef_).flatten()
})
coeficientes["Importância"] = coeficientes["Coeficiente (escala normalizada)"].abs()
coeficientes = coeficientes.sort_values("Importância", ascending=False)

print("Top 10 coeficientes (escala padronizada):")
print(coeficientes.head(10).to_string(index=False))

# =============================
# FINAL DA SEÇÃO
# =============================

#%% SEÇÃO 8 - DIAGNÓSTICO E AJUSTES DO MODELO (Box-Cox, VIF, Diagnósticos extras)

print("=" * 80)
print("SEÇÃO 8 - DIAGNÓSTICO E AJUSTES DO MODELO")
print("=" * 80)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# ------------------------------------------------------------------
# Recupera df base (priorize resultado do pré-processamento se existir)
# ------------------------------------------------------------------
df = None
try:
    df = resultados_preprocessamento.get("df_original").copy()
    print("Usando df de resultados_preprocessamento['df_original'].")
except Exception:
    df = dados_consolidados.copy()
    print("Usando dados_consolidados como base.")

TARGET = "Diesel B (Volume em litros)"

# Colunas descartadas (consistência com Seção 6)
colunas_descartar = ["Data","Ano","Mês","Mês/Ano","Trimestre","Semestre","Quantidade_Diesel_B (mil m³)"]
cols_to_drop = [c for c in colunas_descartar if c in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"Colunas descartadas (padronização): {cols_to_drop}")

if TARGET not in df.columns:
    raise KeyError(f"Variável alvo '{TARGET}' não encontrada no dataframe.")

df = df.dropna(subset=[TARGET]).copy()

# ------------------------------------------------------------------
# Imputação + Padronização das features
# ------------------------------------------------------------------
X_all = df.drop(columns=[TARGET]).select_dtypes(include=[np.number]).copy()
y_all = df[TARGET].astype(float)

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X_all), columns=X_all.columns, index=X_all.index)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns, index=X_imputed.index)

# Split temporal 80/20
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_all, test_size=0.2, shuffle=False)
print(f"Train / Test shapes: {X_train.shape} / {X_test.shape}")

# ------------------------------------------------------------------
# Box-Cox no target (normalidade)
# ------------------------------------------------------------------
print("\nAplicando Box-Cox (treino) para verificar lambda ótimo e normalidade:")

min_y = y_train.min()
shift = 0.0
if min_y <= 0:
    shift = abs(min_y) + 1e-6
    print(f"Target contém valores <=0. Será aplicado shift = {shift:.6f} antes do Box-Cox.")

y_train_pos = y_train + shift
y_test_pos  = y_test  + shift

try:
    lmbda_opt = stats.boxcox_normmax(y_train_pos, method='mle')
    y_train_boxcox = stats.boxcox(y_train_pos, lmbda_opt)
    y_test_boxcox  = stats.boxcox(y_test_pos,  lmbda_opt)
    print(f"Lambda ótimo (Box-Cox) estimado: {lmbda_opt:.4f}")
except Exception as e:
    lmbda_opt = None
    print("Falha ao aplicar Box-Cox:", e)

# Plots Box-Cox (paleta azul)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(y_train, kde=True, color="#1f77b4", edgecolor="black")
plt.title("Histograma target (train) - original")
plt.xlabel(TARGET)

plt.subplot(1,2,2)
if lmbda_opt is not None:
    sns.histplot(y_train_boxcox, kde=True, color="#1f77b4", edgecolor="black")
    plt.title(f"Histograma target (train) - Box-Cox (λ={lmbda_opt:.3f})")
else:
    plt.text(0.2,0.5,"Box-Cox não disponível", fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sm.qqplot(y_train, line='s', ax=plt.gca(), markerfacecolor="#1f77b4", markeredgecolor="black")
plt.title("QQ-plot target (train) - original")

plt.subplot(1,2,2)
if lmbda_opt is not None:
    sm.qqplot(y_train_boxcox, line='s', ax=plt.gca(), markerfacecolor="#1f77b4", markeredgecolor="black")
    plt.title("QQ-plot target (train) - Box-Cox")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# VIF e Tolerância
# ------------------------------------------------------------------
print("\nCalculando VIF e tolerância para cada variável (usando X_train).")

def compute_vif(df_X):
    if df_X.shape[1] == 0:
        return pd.DataFrame(columns=["feature","VIF","Tolerance"])
    Xc = sm.add_constant(df_X)
    cols = Xc.columns
    vals = []
    for i in range(Xc.shape[1]):
        try:
            v = variance_inflation_factor(Xc.values, i)
        except Exception:
            v = np.nan
        vals.append(v)
    out = pd.DataFrame({"feature": cols, "VIF": vals})
    out = out[out["feature"]!="const"].copy()
    out["Tolerance"] = 1.0 / out["VIF"]
    out = out.sort_values("VIF", ascending=False).reset_index(drop=True)
    return out

vif_table = compute_vif(X_train)
print("\nVIF (ordenado):")
print(vif_table.head(30).to_string(index=False))

# Heatmap 1-coluna: correlação das removidas com o alvo
variaveis_removidas = list(vif_table[vif_table["VIF"] > 10]["feature"])
if variaveis_removidas:
    print(f"\nHeatmap de correlação das {len(variaveis_removidas)} variáveis removidas (alto VIF).")
    corr_target = X_train[variaveis_removidas].corrwith(y_train).sort_values(ascending=False).to_frame(name="Correlação com Diesel B")
    plt.figure(figsize=(6, 0.5*len(corr_target)))
    sns.heatmap(corr_target, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                cbar=True, linewidths=0.5, linecolor='white')
    plt.title("Correlação das variáveis removidas com Diesel B (visual)", fontsize=12, pad=12)
    plt.xlabel("Correlação")
    plt.ylabel("")
    plt.xticks(rotation=0); plt.yticks(rotation=0)
    plt.tight_layout(); plt.show()
else:
    print("\nNenhuma variável com VIF > 10 identificada no primeiro diagnóstico.")

# ------------------------------------------------------------------
# Condição numérica (antes da poda)
# ------------------------------------------------------------------
print("\nDiagnóstico numérico da matriz de X_train (antes da poda):")
cond_number_before = np.linalg.cond(X_train.values)
matrix_rank_before  = np.linalg.matrix_rank(X_train.values)
print(f"Número de condição da matriz (antes): {cond_number_before:.2e}")
print(f"Rank da matriz (antes): {matrix_rank_before} (de {X_train.shape[1]} variáveis)")

# ------------------------------------------------------------------
# Poda iterativa por VIF
# ------------------------------------------------------------------
print("\nAplicando poda iterativa de VIF (removendo maior VIF até que todos <= 10).")

def iterative_vif(df_X, threshold=10.0):
    df_work = df_X.copy()
    dropped = []
    while True:
        vif_df = compute_vif(df_work)
        if vif_df.empty: break
        max_vif = vif_df["VIF"].max()
        if pd.isna(max_vif): break
        if max_vif > threshold:
            var_remove = vif_df.iloc[0]["feature"]
            print(f"Removendo '{var_remove}' com VIF={max_vif:.2f}")
            df_work = df_work.drop(columns=[var_remove])
            dropped.append(var_remove)
        else:
            break
    return df_work, dropped, compute_vif(df_work)

X_train_reduzido, variaveis_removidas_iter, vif_final = iterative_vif(X_train)

X_test_reduzido = X_test[X_train_reduzido.columns] if X_train_reduzido.shape[1] > 0 else pd.DataFrame(index=X_test.index)

print("\nVariáveis removidas iterativamente:")
print(variaveis_removidas_iter)

print("\nVIF final após poda iterativa:")
print(vif_final.to_string(index=False))

# ------------------------------------------------------------------
# Pós-poda: condição e rank
# ------------------------------------------------------------------
print("\nRecalculo: condição e rank após poda:")
if X_train_reduzido.shape[1] > 0:
    cond_after = np.linalg.cond(X_train_reduzido.values)
    rank_after = np.linalg.matrix_rank(X_train_reduzido.values)
    print(f"Cond (depois): {cond_after:.2e}")
    print(f"Rank (depois): {rank_after} (de {X_train_reduzido.shape[1]} variáveis)")
else:
    print("X_train_reduzido está vazio — não foi possível calcular condição/rank.")

# Pares com correlação > 0.95 entre removidas
print("\nPares com correlação absoluta > 0.95 entre variáveis removidas (near-duplicates):")
if len(variaveis_removidas_iter) > 1:
    corr = X_train[variaveis_removidas_iter].corr().abs()
    high_pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                    .stack().reset_index()
                    .rename(columns={'level_0':'var1','level_1':'var2',0:'corr'})
                    .query('corr > 0.95'))
    if not high_pairs.empty:
        print(high_pairs.sort_values('corr', ascending=False).to_string(index=False))
    else:
        print("Nenhum par com correlação > 0.95 encontrado entre as removidas.")
else:
    print("Pares não aplicáveis (menos de 2 variáveis removidas).")

# ------------------------------------------------------------------
# PCA no bloco removido (compressão de informação)
# ------------------------------------------------------------------
print("\nPCA no bloco de variáveis removidas (avaliar compressão de informação):")
if len(variaveis_removidas_iter) >= 2:
    bloco_train = X_train[variaveis_removidas_iter].copy()
    bloco_test  = X_test[variaveis_removidas_iter].copy()

    scaler_block = StandardScaler()
    Z_train = scaler_block.fit_transform(bloco_train)

    pca_full = PCA()
    pca_full.fit(Z_train)

    # Variância explicada
    evr = pca_full.explained_variance_ratio_           # por componente
    cum = evr.cumsum()                                 # acumulada

    evr_fmt = ", ".join([f"{v:.6f}" for v in evr])
    cum_fmt = ", ".join([f"{v:.6f}" for v in cum])
    print("Explained variance ratios (removed block):", evr_fmt)
    print("Cumulative (removed block):", cum_fmt)

    # Sugestão de nº de PCs para >= 85%
    thresh = 0.85
    if cum.max() >= thresh:
        n_comp_suggest = int((cum >= thresh).argmax() + 1)
    else:
        n_comp_suggest = min(3, len(evr))
    print(f"Sugestão de n_components para reter >= {int(thresh*100)}% variância: {n_comp_suggest}")

    # Loadings (quem forma cada PC)
    loadings = pd.DataFrame(
        pca_full.components_.T,
        index=bloco_train.columns,
        columns=[f"PC{i+1}" for i in range(len(evr))]
    )

    # ---- Scree plot com rótulos PC1..PCk e anotações ----
    labels_pc = [f"PC{i}" for i in range(1, len(evr)+1)]
    plt.figure(figsize=(9,5))
    plt.plot(range(1, len(evr)+1), cum, marker='o', color="#1f77b4")
    plt.axhline(y=0.85, color='red', linestyle='--', label='85% da variância')
    plt.xticks(range(1, len(evr)+1), labels_pc)
    for i in range(min(6, len(evr))):
        plt.annotate(f"{cum[i]*100:.1f}%", (i+1, cum[i]), textcoords="offset points",
                     xytext=(0,8), ha='center', fontsize=9)
    plt.xlabel('Componente Principal')
    plt.ylabel('Variância Explicada Acumulada')
    plt.title('Scree Plot – Variância Explicada (bloco removido)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Tabela (console): top loadings por componente ----
    top_k = 5
    n_show = min(n_comp_suggest, 5)
    resumos = []
    for j in range(n_show):
        s = loadings.iloc[:, j].copy()
        s_abs = s.abs().sort_values(ascending=False).head(top_k)
        for var in s_abs.index:
            resumos.append({"PC": f"PC{j+1}", "Variável": var, "Loading": s[var], "|Loading|": abs(s[var])})
    tabela_loadings = pd.DataFrame(resumos).sort_values(["PC","|Loading|"], ascending=[True, False])
    print("\nTop loadings (quem compõe cada PC):")
    print(tabela_loadings.drop(columns="|Loading|").to_string(index=False))

    # ---- Heatmap compacto dos loadings para slide ----
    pcs_para_plot = [f"PC{i}" for i in range(1, n_show+1)]
    plt.figure(figsize=(8, 0.45 * len(loadings.index)))
    sns.heatmap(loadings[pcs_para_plot], cmap="RdBu_r", center=0, annot=False,
                linewidths=0.4, linecolor="white")
    plt.title("Loadings dos principais componentes (bloco removido)")
    plt.xlabel("Componentes"); plt.ylabel("Variáveis")
    plt.tight_layout()
    plt.show()

else:
    print("PCA não aplicável (menos de 2 variáveis removidas).")
    evr = cum = None
    loadings = pd.DataFrame()

# ------------------------------------------------------------------
# Objetos úteis para reutilização
# ------------------------------------------------------------------
diagnostic_objects = {
    "X_scaled": X_scaled, "X_train": X_train, "X_test": X_test,
    "y_train": y_train, "y_test": y_test, "lmbda_boxcox": lmbda_opt,
    "shift_boxcox": shift, "vif_table": vif_table,
    "X_train_reduzido": X_train_reduzido, "X_test_reduzido": X_test_reduzido,
    "variaveis_removidas_iter": variaveis_removidas_iter,
    "pca_evr": evr, "pca_cum": cum, "pca_loadings": loadings
}

# ------------------------------------------------------------------
# Interpretação Didática (console)
# ------------------------------------------------------------------
print("\n--- Interpretação Didática ---")
if lmbda_opt is not None:
    print(f"O Box-Cox encontrou λ = {lmbda_opt:.2f}. Valores próximos de 1 indicam distribuição já próxima da normalidade.")
else:
    print("Não foi possível calcular Box-Cox. Seguimos com os dados originais.")

print("O diagnóstico inicial de VIF mostrou valores muito altos (alguns infinitos), confirmando multicolinearidade forte.")
if variaveis_removidas:
    print(f"Foram identificadas {len(variaveis_removidas)} variáveis com VIF > 10 (heatmap exibido).")
else:
    print("Nenhuma variável com VIF > 10 no diagnóstico inicial.")

print(f"A poda iterativa removeu {len(variaveis_removidas_iter)} variáveis.")
if X_train_reduzido.shape[1] > 0:
    print(f"O conjunto final tem {X_train_reduzido.shape[1]} variáveis e condição numérica estável (ver valores).")
else:
    print("Conjunto vazio após poda — reavaliar critérios ou aplicar PCA antes da poda.")

print("Recomendações:")
print("- Verificar pares com correlação > 0.95 e manter apenas representações canônicas.")
print("- Para previsão, reintroduzir 1–2 PCs do bloco petrolífero/derivados; para interpretação, usar o conjunto reduzido.")
print("--- Fim da Interpretação ---\n")

#%% SEÇÃO 9 - SELEÇÃO DE VARIÁVEIS (STEPWISE)

print("\n" + "=" * 80)
print("SEÇÃO 9 - SELEÇÃO DE VARIÁVEIS (Stepwise)")
print("=" * 80)

# -------------------------------------------------------------------------
# Dependências
# -------------------------------------------------------------------------
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Verificar se objetos da Seção 8 estão disponíveis
if "diagnostic_objects" not in globals():
    raise RuntimeError("A Seção 8 precisa ser executada antes da Seção 9.")

X_train_reduzido = diagnostic_objects["X_train_reduzido"]
X_test_reduzido = diagnostic_objects["X_test_reduzido"]
y_train = diagnostic_objects["y_train"]
y_test = diagnostic_objects["y_test"]

# -------------------------------------------------------------------------
# Tentativa de usar statstests.process (Fávero & Santos)
# -------------------------------------------------------------------------
use_statstests = False
try:
    import statstests.process as st_process  # type: ignore
    use_statstests = True
    print("statstests disponível — usando stepwise do pacote statstests.process")
except Exception:
    print("statstests não encontrado — usando stepwise baseado em AIC (implementação própria).")

# Base para seleção
X_sel_base = X_train_reduzido.copy()
y_sel_base = y_train.copy()

# -------------------------------------------------------------------------
# Implementação stepwise forward-backward por AIC (fallback)
# -------------------------------------------------------------------------
def stepwise_aic(X, y, verbose=True, max_iter=200):
    included = []
    current_aic = sm.OLS(y, sm.add_constant(pd.DataFrame(np.ones(len(y)), index=X.index))).fit().aic
    if verbose:
        print(f"AIC inicial (const only): {current_aic:.3f}")
    it = 0
    while True and it < max_iter:
        it += 1
        changed = False
        # forward
        excluded = [c for c in X.columns if c not in included]
        best_aic = current_aic
        best_feature = None
        for new_col in excluded:
            try:
                model = sm.OLS(y, sm.add_constant(X[included + [new_col]])).fit()
                aic = model.aic
                if aic < best_aic - 1e-6:
                    best_aic = aic
                    best_feature = new_col
            except Exception:
                continue
        if best_feature is not None:
            included.append(best_feature)
            current_aic = best_aic
            changed = True
            if verbose:
                print(f" Forward: adicionada {best_feature} -> AIC {current_aic:.3f}")
        # backward
        while True:
            if len(included) == 0:
                break
            worst_feature = None
            worst_aic = current_aic
            for col in included:
                trial = [c for c in included if c != col]
                try:
                    model = sm.OLS(y, sm.add_constant(X[trial])).fit()
                    aic = model.aic
                    if aic < worst_aic - 1e-6:
                        worst_aic = aic
                        worst_feature = col
                except Exception:
                    continue
            if worst_feature is not None:
                included.remove(worst_feature)
                current_aic = worst_aic
                changed = True
                if verbose:
                    print(f" Backward: removida {worst_feature} -> AIC {current_aic:.3f}")
            else:
                break
        if not changed:
            break
    return included, current_aic

# -------------------------------------------------------------------------
# Aplicação do stepwise
# -------------------------------------------------------------------------
if use_statstests:
    try:
        # OBS: removendo argumento 'criterion' que deu erro
        result = st_process.stepwise(X_sel_base, y_sel_base, verbose=True)
        selected_vars = result.selected
        print("Variáveis selecionadas (statstests):", selected_vars)
    except Exception as e:
        print("Erro ao usar statstests:", e)
        selected_vars, final_aic = stepwise_aic(X_sel_base, y_sel_base, verbose=True)
        print("Variáveis selecionadas (fallback AIC):", selected_vars)
else:
    selected_vars, final_aic = stepwise_aic(X_sel_base, y_sel_base, verbose=True)
    print("\nStepwise finalizado. Variáveis selecionadas:", selected_vars)
    print(f"AIC final (modelo stepwise): {final_aic:.3f}")

if len(selected_vars) == 0:
    print("Stepwise retornou nenhuma variável. Usando todas as variáveis de X_train_reduzido.")
    selected_vars = list(X_sel_base.columns)

# -------------------------------------------------------------------------
# Ajuste final (OLS) com as variáveis escolhidas
# -------------------------------------------------------------------------
X_step_train = X_sel_base[selected_vars]
X_step_test = X_test_reduzido[selected_vars] if set(selected_vars).issubset(set(X_test_reduzido.columns)) else X_test_reduzido[list(set(selected_vars) & set(X_test_reduzido.columns))]

model_step = sm.OLS(y_sel_base, sm.add_constant(X_step_train)).fit()
print("\nResumo do modelo stepwise (amostra de treino):")
print(model_step.summary())

# -------------------------------------------------------------------------
# Avaliação no conjunto de teste
# -------------------------------------------------------------------------
y_pred_step = model_step.predict(sm.add_constant(X_step_test))
rmse_step = np.sqrt(((y_test - y_pred_step) ** 2).mean())
mae_step = np.abs(y_test - y_pred_step).mean()
r2_step = 1 - ((y_test - y_pred_step) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()

# Novo: cálculo do MAPE (Mean Absolute Percentage Error)
mape_step = (np.abs((y_test - y_pred_step) / y_test).mean()) * 100

print("\nMétricas no teste (modelo stepwise):")
print(f" RMSE = {rmse_step:.2f} litros  | {rmse_step/1e9:.4f} bilhões")
print(f" MAE  = {mae_step:.2f} litros  | {mae_step/1e9:.4f} bilhões")
print(f" R2   = {r2_step:.4f}")
print(f" MAPE = {mape_step:.2f}%")

# -------------------------------------------------------------------------
# Interpretação Didática
# -------------------------------------------------------------------------
print("\n--- Interpretação Didática ---")
print("O stepwise foi aplicado sobre o conjunto reduzido (sem multicolinearidade severa),")
print("o que aumenta a estabilidade numérica do modelo e melhora a interpretabilidade.")
print("As variáveis selecionadas refletem fatores estruturais de oferta e demanda do diesel,")
print("captando tanto indicadores macroeconômicos quanto de transporte e produção.")
print(f"O R² de treino foi {model_step.rsquared:.3f}, confirmando bom ajuste.")
print(f"No teste, o R² foi {r2_step:.2f}, com RMSE ≈ {rmse_step/1e9:.3f} bilhões de litros.")
print(f"O MAPE foi de {mape_step:.2f}%, indicando erro percentual médio na previsão.")
print("Isso indica que o modelo mantém poder explicativo fora da amostra, mesmo após a poda de variáveis.")
print("--- Fim da Interpretação ---\n")

# -------------------------------------------------------------------------
# Salvar resultados para uso posterior
# -------------------------------------------------------------------------
selection_objects = {
    "selected_vars": selected_vars,
    "model_step": model_step,
    "rmse_step": rmse_step,
    "mae_step": mae_step,
    "r2_step": r2_step,
    "mape_step": mape_step
}

#%% Seção 9a - Diagnóstico rápido: VIF das variáveis selecionadas e tabela resumida
print("\n" + "="*80)
print("SEÇÃO 9a - VIF das variáveis selecionadas e resumo")
print("="*80)

# garante compute_vif disponível (definida em Seção 8)
try:
    compute_vif
except NameError:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
    def compute_vif(df_X):
        Xc = sm.add_constant(df_X)
        cols = Xc.columns
        vif_vals = []
        for i in range(Xc.shape[1]):
            try:
                vif = variance_inflation_factor(Xc.values, i)
            except Exception:
                vif = np.nan
            vif_vals.append(vif)
        vif_df = pd.DataFrame({"feature": cols, "VIF": vif_vals})
        vif_df = vif_df[vif_df["feature"] != "const"].copy()
        vif_df["Tolerance"] = 1.0 / vif_df["VIF"]
        return vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)

sel = selection_objects["selected_vars"]
X_sel = X_train_reduzido[sel].copy()
vif_sel = compute_vif(X_sel)
print("\nVIF (variáveis selecionadas):")
print(vif_sel.to_string(index=False))

# salvar no objeto de seleção
selection_objects["vif_selected"] = vif_sel

#%% Seção 9b - Diagnóstico dos resíduos do modelo stepwise (teste)

print("\n" + "="*80)
print("SEÇÃO 9b - Diagnóstico de resíduos (teste)")
print("="*80)

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

# obter previsões e resíduos (já calculados, mas recalcula para segurança)
y_pred_step = model_step.predict(sm.add_constant(X_step_test))
resid_test = y_test - y_pred_step

print("Resumo métricas resíduos (teste):")
print(f" Mean resid = {resid_test.mean():.2e}, Std resid = {resid_test.std():.2e}")
print(f" Durbin-Watson (teste residual): {durbin_watson(resid_test):.4f}")

# Breusch-Pagan no modelo treinado
bp = het_breuschpagan(model_step.resid, model_step.model.exog)
print(f"Breusch-Pagan LM p-value (treino): {bp[1]:.4f}")

# Paleta azul padronizada
cor_real = "black"
cor_pred = "#1f77b4"    # azul médio
cor_resid = "#0b3c5d"   # azul escuro para os pontos (ajuste solicitado)

plt.figure(figsize=(12,8))

# 1. Série temporal - Real vs Previsto
plt.subplot(2,2,1)
plt.plot(y_test.index, y_test, label="Real", color=cor_real, linewidth=2.2)
plt.plot(y_test.index, y_pred_step, label="Previsto", color=cor_pred, linewidth=2.0)
plt.title("Real vs Previsto (teste)")
plt.xlabel("Índice temporal")
plt.ylabel("Volume Diesel B (bi L)")
plt.legend()

# 2. Resíduo vs Predito (pontos em azul escuro)
plt.subplot(2,2,2)
sns.scatterplot(x=y_pred_step, y=resid_test, color=cor_resid, edgecolor="white", s=50)
plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("Predito")
plt.ylabel("Resíduo")
plt.title("Resíduo vs Predito (teste)")

# 3. Histograma dos resíduos
plt.subplot(2,2,3)
sns.histplot(resid_test, kde=True, color=cor_pred, edgecolor="white")
plt.title("Histograma dos resíduos (teste)")
plt.xlabel("Resíduo")

# 4. QQ-plot (pontos em azul escuro)
plt.subplot(2,2,4)
sm.qqplot(resid_test, line='s', ax=plt.gca(), color=cor_resid)
plt.title("QQ-plot (resíduos teste)")

plt.tight_layout()
plt.show()

# salvar
selection_objects["resid_test"] = resid_test

#%% Seção 9c - EXPANDING-WINDOW COMPARATIVO (FULL REDUCED / STEPWISE / PCs / LASSO)

print("\n" + "=" * 80)
print("SEÇÃO 9c - Expanding-window comparativo (robustez preditiva)")
print("=" * 80)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

# Parâmetros
INITIAL_FRAC = 0.6        # fração inicial para expanding-window (treino inicial = 60% da amostra)
N_PCS = 5                 # nº máximo de componentes PCA quando usado
LASSO_CV_SPLITS = 5       # folds para LassoCV (time-series split)
PRINT_PROGRESS = True
BILHAO = 1e9              # conversão para bilhões de litros

# Verificações prévias (variáveis que o código espera existir)
try:
    X_train_reduzido.shape
    selected_vars
    X_train.shape
    y_train.shape
except Exception as e:
    raise RuntimeError("Variáveis prévias necessárias (X_train_reduzido, selected_vars, X_train, y_train) "
                       "não encontradas. Execute Seções anteriores antes desta seção.") from e

print("Configurando expanding-window. Pode demorar dependendo do tamanho da série e do LassoCV...")

# Função utilitária: rodar OLS garantindo alinhamento de colunas (const incluída)
def ols_fit_predict(Xtr, ytr, Xp):
    Xtr_const = sm.add_constant(Xtr)
    model = sm.OLS(ytr, Xtr_const).fit()
    Xp_const = sm.add_constant(Xp)
    Xp_const = Xp_const.reindex(columns=Xtr_const.columns, fill_value=0)
    yhat = model.predict(Xp_const)
    return model, np.asarray(yhat).ravel()

# Função para executar expanding-window robusta
def expanding_eval(X_full, y_full, model_type="ols", initial_frac=INITIAL_FRAC,
                   use_pcs=False, n_pcs=N_PCS, lasso_cv_splits=LASSO_CV_SPLITS, verbose=False):
    n = len(y_full)
    start = int(n * initial_frac)
    preds, trues, times = [], [], []

    for t in range(start, n):
        Xtr = X_full.iloc[:t, :].copy()
        ytr = y_full.iloc[:t].copy()
        Xp = X_full.iloc[t:t+1, :].copy()
        y_true = y_full.iloc[t:t+1].copy()

        try:
            if model_type == "ols":
                model, yhat = ols_fit_predict(Xtr, ytr, Xp)

            elif model_type == "stepwise":
                try:
                    cols, _ = stepwise_aic(Xtr, ytr, verbose=False)  # função definida em seções anteriores
                except Exception:
                    cols = list(Xtr.columns)
                if len(cols) == 0:
                    cols = list(Xtr.columns)
                Xtr_fold = Xtr[cols]
                Xp_fold = Xp.reindex(columns=cols, fill_value=0)
                model, yhat = ols_fit_predict(Xtr_fold, ytr, Xp_fold)

            elif model_type == "pcs" and use_pcs:
                n_comp = min(n_pcs, Xtr.shape[1])
                scaler = StandardScaler()
                Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr), columns=Xtr.columns, index=Xtr.index)
                Xp_s = pd.DataFrame(scaler.transform(Xp), columns=Xp.columns, index=Xp.index)
                pca = PCA(n_components=n_comp)
                pcs_tr = pd.DataFrame(pca.fit_transform(Xtr_s),
                                      columns=[f"PC{i+1}" for i in range(n_comp)],
                                      index=Xtr.index)
                pcs_xp = pd.DataFrame(pca.transform(Xp_s),
                                      columns=[f"PC{i+1}" for i in range(n_comp)],
                                      index=Xp.index)
                model, yhat = ols_fit_predict(pcs_tr, ytr, pcs_xp)

            elif model_type == "lasso":
                tscv = TimeSeriesSplit(n_splits=lasso_cv_splits)
                lasso = LassoCV(cv=tscv, n_jobs=-1, random_state=0)
                lasso.fit(Xtr.values, ytr.values.ravel())
                Xp_fold = Xp.reindex(columns=Xtr.columns, fill_value=0)
                yhat = lasso.predict(Xp_fold.values)

            else:
                raise ValueError(f"Modelo não reconhecido: {model_type}")

            preds.append(float(np.asarray(yhat).ravel()[0]) if (yhat is not None and len(yhat) > 0) else np.nan)

        except Exception:
            preds.append(np.nan)

        trues.append(float(y_true.values.ravel()[0]))
        times.append(Xp.index[0])

        if verbose and ((t - start + 1) % 10 == 0):
            print(f"  expanding progress: predições geradas {t - start + 1}/{n - start}")

    preds = np.array(preds, dtype=float)
    trues = np.array(trues, dtype=float)

    # métricas (ignora NaN automaticamente)
    mask = ~np.isnan(preds)
    preds_valid, trues_valid = preds[mask], trues[mask]

    rmse = np.sqrt(np.mean((trues_valid - preds_valid) ** 2))
    mae = np.mean(np.abs(trues_valid - preds_valid))
    denom = np.sum((trues_valid - trues_valid.mean()) ** 2)
    r2 = 1 - np.sum((trues_valid - preds_valid) ** 2) / denom if denom != 0 else np.nan
    # MAPE (%), evitando divisão por zero
    denom_mape = np.where(trues_valid == 0, np.nan, trues_valid)
    mape = np.nanmean(np.abs((trues_valid - preds_valid) / denom_mape)) * 100.0

    return {"times": np.array(times),
            "y_true": trues,
            "y_pred": preds,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape}

# Preparar os conjuntos
X_full = X_train_reduzido.copy()
X_step = X_train[selected_vars].copy()
X_pcs  = X_train_reduzido.copy()
X_lasso= X_train_reduzido.copy()

# Executar expanding-window
print("\nExecutando expanding-window para cada modelo...")

res_full  = expanding_eval(X_full,  y_train, model_type="ols",      initial_frac=INITIAL_FRAC, verbose=PRINT_PROGRESS)
res_step  = expanding_eval(X_step,  y_train, model_type="stepwise", initial_frac=INITIAL_FRAC, verbose=PRINT_PROGRESS)
res_pcs   = expanding_eval(X_pcs,   y_train, model_type="pcs",      initial_frac=INITIAL_FRAC, use_pcs=True, n_pcs=N_PCS, verbose=PRINT_PROGRESS)
res_lasso = expanding_eval(X_lasso, y_train, model_type="lasso",    initial_frac=INITIAL_FRAC, lasso_cv_splits=LASSO_CV_SPLITS, verbose=PRINT_PROGRESS)

# Compilar resultados (litros + colunas já convertidas para bi L e %)
results_df = pd.DataFrame({
    "Modelo": ["Full Reduced", "Stepwise", "PCs Augmented", "LASSO"],
    "RMSE (litros)": [res_full["rmse"], res_step["rmse"], res_pcs["rmse"], res_lasso["rmse"]],
    "MAE (litros)":  [res_full["mae"],  res_step["mae"],  res_pcs["mae"],  res_lasso["mae"]],
    "MAPE (%)":      [res_full["mape"], res_step["mape"], res_pcs["mape"], res_lasso["mape"]],
    "R2":            [res_full["r2"],   res_step["r2"],   res_pcs["r2"],   res_lasso["r2"]],
})
results_df["RMSE (bi L)"] = results_df["RMSE (litros)"] / BILHAO
results_df["MAE (bi L)"]  = results_df["MAE (litros)"]  / BILHAO

print("\n--- Expanding-window resultados (em bi L e %) ---")
print(results_df[["Modelo","R2","RMSE (bi L)","MAE (bi L)","MAPE (%)"]].to_string(index=False))

# Paleta de cores (contraste mais forte e elegante)
cores = {
    "real": "black",
    "full": "#1f77b4",   # azul médio
    "step": "#ff7f0e",   # laranja
    "pcs":  "#2ca02c",   # verde
    "lasso":"#d62728"    # vermelho
}

# Plot comparativo de séries (em bi L)
plt.figure(figsize=(12,6))
ts_idx = res_full["times"]
plt.plot(ts_idx, res_full["y_true"]/BILHAO, label="Real", color=cores["real"], linewidth=2.2)
plt.plot(ts_idx, res_full["y_pred"]/BILHAO, label="Full Reduced",  color=cores["full"], linewidth=1.8)
plt.plot(ts_idx, res_step["y_pred"]/BILHAO, label="Stepwise",      color=cores["step"], linewidth=1.8)
plt.plot(ts_idx, res_pcs["y_pred"]/BILHAO,  label="PCs Augmented", color=cores["pcs"],  linewidth=1.8)
plt.plot(ts_idx, res_lasso["y_pred"]/BILHAO,label="LASSO",         color=cores["lasso"],linewidth=1.8)
plt.title("Expanding-window comparativo - Previsão de Diesel B")
plt.xlabel("Índice temporal")
plt.ylabel("Volume Diesel B (bi L)")
plt.legend()
plt.tight_layout()
plt.show()

# Plot erro acumulado (em bi L)
plt.figure(figsize=(10,5))
err_cumsum_full  = np.cumsum((res_full["y_pred"]  - res_full["y_true"])  / BILHAO)
err_cumsum_step  = np.cumsum((res_step["y_pred"]  - res_step["y_true"])  / BILHAO)
err_cumsum_pcs   = np.cumsum((res_pcs["y_pred"]   - res_pcs["y_true"])   / BILHAO)
err_cumsum_lasso = np.cumsum((res_lasso["y_pred"] - res_lasso["y_true"]) / BILHAO)
plt.axhline(0, linestyle="--", color="gray", linewidth=1)
plt.plot(ts_idx, err_cumsum_full,  label="Full Reduced",  color=cores["full"],  linewidth=1.8)
plt.plot(ts_idx, err_cumsum_step,  label="Stepwise",      color=cores["step"],  linewidth=1.8)
plt.plot(ts_idx, err_cumsum_pcs,   label="PCs Augmented", color=cores["pcs"],   linewidth=1.8)
plt.plot(ts_idx, err_cumsum_lasso, label="LASSO",         color=cores["lasso"], linewidth=1.8)
plt.title("Erro acumulado (cumsum) - expanding-window")
plt.xlabel("Índice temporal")
plt.ylabel("Erro acumulado (bi L)")
plt.legend()
plt.tight_layout()
plt.show()

# Interpretação didática (em bi L e %)
print("\n--- Interpretação Didática ---")
print(f"Full Reduced: RMSE={res_full['rmse']/BILHAO:.3f} bi L, MAE={res_full['mae']/BILHAO:.3f} bi L, MAPE={res_full['mape']:.2f}%, R2={res_full['r2']:.4f}")
print(f"Stepwise   : RMSE={res_step['rmse']/BILHAO:.3f} bi L, MAE={res_step['mae']/BILHAO:.3f} bi L, MAPE={res_step['mape']:.2f}%, R2={res_step['r2']:.4f}")
print(f"PCs Augmen.: RMSE={res_pcs['rmse']/BILHAO:.3f} bi L, MAE={res_pcs['mae']/BILHAO:.3f} bi L, MAPE={res_pcs['mape']:.2f}%, R2={res_pcs['r2']:.4f}")
print(f"LASSO      : RMSE={res_lasso['rmse']/BILHAO:.3f} bi L, MAE={res_lasso['mae']/BILHAO:.3f} bi L, MAPE={res_lasso['mape']:.2f}%, R2={res_lasso['r2']:.4f}")

print("\nInterpretação (resumo):")
print("- Compare RMSE/MAE (em bilhões de litros) e MAPE (%) para a performance fora da amostra.")
print("- R2 negativo ou muito baixo indica overfitting no treino (modelo não generaliza).")
print("- LASSO tende a reduzir overfitting em presença de muitas variáveis correlacionadas.")
print("- PCs podem estabilizar predições, mas sacrificam interpretabilidade direta.")
print("--- Fim da Interpretação ---\n")

# Salvar resultados
expanding_results = {
    "res_full": res_full,
    "res_step": res_step,
    "res_pcs": res_pcs,
    "res_lasso": res_lasso,
    "results_table": results_df
}

#%% Seção 9d - Incluir lags do target e reavaliar (expanding-window robusto)
# =============================================================================
print("\n" + "="*80)
print("SEÇÃO 9d - Incluir lags do target e reavaliar (expanding-window robusto)")
print("="*80)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Configurações
# ---------------------------
TARGET = "Diesel B (Volume em litros)"   # seu target (ajuste se necessário)
LAGS = [1, 2, 3]
INITIAL_FRAC = 0.6            # janela inicial (60% treino -> expanding)
N_PCS = 5                     # nº max de PCs para o modelo PCA
CV_SPLITS = 5                 # máximo de folds para CV temporal
RIDGE_ALPHAS = np.logspace(-6, 6, 25)
LASSO_ALPHAS = np.logspace(-6, 1, 25)
BILHAO = 1e9                  # conversão para bilhões de litros

# ---------------------------
# 0) cópia e checagem básica
# ---------------------------
df = dados_consolidados.copy().reset_index(drop=True)  # use sua base
if TARGET not in df.columns:
    raise KeyError(f"Target '{TARGET}' não encontrado em dados_consolidados.columns")

# --- remover coluna indesejada ---
if "Quantidade_Diesel_B (mil m³)" in df.columns:
    df = df.drop(columns=["Quantidade_Diesel_B (mil m³)"])
    print("[INFO] Coluna 'Quantidade_Diesel_B (mil m³)' removida da base.")

# ---------------------------
# 1) criar lags do target
# ---------------------------
for lag in LAGS:
    df[f"{TARGET}_lag{lag}"] = df[TARGET].shift(lag)

# remover linhas com NaN (devido a lags)
df = df.dropna().reset_index(drop=True)
n_obs = len(df)
print(f"[INFO] Observações após criar lags e dropna: {n_obs}")

# ---------------------------
# 2) preparar X numeric (remover colunas não-numéricas automaticamente)
# ---------------------------
non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"[INFO] Colunas não-numéricas detectadas e descartadas: {non_numeric}")

X_all = df.select_dtypes(include=[np.number]).copy()
if TARGET in X_all.columns:
    X_all = X_all.drop(columns=[TARGET])
X_all = X_all.astype(float)

y = df[TARGET].astype(float).reset_index(drop=True)

print(f"[INFO] X numeric shape: {X_all.shape} | y length: {len(y)}")

# montar dois conjuntos: sem lags e com lags
cols_with_lags = X_all.columns.tolist()
cols_without_lags = [c for c in cols_with_lags if f"{TARGET}_lag" not in c]

X_with_lags = X_all[cols_with_lags].copy()
X_without_lags = X_all[cols_without_lags].copy()

print(f"[INFO] Variáveis (com lags): {len(cols_with_lags)} | (sem lags): {len(cols_without_lags)}")

# ---------------------------
# util: stepwise por AIC
# ---------------------------
def stepwise_aic(X, y, verbose=False, max_iter=200):
    included = []
    current_aic = sm.OLS(y, sm.add_constant(pd.DataFrame(np.ones(len(y)), index=X.index))).fit().aic
    it = 0
    while True and it < max_iter:
        it += 1
        changed = False
        excluded = [c for c in X.columns if c not in included]
        best_aic = current_aic
        best_feature = None
        for new_col in excluded:
            try:
                model = sm.OLS(y, sm.add_constant(X[included + [new_col]])).fit()
                aic = model.aic
                if aic < best_aic - 1e-6:
                    best_aic = aic
                    best_feature = new_col
            except Exception:
                continue
        if best_feature is not None:
            included.append(best_feature)
            current_aic = best_aic
            changed = True
        # backward
        while True:
            if len(included) == 0:
                break
            worst_feature = None
            worst_aic = current_aic
            for col in included:
                trial = [c for c in included if c != col]
                try:
                    model = sm.OLS(y, sm.add_constant(X[trial])).fit()
                    aic = model.aic
                    if aic < worst_aic - 1e-6:
                        worst_aic = aic
                        worst_feature = col
                except Exception:
                    continue
            if worst_feature is not None:
                included.remove(worst_feature)
                current_aic = worst_aic
                changed = True
            else:
                break
        if not changed:
            break
    return included if len(included) > 0 else list(X.columns)

# ---------------------------
# função de expanding-window
# ---------------------------
def expanding_eval(X, y, model_key, start_frac=INITIAL_FRAC, n_pcs=N_PCS, verbose=False):
    """
    Retorna: rmse, mae, r2, mape(%), n_valid e DataFrame de folds
    """
    n = len(y)
    start = int(n * start_frac)
    preds, trues, folds = [], [], []

    for t in range(start, n):
        Xtr, Xte = X.iloc[:t, :].copy(), X.iloc[t:t+1, :].copy()
        ytr, yte = y.iloc[:t].copy(), float(y.iloc[t])

        if Xtr.shape[0] < 3:
            preds.append(np.nan)
            trues.append(yte)
            folds.append({"fold": t, "y_true": yte, "y_pred": np.nan, "note": "train_too_small"})
            continue

        n_splits = min(CV_SPLITS, max(2, Xtr.shape[0] - 1))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        try:
            if model_key == "Full_RidgeCV":
                try:
                    gs = GridSearchCV(Ridge(), param_grid={"alpha": RIDGE_ALPHAS}, cv=tscv,
                                      scoring="neg_mean_squared_error", n_jobs=1, error_score="raise")
                    gs.fit(Xtr.values, ytr.values.ravel())
                    best = gs.best_estimator_
                    yhat = best.predict(Xte.values)[0]
                except Exception:
                    yhat = Ridge(alpha=1.0).fit(Xtr.values, ytr.values.ravel()).predict(Xte.values)[0]
            elif model_key == "LassoCV":
                try:
                    from sklearn.linear_model import LassoCV as LassoCVsk
                    lasso = LassoCVsk(alphas=LASSO_ALPHAS, cv=tscv, max_iter=10000, n_jobs=1)
                    lasso.fit(Xtr.values, ytr.values.ravel())
                    yhat = lasso.predict(Xte.values)[0]
                except Exception:
                    yhat = Lasso(alpha=0.01, max_iter=10000).fit(Xtr.values, ytr.values.ravel()).predict(Xte.values)[0]
            elif model_key == "Stepwise":
                sel = stepwise_aic(Xtr, ytr, verbose=False)
                if len(sel) == 0:
                    sel = list(Xtr.columns)[:1]
                mdl = LinearRegression().fit(Xtr[sel].values, ytr.values.ravel())
                Xte_sel = Xte.reindex(columns=sel, fill_value=0).values
                yhat = mdl.predict(Xte_sel)[0]
            elif model_key == "PCs_RidgeCV":
                n_comp = min(n_pcs, max(1, Xtr.shape[1] - 1))
                scaler = StandardScaler()
                Xtr_s, Xte_s = scaler.fit_transform(Xtr.values), scaler.transform(Xte.values)
                pca = PCA(n_components=n_comp)
                Xtr_p, Xte_p = pca.fit_transform(Xtr_s), pca.transform(Xte_s)
                try:
                    gs = GridSearchCV(Ridge(), param_grid={"alpha": RIDGE_ALPHAS}, cv=tscv,
                                      scoring="neg_mean_squared_error", n_jobs=1, error_score="raise")
                    gs.fit(Xtr_p, ytr.values.ravel())
                    best = gs.best_estimator_
                    yhat = best.predict(Xte_p)[0]
                except Exception:
                    yhat = Ridge(alpha=1.0).fit(Xtr_p, ytr.values.ravel()).predict(Xte_p)[0]
            else:
                raise ValueError("model_key inválido")
            preds.append(float(yhat))
            trues.append(float(yte))
            folds.append({"fold": t, "y_true": yte, "y_pred": float(yhat), "note": ""})
        except Exception as e:
            preds.append(np.nan)
            trues.append(float(yte))
            folds.append({"fold": t, "y_true": yte, "y_pred": np.nan, "note": str(e)})

    preds_arr, trues_arr = np.array(preds, dtype=float), np.array(trues, dtype=float)
    mask = ~np.isnan(preds_arr)
    if mask.sum() == 0:
        rmse, mae, r2, mape = np.nan, np.nan, np.nan, np.nan
    else:
        y_true_v, y_pred_v = trues_arr[mask], preds_arr[mask]
        rmse = mean_squared_error(y_true_v, y_pred_v, squared=False)
        mae  = mean_absolute_error(y_true_v, y_pred_v)
        r2   = r2_score(y_true_v, y_pred_v)
        denom_mape = np.where(y_true_v == 0, np.nan, y_true_v)
        mape = np.nanmean(np.abs((y_true_v - y_pred_v) / denom_mape)) * 100.0

    df_folds = pd.DataFrame(folds)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "n_valid": int(mask.sum()), "folds": df_folds}

# ---------------------------
# executar expanding-window
# ---------------------------
models = ["Full_RidgeCV", "Stepwise", "PCs_RidgeCV", "LassoCV"]
scenarios = {"WithLags": X_with_lags, "WithoutLags": X_without_lags}

summary_rows, details = [], {}
for scen_name, Xmat in scenarios.items():
    print("\n" + "-"*60)
    print(f"[INFO] Cenário: {scen_name} | X shape = {Xmat.shape}")
    for m in models:
        print(f"\n[INFO] Executando expanding-window para: {m} (cenário={scen_name})")
        res = expanding_eval(Xmat.reset_index(drop=True), y.reset_index(drop=True),
                             model_key=m, start_frac=INITIAL_FRAC, n_pcs=N_PCS, verbose=True)
        row_name = f"{m}_{scen_name}"
        summary_rows.append((row_name, res["r2"], res["rmse"], res["mae"], res["mape"], res["n_valid"]))
        details[row_name] = res["folds"]
        print(f"[RESULTADO] {row_name}: previsões válidas = {res['n_valid']} "
              f"| RMSE={res['rmse']/BILHAO:.4f} bi L | MAE={res['mae']/BILHAO:.4f} bi L | "
              f"MAPE={res['mape']:.2f}% | R2={res['r2']:.4f}")

# ---------------------------
# montar summary DataFrame
# ---------------------------
summary_df = pd.DataFrame(
    summary_rows, columns=["Modelo", "R2", "RMSE", "MAE", "MAPE", "n_valid"]
).set_index("Modelo")
summary_df["RMSE_bi"] = summary_df["RMSE"] / BILHAO
summary_df["MAE_bi"]  = summary_df["MAE"]  / BILHAO

print("\n\n--- Resumo comparativo (expanding-window, Seção 9d) ---")
print(
    summary_df[["R2", "RMSE_bi", "MAE_bi", "MAPE", "n_valid"]]
    .sort_values("R2", ascending=False)
    .rename(columns={"RMSE_bi":"RMSE (bi L)", "MAE_bi":"MAE (bi L)", "MAPE":"MAPE (%)"})
    .to_string()
)

# tabela lado-a-lado (Model x Scenario)
rows = []
for _, r in summary_df.reset_index().iterrows():
    model_base, scenario = r["Modelo"].rsplit("_", 1)
    rows.append({
        "ModelBase": model_base, "Scenario": scenario,
        "R2": r["R2"], "RMSE_bi": r["RMSE"]/BILHAO, "MAE_bi": r["MAE"]/BILHAO, "MAPE (%)": r["MAPE"]
    })
comp_df = pd.DataFrame(rows).pivot(index="ModelBase", columns="Scenario", values=["R2","RMSE_bi","MAE_bi","MAPE (%)"])
comp_df.columns = ["_".join(col).strip() for col in comp_df.columns.values]
print("\n\n--- Comparação lado-a-lado (Model x Scenario) ---")
print(comp_df.to_string())

# ---------------------------
# salvar objeto global
# ---------------------------
expanding_results_9d = {
    "summary": summary_df,
    "details": details,
    "X_numeric_columns": list(X_all.columns),
    "dropped_non_numeric": non_numeric,
    "start_index": int(n_obs * INITIAL_FRAC)
}

print("\n[INFO] expanding_results_9d criado. Chaves:", list(expanding_results_9d.keys()))
print("[INFO] Para inspecionar folds: expanding_results_9d['details']['Full_RidgeCV_WithLags'].head()")

# ---------------------------
# ANÁLISE DIDÁTICA AUTOMÁTICA (colar logo após summary_df / comp_df)
# ---------------------------
print("\n\n--- Análise Didática Automática (Seção 9d) ---")

# Melhor pelo RMSE absoluto (menor RMSE)
best_rmse_idx = summary_df["RMSE"].idxmin()
best_rmse = summary_df.loc[best_rmse_idx, "RMSE"]
print(f" Melhor RMSE (menor erro absoluto): {best_rmse_idx} -> RMSE = {best_rmse/BILHAO:.3f} bi L")

# Melhor por R2 (maior R2)
best_r2_idx = summary_df["R2"].idxmax()
best_r2 = summary_df.loc[best_r2_idx, "R2"]
print(f" Melhor R² (maior explicação fora da amostra): {best_r2_idx} -> R² = {best_r2:.4f}")

# Efeito das lags (comparação rápida)
print("\n Efeito dos lags (comparação WithLags vs WithoutLags):")
models_base = sorted(list({m.rsplit("_",1)[0] for m in summary_df.reset_index()['Modelo'].values}))
for mb in models_base:
    try:
        r_with = summary_df.loc[f"{mb}_WithLags", "R2"]
        r_wo   = summary_df.loc[f"{mb}_WithoutLags", "R2"]
        rmse_with = summary_df.loc[f"{mb}_WithLags", "RMSE"]/BILHAO
        rmse_wo   = summary_df.loc[f"{mb}_WithoutLags", "RMSE"]/BILHAO
        print(f"  {mb}: R2_with={r_with:.4f}, R2_wo={r_wo:.4f} | RMSE_with={rmse_with:.3f} bi L, RMSE_wo={rmse_wo:.3f} bi L")
        if r_with > r_wo and rmse_with < rmse_wo:
            print(f"   -> Com lags: MELHORA consistente. Recomenda-se incluir lags para {mb}.")
        elif r_with < r_wo and rmse_with > rmse_wo:
            print(f"   -> Sem lags: MELHORA consistente. Avaliar remoção de lags para {mb}.")
        else:
            print(f"   -> Efeito misto. Interpretar com cuidado; investigar tuning e multicolinearidade.")
    except KeyError:
        continue

# Recomendação final com heurística simples (R2 e RMSE)
sc = summary_df.copy()
# normalizar R2 e RMSE (RMSE neg porque menor é melhor)
sc["R2_s"] = (sc["R2"] - sc["R2"].min()) / (sc["R2"].max() - sc["R2"].min() + 1e-12)
sc["RMSE_s"] = 1 - (sc["RMSE"] - sc["RMSE"].min()) / (sc["RMSE"].max() - sc["RMSE"].min() + 1e-12)
sc["score"] = 0.7 * sc["R2_s"] + 0.3 * sc["RMSE_s"]
best = sc["score"].idxmax()
print(f"\nRecomendação automática (heurística R2/RMSE): prefira apresentar '{best}' como modelo principal no TCC (justifique economicamente).")

print("--- Fim da análise didática automática ---\n")


#%% Seção 10 - VALIDAÇÃO DOS PRESSUPOSTOS DA REGRESSÃO (diagnósticos por modelo via expanding_results_9d)

print("\n" + "=" * 80)
print("SEÇÃO 10 - VALIDAÇÃO DOS PRESSUPOSTOS DA REGRESSÃO (diagnósticos por modelo)")
print("=" * 80)

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
import pandas as pd

# garantir que expanding_results_9d exista
if "expanding_results_9d" not in globals():
    raise RuntimeError("Objeto 'expanding_results_9d' não encontrado. Rode Seção 9d primeiro.")

details = expanding_results_9d["details"]
summary = expanding_results_9d["summary"]

model_keys = sorted(details.keys())  # ex: 'Full_RidgeCV_WithLags', etc.
models_base = sorted({k.rsplit("_",1)[0] for k in model_keys})

print(f"[INFO] Encontrados modelos em expanding_results_9d['details']: {list(details.keys())}")

# ========================
# Paleta azul padronizada
# ========================
azul_escuro = "#1f77b4"
azul_medio  = "#2e86de"
azul_claro  = "#5dade2"

def _safe_mape(y_true, y_pred):
    """MAPE em %, ignorando divisões por zero."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.where(yt == 0, np.nan, np.abs(yt))
    ape = np.abs(yt - yp) / denom
    return float(np.nanmean(ape) * 100)

def _pretty_model_name(key: str) -> str:
    """Deixa o nome do modelo legível no título."""
    name = key.replace("_", " ")
    name = name.replace("WithLags", "— com Lags")
    name = name.replace("WithoutLags", "— sem Lags")
    name = name.replace("LassoCV", "LASSO")
    name = name.replace("Full RidgeCV", "Full RidgeCV")  # mantém
    name = name.replace("PCs RidgeCV", "PCs RidgeCV")
    return name

# processar cada chave (cenário/variante)
diagnostics = {}
for key, df_folds in details.items():
    df_valid = df_folds.dropna(subset=["y_pred"]).copy()
    if df_valid.shape[0] == 0:
        print(f"[WARN] {key}: sem previsões válidas para diagnóstico.")
        continue

    y_true = df_valid["y_true"].values
    y_pred = df_valid["y_pred"].values
    resid = y_true - y_pred

    # ===== métricas (R2, RMSE, MAE, MAPE) =====
    rmse = float(np.sqrt(np.mean((resid)**2)))
    mae  = float(np.mean(np.abs(resid)))
    denom = float(np.sum((y_true - np.mean(y_true))**2))
    r2   = float(1 - np.sum(resid**2) / denom) if denom != 0 else np.nan
    mape = _safe_mape(y_true, y_pred)

    # testes de pressupostos
    dw = float(durbin_watson(resid))
    try:
        sf_stat, sf_p = shapiro(resid)
    except Exception:
        sf_stat, sf_p = np.nan, np.nan
    try:
        lb = acorr_ljungbox(resid, lags=[12], return_df=True)
    except Exception:
        lb = None

    diagnostics[key] = {
        "n": len(resid), "rmse": rmse, "mae": mae, "r2": r2, "mape": mape,
        "dw": dw, "shapiro_stat": sf_stat, "shapiro_p": sf_p, "ljungbox": lb
    }

    # =====================
    # Gráficos com paleta azul (títulos claros)
    # =====================
    model_name = _pretty_model_name(key)
    plt.figure(figsize=(12,4))

    # 1) Real vs Predito
    plt.subplot(1,3,1)
    plt.plot(df_valid["fold"], y_true,  label="Real",    color=azul_escuro, linewidth=1.8)
    plt.plot(df_valid["fold"], y_pred,  label="Predito", color=azul_claro,  linewidth=1.8, alpha=0.95)
    plt.title(f"{model_name}\nReal vs Predito")
    plt.legend()
    plt.grid(alpha=0.15)

    # 2) Resíduo vs Predito
    plt.subplot(1,3,2)
    plt.scatter(y_pred, resid, s=22, color=azul_escuro, alpha=0.85)
    plt.axhline(0, linestyle="--", color="black", linewidth=0.8)
    plt.xlabel("Predito"); plt.ylabel("Resíduo")
    plt.title(f"{model_name}\nResíduo vs Predito")
    plt.grid(alpha=0.15)

    # 3) QQ-plot (resíduos)
    plt.subplot(1,3,3)
    ax = plt.gca()
    sm.qqplot(resid, line='s', ax=ax)
    # estiliza pontos/linha do QQ-plot para tons de azul
    for l in ax.lines:
        l.set_color(azul_escuro)
        try:
            l.set_markerfacecolor(azul_claro)
            l.set_markeredgecolor(azul_escuro)
        except Exception:
            pass
    plt.title(f"{model_name}\nQQ-Plot (Resíduos)")
    plt.grid(alpha=0.15)

    plt.tight_layout()
    plt.show()

    # imprimir sumário (inclui R² e MAPE)
    print(
        f"\n[INFO] Diagnóstico {key}: n={len(resid)}, "
        f"RMSE={rmse:.2e}, MAE={mae:.2e}, MAPE={mape:.2f}%, R2={r2:.4f}, "
        f"DW={dw:.4f}, Shapiro p={sf_p:.4f}"
    )
    if lb is not None:
        print(" Ljung-Box (lags=12):")
        print(lb)

# Breusch-Pagan: usar model_step (se existir) — útil para heterocedasticidade do OLS de treino
if "model_step" in globals():
    resid_step = model_step.resid
    exog_for_test = model_step.model.exog
    bp = het_breuschpagan(resid_step, exog_for_test)
    bp_names = ["LM stat", "LM p-value", "f-stat", "f p-value"]
    print("\nBreusch-Pagan (modelo stepwise - treino):")
    for n,v in zip(bp_names, bp):
        print(f" {n}: {v}")
else:
    print("\n[WARN] 'model_step' não encontrado: não é possível rodar Breusch-Pagan no OLS de treino.")

# salvar diagnosticos para relatório
diagnostics_objects = {
    "per_model": diagnostics,
    "bp_test": bp if "bp" in locals() else None
}
print("\n[INFO] Diagnósticos da Seção 10 guardados em 'diagnostics_objects'.")

# ---------------------------
# Análise didática automática (tabela formatada com MAPE)
# ---------------------------

if "expanding_results_9d" not in globals():
    print("[WARN] expanding_results_9d não encontrado. Execute Seção 9d primeiro.")
else:
    summary = expanding_results_9d["summary"].copy()
    # garantir nomes coerentes (Modelo index)
    if "Modelo" in summary.columns:
        summary = summary.set_index("Modelo")

    # valor médio de y (para % do RMSE/MAE)
    try:
        any_key = next(iter(expanding_results_9d["details"]))
        mean_y = float(np.nanmean(expanding_results_9d["details"][any_key]["y_true"]))
    except Exception:
        try:
            mean_y = float(np.mean(y))
        except Exception:
            mean_y = np.nan

    out = summary.copy()
    out["RMSE_mil"] = (out["RMSE"] / 1e6).round(2)
    out["MAE_mil"]  = (out["MAE"] / 1e6).round(2)
    out["RMSE_bi"]  = (out["RMSE"] / 1e9).round(4)
    out["MAE_bi"]   = (out["MAE"]  / 1e9).round(4)
    out["R2"]       = out["R2"].round(4)

    # acrescenta MAPE (%) a partir dos diagnósticos calculados acima
    mape_series = pd.Series({k: diagnostics.get(k, {}).get("mape", np.nan) for k in out.index})
    out["MAPE_%"] = mape_series.round(2)

    if not np.isnan(mean_y):
        out["RMSE_pct_of_mean"] = ((out["RMSE"] / mean_y) * 100).round(2)
        out["MAE_pct_of_mean"]  = ((out["MAE"]  / mean_y) * 100).round(2)
    else:
        out["RMSE_pct_of_mean"] = np.nan
        out["MAE_pct_of_mean"]  = np.nan

    print("\n--- Tabela resumida (formatada) ---")
    display_cols = ["R2", "RMSE_mil", "MAE_mil", "RMSE_bi", "MAE_bi", "MAPE_%", "RMSE_pct_of_mean"]
    print(out[display_cols].sort_values("R2", ascending=False).to_string())

    # Escolha automática do modelo recomendado (R2 alto + RMSE baixo)
    cand = out[out["R2"] > 0.60]
    preferred = cand["R2"].idxmax() if not cand.empty else out["R2"].idxmax()

    print(f"\n--- Recomendação automatizada ---")
    print(f"Modelo recomendado (regra simples): {preferred}")
    print("Interpretação rápida:")
    if "Stepwise" in preferred:
        print("- Stepwise: bom compromisso entre previsão e interpretação — indicado como modelo principal para o TCC.")
    if "LASSO" in preferred or "Lasso" in preferred:
        print("- LASSO: boa penalização; use como robustez.")
    if "PCs" in preferred:
        print("- PCs: útil para redução de dimensionalidade, mas perde interpretabilidade.")

    # Avisos auxiliares se houver objetos de pressupostos globais
    try:
        if ("bp_test" in globals().get("assumption_objects", {})):
            bp_p = assumption_objects["bp_test"][1]
            if bp_p < 0.05:
                print("- Heterocedasticidade detectada (Breusch-Pagan p<0.05). Use SE robustos (ex.: HC3).")
            elif bp_p < 0.10:
                print("- Heterocedasticidade borderline (p<0.10). Considere SE robustos.")
        lb_aux = assumption_objects.get("ljung_box", None) if 'assumption_objects' in globals() else None
        if lb_aux is not None:
            p = lb_aux["lb_pvalue"].values[0]
            if p < 0.05:
                print("- Atenção: Ljung-Box indica autocorrelação nos resíduos (p<0.05).")
    except Exception:
        pass

    print("\nObservação final: para o TCC recomendo reportar **Stepwise** como modelo principal,")
    print("incluir **LASSO** e **Ridge/PCA** como checks de robustez, e mostrar gráficos de resíduos (QQ, resid vs pred).")

#%% Gráficos do modelo Stepwise — com Lags (para slide, métricas embaixo)

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# requer 'expanding_results_9d' da Seção 9d
if "expanding_results_9d" not in globals():
    raise RuntimeError("Objeto 'expanding_results_9d' não encontrado. Rode a Seção 9d primeiro.")

details = expanding_results_9d["details"]

# localizar a chave do Stepwise com lags
cand = [k for k in details.keys() if ("Stepwise" in k and "WithLags" in k)]
if not cand:
    raise RuntimeError("Chave do modelo 'Stepwise_WithLags' não encontrada em expanding_results_9d['details'].")
key = cand[0]

# paleta azul
azul_escuro = "#1f77b4"
azul_claro  = "#5dade2"

def _safe_mape(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.where(yt == 0, np.nan, np.abs(yt))
    return float(np.nanmean(np.abs(yt - yp) / denom) * 100)

# dados válidos
df_valid = details[key].dropna(subset=["y_pred"]).copy()
y_true = df_valid["y_true"].values
y_pred = df_valid["y_pred"].values
resid  = y_true - y_pred

# métricas
rmse = float(np.sqrt(np.mean(resid**2)))
mae  = float(np.mean(np.abs(resid)))
den  = float(np.sum((y_true - np.mean(y_true))**2))
r2   = float(1 - np.sum(resid**2) / den) if den != 0 else np.nan
mape = _safe_mape(y_true, y_pred)

titulo_metricas = f"R²={r2:.3f} | RMSE={rmse/1e9:.3f} bi L | MAE={mae/1e9:.3f} bi L | MAPE={mape:.2f}%"
model_name = "Stepwise — com Lags"

# figura para o slide
plt.figure(figsize=(13.5,4.2))

# 1) Real vs Predito
ax1 = plt.subplot(1,3,1)
ax1.plot(df_valid["fold"], y_true, label="Real",    color=azul_escuro, linewidth=1.9)
ax1.plot(df_valid["fold"], y_pred, label="Predito", color=azul_claro,  linewidth=1.9, alpha=0.95)
ax1.set_title(f"{model_name}\nReal vs Predito")
ax1.set_xlabel("Índice temporal")
ax1.set_ylabel("Volume (litros)")
ax1.grid(alpha=0.15)
ax1.legend()

# 2) Resíduo vs Predito
ax2 = plt.subplot(1,3,2)
ax2.scatter(y_pred, resid, s=24, color=azul_escuro, alpha=0.9)
ax2.axhline(0, linestyle="--", color="black", linewidth=0.8)
ax2.set_title(f"{model_name}\nResíduo vs Predito")
ax2.set_xlabel("Predito")
ax2.set_ylabel("Resíduo")
ax2.grid(alpha=0.15)

# 3) QQ-Plot (resíduos)
ax3 = plt.subplot(1,3,3)
sm.qqplot(resid, line='s', ax=ax3)
for l in ax3.lines:
    l.set_color(azul_escuro)
    try:
        l.set_markerfacecolor(azul_claro)
        l.set_markeredgecolor(azul_escuro)
    except Exception:
        pass
ax3.set_title(f"{model_name}\nQQ-Plot (Resíduos)")
ax3.grid(alpha=0.15)

# Adiciona as métricas na parte de baixo da figura
plt.figtext(0.5, -0.05, titulo_metricas, ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("stepwise_withlags_diagnostics_bottom.png", dpi=240, bbox_inches="tight")
plt.show()

print("Figura salva em: stepwise_withlags_diagnostics_bottom.png")

#%% SEÇÃO 10b - Métricas adicionais, seleção final e salvar melhor modelo (compatível com 9d)

print("\n" + "="*80)
print("SEÇÃO 10b - Métricas adicionais, seleção final e salvar melhor modelo")
print("="*80)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Constantes / utils
# ---------------------------
BILHAO = 1e9

def _safe_mape(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.where(yt == 0, np.nan, np.abs(yt))
    ape = np.abs(yt - yp) / denom
    return float(np.nanmean(ape) * 100)

# ---------------------------
# Checagens básicas
# ---------------------------
if "expanding_results_9d" not in globals():
    raise RuntimeError("Objeto 'expanding_results_9d' não encontrado. Rode Seção 9d antes desta Seção 10b.")

summary = expanding_results_9d["summary"].copy()
# Ajustar índice se necessário
if "Modelo" in summary.columns:
    summary = summary.set_index("Modelo")

# Garantir colunas de interesse
for col in ["R2", "RMSE", "MAE"]:
    if col not in summary.columns:
        raise RuntimeError(f"A coluna '{col}' não existe em expanding_results_9d['summary']. Verifique Seção 9d.")

# ---------------------------
# Completar MAPE no summary (usando os folds de expanding_results_9d)
# ---------------------------
if "MAPE_%" not in summary.columns:
    mape_map = {}
    for name in summary.index:
        df_det = expanding_results_9d.get("details", {}).get(name)
        if df_det is not None and not df_det.empty:
            dfv = df_det.dropna(subset=["y_pred"])
            mape_map[name] = _safe_mape(dfv["y_true"].values, dfv["y_pred"].values)
        else:
            mape_map[name] = np.nan
    summary["MAPE_%" ] = pd.Series(mape_map)

# ---------------------------
# Ranking (R2 desc; RMSE asc) com formatação padronizada
# ---------------------------
rank = summary.sort_values(["R2", "RMSE"], ascending=[False, True]).copy()
rank_display = pd.DataFrame({
    "R²": rank["R2"].round(3),
    "RMSE (bi L)": (rank["RMSE"] / BILHAO).round(3),
    "MAE (bi L)":  (rank["MAE"]  / BILHAO).round(3),
    "MAPE (%)":    rank["MAPE_%"].round(2)
}).head(3)

print("\n[INFO] Ranking (Top 3) — R² desc / RMSE asc:")
print(rank_display.to_string(index=True))

# Escolha do melhor (maior R2, desempate por menor RMSE)
best_r2 = summary["R2"].max()
cands = summary[summary["R2"] == best_r2]
best_name = cands["RMSE"].idxmin() if len(cands) > 1 else summary["R2"].idxmax()
print(f"\n[INFO] Modelo recomendado automaticamente: {best_name}")

# ---------------------------
# Parse do nome (ex.: "Stepwise_WithLags")
# ---------------------------
try:
    model_base, scenario = best_name.rsplit("_", 1)
except Exception:
    model_base = best_name
    scenario = None

# ---------------------------
# 2) Recuperar/montar X e y para treinar modelo final
# ---------------------------
TARGET = globals().get("TARGET", "Diesel B (Volume em litros)")
LAGS = globals().get("LAGS", [1,2,3])  # se não existir, assume 1,2,3

# Preferir usar X_with_lags / X_without_lags se estiverem no ambiente
if scenario == "WithLags" and "X_with_lags" in globals():
    X_all_for_model = X_with_lags.copy()
elif scenario == "WithoutLags" and "X_without_lags" in globals():
    X_all_for_model = X_without_lags.copy()
else:
    # Fallback: reconstruir a partir de dados_consolidados (como fez a Seção 9d)
    if "dados_consolidados" not in globals():
        raise RuntimeError("Nem X_with_lags/X_without_lags nem dados_consolidados estão disponíveis para reconstruir X.")
    df_tmp = dados_consolidados.copy().reset_index(drop=True)
    # criar lags
    for lag in LAGS:
        df_tmp[f"{TARGET}_lag{lag}"] = df_tmp[TARGET].shift(lag)
    df_tmp = df_tmp.dropna().reset_index(drop=True)
    # manter apenas numéricos e remover TARGET
    X_tmp = df_tmp.select_dtypes(include=[np.number]).copy()
    if TARGET in X_tmp.columns:
        X_tmp = X_tmp.drop(columns=[TARGET])
    cols_with_lags = X_tmp.columns.tolist()
    cols_without_lags = [c for c in cols_with_lags if f"{TARGET}_lag" not in c]
    if scenario == "WithLags":
        X_all_for_model = X_tmp[cols_with_lags].astype(float)
    else:
        X_all_for_model = X_tmp[cols_without_lags].astype(float)
    # y recreation
    y_full_for_model = df_tmp[TARGET].astype(float).reset_index(drop=True)

# ensure y for training
if "y_full_for_model" not in locals():
    if "dados_consolidados" in globals():
        df_tmp2 = dados_consolidados.copy().reset_index(drop=True)
        for lag in LAGS:
            df_tmp2[f"{TARGET}_lag{lag}"] = df_tmp2[TARGET].shift(lag)
        df_tmp2 = df_tmp2.dropna().reset_index(drop=True)
        y_full_for_model = df_tmp2[TARGET].astype(float).reset_index(drop=True)
        if 'X_all_for_model' not in locals():
            X_all_for_model = df_tmp2.select_dtypes(include=[np.number]).copy()
            if TARGET in X_all_for_model.columns:
                X_all_for_model = X_all_for_model.drop(columns=[TARGET])
    else:
        raise RuntimeError("Não foi possível recuperar y para treinar o modelo final. Rode Seção 9d novamente.")

# ---------------------------
# 3) Split temporal: 80/20 (últimos 20% como teste)
# ---------------------------
n = len(X_all_for_model)
test_size = max(1, int(np.ceil(0.2 * n)))
train_size = n - test_size
X_train = X_all_for_model.iloc[:train_size].reset_index(drop=True)
X_test  = X_all_for_model.iloc[train_size:].reset_index(drop=True)
y_train = y_full_for_model.iloc[:train_size].reset_index(drop=True)
y_test  = y_full_for_model.iloc[train_size:].reset_index(drop=True)

print(f"\n[INFO] Treino final: {len(X_train)} | Teste final: {len(X_test)} | Cenário: {scenario}")

# ---------------------------
# 4) stepwise_aic (caso não exista) — corrigido p/ constante alinhada ao y
# ---------------------------
def _stepwise_aic_local(X, y, verbose=False, max_iter=200):
    included = []
    # constante com índice do y (evita erro de shape/índice)
    const_only = sm.add_constant(pd.DataFrame({"const": np.ones(len(y))}, index=y.index), has_constant='add')
    current_aic = sm.OLS(y, const_only).fit().aic
    it = 0
    while True and it < max_iter:
        it += 1
        changed = False
        excluded = [c for c in X.columns if c not in included]
        best_aic = current_aic
        best_feature = None
        # forward
        for new_col in excluded:
            cols = included + [new_col]
            try:
                model = sm.OLS(y, sm.add_constant(X[cols], has_constant='add')).fit()
                aic = model.aic
                if aic < best_aic - 1e-6:
                    best_aic = aic
                    best_feature = new_col
            except Exception:
                continue
        if best_feature is not None:
            included.append(best_feature)
            current_aic = best_aic
            changed = True
        # backward
        while True:
            if len(included) == 0:
                break
            worst_feature = None
            worst_aic = current_aic
            for col in included:
                trial = [c for c in included if c != col]
                try:
                    model = sm.OLS(y, sm.add_constant(X[trial], has_constant='add')).fit()
                    aic = model.aic
                    if aic < worst_aic - 1e-6:
                        worst_aic = aic
                        worst_feature = col
                except Exception:
                    continue
            if worst_feature is not None:
                included.remove(worst_feature)
                current_aic = worst_aic
                changed = True
            else:
                break
        if not changed:
            break
    return included if len(included) > 0 else list(X.columns)

# ---------------------------
# 5) Treinar o modelo final conforme model_base
# ---------------------------
def fit_final_model(model_base, X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(X_train)-1)))
    if "Ridge" in model_base:
        alphas = globals().get("RIDGE_ALPHAS", np.logspace(-6,6,25))
        gs = GridSearchCV(Ridge(), param_grid={"alpha": alphas}, cv=tscv,
                          scoring="neg_mean_squared_error", n_jobs=1)
        gs.fit(X_train.values, y_train.values.ravel())
        return gs.best_estimator_, {"alpha": gs.best_params_}
    elif "Lasso" in model_base:
        alphas = globals().get("LASSO_ALPHAS", np.logspace(-6,1,25))
        try:
            lcv = LassoCV(alphas=alphas, cv=tscv, max_iter=20000, n_jobs=1)
            lcv.fit(X_train.values, y_train.values.ravel())
            return lcv, {"alpha": getattr(lcv, "alpha_", None)}
        except Exception:
            mdl = Lasso(alpha=0.01, max_iter=20000).fit(X_train.values, y_train.values.ravel())
            return mdl, {"alpha": 0.01}
    elif "Stepwise" in model_base:
        sel = _stepwise_aic_local(X_train, y_train, verbose=False)
        if len(sel) == 0:
            sel = [X_train.columns[0]]
        mdl = LinearRegression().fit(X_train[sel].values, y_train.values.ravel())
        return (("stepwise", mdl, sel), {"selected": sel})
    elif "PCs" in model_base:
        n_comp = min(globals().get("N_PCS", 5), max(1, X_train.shape[1]-1))
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X_train.values)
        pca = PCA(n_components=n_comp)
        Xtr_p = pca.fit_transform(Xtr_s)
        alphas = globals().get("RIDGE_ALPHAS", np.logspace(-6,6,25))
        gs = GridSearchCV(Ridge(), param_grid={"alpha": alphas}, cv=tscv,
                          scoring="neg_mean_squared_error", n_jobs=1)
        gs.fit(Xtr_p, y_train.values.ravel())
        best = gs.best_estimator_
        wrapper = {"scaler": scaler, "pca": pca, "reg": best}
        return ("pca_wrapper", wrapper), {"alpha": gs.best_params_, "n_comp": n_comp}
    else:
        mdl = LinearRegression().fit(X_train.values, y_train.values.ravel())
        return mdl, {}

# ---------------------------
# 6) Fit final
# ---------------------------
try:
    trained, train_info = fit_final_model(model_base, X_train, y_train)
except Exception as e:
    raise RuntimeError(f"Falha ao treinar modelo final '{model_base}': {e}")

# ---------------------------
# 7) Predição uniforme (wrappers stepwise/pca)
# ---------------------------
def predict_from_trained(trained_obj, X):
    if isinstance(trained_obj, tuple):
        tag = trained_obj[0]
        if tag == "stepwise":
            _, mdl, sel = trained_obj
            X_sel = X.reindex(columns=sel, fill_value=0).values
            return mdl.predict(X_sel)
        elif tag == "pca_wrapper":
            _, wrapper = trained_obj
            scaler = wrapper["scaler"]
            pca = wrapper["pca"]
            reg = wrapper["reg"]
            Xs = scaler.transform(X.values)
            Xp = pca.transform(Xs)
            return reg.predict(Xp)
        else:
            raise RuntimeError("Wrapper desconhecido no modelo treinado.")
    else:
        return trained_obj.predict(X.values)

# ---------------------------
# 8) Métricas de teste
# ---------------------------
y_pred_test = predict_from_trained(trained, X_test)
rmse_final = mean_squared_error(y_test, y_pred_test, squared=False)
mae_final  = mean_absolute_error(y_test, y_pred_test)
r2_final   = r2_score(y_test, y_pred_test)
mape_final = _safe_mape(y_test, y_pred_test)

# ---------------------------
# 9) Salvar objetos globais
# ---------------------------
melhor_modelo = trained
best_model_results = {
    "nome": best_name,
    "model_base": model_base,
    "scenario": scenario,
    "modelo": trained,          # estimator, wrapper ou tuple
    "train_info": train_info,
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "y_pred": y_pred_test,
    "metrics": {"RMSE": rmse_final, "MAE": mae_final, "R2": r2_final, "MAPE_%": mape_final}
}
# compatibilidade
X_train_global, X_test_global = X_train, X_test
y_train_global, y_test_global = y_train, y_test

# ---------------------------
# 10) Impressão didática resumida
# ---------------------------
print("\n[INFO] Modelo final treinado e salvo em 'best_model_results'.")
print(f" Nome selecionado: {best_name}")
print(f" Tipo (base): {model_base} | Cenário: {scenario}")
print(f" Métricas no teste final:"
      f" RMSE={rmse_final:.3e} ({rmse_final/BILHAO:.3f} bi L),"
      f" MAE={mae_final:.3e} ({mae_final/BILHAO:.3f} bi L),"
      f" MAPE={mape_final:.2f}%, R2={r2_final:.4f}")
print("[INFO] Para interpretação: use best_model_results['modelo'] e X_train_global/y_train_global.")


# ---------------------------
# 11) Ranking e desempenho do modelo final (paleta azul
# ---------------------------


# Paleta azul padronizada

azul_escuro = "#1f77b4"
azul_medio  = "#2e86de"
azul_claro  = "#5dade2"
BILHAO = 1e9

# -------------------------------
# 1) RANKING TOP-3 — R² e RMSE
# -------------------------------
summary_10b = expanding_results_9d["summary"].copy()
if "Modelo" in summary_10b.columns:
    summary_10b = summary_10b.set_index("Modelo")

# Função auxiliar para calcular MAPE se necessário
def _safe_mape(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.where(yt == 0, np.nan, np.abs(yt))
    ape = np.abs(yt - yp) / denom
    return float(np.nanmean(ape) * 100)

# Garante que MAPE_% está presente
if "MAPE_%" not in summary_10b.columns:
    mape_map = {}
    for name in summary_10b.index:
        det = expanding_results_9d["details"].get(name)
        if det is not None and not det.empty:
            dv = det.dropna(subset=["y_pred"])
            mape_map[name] = _safe_mape(dv["y_true"].values, dv["y_pred"].values)
        else:
            mape_map[name] = np.nan
    summary_10b["MAPE_%"] = pd.Series(mape_map)

# Selecionar Top-3 modelos pelo R² desc / RMSE asc
rank = summary_10b.sort_values(["R2","RMSE"], ascending=[False, True]).head(3).copy()
labels = rank.index.tolist()
r2_vals = rank["R2"].values
rmse_bi = (rank["RMSE"] / BILHAO).values
mape_vals = rank["MAPE_%"].values

# Plot dos gráficos de barras
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# --- R² ---
axes[0].barh(labels, r2_vals, color=azul_escuro, alpha=0.95)
axes[0].set_title("Ranking Top-3 — R² (maior é melhor)")
axes[0].set_xlabel("R²")
axes[0].invert_yaxis()
for i, v in enumerate(r2_vals):
    axes[0].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

# --- RMSE ---
axes[1].barh(labels, rmse_bi, color=azul_claro, alpha=0.95)
axes[1].set_title("Ranking Top-3 — RMSE (bi L, menor é melhor)")
axes[1].set_xlabel("RMSE (bi L)")
axes[1].invert_yaxis()
for i, v in enumerate(rmse_bi):
    axes[1].text(v + max(rmse_bi)*0.02, i, f"{v:.3f}", va="center", fontsize=9)

plt.show()

# -------------------------------
# 2) GRÁFICO LINHA — Modelo Final
# -------------------------------
nome = best_model_results["nome"]
y_test = best_model_results["y_test"].reset_index(drop=True)
y_pred = pd.Series(best_model_results["y_pred"]).reset_index(drop=True)

# Métricas do modelo final
rmse = best_model_results["metrics"]["RMSE"]
mae = best_model_results["metrics"]["MAE"]
r2 = best_model_results["metrics"]["R2"]
mape = best_model_results["metrics"]["MAPE_%"]

plt.figure(figsize=(12, 4))
plt.plot(y_test.index, y_test.values / BILHAO, label="Real", color=azul_escuro, linewidth=2.0)
plt.plot(y_test.index, y_pred.values / BILHAO, label="Predito", color=azul_claro, linewidth=2.0, alpha=0.95)

plt.title(f"Modelo Selecionado: {nome} — Desempenho no Teste")
plt.xlabel("Observações (janela de teste)")
plt.ylabel("Volume Diesel B (bi L)")
plt.grid(True, linestyle="--", alpha=0.5)   # <<< grade no gráfico
plt.legend()

# Métricas no rodapé — texto preto
txt = f"R²={r2:.3f} | RMSE={rmse/BILHAO:.3f} bi L | MAE={mae/BILHAO:.3f} bi L | MAPE={mape:.2f}%"
plt.gcf().text(0.5, 0.02, txt, ha="center", va="bottom", fontsize=10, color="black")

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

#%% Seção 11a - Interpretação dos preditores (robusta, com diagnóstico de sinal e VIF)
print("\n" + "="*80)
print("SEÇÃO 11a - Interpretação dos preditores (robusta)")
print("="*80)

import warnings
warnings.filterwarnings("ignore")  # manter console limpo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Paleta
AZUL_ESCURO = "#1f77b4"
AZUL_CLARO  = "#5dade2"

def coef_betas(modelo, X, y):
    """Retorna DataFrame com coeficientes e betas padronizados."""
    coefs = pd.Series(modelo.coef_, index=X.columns, name="coef")
    sd_x  = X.std(ddof=0).replace(0, np.nan)
    sd_y  = float(np.std(y, ddof=0))
    betas = (coefs * (sd_x / sd_y)).rename("beta")
    out = pd.concat([coefs, betas], axis=1)
    out["abs_beta"] = out["beta"].abs()
    return out.sort_values("abs_beta", ascending=False)

def compute_vif(X):
    """VIF por variável (usa constante)."""
    Xc = sm.add_constant(X, has_constant='add')
    vifs = []
    for i, col in enumerate(Xc.columns):
        if col == "const": 
            continue
        vifs.append((col, variance_inflation_factor(Xc.values, i)))
    return pd.Series(dict(vifs)).rename("VIF")

try:
    # Recupera melhor modelo salvo na 10b
    raw_model   = best_model_results.get("modelo", None)
    nome_modelo = best_model_results.get("nome", "Modelo_desconhecido")
    X_train     = best_model_results.get("X_train", None)
    y_train     = best_model_results.get("y_train", None)

    if raw_model is None or X_train is None or y_train is None:
        raise ValueError("Modelo ou dados de treino não encontrados em best_model_results. Rode a Seção 10b primeiro.")

    # Se for wrapper stepwise, recuperar colunas selecionadas e refitar com DataFrame (com nomes)
    if isinstance(raw_model, tuple) and raw_model[0] == "stepwise":
        _, mdl_orig, selected = raw_model
        X_used = X_train[selected].copy()
        modelo = LinearRegression().fit(X_used, y_train)  # refit com nomes (elimina warnings)
        print(f"[INFO] Stepwise detectado. Variáveis selecionadas: {len(selected)}")
    else:
        modelo = raw_model
        X_used = X_train.copy()
        # se o modelo for LinearRegression treinado via arrays, refit com nomes:
        if isinstance(modelo, LinearRegression) and not isinstance(getattr(modelo, "coef_", None), type(None)):
            modelo = LinearRegression().fit(X_used, y_train)
        print(f"[INFO] Modelo detectado: {nome_modelo}")

    # ---------------------------
    # Betas padronizados (parciais)
    # ---------------------------
    if hasattr(modelo, "coef_"):
        imp_df = coef_betas(modelo, X_used, y_train)
        top = imp_df.head(12)
        plt.figure(figsize=(10, 6))
        plt.bar(top.index, top["beta"], color=AZUL_ESCURO, alpha=0.9)
        plt.axhline(0, color="k", linewidth=0.8)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Coeficiente padronizado (beta)")
        plt.title(f"Importância — {nome_modelo} (betas parciais)")
        plt.grid(axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
        plt.show()
        print("[INFO] Importância baseada em coeficientes padronizados (betas).")
    else:
        imp_df = None
        print("[WARN] Modelo não linear com .coef_. Pule para permutation importance.")

    # ---------------------------
    # Permutation importance
    # ---------------------------
    try:
        perm = permutation_importance(modelo, X_used, y_train, n_repeats=50, random_state=42, n_jobs=-1)
        perm_df = pd.Series(perm.importances_mean, index=X_used.columns).sort_values(ascending=False).rename("perm_mean")
        plt.figure(figsize=(10, 6))
        plt.bar(perm_df.head(12).index, perm_df.head(12).values, color=AZUL_CLARO, alpha=0.9)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Permutation importance (Δ erro)")
        plt.title(f"Importância (Permutation) — {nome_modelo}")
        plt.grid(axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
        plt.show()
        print("[INFO] Importância calculada com permutation_importance.")
    except Exception as e:
        perm_df = None
        print(f"[WARN] Falha no permutation importance: {e}")

    # ---------------------------
    # Diagnóstico de sinal e VIF
    # ---------------------------
    corr_simple = X_used.corrwith(y_train).rename("corr_y")
    try:
        vif = compute_vif(X_used)
    except Exception:
        vif = pd.Series(index=X_used.columns, dtype=float, name="VIF")

    resumo = None
    if imp_df is not None:
        resumo = imp_df.join([corr_simple, vif], how="left")
        # Marca sinal invertido
        resumo["sinal_invertido"] = np.sign(resumo["beta"]).ne(np.sign(resumo["corr_y"]))
        # Ordena por |beta|
        resumo = resumo.sort_values("abs_beta", ascending=False)

        print("\n--- Análise Didática Automática ---")
        for var, row in resumo.head(10).iterrows():
            beta = row["beta"]
            corr = row["corr_y"]
            vifv = row["VIF"]
            tag = " (sinal invertido: possível supressão/multicolinearidade)" if row["sinal_invertido"] else ""
            sentido = "AUMENTAR" if beta > 0 else "REDUZIR"
            print(f"- {var}: beta={beta:+.3f}, corr={corr:+.3f}, VIF={vifv:.2f}{tag} → efeito parcial tende a {sentido} o consumo.")

        # Tabela top-10
        fmt = lambda x: f"{x:,.3f}"
        print("\nTop 10 (coef / beta / corr / VIF):")
        print(resumo[["coef","beta","corr_y","VIF"]].head(10).to_string(float_format=fmt))

    else:
        # fallback usando permutation apenas
        if perm_df is not None:
            print("\n--- Análise Didática (Permutation) ---")
            for var, val in perm_df.head(10).items():
                print(f"- {var}: impacto médio em erro = {val:.3e}")

except Exception as e:
    print(f"[ERRO na Seção 11a] {e}")

#%% Seção 11b (ENXUTA) — Projeção Futura com o MELHOR MODELO (Stepwise + Lags) + gráficos
print("\n" + "="*80)
print("SEÇÃO 11b (ENXUTA) — Projeção Futura com o melhor modelo salvo (Stepwise + Lags)")
print("="*80)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ----------------------------
# Checagens e objetos de base
# ----------------------------
TARGET = "Diesel B (Volume em litros)"

if "dados_consolidados" not in globals():
    raise RuntimeError("Base 'dados_consolidados' não encontrada. Rode as seções anteriores.")

if "best_model_results" not in globals():
    raise RuntimeError("Objeto 'best_model_results' não encontrado. Rode a Seção 10b antes.")

# Modelo e metadados salvos pela 10b
nome_modelo = best_model_results.get("nome", "Modelo_desconhecido")
modelo_bruto = best_model_results.get("modelo", None)
X_train = best_model_results.get("X_train", None)
y_train = best_model_results.get("y_train", None)

if modelo_bruto is None or X_train is None or y_train is None:
    raise RuntimeError("Modelo/dados de treino ausentes em 'best_model_results'. Reexecute a 10b.")

print(f"[INFO] Melhor modelo (10b): {nome_modelo}")

# Extrair estimador e features usadas
if isinstance(modelo_bruto, tuple) and modelo_bruto[0] == "stepwise":
    # ("stepwise", LinearRegression(), selected_features)
    model = modelo_bruto[1]
    selected_features = list(modelo_bruto[2])
else:
    model = modelo_bruto
    selected_features = list(X_train.columns)

if "WithLags" not in nome_modelo and any(f"{TARGET}_lag" in c for c in selected_features) is False:
    print("[WARN] O modelo salvo não parece ser 'Stepwise_WithLags'. A projeção seguirá com o melhor modelo disponível.")

# ----------------------------
# Preparar dados e utilitários
# ----------------------------
df = dados_consolidados.copy()
if "Data" not in df.columns:
    raise KeyError("Coluna 'Data' não encontrada em 'dados_consolidados'.")

df["Data"] = pd.to_datetime(df["Data"])
df = df.sort_values("Data").reset_index(drop=True)
df["Ano"] = df["Data"].dt.year
df["Mês"] = df["Data"].dt.month
df["Semestre"] = np.where(df["Mês"] <= 6, 1, 2)

last_date = df["Data"].max()
last_year, last_month = last_date.year, last_date.month
print(f"[INFO] Último dado disponível: {last_month:02d}/{last_year}")

# Formatador para eixos (bi litros)
def yfmt_billions(y, _pos=None):
    return f"{y/1e9:.1f}"

# Média do mesmo mês dos últimos até 3 anos (fallback: média histórica do mês)
def moy_avg_3y(col, mes, ano_ref):
    vals = []
    for a in [ano_ref-1, ano_ref-2, ano_ref-3]:
        v = df.loc[(df["Ano"]==a) & (df["Mês"]==mes), col]
        if len(v) and pd.notna(v.values[0]):
            vals.append(float(v.values[0]))
    if vals:
        return float(np.mean(vals))
    v = df.loc[df["Mês"]==mes, col].mean()
    return float(v) if pd.notna(v) else float(df[col].mean())

# Monta a linha X (na ordem das features do modelo) para (ano, mes)
def construir_X_future_row(ano, mes, y_hist_e_pred):
    row = {}
    semestre = 1 if mes <= 6 else 2
    for col in selected_features:
        if col == "Semestre":
            row[col] = semestre
        elif col.startswith(TARGET + "_lag"):
            # extrai k de "..._lagK"
            try:
                k = int(col.split("_lag")[-1])
            except Exception:
                k = 1
            # usa os valores historico+predito para preencher lag
            if len(y_hist_e_pred) >= k:
                row[col] = float(y_hist_e_pred[-k])
            else:
                row[col] = float(y_hist_e_pred[-1])
        else:
            if col in df.columns:
                row[col] = moy_avg_3y(col, mes, ano)
            else:
                row[col] = 0.0
    # DataFrame com colunas na MESMA ordem das features do modelo
    return pd.DataFrame([row], columns=selected_features)

# ----------------------------------------
# 1) Projetar meses faltantes do ano atual
# ----------------------------------------
faltantes_ano_atual = [(last_year, m) for m in range(last_month+1, 12+1)]
proje_atual = []

# histórico do TARGET (para alimentar lags durante a simulação)
y_hist = list(df[TARGET].astype(float).values)

for (yy, mm) in faltantes_ano_atual:
    X_future = construir_X_future_row(yy, mm, y_hist)
    # predição — usa .values para evitar erro quando modelo foi treinado sem nomes
    y_hat = float(model.predict(X_future.values)[0])
    proje_atual.append({"Ano": yy, "Mês": mm, "Predito": y_hat})
    y_hist.append(y_hat)  # append para alimentar lags

proje_atual = pd.DataFrame(proje_atual) if len(proje_atual) else pd.DataFrame(columns=["Ano","Mês","Predito"])

# ------------------------------
# 2) Projetar TODO o próximo ano
# ------------------------------
ano_futuro = last_year + 1
proje_next = []
for mm in range(1, 12+1):
    X_future = construir_X_future_row(ano_futuro, mm, y_hist)
    y_hat = float(model.predict(X_future.values)[0])
    proje_next.append({"Ano": ano_futuro, "Mês": mm, "Predito": y_hat})
    y_hist.append(y_hat)

proje_next = pd.DataFrame(proje_next)

# --------------------------
# 3) Consolidação e Tabelas
# --------------------------
proje_total = pd.concat([proje_atual, proje_next], ignore_index=True)

saida = proje_total.copy()
saida["Predito (bi litros)"] = (saida["Predito"]/1e9).round(2)
saida["Predito (mi litros)"] = (saida["Predito"]/1e6).round(1)

print("\n--- Projeção Futura (bilhões de litros) ---")
print(saida[["Ano","Mês","Predito (bi litros)","Predito (mi litros)"]])

# Resumo anual (histórico + projetado)
hist_annual = (df.groupby("Ano")[TARGET].sum()/1e9).round(2)
fut_annual  = (proje_total.groupby("Ano")["Predito"].sum()/1e9).round(2)
annual = pd.concat([hist_annual, fut_annual]).sort_index()
print("\n--- Resumo Anual (bilhões de litros) ---")
print(annual)

# Pequeno resumo automático vs último ano COMPLETO
anos_idx = annual.index.tolist()
if ano_futuro in anos_idx:
    # pega o último ano completo (12 meses) como base
    anos_comp = [a for a in sorted(df["Ano"].unique()) if (df["Ano"]==a).sum()==12]
    base_year = anos_comp[-1] if anos_comp else df["Ano"].min()
    v_base = float(hist_annual.get(base_year, np.nan))
    v_fut = float(annual.loc[ano_futuro])
    if not np.isnan(v_base) and v_base != 0:
        var = (v_fut - v_base)/v_base*100
        sentido = "aumentar" if var>0 else "reduzir"
        print(f"\n[RESUMO] {ano_futuro}: {v_fut:.2f} bi L — deve {sentido} {abs(var):.2f}% vs {base_year} ({v_base:.2f} bi L).")

# =======================================
# 4) GRÁFICOS (3 visões complementares)
# =======================================

# A) Barras anuais (histórico + projetado)
def plot_barras_anuais(df_hist, df_proj):
    hist_tot = df_hist.groupby("Ano")[TARGET].sum()          # litros
    proj_tot = df_proj.groupby("Ano")["Predito"].sum()       # litros

    anos_all = sorted(set(hist_tot.index.tolist()) | set(proj_tot.index.tolist()))
    valores = []
    cores = []
    hatches = []
    for a in anos_all:
        if a in proj_tot.index and a not in hist_tot.index:
            valores.append(proj_tot.loc[a]); cores.append("#6c757d"); hatches.append("//")
        elif a in proj_tot.index and (df_hist["Ano"]==a).sum() < 12:
            # ano corrente incompleto (real+projeção dos meses faltantes)
            valores.append(hist_tot.get(a, 0.0) + proj_tot.get(a, 0.0)); cores.append("#6c757d"); hatches.append("//")
        else:
            valores.append(hist_tot.get(a, np.nan)); cores.append("#1f77b4"); hatches.append(None)

    fig, ax = plt.subplots(figsize=(12,6))
    x = np.arange(len(anos_all))
    bars = ax.bar(x, valores, color=cores, edgecolor="black")
    for bar, ht in zip(bars, hatches):
        if ht: bar.set_hatch(ht)

    ax.set_xticks(x); ax.set_xticklabels(anos_all)
    ax.yaxis.set_major_formatter(FuncFormatter(yfmt_billions))
    ax.set_ylabel("Volume (bilhões de litros)")
    ax.set_title("Consumo de Diesel B por Ano (histórico + projetado)")

    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.text(b.get_x()+b.get_width()/2, h*1.005, f"{h/1e9:.2f}", ha="center", va="bottom", fontsize=9)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#1f77b4", edgecolor="black", label="Histórico"),
        Patch(facecolor="#6c757d", edgecolor="black", hatch="//", label="Projetado")
    ], loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

print("\n1) Barras anuais")
plot_barras_anuais(df_hist=df, df_proj=proje_total)

# B) Últimos 5 anos + projeção (faixa min–máx + média)
def plot_ultimos5_com_projecao(df_hist, df_next, ano_next):
    dados = df_hist.copy()
    dados["Ano"] = dados["Data"].dt.year
    dados["Mês"] = dados["Data"].dt.month

    anos_disponiveis = sorted(dados["Ano"].unique())
    ult5 = anos_disponiveis[-5:] if len(anos_disponiveis) >= 5 else anos_disponiveis

    pivot = dados[dados["Ano"].isin(ult5)].pivot_table(index="Mês", columns="Ano", values=TARGET, aggfunc="sum")
    min_m, max_m, mean_m = pivot.min(axis=1), pivot.max(axis=1), pivot.mean(axis=1)

    proj_next = df_next[df_next["Ano"] == ano_next].set_index("Mês")["Predito"].reindex(range(1, 13))

    meses = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.fill_between(meses, min_m.reindex(meses), max_m.reindex(meses), color="lightgray", alpha=0.35, label="Faixa min–máx (5 anos)")
    ax.plot(meses, mean_m.reindex(meses), color="black", linestyle="--", linewidth=1.5, label="Média (5 anos)")

    # Anos históricos (cores discretas)
    cores = ['#1f77b4','#2ca02c','#9467bd','#d62728']
    for i, ano in enumerate(ult5[:-1]):
        if ano in pivot.columns:
            ax.plot(meses, pivot[ano].reindex(meses), linestyle="--", linewidth=1.5, color=cores[i % len(cores)], alpha=0.75, label=str(ano))

    ano_recente = ult5[-1]
    if ano_recente in pivot.columns:
        ax.plot(meses, pivot[ano_recente].reindex(meses), color="#1f77b4", linewidth=2.5, label=str(ano_recente))

    ax.plot(meses, proj_next, color="#6c757d", linestyle="--", linewidth=2.5, label=str(ano_next))

    ax.set_xticks(meses); ax.set_xticklabels(['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'])
    ax.yaxis.set_major_formatter(FuncFormatter(yfmt_billions))
    ax.set_xlabel("Mês"); ax.set_ylabel("Volume (bilhões de litros)")
    ax.set_title("Consumo de Diesel B – Últimos 5 anos + Projeção")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=6, fontsize=10, frameon=False)
    plt.tight_layout()
    plt.show()

print("2) Últimos 5 anos + projeção")
plot_ultimos5_com_projecao(df_hist=df, df_next=proje_next if not proje_next.empty else proje_total, ano_next=ano_futuro)

# C) Série contínua: real (azul) vs projeção (cinza tracejada)
proje_total_plot = proje_total.copy()
proje_total_plot["Data"] = pd.to_datetime(proje_total_plot["Ano"].astype(str) + "-" + proje_total_plot["Mês"].astype(str) + "-01")
proje_total_plot = proje_total_plot.sort_values(["Ano","Mês"])

plt.figure(figsize=(12,6))
plt.plot(df["Data"], df[TARGET]/1e9, color="#1f77b4", label="Real (bi L)")
plt.plot(proje_total_plot["Data"], proje_total_plot["Predito"]/1e9, linestyle="--", color="#6c757d", label="Projeção (bi L)")
plt.ylabel("Consumo de Diesel B (bilhões de litros)")
plt.title(f"Projeção futura — {nome_modelo}")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

#%% FIM DO PROJETO