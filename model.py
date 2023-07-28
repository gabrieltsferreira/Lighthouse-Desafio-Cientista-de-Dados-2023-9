import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data_model_fix = pd.read_csv('cars_train - fixed.csv')

# Retira colunas desconsideradas durante o treinamento
drop_cols = ['veiculo_alienado','elegivel_revisao','id','num_fotos','num_portas']
data_model_fix = data_model_fix.drop(drop_cols, axis=1)


# Trata variáveis booleanas
bool_cols = ['dono_aceita_troca','veiculo_único_dono','revisoes_concessionaria','ipva_pago',
'veiculo_licenciado','garantia_de_fábrica','revisoes_dentro_agenda']

data_model_fix[bool_cols] = data_model_fix[bool_cols].notnull().astype('int')

# Substituindo valor lógico para valor numérico
data_model_fix[['entrega_delivery', 'troca']] = data_model_fix[['entrega_delivery', 'troca']].astype('int')

# Substitui valores "N" por "0"
data_model_fix['blindado'] = data_model_fix['blindado'].replace('N', 0)

# Substitui valores "S" por "1"
data_model_fix['blindado'] = data_model_fix['blindado'].replace('S', 1)

colunas_categoricas = ['marca','modelo','versao','cambio','tipo','cor','tipo_vendedor','cidade_vendedor','estado_vendedor','anunciante']

# Salva valores de preco para dropar coluna antes da codificação
y = data_model_fix['preco']
data_model_fix = data_model_fix.drop('preco', axis=1)

one_hot_enc = make_column_transformer(
    (OneHotEncoder(handle_unknown = 'ignore'),
    colunas_categoricas),
    remainder='passthrough')

data_model_fix = one_hot_enc.fit_transform(data_model_fix).toarray()


feature_labels = one_hot_enc.get_feature_names_out()
data_model_fix = pd.DataFrame(data_model_fix, columns=feature_labels)
data_model_fix

X = data_model_fix

X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_model.score(X_test, y_test)