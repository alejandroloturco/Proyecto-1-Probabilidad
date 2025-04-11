import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import seaborn as sns
#Se crean varas funciones para el tratamiento de datos

def convertir_NAN_MEAN(nombres,df):
    """En caso de que la columna no tome los valores como int o float los convierte a valores numericos, luego convierte los valores NaN de una columna a su media."""
    for i in nombres:
        df[i] = pd.to_numeric(df[i], errors='coerce')  # Convertir a numérico, forzando NaN donde no se puede
        df[i].fillna(np.mean(df[i]), inplace=True) #Reemplaza los Nan con la media 

def informacion(dat,categorico=False):
    """Con el ingreso de una serie de datos, devuelve una lista con la media, mediana, moda, SD, MAD, varianza, IQR, CV y CVM."""
    lista = []    
    if categorico == False:
        # Intentamos tratar los datos como numéricos
        dat = pd.to_numeric(dat)
        lista.append(np.mean(dat))  # media
        lista.append(np.median(dat))  # mediana
        moda = dat.mode().iloc[0] if not dat.mode().empty else np.nan #Si no es categorico calcula la moda
        lista.append(moda)
        lista.append(np.std(dat))  # SD
        lista.append(stats.median_abs_deviation(dat))  # MAD
        lista.append(np.var(dat))  # Varianza
        Q1 = np.percentile(dat, 25)
        Q3 = np.percentile(dat, 75)
        lista.append(Q3 - Q1)  # IQR
        lista.append(dat.std(ddof=1) / dat.mean() * 100)  # CV
        lista.append((stats.median_abs_deviation(dat)/ np.median(dat)) if np.median(dat) != 0 else np.nan)  # CVM
    elif categorico == True:
        lista = [np.nan] * 9  
        lista[2] = dat.mode().iloc[0] if not dat.mode().empty else np.nan  
    
    return lista

def atipicos (valor):  
    """""""Con el ingreso de una serie de datos, devuelve una lista con los datos atipicos y la cantidad de estos."""  
    atipico = []
    valor = pd.to_numeric(valor)
    Q1 = np.nanpercentile(valor,25)    
    Q3 =np.nanpercentile(valor,75)
    iqr = Q3-Q1

    for x in valor:
        if x < (Q1 - 1.5*iqr) :
            atipico.append(x)
        elif x > (Q3 + 1.5*iqr):
            atipico.append(x)

    cont = len(atipico)
    return  cont,atipico

def mostrar_atipicos(lista : list, df):
    """Recibe una lista de variables y un dataframe, y muestra la cantidad de datos atipicos y los datos atipicos para cada variable."""
    for i in range(len(lista)):
        cont,datos = atipicos(df[lista[i]])
        print(f"{lista[i]} tiene {cont} datos atipicos")
        print("Los datos atipicos son:")
        print(datos)

def comparacion(data: str, dat1: dict, dat2: dict):
        """"Recibe dos diccionarios con los resultados de la poblacion y la muestra, y devuelve una lista con el porcentaje de diferencia entre ambos."""
        lista_p = []
        for x in range(9): 
            dif = (100 * np.abs(dat2[data][x]-dat1[data][x]))/dat1[data][x]         
            lista_p.append(dif)
        return lista_p

#Se crean listas de variables a usar como llaves en los multiples dataframes que se crean a partir del csv

#Diccionario para almacenar los indices de las variables, que corresponderan a las columnas del csv para almacenarlos de forma adecuada
aqui = {'aveOralM':115,
'Gender':116, 
'Age':117, 
'Ethnicity':118,
'T_atm':119 , 
'Humidity':120,
'Cosmetics':122, 
'Max1R13_1':3,
'Max1R13_2':31,
'Max1R13_3':59,
'Max1R13_4':87}


#Listas de variables categoricas y numericas, corresponden a las llaves a usar en las funciones de analisis
variables_numericas = [
    'aveOralM',  
    'T_atm', 
    'Humidity',
    'PromMax1R13'
]

variables_categoricas = ['Gender',
'Age',
'Ethnicity',
'Cosmetics']

#lectura de la base de datos y conversion a dataframe
csv = pd.read_csv(r'FLIR_groups1and2.csv', delimiter=';')
csv = csv.iloc[2:].reset_index(drop=True)
 
#establecemos el tipo de datos de las columnas a usar, en este caso son todas categoricas 
csv_filter = csv.iloc[:,[aqui['aveOralM'], aqui['Gender'], aqui['Age'], aqui['Ethnicity'], aqui['T_atm'], aqui['Humidity'], aqui['Cosmetics']]]
csv_filter.columns = ["aveOralM", "Gender", "Age", "Ethnicity", "T_atm", "Humidity", "Cosmetics"]

#crea un dataframe temporal para realizar el promedio de las columnas Max1R13, y luego lo agrega al dataframe principal
data_temp = csv.iloc[:, [aqui['Max1R13_1'], aqui['Max1R13_2'], aqui['Max1R13_3'], aqui['Max1R13_4']]].astype(float)
data_temp.columns = ["Max1R13_1","Max1R13_2","Max1R13_3","Max1R13_4"]
convertir_NAN_MEAN(data_temp.columns,data_temp) 

csv_filter['PromMax1R13'] = data_temp.mean(axis=1)

#convierte los np.nan de la columna de cosmetics en string 'Nan' para evitar problemas al graficar
csv_filter["Cosmetics"] = csv_filter["Cosmetics"].fillna('Nan')

#se encarga de convertir los valores de las columnas numericas a floats o ints para evitar problemas al graficar
for i in variables_numericas:
    csv_filter[i] = pd.to_numeric(csv_filter[i],errors='coerce')

estrato = "Ethnicity"

# Muestreo estratificado para n=500
muestra_estratificada, _ = train_test_split(
    csv_filter,
    stratify=csv_filter[estrato],
    test_size=(len(csv_filter) - 500) / len(csv_filter),
    random_state=42
)

# Graficar la comparación entre población y muestra
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

sns.countplot(x=csv_filter[estrato], order=csv_filter[estrato].value_counts().index, palette="pastel", ax=axes[0])
axes[0].set_title("Distribución en la Población")
axes[0].set_xlabel("Ethnicity")
axes[0].set_ylabel("Frecuencia")
axes[0].tick_params(axis='x', rotation=45)

sns.countplot(x=muestra_estratificada[estrato], order=csv_filter[estrato].value_counts().index, palette="pastel", ax=axes[1])
axes[1].set_title("Distribución en la Muestra Estratificada")
axes[1].set_xlabel("Ethnicity")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()