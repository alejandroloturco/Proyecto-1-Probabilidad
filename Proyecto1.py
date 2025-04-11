#%%
#Librerias importadas
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

#%%
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

#se crea un csv con los datos filtrados y convertidos a float o int para visualizar
csv_filter.to_csv('FLIR_groups1and2_replace.csv',  sep=';', decimal=',',index=False)

#%%
#PUNTO 1
#se crea diccionario con los valores que corresponderan al dataframe pedido
resultados = {'': ["Media", "Mediana", "Moda", "SD", "MAD", "Varianza", "IQR", "CV", "CVM"],    
    'aveOralM': informacion(csv_filter['aveOralM']),
    'Gender': informacion(csv_filter['Gender'], categorico=True),
    'Age': informacion(csv_filter['Age'], categorico=True),
    'Ethnicity': informacion(csv_filter['Ethnicity'], categorico=True), 
    'T_atm': informacion(csv_filter['T_atm']),
    'Humidity': informacion(csv_filter['Humidity']),
    'Cosmetics': informacion(csv_filter['Cosmetics'], categorico=True),
    'PromMax1R13': informacion(csv_filter['PromMax1R13'])
}
#se crea datframe a partir del diccionario creado anteriormente
df = pd.DataFrame(resultados)
#%%
#PUNTO 2
#Datos atipicos pedidos
mostrar_atipicos(variables_numericas, csv_filter)

#%%
#PUNTO 3
#Histograma para variables numericas 
plt.figure(figsize=(12, 8))
for i, col in enumerate(variables_numericas):
    plt.subplot(2, 2, i + 1)    
    sns.histplot(csv_filter[col], bins=30, color="#F7C6D9", stat="density")
    sns.kdeplot(csv_filter[col], color="purple", linewidth=2)
    plt.title(f"{col}")
    plt.ylabel("Densidad")
plt.tight_layout()
plt.show()

#%%
#PUNTO 4
#Gráficos de caja (boxplots)
plt.figure(figsize=(12, 8))
for i, col in enumerate(variables_numericas):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x=csv_filter[col], color="#25beee")
    plt.title(f"{col}")
plt.tight_layout()
plt.show()

#%%
#PUNTO 5
#Histograma para variables categoricas
plt.figure(figsize=(12, 10))
for i, col in enumerate(variables_categoricas):
    plt.subplot(2, 2, i + 1)
    sns.countplot(x=csv_filter[col], hue=csv_filter[col], palette=["#F7C6D9", "#DDA0DD", "#E6B0AA", "#C39BD3"], legend=False, order=csv_filter[col].value_counts().index)
    plt.title(f"Distribución de {col}")
    plt.xticks(rotation=90)
    plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

#%%
#PUNTO 6
# Matriz de correlación
matrix = csv_filter[variables_numericas].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    matrix,
    annot=True,
    fmt=".2f",
    cmap='magma',      
    center=0,
    linewidths=0.5,
    linecolor='white',
    square=True,
    cbar_kws={"shrink": .75})
plt.title("Matriz de Correlación", fontsize=14, weight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
# %%
#Muestreo aleatorio simple
simple = csv_filter.sample(n=500, random_state=42) #42 se asegura que el experimento sea replicable

resultados2 = {'': ["Media", "Mediana", "Moda", "SD", "MAD", "Varianza", "IQR", "CV", "CVM"],    
    'aveOralM': informacion(simple['aveOralM']),
    'Gender': informacion(simple['Gender'], categorico=True),
    'Age': informacion(simple['Age'], categorico=True),
    'Ethnicity': informacion(simple['Ethnicity'], categorico=True), 
    'T_atm': informacion(simple['T_atm']),
    'Humidity': informacion(simple['Humidity']),
    'Cosmetics': informacion(simple['Cosmetics'], categorico=True),
    'PromMax1R13': informacion(simple['PromMax1R13'])
}

df_muestra_500 = pd.DataFrame(resultados2)
# %%
#Función que compara muestra vs población

#Crea un dataframe con los resultados de la comparación entre la población y la muestra, mostrando el porcentaje de diferencia entre ambos
resultados3 = {'': ["Media", "Mediana", "Moda", "SD", "MAD", "Varianza", "IQR", "CV", "CVM"],    
    'aveOralM': comparacion('aveOralM',resultados,resultados2),     
    'T_atm': comparacion('T_atm',resultados,resultados2),
    'Humidity': comparacion('Humidity',resultados,resultados2),    
    'PromMax1R13': comparacion('PromMax1R13',resultados,resultados2)}

df_porcentajes = pd.DataFrame(resultados3)
print("Porcentaje de diferencia entre poblacion y la muestra de 500 elementos\n")
print(df_porcentajes)

# %%
#ESTRATIFICACION
variable = 'Ethnicity'
# Muestreo estratificado para n=500
mu_estra, _ = train_test_split(
    csv_filter,
    stratify=csv_filter[variable],
    test_size=(len(csv_filter) - 500) / len(csv_filter),
    random_state=42)


fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

sns.countplot(x=csv_filter[variable], order=csv_filter[variable].value_counts().index, palette=["#F7C6D9", "#DDA0DD", "#E6B0AA", "#C39BD3"], ax=axes[0])
axes[0].set_title("Distribución en la Población")
axes[0].set_xlabel("Ethnicity")
axes[0].set_ylabel("Frecuencia")
axes[0].tick_params(axis='x', rotation=45)

sns.countplot(x=mu_estra[variable], order=csv_filter[variable].value_counts().index, palette=["#F7C6D9", "#DDA0DD", "#E6B0AA", "#C39BD3"], ax=axes[1])
axes[1].set_title("Distribución en la Muestra Estratificada")
axes[1].set_xlabel("Ethnicity")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('estratificacion.png', dpi=300, bbox_inches='tight') # Guarda la figura como PNG con alta calidad
plt.show()

# %%
#Metricas para datos estratificados
resultados4 = {'': ["Media", "Mediana", "Moda", "SD", "MAD", "Varianza", "IQR", "CV", "CVM"],    
    'aveOralM': informacion(mu_estra['aveOralM']),
    'Gender': informacion(mu_estra['Gender'], categorico=True),
    'Age': informacion(mu_estra['Age'], categorico=True),
    'Ethnicity': informacion(mu_estra['Ethnicity'], categorico=True), 
    'T_atm': informacion(mu_estra['T_atm']),
    'Humidity': informacion(mu_estra['Humidity']),
    'Cosmetics': informacion(mu_estra['Cosmetics'], categorico=True),
    'PromMax1R13': informacion(mu_estra['PromMax1R13'])}

df_estratificado = pd.DataFrame(resultados4)

resultados5 = {'': ["Media", "Mediana", "Moda", "SD", "MAD", "Varianza", "IQR", "CV", "CVM"],    
    'aveOralM': comparacion('aveOralM',resultados,resultados4),     
    'T_atm': comparacion('T_atm',resultados,resultados4),
    'Humidity': comparacion('Humidity',resultados,resultados4),    
    'PromMax1R13': comparacion('PromMax1R13',resultados,resultados4)}

df_porcentajes_estra = pd.DataFrame(resultados5)
print("Porcentaje de diferencia entre poblacion y la muestra de 500 elementos estratificado\n")
print(df_porcentajes_estra)
# %%
