#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats

def convertir_NAN_MEAN(nombres,df):
    """Convierte los valores NaN de una columna a su media."""
    for i in nombres:
        df[i] = pd.to_numeric(df[i], errors='coerce')  # Convertir a numérico, forzando NaN donde no se puede
        df[i].fillna(np.mean(df[i]), inplace=True)

def informacion(dat,categorico=False):
    lista = []    
    if categorico == False:
        # Intentamos tratar los datos como numéricos
        dat = pd.to_numeric(dat)
        lista.append(np.mean(dat))  # media
        lista.append(np.median(dat))  # mediana
        moda = dat.mode().iloc[0] if not dat.mode().empty else np.nan
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
        lista = [np.nan] * 9  # media, mediana, moda, SD, MAD, varianza, IQR, CV, CVM
        lista[2] = dat.mode().iloc[0] if not dat.mode().empty else np.nan  # moda
    
    return lista

def atipicos (valor):    
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
    for i in range(len(lista)):
        cont,datos = atipicos(df[lista[i]])
        print(f"{lista[i]} tiene {cont} datos atipicos")
        print("Los datos atipicos son:")
        print(datos)

#%%
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

csv = pd.read_csv(r'FLIR_groups1and2.csv', delimiter=';')
csv = csv.iloc[2:].reset_index(drop=True)
 
csv_filter = csv.iloc[:,[aqui['aveOralM'], aqui['Gender'], aqui['Age'], aqui['Ethnicity'], aqui['T_atm'], aqui['Humidity'], aqui['Cosmetics']]]
csv_filter.columns = ["aveOralM", "Gender", "Age", "Ethnicity", "T_atm", "Humidity", "Cosmetics"]

data_temp = csv.iloc[:, [aqui['Max1R13_1'], aqui['Max1R13_2'], aqui['Max1R13_3'], aqui['Max1R13_4']]].astype(float)
data_temp.columns = ["Max1R13_1","Max1R13_2","Max1R13_3","Max1R13_4"]
convertir_NAN_MEAN(data_temp.columns,data_temp) 

csv_filter['PromMax1R13'] = data_temp.mean(axis=1)

csv_filter["Cosmetics"] = csv_filter["Cosmetics"].fillna('Nan')
for i in variables_numericas:
    csv_filter[i] = pd.to_numeric(csv_filter[i],errors='coerce')
csv_filter.to_csv('FLIR_groups1and2_replace.csv',  sep=';', decimal=',',index=False)

#%%

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

df = pd.DataFrame(resultados)
#%%
#Datos atipicos

mostrar_atipicos(variables_numericas, csv_filter)

#%%
# Histograma para cada variable numérica
plt.figure(figsize=(12, 8))
for i, col in enumerate(variables_numericas):
    plt.subplot(2, 2, i + 1)
    sns.histplot(csv_filter[col], kde=True, bins=30, color="skyblue")
    plt.title(f"Histograma de {col}")
plt.tight_layout()
plt.show()

#%%
# --- Gráficos de caja (boxplots) ---
plt.figure(figsize=(12, 8))
for i, col in enumerate(variables_numericas):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x=csv_filter[col], color="lightgreen")
    plt.title(f"Caja y Bigotes de {col}")
plt.tight_layout()
plt.show()
#%%

# --- Gráficos de barras para variables categóricas ---
plt.figure(figsize=(12, 10))
for i, col in enumerate(variables_categoricas):
    plt.subplot(2, 2, i + 1)
    sns.countplot(x=csv_filter[col], hue=csv_filter[col], palette="pastel", legend=False, order=csv_filter[col].value_counts().index)
    plt.title(f"Distribución de {col}")
    plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
#%%
# --- Matriz de correlación para variables numéricas ---
corr_matrix = csv_filter[variables_numericas].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Matriz de Correlación entre Variables Numéricas")
plt.show()
# %%
muestra_simple = csv_filter.sample(n=500, random_state=42)
resultados2 = {'': ["Media", "Mediana", "Moda", "SD", "MAD", "Varianza", "IQR", "CV", "CVM"],    
    'aveOralM': informacion(muestra_simple['aveOralM']),
    'Gender': informacion(muestra_simple['Gender'], categorico=True),
    'Age': informacion(muestra_simple['Age'], categorico=True),
    'Ethnicity': informacion(muestra_simple['Ethnicity'], categorico=True), 
    'T_atm': informacion(muestra_simple['T_atm']),
    'Humidity': informacion(muestra_simple['Humidity']),
    'Cosmetics': informacion(muestra_simple['Cosmetics'], categorico=True),
    'PromMax1R13': informacion(muestra_simple['PromMax1R13'])
}

df_muestra = pd.DataFrame(resultados2)
# %%
def comparacion(data, dat1, dat2):
        lista_p = []
        for x in range(9): 
            dif = (100 * np.abs(dat2[data][x]-dat1[data][x]))/dat1[data][x]         
            lista_p.append(dif)
        return lista_p

print(comparacion('aveOralM',resultados,resultados2))
resultados3 = {'': ["Media", "Mediana", "Moda", "SD", "MAD", "Varianza", "IQR", "CV", "CVM"],    
    'aveOralM': comparacion('aveOralM',resultados,resultados2),
    #'Gender': comparacion('Gender',resultados,resultados2),
    #'Age': comparacion('Age', resultados,resultados2),
    #'Ethnicity': comparacion('Ethnicity',resultados,resultados2), 
    'T_atm': comparacion('T_atm',resultados,resultados2),
    'Humidity': comparacion('Humidity',resultados,resultados2),
    #'Cosmetics': comparacion('Cosmetics', resultados,resultados2),
    'PromMax1R13': comparacion('PromMax1R13',resultados,resultados2)
}

df_preuba = pd.DataFrame(resultados3)
# %%
