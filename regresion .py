#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 


# In[2]:


import matplotlib.pyplot as plt
import math 

# Creamos una funcion logistica vectorial (ufuncs)
logistica = np.frompyfunc(lambda b0, b1, x:
                         1 /import numpy as np  (1 + math.exp (- (b0 + b1 * x))),
                         3, 1)
#Graficamos la funcion logistica con diferentes pendientes
plt.figure(figsize=(8, 4))
#grafica de dispersion para el eje x
plt.scatter(np.arange(-5, 5, 0.1),
           logistica(0, 1, np.arange(-5, 5, 0.1)),
          color='green')
plt.scatter(np.arange(-5, 5, 0.1),
           logistica(0, 2, np.arange(-5, 5, 0.1)),
          color='gold')
plt.scatter(np.arange(-5, 5, 0.1),
           logistica(0, 3, np.arange(-5, 5, 0.1)),
          color='red')
plt.title("Funcion Logistica Estandar - Diferentes 'Pendientes'", fontsize=14.0)
plt.ylabel("Probabilidad", fontsize=13.0)
plt.xlabel("Valores", fontsize=13.0)
plt.show()


# In[5]:


#Taquicardia Probabilidad y clase 
#Persona normal de 60 a 100 latidos porminuto.
#Persona con taquicardia de hasta 220 latidos por minuto.
personas_normal = [65, 70, 80, 80, 80,
                  90, 95, 100, 105, 110]
personas_taquicardia = [105, 110, 110, 120, 120,
                       130, 140, 180, 185, 190]
#graficamos una funcion logistica 
plt.figure(figsize=(6, 4))

# y = b0 + b1x
#
# y = -46.68057196 + 0.42460226x
plt.scatter(np.arange(60, 200, 0.1),
           logistica(-46.68057196, 0.42460226,
                    np.arange(60, 200, 0.1)))
#Graficamos la frecuencia cardíaca de las personas
plt.scatter(personas_normal, [0]*10,
           marker="o", c="green", s=250, label="Normal")
plt.scatter(personas_taquicardia, [1]*10,
           marker="o", c="red", s=250, label="Taquicardia")
#Graficamos las probabilidades para tres (3) individuos
individuos = [80, 110, 180]
probabilidades = logistica (-46.68057196, 0.42460226, individuos)
plt.scatter(individuos, probabilidades, 
           marker="*", c="darkorange", s=500)
plt.text(individuos[0]+7, 0.05, "%0.2f" % probabilidades [0],
        size=12, color="black")
plt.text(individuos[1]+7, 0.48, "%0.2f" % probabilidades [1],
        size=12, color="black")
plt.text(individuos[2]+7, 0.90, "%0.2f" % probabilidades [2],
        size=12, color="black")

plt.text(0, 1, "TAQUICARDIA", size=12, color="red")
plt.text(0, 0, "NORMAL", size=12, color="red")
plt.ylabel("Probabilidad de Taquicardia", fontsize=13.0)
plt.xlabel("Frecuencia Cardiaca (latidos por minuto)", fontsize=13.0)
plt.legend(bbox_to_anchor=(1, 0.2))
plt.show()


# In[7]:


#Maxima Verosimilitud
#Diferentes funciones logísticas con diferentes "pendientes"
plt.figure(figsize=(6, 4))

for b1 in np.arange(0.35, 0.49, 0.025):
    plt.scatter(np.arange(60, 200, 0.1),
                logistica(-46.68057196,
                         b1,
                         np.arange(60, 200, 0.1)),
                label="b_1=%0.2f" % b1)
    #Graficamos la frecuencia cardiaca de las personas
    plt.scatter(personas_normal, [0]*10,
               marker="o", c="green", s=250, label="NORMAL")
    plt.scatter(personas_taquicardia, [1]*10,
               marker="o", c="red", s=250, label="TAQUICARDIA")
    plt.title("Máxima Verosimilitud", fontsize=18.0)
    plt.text(0, 1, "TAQUICARDIA", size=12, color="red")
    plt.text(0, 0, "NORMAL", size=12, color="red")
    plt.ylabel("Probabilidad de Taquicardia", fontsize=13.0)
    plt.xlabel("Frecuencia cardíaca (latidos por minuto)", fontsize=13.0)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()
    
    


# In[13]:


# Modelo de Regresión logística
#importacion de libreriad e regresion logistica
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

frecuencias_cardiacas = [[65], [70], [80], [80], [80],
                       [90], [95], [100], [105], [110],
                       [105], [110], [110], [120], [120],
                       [130], [140], [180], [185], [190]]
# vector de clase
clase = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

#Creamos conjuntos de entrenamiento de prueba del modelo
#train_test_split separa los datos de forma aleatoria (muestreo de datos de una forma aleatoria)
datos_entrena, datos_prueba, clase_entrena, clase_prueba = train_test_split(frecuencias_cardiacas, 
                clase,
                test_size=0.30) #parametro para crear el modelo

#Creamos el modelo de regresion logistica


modelo = LogisticRegression().fit(datos_entrena, clase_entrena)
#suprimir la notacion cientifica
np.set_printoptions(suppress=True)
#utilizamos nuestro modelo para predecir los datos de prueba que no fueron introducidos para evaluarlos utilizamos el metodo predict
print(modelo.predict(datos_prueba))
#obtencion de datos de probablidadad de cada uno de los individuos que pertenecen  a la clase taquicardia
print(modelo.predict_proba(datos_prueba))
#introducimos la metrica de cuantos elementos clasifici correctamente
print(modelo.score(datos_prueba, clase_prueba))
#obtencion de b0 y b1
print(modelo.intercept_, modelo.coef_)


# 
