# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import sklearn as skl
from sklearn import datasets
import matplotlib.pyplot as plt 
import math as mt

#Parte 1
#primero
iris = skl.datasets.load_iris()

#segundo
X = iris.data
y = iris.target

#tercero
X = X[:, 1:4:2]

#cuarto
def cal_tamanio(arr_clase, id_clase):
    cont = 0
    for b in arr_clase:
        if(b==id_clase):
            cont = cont+1;
    return cont

primer_tamanio = cal_tamanio(y, 0)
segundo_tamanio = primer_tamanio + cal_tamanio(y, 1)
tercer_tamanio = segundo_tamanio + cal_tamanio(y, 2)

plt.scatter(X[:primer_tamanio, 0], X[:primer_tamanio,1], label = 'PRIMERA CLASE', c='orange', linewidths=2)
plt.scatter(X[primer_tamanio:segundo_tamanio, 0], X[primer_tamanio:segundo_tamanio,1], label = 'SEGUNDA CLASE', c='k', linewidths=2)
plt.scatter(X[segundo_tamanio:, 0], X[segundo_tamanio:,1], label = 'TERCERA CLASE', c='g', linewidths=2)

plt.legend()
plt.show()
print("\n")

#Parte2
#primero
#Unir ambas listas
segundo_ejer = list(zip(X,y))

np.random.shuffle(segundo_ejer)

cont_ctrain = 0
cont_utrain = 0
cont_dtrain = 0

cont_ctest = 0
cont_utest = 0
cont_dtest = 0

x_train = []
y_train = []
x_test = []
y_test = []

for a in segundo_ejer:
    if(a[1]==0):
        if(cont_ctrain < 38):
            x_train.append(a[0])
            y_train.append(a[1])
            cont_ctrain = cont_ctrain + 1
        
        elif(cont_ctrain > 37):
            x_test.append(a[0])
            y_test.append(a[1])
            cont_ctest = cont_ctest + 1
            
            
    if(a[1]==1):
        if(cont_utrain < 38):
            x_train.append(a[0])
            y_train.append(a[1])
            cont_utrain = cont_utrain + 1
        
        elif(cont_utrain > 37):
            x_test.append(a[0])
            y_test.append(a[1])
            cont_utest = cont_utest + 1
            
    if(a[1]==2):
        if(cont_dtrain < 38):
            x_train.append(a[0])
            y_train.append(a[1])
            cont_dtrain = cont_dtrain + 1
        
        elif(cont_dtrain > 37):
            x_test.append(a[0])
            y_test.append(a[1])
            cont_dtest = cont_dtest + 1
            
            
    
print("Tamanio del training: ", len(x_train), "Tamanio de Test: ", len(x_test))
print("\n")
print("Training 0: ", cont_ctrain, "Training 1: ", cont_utrain,  "Training 2: ", cont_dtrain)
print("Test 0: ", cont_ctest, "Test 1: ", cont_utest,  "Test 2: ", cont_dtest)
print("\n")
    
    
#Parte3
#primero
a = np.linspace(0, 4*mt.pi, 100)

#segundo
seno = np.sin(a)
coseno = np.cos(a)
suma = seno + coseno

#tercero
plt.plot(a, seno, 'k--', a, coseno, 'r--' , a, suma, 'g--')
plt.show()

