#%matplotlib inline
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from timeit import default_timer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE,f_classif
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import  accuracy_score, precision_score
from sklearn.linear_model import LinearRegression, LogisticRegression,SGDClassifier, RidgeClassifier, Perceptron
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC


#Añadido para evitar los warnigns producidos por alcanzar el numero maximo de iteraciones antes de converger
#El programa nos recomienda aumentarlo para mejorar el ajuste
import warnings

# Fijamos la semilla
np.random.seed(1)

#Funcion para leer los datos de un fichero que los contiene
def cargar_datos(nom_archivo, remove_header=False):
    # Leemos el cotenido del archivo y lo cargamos en una variable
    archivo_cargado = np.loadtxt(nom_archivo, delimiter=' ')
    archivo_cargado = np.array(archivo_cargado)

    #cargamos los datos de la ultima columna (las etiquetas de la temperatura) en el conjunto y
    y = archivo_cargado[:,-1]
    #cargamos los datos de todas las columnas menos la ultima en el conjunto x (los atributos)
    X = archivo_cargado[:,:-1]
    
    #Devolvemos el conjunto de datos, de etiquetas y el conjunto total
    return X, y, archivo_cargado
    
#Funcion que ordena un conjunto de datos en base a la importancia de sus caracteristicas
def obtener_datos_importantes(X, y, model):
    #Ajustamos el modelo (DecisionTreeRegressor) al conjunto de datos y etiquetas
    model.fit(X, y)
    consecutivos = list(range(X.shape[1]))
    #Llamamos a la funcion que devuelve la importancia de cada variable
    #La importancia se calcula como la reduccion total del criterio que trae esa caracteristica
    importances = list(zip(consecutivos, model.feature_importances_))
    #Ordena el conjunto de mayor a menor para poner por delante las mas importantes 
    importances.sort(key=lambda x: x[1], reverse=True)
    
    #Devolvemos el vector ordenado por el indice de importancia
    return importances

#Funcion para el preprocesado de datos
def preprocessin_data(X_train,y_train, X_test):
    
    #Estandarizacion de los datos para que pertenezcan a una distribucion normal estandar
    scaler = StandardScaler()
    X_train_modificado = scaler.fit_transform(X_train, y_train)
    X_test_modificado = scaler.transform(X_test)
    
    #Aplicamos PCA para la reduccion de dimensionalidad
    pca=PCA(0.6) 
    X_train_modificado = pca.fit_transform(X_train_modificado, y_train)
    X_test_modificado = pca.transform(X_test_modificado)
    
    
    #Usamos transformaciones cuadraticas para adaptarnos mejor a nuestros datos
    trans = PolynomialFeatures(2)
    X_train_modificado = trans.fit_transform(X_train_modificado, y_train)
    X_test_modificado = trans.transform(X_test_modificado)
    
    #Eliminamos las variable que tienen una varianza muy baja o nula
    selector = VarianceThreshold(0.1)
    X_train_modificado = selector.fit_transform(X_train_modificado, y_train)
    X_test_modificado = selector.transform(X_test_modificado)

    #Devolvemos el conjunto modificado
    return X_train_modificado, X_test_modificado




###################################################################################################
#################################### CARGAMOS LOS DATOS TOTALES ###################################
###################################################################################################
input("\n--- Pulsar tecla para cargar los datos ---\n")

#Llamamos a nuestra funcion de cargar datos
X, y, archivo_cargado = cargar_datos('./data/clasificacion/Sensorless_drive_diagnosis.txt')

print("CARGADOS CON EXITO")
print("         Datos leidos:", archivo_cargado.shape[0])
print("         Atributos leidos:", archivo_cargado.shape[1])

values = []
tags = list(set(archivo_cargado[:,-1]))
for tag in tags:
    values.append(list(archivo_cargado[:,-1]).count(tag))
plt.figure(dpi=200)
plt.bar(tags, values, color = 'green')
plt.xticks(tags)
plt.show()

input("\n--- Pulsar tecla para dividir los datos ---\n")
#Hacemos la division de datos del conjunto total de datos y etiquetas en test y training
#Se usa la funcion train_test_split y dividimos un 80% para training y 20% en test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("DIVIDIDOS CON EXITO")
print("     Numero de datos en entrenamiento: ", X_train.shape[0] )
print("     Numero de datos en test: ", X_test.shape[0] )
print("     Numero de atributos por dato: ", X_test.shape[1] )


input("\n--- Pulsar tecla para preprocesar los datos ---\n")
#Preprocesamos los datos para eliminar varianzas bajas, adaptarlos a una distribucion, etc
#Paso necesario para trabajar con los datos
X_train_modificado, X_test_modificado = preprocessin_data(X_train, y_train, X_test)


print("PREPROCESADOS CON LA PRIMERA FUNCION CON EXITO")
print("     Numero de datos en entrenamiento preprocesados: ", X_train_modificado.shape[0] )
print("     Numero de datos en test preprocesados: ", X_test_modificado.shape[0] )
print("     Numero de atributos por dato preprocesados: ", X_test_modificado.shape[1] )
print("     Se han reducido ", X_test.shape[1] - X_test_modificado.shape[1] , " atributos")



#Declaramos el vector que contiene las transformaciones que hemos hecho en el preprocesado
#Sera util para buscar el mejor modelo. Queremos evitar data snooping, por lo que se indexara
#este vector con el de los modelos a probar y se hara todo a la vez. El ultimo elemento es un placeholder
#en el que iran los modelos a comprobar
vector_preprocesamiento_modelo = [("standardization", StandardScaler()),  ("dim_reduction", PCA(0.6)),
                                  ("zspace", PolynomialFeatures(2)),      ("variance_thresh", VarianceThreshold(0.1)),
                                  ("clas", LogisticRegression())]


modelos_de_prueba = [ {"clas": [LogisticRegression(max_iter=2500)], "clas__penalty": ['l2']}, #LogisticRegression
                      {"clas": [Perceptron(max_iter=2500)], "clas__penalty": ['l1', 'l2'],
                       'clas__alpha':[0.1,0.001,0.0001,0.00001]},
                      {"clas": [SGDClassifier(loss='hinge', penalty='l2', max_iter=2500 )],
                       "clas__alpha": np.logspace(-5, 5, 3)}]


#canalización para ensamblar varios pasos que se pueden validar de forma cruzada mientras se 
#establecen diferentes parámetros
preprocesamiento_con_modelos = Pipeline(vector_preprocesamiento_modelo)

#Buscamos el mejor modelo con los parametros, modelos y pasos indicados
best_model = GridSearchCV(preprocesamiento_con_modelos, modelos_de_prueba,scoring='accuracy', cv=5, n_jobs=-1, verbose=10)

input("\n--- Pulsar tecla para aplicar el mejor modelo ---\n")
#Ignora los warnings que muestra por el numero de iteraciones indicado. Descomentar para evitarlos
#warnings.filterwarnings("ignore")

#Ajustamos el mejor modelo obtenido a nuestro conjunto de datos. Se hace sobre el que no esta procesado 
#para obtener mejor rendimiento
best_model.fit(X_train, y_train)

#Obtenemos los parametros con los que mejor resultado ha dado
input("\n--- Pulsar tecla para ver los mejores parametros ---\n")
best_params = best_model.best_params_
print(best_params)

#Vemos cual ha sido el mejor error que hemos obtenido en la validacion
input("\n--- Pulsar tecla para ver el menor error obtenido por el modelo ---\n")
print(best_model.best_score_)

#Mostramos los errores dentro y fuera del conjunto. Ahora usamos el conjunto test despues de no
#tocarlo en todo el proceso
input("\n--- Pulsar tecla para ver los errores obtenidos ---\n")

#Este es el error dentro de la muestra con precision y accuracy
print("TRAIN")
print("Error dentro de la muestra con accuracy: ", accuracy_score(y_train, best_model.predict(X_train)))


#Este es el error fuera de la muestra con precision y accuracy
print("TEST")
print("Error fuera de la muestra con accuracy: ", accuracy_score(y_test, best_model.predict(X_test)))





    
    
    
