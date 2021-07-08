#%matplotlib inline
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
#from timeit import default_timer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

#Añadido para evitar los warnigns producidos por alcanzar el numero maximo de iteraciones antes de converger
#El programa nos recomienda aumentarlo para mejorar el ajuste
import warnings

# Fijamos la semilla
np.random.seed(1)

#Funcion para leer los datos de un fichero que los contiene
def cargar_datos(nom_archivo, remove_header=False):
    # Leemos el cotenido del archivo y lo cargamos en una variable
    archivo_cargado = pd.read_csv(nom_archivo)
    
    #obj tendra el nombre del ultimo atributo que es el importante ('critical_temp')
    obj = archivo_cargado.columns.values[-1]
    #carga en caract los nombres de todos los atributos que sean diferentes a 'critical_temp' para
    #coger el nombre de todos los atributos que no sean objetivo
    caract = [f for f in archivo_cargado.columns if f != obj]
    
    #cargamos los datos de la ultima columna (las etiquetas de la temperatura) en el conjunto y
    y = archivo_cargado.values[:,-1]
    #cargamos los datos de todas las columnas menos la ultima en el conjunto x (los atributos)
    X = archivo_cargado.values[:,:-1]
    
    #Devolvemos el conjunto de datos, de etiquetas y el de los nombres de los atributos para obtener los 
    #importantes
    return X, y, archivo_cargado, caract
    
#Funcion que ordena un conjunto de datos en base a la importancia de sus caracteristicas
def obtener_datos_importantes(X, y, caract, model):
    #Ajustamos el modelo (DecisionTreeRegressor) al conjunto de datos y etiquetas
    model.fit(X, y)
    #Llamamos a la funcion que devuelve la importancia de cada variable
    #La importancia se calcula como la reduccion total del criterio que trae esa caracteristica
    importances = list(zip(caract, model.feature_importances_))
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
    pca=PCA(0.8) 
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
X, y, archivo_cargado, caract = cargar_datos('./data/regresion/train.csv')

print("CARGADOS CON EXITO")
print("         Datos leidos:", archivo_cargado.shape[0])
print("         Atributos leidos:", archivo_cargado.shape[1])


#Declaramos cual sera nuestro modelo para tener datos importantes
modelo_importancia = DecisionTreeRegressor()
#Llamamos a nuestra funcion para ver las variables mas importantes
importances_sorted = obtener_datos_importantes(X, y, caract ,modelo_importancia)

#Mostramos los 10 mas importantes
print("\nLos atributos más importantes son:\n")
for i in range(10):
    print(importances_sorted[i])

#Vector que contenga los 10 mas importantes para mostrarlos en una grafica
mostrar_importantes = importances_sorted[:10]

#Pintamos un histograma para ver la distribucion de los datos para intentar buscar una relacion 
#entre los datos a simple vista
input("\n--- Pulsar tecla para hacer la grafica sobre la distribucion de los datos ---\n")
plt.xlabel("temperatura")
plt.ylabel('Veces que aparece esta temperatura')
plt.hist(y, color='green')
plt.show()

#Pintamos la grafica de las 10 variables mas importantes para compararlas 
input("\n--- Pulsar tecla para hacer la grafica sobre los atributos importantes ---\n")
plt.figure(figsize=(20,6), dpi=200)
plt.xticks(rotation=45)
plt.bar(list(list(zip(*mostrar_importantes)))[0], list(zip(*mostrar_importantes))[1], color='green')
plt.show()



#Con esta linea miramos en todo el conjunto de datos que no haya valores perdidos. 
#Si no los hay, muestra TRUE. De haber datos perdidos habria que procesar de otra forma, 
# pero no es el caso. DESCOMENTAR PARA VERLO

#input("\n--- Pulsar tecla para ver si existen valores perdidos ---\n")
#print("No existen datos perdidos: ", np.all(archivo_cargado.notnull()))



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

print("PREPROCESADOS CON EXITO")
print("     Numero de datos en entrenamiento preprocesados: ", X_train_modificado.shape[0] )
print("     Numero de datos en test preprocesados: ", X_test_modificado.shape[0] )
print("     Numero de atributos por dato preprocesados: ", X_test_modificado.shape[1] )
print("     Se han reducido ", X_test.shape[1] - X_test_modificado.shape[1] , " atributos")


#Declaramos el vector que contiene las transformaciones que hemos hecho en el preprocesado
#Sera util para buscar el mejor modelo. Queremos evitar data snooping, por lo que se indexara
#este vector con el de los modelos a probar y se hara todo a la vez. El ultimo elemento es un placeholder
#en el que iran los modelos a comprobar
vector_preprocesamiento_modelo = [("standardization", StandardScaler()),  ("dim_reduction", PCA(0.8)),
                                  ("zspace", PolynomialFeatures(2)),      ("variance_thresh", VarianceThreshold(0.1)),
                                  ("reg", SGDRegressor())]

#Espacio de busqueda. Aqui especificamos los modelos que vamos a probar y sus parametros
modelos_de_prueba = [ {"reg": [Lasso(max_iter = 2500)], "reg__alpha": np.logspace(-4, 4, 15)},
                 {"reg": [Ridge(max_iter = 2500)], "reg__alpha": np.logspace(-4, 4, 15)}, 
                 {"reg": [SGDRegressor()], "reg__alpha": np.logspace(-4, 4, 15),
                  "reg__loss": ['squared_loss', 'epsilon_insensitive'],
                  "reg__penalty": ['l1', 'l2'],
                  "reg__learning_rate": ['optimal', 'adaptive']}    
]

#canalización para ensamblar varios pasos que se pueden validar de forma cruzada mientras se 
#establecen diferentes parámetros
preprocesamiento_con_modelos = Pipeline(vector_preprocesamiento_modelo)

#Buscamos el mejor modelo con los parametros, modelos y pasos indicados
best_model = GridSearchCV(preprocesamiento_con_modelos, modelos_de_prueba,scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=10)

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
print(-best_model.best_score_)

#Mostramos los errores dentro y fuera del conjunto. Ahora usamos el conjunto test despues de no
#tocarlo en todo el proceso
input("\n--- Pulsar tecla para ver los errores obtenidos ---\n")
#vector que contendra los errores
model_metrics = {}

#Este es el error dentro de la muestra con MSE y con coeficiente de determinacion
y_pred = best_model.predict(X_train)
model_metrics[mean_squared_error.__name__] = mean_squared_error(y_train, y_pred)
model_metrics[r2_score.__name__] = r2_score(y_train, y_pred)

print("TRAIN")
print("Error de train con MSE: ", model_metrics[mean_squared_error.__name__])
print("Error de train con R^2: ",model_metrics[r2_score.__name__])


#Este es el error fuera de la muestra con MSE y con coeficiente de determinacion
y_pred = best_model.predict(X_test)
model_metrics[mean_squared_error.__name__] = mean_squared_error(y_test, y_pred)
model_metrics[r2_score.__name__] = r2_score(y_test, y_pred)

print("TEST")
print("Error de test con MSE: ", model_metrics[mean_squared_error.__name__])
print("Error de test con R^2: ",model_metrics[r2_score.__name__])

