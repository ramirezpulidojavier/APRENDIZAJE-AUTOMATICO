import pandas as pd
import seaborn as sns
import numpy as np

# To plot pretty figures
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

# Machine learn packages
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, PassiveAggressiveRegressor, SGDRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.manifold import TSNE
import multiprocessing

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

#SI SE PONE A TRUE MANEJAMOS EL AVANCE DE LA EVOLUCION PARANDOSE CADA VEZ QUE
#CALCULE O MUESTRE ALGO. SI ESTA A FALSE HACE LA EJECUCION DE CORRIDO DE 
#PRINCIPIO A FIN
detener_por_pasos = False

# Same random seed state
np.random.seed(1)

######################################################################
######################################################################
######################################################################
######################################################################
######################################################################

#Funcion para leer los datos de un archivo .txt 
def cargar_dividir_datos(ruta, remove_header=False):
    #Cargamos todos los datos en 'datos'
    datos = np.loadtxt(ruta, delimiter='\t')
    datos = np.array(datos)
    
    #Metemos en 'x' aquellos que son atributos
    x = datos[:,:-1]
    #Metemos en 'y' aquellos que son etiquetas (ultima columna en nuestro caso)
    y = datos[:,-1]
    
    #Dividimos el conjunto en training y test para entrenar y probar 
    #nuestros modelos. Respectivamente será un 80% y un 20%
    x_tr, x_ts, y_tr, y_ts = train_test_split(np.array(x),np.array(y),test_size=0.2)
    
    #Devolvemos tanto el conjunto completo de datos, de etiquetas, de train y de test
    return datos, y, np.array(x_tr), np.array(y_tr), np.array(x_ts), np.array(y_ts)

######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
if detener_por_pasos:
    print("------------------------------------------------------------------")
    print("--------------------PULSA PARA CARGAR LOS DATOS-------------------")
    print("------------------------------------------------------------------")
    input()
#Usamos nuestra funcion para cargar los datos de la base de datos
datos, etiq, x_train, y_train, x_test, y_test = cargar_dividir_datos("./airfoil_self_noise.txt")

#Si llegamos aqui, se habran cargado con exito
print("¡CARGADO CON EXITO!")

#Mostramos la dimension de nuestro conjunto de datos 
print("\nDATASET: ")
print('\t\tDATOS:' ,datos.shape[0])         #Cantidad de datos
print('\t\tATRIBUTOS:' ,datos.shape[1])     #Cantidad de atributos

#Convertimos los datos en dataframe para acceder a informacion estadistica sobre ellos
datospru = pd.DataFrame(datos) 

#Como sabemos los nombres de las columnas por la base de datos, le ponemos nombre a las variables de nuestro conjunto
datospru.columns = ['frequency', 'angle_of_attack', 'chord', 'velocity', 'suc_displacement', 'sound_pressure'] 

#Para que muestre todas las columnas que hay y nos las limite
pd.set_option('display.max_columns', None)

if detener_por_pasos:
    print("\n------------------------------------------------------------------")
    print("---------PULSA PARA MOSTRAR LOS VALORES DE CADA ATRIBUTO----------")
    print("------------------------------------------------------------------")
    input()

print("- VALOR DE ATRIBUTOS\n\n")
#Mostramos las n primeras filas de valores para todos los atributos
#Si no se le pasa ningun n como parametro, es 5 por defecto (5x6 en nuestro caso)
print(datospru.head(), "\n")


if detener_por_pasos:
    print("\n------------------------------------------------------------------")
    print("----------PULSA PARA MOSTRAR ESTADISTICAS DEL CONJUNTO------------")
    print("------------------------------------------------------------------")
    input()


print("- ESTADISTICAS DEL CONJUNTO\n\n")
#Muestra estadisticas como cuartiles, medias, maximos, minimos, etc para conocer
#mejor el conjunto de datos
print(datospru.describe(), "\n")

if detener_por_pasos:
    print("------------------------------------------------------------------")
    print("------------PULSA PARA MOSTRAR RESUMEN DE LOS DATOS---------------")
    print("------------------------------------------------------------------")
    input()


print("- RESUMEN DE LOS DATOS DEL PROBLEMA\n\n")
#‎Este método imprime información sobre un DataFrame, incluidos el índice 
#dtype y las columnas, los valores no NULL y el uso de memoria.‎
print(datospru.info(), "\n")


# Comprobamos si tenemos valores perdidos en el conjunto. De esto dependera
# el tratamiento que se le realice a los datos para los modelos
if detener_por_pasos:
    print("------------------------------------------------------------------")
    print("--------PULSA PARA COMPROBAR SI TENEMOS DATOS PERDIDOS------------")
    print("------------------------------------------------------------------")
    input()
    
    
if (np.all(datospru.notnull())):
    print('\t\t\t¡NO HAY DATOS PERDIDOS!\n\n')
else:
    print('\t\t\t¡HAY DATOS PERDIDOS!\n\n')


if detener_por_pasos:
    print("------------------------------------------------------------------")
    print("------------PULSA PARA MOSTRAR RELACIONES EN PARES----------------")
    print("------------------------------------------------------------------")
    input()
# Las gráficas diagonales se tratan de manera diferente: se dibuja 
# una gráfica de distribución univariante para mostrar la distribución
# marginal de los datos en cada columna.
sns.pairplot(datospru)
plt.show()

if detener_por_pasos:
    print("------------------------------------------------------------------")
    print("---------PULSA PARA MOSTRAR ATRIBUTOS MAS IMPORTANTES-------------")
    print("------------------------------------------------------------------")
    input()
#Sacamos del conjunto todos los valores que se corresponden a presion de sonido (valor a predecir)
sound_pressure = datospru.columns.values[-1] #ultima columna

#Vector que contendra todos los atributos que no sean etiqueta 
atributos = []
#Introducimos en el vector de caracteristicas todas aquellas que se encuentren 
# en nuestro conjunto de datos a excepcion de la temperatura critica,
# que ha sido seleccionada previamente
for i in datospru.columns:    
    if i != sound_pressure:
        atributos.append(i)
        
#Vamos a seleccionar los atributos mas predictivos de nuestro problema
#Esto nos ayudara a comprender mejor los datos, entender mejor nuestro modelo 
# y reducir el numero de funciones de entrada
#Para ello empleamos DecisionTreeRegressor de Scikit-Learn
modelo = DecisionTreeRegressor()
modelo.fit(datospru,etiq)


#Sacamos en una lista de entre todos los atributos aquellos mas predictivos 
#Gracias a la funcion feature_importances_ que incorpora nuestro modelo
importances = list(zip(atributos, modelo.feature_importances_))


#Y las ordenamos en orden descendente para priorizar las mas importantes
#key=lambda x: x[1] crea una funcion inline para obtener el elemento x[1]
#Reverse True saca la lista descendente 
importances.sort(key=lambda x: x[1], reverse=True) 


#Grafica con todos los atributos ordenados por importancia
plt.xticks(rotation=45)
plt.bar(list(
            list(zip(*importances)))[0],
            list(zip(*importances))[1])
plt.show()


#Sacamos los dos mas predictivos
atri_predi = [importance[0] for importance in importances[:2]]

#Mostramos el nombre de los dos atributos con mayor relevancia en el conjunto
#Nos puede ayudar a buscar relaciones para predecir la presion de ruido
print("LOS DOS ATRIBUTOS CON MAYOR IMPORTANCIA SON:")
print("\t\t- ", atri_predi[0])
print("\t\t- ", atri_predi[1])

if detener_por_pasos:
    print("\n\n------------------------------------------------------------------")
    print("PULSA PARA LA RELACION DE ESTOS ATRIBUTOS CON EL VALOR A PREDECIR-")
    print("------------------------------------------------------------------")
    input()
    
#Funcion que muestra la relacion que tienen los atributos mas importantes con
# el valor a predecir
fig, ax = plt.subplots(1,2, figsize=(10,4), dpi=200)
#Primera grafica
sns.scatterplot(x=atri_predi[0], y=sound_pressure, data=datospru, ax=ax[0])
#Segunda grafica
sns.scatterplot(x=atri_predi[1], y=sound_pressure, data=datospru, ax=ax[1])
fig.tight_layout()
plt.show()

if detener_por_pasos:
    print("------------------------------------------------------------------")
    print("------------PULSA PARA VER LA DISTRIBUCION DE LOS DATOS-----------")
    print("------------------------------------------------------------------")
    input()
    
#Histrograma con la distribucion de los datos
plt.xlabel(sound_pressure)
plt.ylabel('Veces que aparece')
plt.hist(etiq, edgecolor='black')
plt.show()

######################################################################
######################################################################
########               PREPROCESADO                    ###############
######################################################################
######################################################################
if detener_por_pasos:
    print("------------------------------------------------------------------")
    print("------------------PULSA PARA PREPROCESAR LOS DATOS----------------")
    print("------------------------------------------------------------------")
    input()
    
# Preprocesado que aplicaremos al conjunto de datos antes de trabajar con el
pipeline_preprocesado = [
    
    ("aumento_complejidad", PolynomialFeatures()),
    ("escalado", MinMaxScaler()),
    ("reduccion_dimensionalidad", PCA(0.9))

]
  
#Creamos nuestro Pipeline de preprocesado con las condiciones que hemos 
# definido anteriormente
preprocessing_pipeline = Pipeline(pipeline_preprocesado)

#Aplicamos el preprocesado sobre los datos 
x_train_preprocesado = preprocessing_pipeline.fit_transform(x_train, y_train)
x_test_preprocesado = preprocessing_pipeline.transform(x_test)

#Si llegamos a este punto es porque se ha preprocesado con exito
print("¡PREPROCESADO CON EXITO!")

#Mostramos la dimension de nuestro conjunto de datos 
print("\nDATASET PREPOCESADO: ")
print('\t\tDATOS:'      ,x_train_preprocesado.shape[0])         #Cantidad de datos
print('\t\tATRIBUTOS:'  ,x_train_preprocesado.shape[1])     #Cantidad de atributos

if detener_por_pasos:
    print("\n\n------------------------------------------------------------------")
    print("------PULSA PARA VER TSNE ANTES Y DESPUES DEL PREPROCESADO--------")
    print("------------------------------------------------------------------")
    input()

#TSNE para ver los datos originales
data_tsne = TSNE().fit_transform(x_train)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(data_tsne[:,0], data_tsne[:,1],  c='green', marker='o')
ax.set_xlabel('TSNE 1')
ax.set_ylabel('TSNE 2')
plt.show()

#TSNE para ver los datos preprocesados
data_tsne = TSNE().fit_transform(x_train_preprocesado)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('TSNE 1')
ax.set_ylabel('TSNE 2')
ax.scatter(data_tsne[:,0], data_tsne[:,1],  c='green', marker='o')
plt.show()

if detener_por_pasos:
    print("------------------------------------------------------------------")
    print("--PULSA PARA VER LA CORRELACION ANTES Y DESPUES DEL PREPROCESADO--")
    print("------------------------------------------------------------------")
    input()
    
#Grafica con la matriz de correlacion antes de preprocesar los datos
plt.rcParams['figure.figsize'] = (7.0, 5.0)
plt.title("MATRIZ DE CORRELACION ORIGINAL")
sns.heatmap(datospru.corr(), cmap='viridis')
plt.show()

#Grafica con la matriz de correlacion despues de preprocesar los datos
datospru2 = pd.DataFrame(x_train_preprocesado)
plt.rcParams['figure.figsize'] = (7.0, 5.0)
plt.title("MATRIZ DE CORRELACION PREPROCESADA")
sns.heatmap(datospru2.corr(), cmap='viridis')
plt.show()

if detener_por_pasos:
    print("------------------------------------------------------------------")
    print("----------PULSA PARA CALCULAR LOS ERRORES DE CADA MODELO----------")
    print("------------------------------------------------------------------")
    input()

#Agregamos al conjunto de procedimientos de preprocesado un placeholder para que
#todos los modelos que apliquemos se realicen a la vez que el preprocesado
#y evitemos data snooping
preprocessing_pipeline2 = Pipeline(pipeline_preprocesado + [("model", MLPRegressor(max_iter=2000))])

#Modelos a probar para nuestro proyecto
parameters_to_train = [
    
            {"model": [MLPRegressor(max_iter=2000)],
                        "model__hidden_layer_sizes": [(50,80)],
                        "model__activation": ['tanh', 'relu', 'logistic'],
                        "model__solver": ['sgd', 'adam'],
                        "model__alpha": [0.0001, 0.05],
                        "model__learning_rate": ['constant','adaptive']},
    
            {"model": [SGDRegressor(max_iter = 25000)] , 
                         "model__loss": ['squared_loss', 'epsilon_insensitive'],
                         "model__penalty": ['l1', 'l2'],
                         "model__learning_rate": ['optimal', 'adaptive']},
            
            {"model": [GradientBoostingRegressor(max_features = 'auto')] ,
                          "model__n_estimators":[10, 50, 100, 500],
                          "model__max_depth":[1, 3, 5], 
                          "model__learning_rate":[0.001, 0.01, 0.1]},
            
            {"model": [Ridge(max_iter=2500, tol=5)] ,
                          "model__alpha": np.logspace(-5, 2, 20)},
            
            {"model": [Lasso(max_iter=2500, tol=5)] ,
                          "model__alpha": np.logspace(-5, 2, 20)},
            
            {"model": [ElasticNet(max_iter=2500, tol=5)] ,
                          "model__alpha": np.logspace(-5, 2, 20),
                          "model__l1_ratio" : [0, 0.2, 0.4, 0.6, 0.8, 1]},
            
            {"model": [SVR()] ,
                          "model__kernel": ['poly','rbf'], 
                          "model__C":[0.1, 0.2, 0.4, 0.7, 0.9, 1] },
            
            {"model": [AdaBoostRegressor()] ,
                          "model__n_estimators": [10, 50, 100, 500], 
                          "model__learning_rate" : [0.001, 0.01, 0.1]},
            
            {"model": [HistGradientBoostingRegressor(max_iter = 250)] ,
                          "model__l2_regularization": [0, 1], 
                          "model__max_depth":[1, 3, 5], 
                          "model__learning_rate":[0.001, 0.01, 0.1]},
            
            {"model": [LinearRegression(n_jobs=-1)]},
            
            {"model": [BayesianRidge(n_iter = 2000)]},
            
            {"model": [PassiveAggressiveRegressor(max_iter=2000)]},
            
            {"model": [KNeighborsRegressor(n_jobs=-1)]},
            
            {"model": [RandomForestRegressor(n_jobs=-1, random_state = 200)],
                         "model__n_estimators": [20, 100, 200, 300, 400, 500, 600, 1000]},
            
            
 
]

#Nombre de las funciones para que cada grafica pueda estar personalizada y reconocible
funciones = ["MLPRegressor","SGDRegressor", "GradientBoostingRegressor", "Ridge", "Lasso", 
             "ElasticNet", "SVR", "AdaBoostRegressor", "HistGradientBoostingRegressor", 
             "LinearRegression", "BayesianRidge", "PassiveAggressiveRegressor",
             "KNeighborsRegressor", "RandomForest"]


#Vector que contendra en cada posicion el nombre de cada modelo y su error con 
#diferentes metricas (en nuestro casi, MSE, MAE, raiz de MSE y R2)
results = []
#Para todos los modelos que tengamos
for i in range(len(funciones)):
    #Aplicamos GDCV
    train_estimator = GridSearchCV(preprocessing_pipeline2, parameters_to_train[i], 
                                   scoring='neg_mean_squared_error', cv=5, n_jobs=-1,
                                   verbose=10)

    #Ajustamos el modelo con mejores parametros que se saque de cross validation
    train_estimator.fit(x_train, y_train)
    y_predict = train_estimator.predict(x_test)
    
    #Grafica que muestra la regresion generada por los valores predichos
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_predict)
    ax.plot([y_test.min(), y_test.max()], [y_predict.min(), y_predict.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.title(funciones[i])
    plt.show()
    
    #Grafica que compara los valores que deberiamos haber obtenido con los que obtenemos realmente
    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_predict, label="predicted")
    plt.title(funciones[i])
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()
    
    #Grafica de la distribucion del error de cada modelo
    y_predict = pd.Series(y_predict.flatten().tolist())
    plt.rcParams['figure.figsize'] = (7.0, 5.0)
    plt.title("Distribucion de error de "+ funciones[i])
    sns.distplot(y_test-y_predict, bins = 20)
    plt.show()
    
    #Comentar para que haga todos los modelos de corrido
    if detener_por_pasos:
        print("------------------------------------------------------------------")
        print("---------------PULSA PARA PASAR AL SIGUIENTE MODELO---------------")
        print("------------------------------------------------------------------")
        input()
    
    #Añadimos los valores del error de cada metrica que tengamos para este modelo
    results.append((funciones[i], r2_score(y_test, y_predict), mean_squared_error(y_test, y_predict), mean_absolute_error(y_test, y_predict), mean_squared_error(y_test, y_predict)**0.5))


#Mostramos la tabla con cada modelo y cada medida del error para comparar el desempeño de cada uno
if detener_por_pasos:
    print("------------------------------------------------------------------")
    print("---------------------PULSA PARA MOSTRAR LA TABLA------------------")
    print("------------------------------------------------------------------")
    input()

print("RMSE score  |  R2 score  |  MAE score   |   MSE score     |    Modelo")
for res in results:
    print(f"{res[4]:.4f}", "\t\t", f"{res[1]:.4f}", "\t\t", f"{res[3]:.4f}", "\t\t", f"{res[2]:.4f}", "\t\t", res[0])

if detener_por_pasos:
    print("------------------------------------------------------------------")
    print("-----------COMPROBACION DE LASSO Y RIDGE CON ELASTIC NET----------")
    print("------------------------------------------------------------------")
    input()
    
#alpha = 1 -> lasso /// alpha = 0 -> ridge
#Aplicamos ElasticNet con diferentes valores de regularizacion para ver si el mejor
# desempeño se obtiene mas cercano a la regularizacion que en la tabla anterior es mejor
parameters_to_train1 = [
    {"model": [ElasticNet(max_iter = 25000, l1_ratio=0, tol=5)] , 
                          "model__alpha": np.logspace(-10, 3, 200)},
    {"model": [ElasticNet(max_iter = 25000, l1_ratio=0.15, tol=5)] , 
                          "model__alpha": np.logspace(-10, 3, 200)},
    {"model": [ElasticNet(max_iter = 25000, l1_ratio=0.25, tol=5)] , 
                          "model__alpha": np.logspace(-10, 3, 200)},
    {"model": [ElasticNet(max_iter = 25000, l1_ratio=0.35, tol=5)] , 
                          "model__alpha": np.logspace(-10, 3, 200)},
    {"model": [ElasticNet(max_iter = 25000, l1_ratio=0.55, tol=5)] , 
                          "model__alpha": np.logspace(-10, 3, 200)},
    {"model": [ElasticNet(max_iter = 25000, l1_ratio=0.75, tol=5)] , 
                          "model__alpha": np.logspace(-10, 3, 200)},
    {"model": [ElasticNet(max_iter = 25000, l1_ratio=0.95, tol=5)] , 
                          "model__alpha": np.logspace(-10, 3, 200)},
    {"model": [ElasticNet(max_iter = 25000, l1_ratio=1, tol=5)] , 
                          "model__alpha": np.logspace(-10, 3, 200)}
]

#Funcion con los nombres de los modelos que aplicamos para poder ponerlos en la tabla
funciones1 = ["ElasticNet_Ridge",        "ElasticNet_Intermedio1", 
              "ElasticNet_Intermedio2",  "ElasticNet_Intermedio3",  
              "ElasticNet_Intermedio4",  "ElasticNet_Intermedio5",
              "ElasticNet_Intermedio6",  "ElasticNet_Lasso"]

results1 = []
for i in range(len(funciones1)):
    train_estimator = GridSearchCV(preprocessing_pipeline2, parameters_to_train1[i], 
                                    scoring='neg_mean_squared_error', cv=5, n_jobs=-1, 
                                    verbose=10)

    train_estimator.fit(x_train, y_train)
    y_predict = train_estimator.predict(x_test)    
        
    results1.append((funciones1[i], r2_score(y_test, y_predict), mean_squared_error(y_test, y_predict), mean_absolute_error(y_test, y_predict), mean_squared_error(y_test, y_predict)**0.5))

#Mostramos la tabla para comparar el desempeño de ElasticNet conforma se acerca o aleja
#a cada tipo de regularizacion y vemos si mejora para la que deberia mejorar
if detener_por_pasos:
    print("------------------------------------------------------------------")
    print("---------------------PULSA PARA MOSTRAR LA TABLA------------------")
    print("------------------------------------------------------------------")
    input()

print("RMSE score  |  R2 score  |  MAE score   |   MSE score     |    Modelo")
for res in results1:
    print(f"{res[4]:.4f}", "\t\t", f"{res[1]:.4f}", "\t\t", f"{res[3]:.4f}", "\t\t", f"{res[2]:.4f}", "\t\t", res[0])



#Aplicamos GSCV de corrido para todos los modelos para obtener cual es el mejor sin 
#la necesidad de la tabla. Con este obtendremos el mejor modelo, parametros y goal
if detener_por_pasos:
    print("------------------------------------------------------------------")
    print("-----------------PULSA PARA CALCULAR EL MEJOR MODELO--------------")
    print("------------------------------------------------------------------")
    input()
#Modelos que vamos a aplicar en la busqueda del mejor. Son los mismos que hasta ahora
#pero no vamos a hacer una tabla ni una grafica por cada uno, solo buscamos el mejor
modelos = [ 
    
            {"model": [MLPRegressor(max_iter=2000)],
                "model__hidden_layer_sizes": [(50,80)],
                "model__activation": ['tanh', 'relu', 'logistic'],
                "model__solver": ['sgd', 'adam'],
                "model__alpha": [0.0001,0.05],
                "model__learning_rate": ['constant','adaptive']},
    
            {"model": [AdaBoostRegressor()], 
                 "model__n_estimators": [10, 50, 100, 500], 
                 "model__learning_rate" : [0.001, 0.01, 0.1]},
            
            {"model": [SVR()], 
                 "model__kernel": ['poly','rbf'], 
                 "model__C":[0.1, 0.2, 0.4, 0.7, 0.9, 1] },
            
            {"model": [HistGradientBoostingRegressor(max_iter  = 250)], 
                 "model__l2_regularization": [0, 1], 
                 "model__max_depth":[1, 3, 5], 
                 "model__learning_rate":[0.001, 0.01, 0.1]},
            
            {"model": [GradientBoostingRegressor(max_features  = 'auto')], 
                 "model__n_estimators":[10, 50, 100, 500],
                 "model__max_depth":[1, 3, 5], 
                 "model__learning_rate":[0.001, 0.01, 0.1]},
            
            {"model": [Ridge(max_iter=2500, tol=10)],
                 "model__alpha": np.logspace(-5, 2, 20)},
            
            {"model": [Lasso(max_iter=2500, tol=10)],
                 "model__alpha": np.logspace(-5, 2, 20)},
            
            {"model": [ElasticNet(max_iter=2000, tol=10)], 
                 "model__l1_ratio": [0, 0.2, 0.4, 0.6, 0.8, 1] , 
                 "model__alpha":np.logspace(-5, 2, 20)},
            
            {"model": [BayesianRidge(n_iter = 2000)]},
            
            {"model": [PassiveAggressiveRegressor(max_iter=2000)]},
            
            {"model": [KNeighborsRegressor(n_jobs=-1)]},
            
            {"model": [SGDRegressor(max_iter = 25000)],
                  "model__loss": ['squared_loss', 'epsilon_insensitive'],
                  "model__penalty": ['l1', 'l2'],
                  "model__learning_rate": ['optimal', 'adaptive']},
            
            {"model": [LinearRegression(n_jobs=-1)]},
            
            {"model": [RandomForestRegressor(n_jobs=-1, random_state = 200)],
                 "model__n_estimators": [20, 100, 200, 300, 400, 500, 600, 1000]},  
            
            
]

#Aplicamos GridSearchCV
best_model = GridSearchCV(preprocessing_pipeline2, modelos,scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=10)

#Ajustamos el modelo elegido como el mejor
best_model.fit(x_train, y_train)

#Obtenemos los parametros con los que mejor resultado ha dado
best_params = best_model.best_params_
best_estimator = best_model.best_estimator_

if detener_por_pasos:
    print("------------------------------------------------------------------")
    print("----------------------MOSTRAR MEJORES PARAMETROS------------------")
    print("------------------------------------------------------------------")
    input()
    
print(best_params)
print("\n\tMEJOR ERROR OBTENIDO EN GSCV: ", -best_model.best_score_)

if detener_por_pasos:
    print("------------------------------------------------------------------")
    print("-------------MOSTRAR GRAFICA DE CURVAS DE APRENDIZAJE-------------")
    print("------------------------------------------------------------------")
    input()

train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(best_estimator, x_train, y_train, cv=5, n_jobs=None, 
                                                                      train_sizes=np.linspace(.1, 1.0, 5), return_times=True)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)

# ESCALABILIDAD. COMPARACION DE LA CANTIDAD DE EJEMPLOS CON EL TIEMPO QUE TARDA. POR LO GENERAL, CUANTOS MAS MODELOS, MAS TIEMPO
plt.grid()
plt.plot(train_sizes, fit_times_mean, 'o-')
plt.fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1)
plt.xlabel("Ejemplos de entrenamiento")
plt.ylabel("Tiempo")
plt.title("ESCALABILIDAD DEL MEJOR MODELO")
plt.show()

# RENDIMIENTO. LA CAPACIDAD DEL MODELO GENERALMENTE CRECE DE FORMA LOGARITMICA
plt.grid()
plt.plot(fit_times_mean, test_scores_mean, 'o-')
plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std,  test_scores_mean + test_scores_std, alpha=0.1)
plt.xlabel("Tiempo")
plt.ylabel("Capacidad")
plt.title("RENDIMIENTO DEL MEJOR MODELO")
plt.show()



################################################################################
################################################################################
################################################################################
#########################EXTRA PARA LA MEMORIA##################################
################################################################################
################################################################################
################################################################################
# if detener_por_pasos:
#   print("------------------------------------------------------------------")
#   print("-------MOSTRAR GRAFICA DE LOS ESTIMADORES DE RANDOM FOREST--------")
#   print("------------------------------------------------------------------")
#   input()

# #Vector que contendra las predicciones reales sobre el conjunto de entrenamiento
# #para compararlas con los resultados obtenidos con RForest
# valores_rforest = []
# #Vector que contendra los errores obtenidos al aplicar cross validation al 
# #modelo de random forest para aplicarlos con los predichos
# valores_crossvalidation = []

# # Diferentes cantidades de estimadores que vamos a probar
# cant_estimadores = [20, 100, 200, 300, 400, 500]

# # Bucle que pruebe el modelo con cada estimador diferente de los que tenemos
# for estim in cant_estimadores:

#     #Establecemos el modelo con los estimadores deseados
#     modelo = RandomForestRegressor(n_estimators = estim, n_jobs=-1, random_state = 200)

#     # Ajustamos el modelo 
#     modelo.fit(x_train, y_train)
#     #Obtenemos las predicciones del modelo en el conjunto de entrenamiento
#     predicciones = modelo.predict(X = x_train)
#     #Calculamos el error entre las predicciones y los datos
#     mse = mean_squared_error(y_true = y_train, y_pred = predicciones, squared = True)
#     #Añadimos los errores para mostrarlos en la grafica
#     valores_rforest.append(mse)
    
#     # Hacemos cross validation que sacara 5 errores diferentes
#     scores = cross_val_score(estimator = modelo, X = x_train, y = y_train,
#                              scoring = 'neg_mean_squared_error', cv = 5,
#                              n_jobs = - 1 )
    
#     # Se hace la media de estos errores, se pasa a positivo y se agrega al vector
#     valores_crossvalidation.append(-scores.mean())

# # Gráfico que representa los errores de los predictores, de cv y el mejor estimador
# fig, ax = plt.subplots(figsize=(6, 3.84))
# ax.plot(cant_estimadores, valores_rforest, label="Error predicciones")
# ax.plot(cant_estimadores, valores_crossvalidation, label="Error cross validation")

# #Pintamos el punto en el que se obtiene el menor error con cv aplicado a randomforest
# ax.plot(cant_estimadores[np.argmin(valores_crossvalidation)], min(valores_crossvalidation),
#         marker='o', color = "k", label="Error mas bajo")

# ax.set_ylabel("Error (MSE)")
# ax.set_xlabel("Cantidad Estimadores")
# ax.set_title("Evolución de errores para obtener estimadores")
# plt.legend();
# plt.show()
# print(f"El estimador con el que se obtiene menor error es: {cant_estimadores[np.argmin(valores_crossvalidation)]}")


#SE APLICA SOBRE EL MEJOR MODELO, POR LO QUE NO HA SIDO USADA HASTA SABER CUAL
# LO ERA SIEMPRE PARA ESTE CONJUNTO DE DATOS
# if detener_por_pasos:
#   print("------------------------------------------------------------------")
#   print("----------MOSTRAR GRAFICA DE ERRORES VS LEARNING RATE-------------")
#   print("------------------------------------------------------------------")
#   input()

# resultados = {}

# # VALORES CON LOS QUE HAREMOS LAS SIMULACIONES
# tasas_prueba = [0.001, 0.01, 0.1]
# cant_estimadores = [10, 50, 100, 500]


# # Sacamos el resultado para cada learning_rate y cada cantidad de estimadores
# #en ambos modelos para pintar como afecta en el error la tasa de aprendizaje
# for learning_rate in tasas_prueba:  #Cada tasa de aprendizaje lleva su propia linea con su comportamiento
    
#     #Vectores en los que se almacenara respectivamente el error de la predicccion
#     #y el error de cross validation
#     valores_gboosting = []
#     valores_crossvalidation = []
    
#     #Cada tasa de aprendizaje prueba con todos los estimadores que vamos a probar
#     #Sera el eje x sobre el que se mueva la linea de esta tasas
#     for n_estimator in cant_estimadores:
    
#         #Establecemos el modelo que vamos a probar
#         modelo = GradientBoostingRegressor(n_estimators = n_estimator, learning_rate = learning_rate,
#                                            random_state = 200)
        
#         #Calculamos el error de prediccion
#         modelo.fit(x_train, y_train)
#         predicciones = modelo.predict(X = x_train)
#         mse = mean_squared_error(y_true = y_train, y_pred = predicciones,squared = True        )
#         valores_gboosting.append(mse)
        
#         #Calculamos el error del modelo con cross validation
#         scores = cross_val_score(estimator = modelo,X = x_train, y = y_train, scoring = 'neg_mean_squared_error', cv = 5, n_jobs = - 1)
#         # Se agregan los scores de cross_val_score() y se pasa a positivo
#         valores_crossvalidation.append(-scores.mean())
    
#     #Introducimos que error hemos obtenido para este learning rate tanto de los predictores como del modelo con cross validation
#     resultados[learning_rate] = {'valores_gboosting': valores_gboosting, 'valores_crossvalidation': valores_crossvalidation}

# print("Valores de predictores:\n ", valores_gboosting)
# print("Valores del modelo:\n ", valores_crossvalidation)


# #Vamos a pintar el grafico de como avanza el error de los predictores y del modelo
# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 3.84))

# #Para cada learning rate sacamos el error de cada estimador de cada "modelo"
# for key, value in resultados.items():
    
#     axs[0].plot(cant_estimadores, value['valores_crossvalidation'], label=f"Learning rate {key}")
#     axs[0].set_ylabel("Error MSE")
#     axs[0].set_xlabel("Cantidad de estimadores")
#     axs[0].set_title("Evolución de crossvalidation segun su learning rate")
    
#     axs[1].plot(cant_estimadores, value['valores_gboosting'], label=f"Learning rate {key}")
#     axs[1].set_ylabel("Error MSE")
#     axs[1].set_xlabel("Cantidad de estimadores")
#     axs[1].set_title("Evolución de predictores segun su learning rate")
#     plt.legend();

# plt.show()



