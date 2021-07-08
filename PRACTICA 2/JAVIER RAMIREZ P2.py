# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: JAVIER RAMIREZ
"""
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")
print("------------------------EJERCICIO 1 APARTADO 1 a)--------------------------------------")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")
# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente
#Simulamos el conjunto de datos con simula_unif
x = simula_unif(50, 2, [-50,50])

#PINTAMOS LA GRAFICA DEL EJERCICIO 1 APARTADO 1 a)
# N = 50, dim = 2, rango = [-50,+50] usando simula_unif(N, dim, rango)
plt.scatter(x[:,0:1], x[:,1:2], c='orange')
plt.axis([-50,50,-50,50])  
plt.title("EJERCICIO 1.1 A)")
plt.xlabel("eje_x")
plt.ylabel("eje_y")
plt.show()


input("\n--- Pulsar tecla para continuar hacia el ejercicio 1 apartado 1 b) ---\n")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")
print("------------------------EJERCICIO 1 APARTADO 1 b)--------------------------------------")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")

#Simulamos el conjunto de datos con simula_gaus
x = simula_gaus(50, 2, np.array([5,7]))

#PINTAMOS LA GRAFICA DEL EJERCICIO 1 APARTADO 1 b)
# N = 50, dim = 2, sigma = [5,7] usando simula_gaus(N, dim, sigma)
plt.scatter(x[:,0:1], x[:,1:2], c='black')
plt.title("EJERCICIO 1.1 B)")
plt.xlabel("eje_x")
plt.ylabel("eje_y")
plt.show()


###############################################################################
###############################################################################
###############################################################################
input("\n--- Pulsar tecla para continuar hacia el ejercicio 1 apartado 2 a) ---\n")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")
print("------------------------EJERCICIO 1 APARTADO 2 a)--------------------------------------")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")

# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)



# Generamos el conjunto de datos con simula_unif()
# N = 100, dim = 2, rango = [-50,+50] usando simula_unif(N, dim, rango)
conjunto_datos = simula_unif(100,2,[-50,50])
# Obtenemos la pendiente y la variable independiente para una recta adaptada al rango en el que se encuentran los datos
#Usar un rango mayor no daria problemas pero es conveniente la adaptacion
pendiente, independiente = simula_recta([-50,50])


#Creamos los vectores necesarios para almacenar el conjunto de datos positivo, negativo y sus respectivas etiquetas
dato_positivo = [] #Conjunto de datos que es positivo con respecto a la recta generada
dato_negativo = [] #Conjunto de datos que es negativo con respecto a la recta generada
etiqueta_positiva1 = [] #Conjunto de etiquetas positivas (+1)
etiqueta_negativa1 = [] #Conjunto de etiquetas negativas (-1)

#PARA CADA DATO GENERADO VAMOS A VER QUE SIGNO TIENE CON RESPECTO A LA RECTA GENERADA
for i in conjunto_datos:
    
    etiqueta_generada = f(i[0], i[1], pendiente, independiente)
    
    #SI GENERAMOS UNA ETIQUETA POSITIVA
    if etiqueta_generada == 1:
        #GUARDARMOS LA ETIQUETA EN EL VECTOR DE ETIQUETAS POSITIVAS
        etiqueta_positiva1.append(etiqueta_generada)
        #GUARDARMOS EL DATO EN EL VECTOR DE DATOS POSITIVOS
        dato_positivo.append(np.array([i[0],i[1]]))
        
    #SI GENERAMOS UNA ETIQUETA NEGATIVA
    else:
        #GUARDARMOS LA ETIQUETA EN EL VECTOR DE ETIQUETAS NEGATIVAS
        dato_negativo.append(np.array([i[0],i[1]]))
        #GUARDARMOS EL DATO EN EL VECTOR DE DATOS NEGATIVOS
        etiqueta_negativa1.append(etiqueta_generada)

#CONVERTIMOS LOS VECTORES EN FLOTANTES DE 64 BITS
dato_positivo = np.array(dato_positivo, np.float64)
dato_negativo = np.array(dato_negativo, np.float64)


#CREAMOS LA RECTA COMO LA UNION DE DOS PUNTOS EN EL EXTREMO DEL RANGO
x = np.array([-50,50])
y = pendiente*x + independiente

#GRAFICO EJERCICIO 1 APARTADO 2 a)
plt.plot(x,y, 'k', label="Recta para etiquetar")
plt.scatter(dato_positivo[:,0:1], dato_positivo[:,1:2], c='b', label="Datos +1")
plt.scatter(dato_negativo[:,0:1], dato_negativo[:,1:2], c='g', label ="Datos -1")
plt.title("EJERCICIO 1 APARTADO 2 a)")
plt.xlabel("eje_x")
plt.ylabel("eje_y")
plt.legend()
plt.show()


###############################################################################
###############################################################################
###############################################################################
input("\n--- Pulsar tecla para continuar hacia el ejercicio 1 apartado 2 b) ---\n")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")
print("------------------------EJERCICIO 1 APARTADO 2 b)--------------------------------------")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")
# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido
etiqueta_positiva2 = np.copy(etiqueta_positiva1)
etiqueta_negativa2 = np.copy(etiqueta_negativa1)


# INTRODUCIMOS RUIDO EN LAS ETIQUETAS DEL 10% DE LOS DATOS PERO EN VECTORES 
diez_por_ciento = round( len(etiqueta_positiva2) * 0.1)
elementos_a_cambiar = np.random.randint(0, len(etiqueta_positiva2)-1, diez_por_ciento) 

for i in range(np.int64(diez_por_ciento)):
        actual_cambiar = elementos_a_cambiar[i]
        if etiqueta_positiva2[actual_cambiar] == 1: 
            etiqueta_positiva2[actual_cambiar] = -1
        else:
            etiqueta_positiva2[actual_cambiar] = 1
        
            
diez_por_ciento = round( len(etiqueta_negativa2) * 0.1)
elementos_a_cambiar = np.random.randint(0, len(etiqueta_negativa2)-1, diez_por_ciento) 


for i in range(np.int64(diez_por_ciento)):
        actual_cambiar = elementos_a_cambiar[i]
        if etiqueta_negativa2[actual_cambiar] == 1: 
            etiqueta_negativa2[actual_cambiar] = -1
        else:
            etiqueta_negativa2[actual_cambiar] = 1



#AGRUPAMOS TANTO LOS DATOS POSITIVOS COMO LOS NEGATIVOS EN UN SOLO VECTOR PARA 
#PARA REAGRUPARLOS POR LAS NUEVAS ETIQUETAS (LAS DE ANTES PERO CON UN 10% MODIFICADO)
datos = np.concatenate((dato_positivo, dato_negativo), axis=0)
etiquetas = np.concatenate((etiqueta_positiva2,etiqueta_negativa2),axis=0)


#REAGRUPAMOS EN DOS VECTORES DE DATOS CON LAS NUEVAS ETIQUETAS
#ESTA REAGRUPACION UNICAMENTE CAMBIA LOS COLORES DE LA GRAFICA PARA SABER CUALES HAN SIDO ALTERADOS
ruido_dato_positivo = []
ruido_dato_negativo = []
contador =0

#PARA CADA DATO REAGRUPAMOS SEGUN LA ETIQUETA QUE TENGA
for i in datos:
    if etiquetas[contador] == 1:
        ruido_dato_positivo.append(np.array([i[0],i[1]]))
    else:
        ruido_dato_negativo.append(np.array([i[0],i[1]]))
    contador+=1
   
    
#CONVERTIMOS LOS VECTORES EN FLOTANTES DE 64 BITS
ruido_dato_positivo = np.array(ruido_dato_positivo, np.float64)
ruido_dato_negativo = np.array(ruido_dato_negativo, np.float64)

datos = np.concatenate((ruido_dato_positivo, ruido_dato_negativo), axis=0)
#GRAFICO EJERCICIO 1 APARTADO 2 b) 
#SIMILAR AL ANTERIOR PERO CON ERROR EN ALGUNAS ETIQUETAS
plt.plot(x,y, 'k', label="Recta para etiquetar")
plt.scatter(ruido_dato_positivo[:,0:1], ruido_dato_positivo[:,1:2], c='b', label="Datos +1")
plt.scatter(ruido_dato_negativo[:,0:1], ruido_dato_negativo[:,1:2], c='g', label="Datos -1")
plt.title("EJERCICIO 1 APARTADO 2 b)")
plt.xlabel("eje_x")
plt.ylabel("eje_y")
plt.legend()
plt.show()



###############################################################################
###############################################################################
###############################################################################
input("\n--- Pulsar tecla para continuar hacia el ejercicio 1 apartado 2 c) ---\n")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")
print("------------------------EJERCICIO 1 APARTADO 2 c)--------------------------------------")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

# DEFINICION DE LA PRIMERA FUNCION  
def funcion1(x):
    return (x[:,0] - 10)**2 + (x[:,1] - 20)**2 - 400
    
# DEFINICION DE LA SEGUNDA FUNCION  
def funcion2(x):
    return 0.5 * (x[:,0] + 10)**2 + (x[:,1] - 20)**2 - 400
    
# DEFINICION DE LA TERCERA FUNCION  
def funcion3(x):
    return 0.5 * (x[:,0] - 10)**2 - (x[:,1] + 20)**2 - 400
    
# DEFINICION DE LA CUARTA FUNCION  
def funcion4(x):
    return x[:,1] - 20 * x[:,0]**2 - 5 * x[:,0] + 3

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()
    

#PINTAMOS EL CONJUNTO DE DATOS CON RUIDO SOBRE LA NUEVA FUNCION 1 
plot_datos_cuad(datos,etiquetas,funcion1,title="PRIMERA FUNCION")

#PINTAMOS EL CONJUNTO DE DATOS CON RUIDO SOBRE LA NUEVA FUNCION 2 
plot_datos_cuad(datos,etiquetas,funcion2,title="SEGUNDA FUNCION")

#PINTAMOS EL CONJUNTO DE DATOS CON RUIDO SOBRE LA NUEVA FUNCION 3 
plot_datos_cuad(datos,etiquetas,funcion3,title="TERCERA FUNCION")

#PINTAMOS EL CONJUNTO DE DATOS CON RUIDO SOBRE LA NUEVA FUNCION 4
plot_datos_cuad(datos,etiquetas,funcion4,title="CUARTA FUNCION")

input("\n--- Pulsar tecla para continuar hacia el ejercicio 2 ---\n")

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#--------------------EJERCICIO 2-----------------------------------------------

def ajusta_PLA(datos, label, max_iter, vini):
    #Contador que lleva el numero de iteraciones
    contador = 0
    #Condicion para parar, se pone a false al no hacer ningun cambio en la recta despues de una epoca 
    seguir = True
    #Guardamos al empezar el w inicial para comparar si tras una epoca este ha cambiado o no
    w_antiguo = np.copy(vini)
    
   
    #Repetimos el proceso un numero maximo de veces o hasta que no encuentre cambios a realizar
    while seguir and contador < max_iter:
        #Avanzamos el contador del numero de iteraciones
        contador += 1

        #Vamos comprobando punto por punto para encontrar los que estan mal clasificados
        for i in range(len(datos)):
            #Si el signo con respecto a la recta actual no coincide con la etiqueta
            if np.sign(np.dot(np.transpose(vini),datos[i])) != label[i]:
                #Actualizamos la recta para corregir el punto incorrecto
                vini = vini + label[i] * datos[i]
               
        # Si tras recorrer todos los puntos la recta no se ha cambiado, no volvemos a dar otra vuelta
        if np.array_equal(w_antiguo, vini):
            seguir = False
        
        # Guardo la recta actual para ver si en la siguiente iteracion hay cambios con respecto a esta
        w_antiguo = vini
    
    #Devolvemos los coeficientes del hiperplano y el numero de iteraciones realizadas para sacar la media
    return -(vini[0]/vini[2])/(vini[0]/vini[1]), -vini[0]/vini[2]  , contador


print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")
print("------------------------EJERCICIO 2 APARTADO A 1)--------------------------------------")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")

# COGEMOS EL CONJUNTO DE DATOS ORIGINAL DEL 1.2.a
sin_ruido_datos = np.concatenate((dato_positivo,dato_negativo), axis=0)
sin_ruido_etiquetas = np.concatenate((etiqueta_positiva1,etiqueta_negativa1),axis=0)

# AGREGAMOS UN 1 AL PRINCIPIO PARA QUE CONTENGA LA MISMA DIMENSION QUE W Y PODER MULTIPLICARLOS
sin_ruido_datos_dimensionado = np.c_[np.ones(sin_ruido_datos.shape[0]), sin_ruido_datos]

#EMPEZAMOS CON EL VECTOR [0,0,0] PARA LA PRIMERA PARTE 2.a.1.a
vini=[0.0,0.0,0.0]

# OBTENEMOS CON LA FUNCION LOS ELEMENTOS DE LA RECTA Y EL NUMERO DE ITERACIONES REALIZADAS
pendiente, independiente, iteraciones_vector_ceros = ajusta_PLA(sin_ruido_datos_dimensionado,sin_ruido_etiquetas,1000,vini)  


#CREAMOS LA RECTA COMO LA UNION DE DOS PUNTOS EN EL EXTREMO DEL RANGO
x = np.array([-50,50])
y = pendiente*x + independiente

# GRAFICA DEL EJERCICIO 2 APARTADO A SUBAPARTADO 1 SECCION A
plt.axis([-50,50,-50,50])  
plt.plot(x,y, 'b--', label="Recta del perceptron")
plt.scatter(dato_positivo[:,0:1], dato_positivo[:,1:2], c='orange', label="Datos +1")
plt.scatter(dato_negativo[:,0:1], dato_negativo[:,1:2], c='pink', label ="Datos -1")
plt.title('EJERCICIO 2.a.1.a (vector ceros)')
plt.xlabel("eje_x")
plt.ylabel("eje_y")
plt.legend()
plt.show()

print("Numero 'medio' de iterationes para el vector 0: ", iteraciones_vector_ceros)

input("\n--- Pulsar tecla para continuar hacia el ejercicio 2 apartado a subapartado 1 B) ---\n")


iterations = []
leyenda = False
for i in range(0,10):
    #AHORA EL VECTOR INICIAL ESTA FORMADO POR NUMEROS ALEATORIOS EN [0,1]
    vini = np.random.uniform(0,1,3)
    # OBTENEMOS CON LA FUNCION LOS ELEMENTOS DE LA RECTA Y EL NUMERO DE ITERACIONES REALIZADAS
    pendiente, independiente, iteraciones_aletarorias = ajusta_PLA(sin_ruido_datos_dimensionado,sin_ruido_etiquetas,1000,vini)
    #VAMOS GUARDANDO EL TOTAL DE LAS ITERACIONES PARA CALCULAR CUAL HA SIDO LA MEDIA EN 10 ITERACIONES
    iterations.append(iteraciones_aletarorias)
    #CREAMOS LA RECTA COMO LA UNION DE DOS PUNTOS EN EL EXTREMO DEL RANGO
    x = np.array([-50,50])
    y = pendiente*x + independiente
    
    #ESTABLECEMOS EL RANGO DE LA GRAFICA PARA LA RECTA ACTUAL OBTENIDA
    plt.axis([-50,50,-50,50])
    
    #SOLO PONEMOS EN LA LEYENDA UNA VEZ QUE SON LAS LINEAS NEGRAS
    if leyenda == False:
        plt.plot(x,y, 'k--',label='Rectas de las iteraciones del perceptron')
        leyenda = True
    #SI YA ESTA PUESTA LA LEYENDA SIMPLEMENTE AÑADIMOS LA RECTA A LA GRAFICA
    else:
        plt.plot(x,y, 'k--')
    
    
# GRAFICA DEL EJERCICIO 2 APARTADO A SUBAPARTADO 1 SECCION B
plt.plot(x,y, 'b--', label="Recta del perceptron")
plt.scatter(dato_positivo[:,0:1], dato_positivo[:,1:2], c='orange', label="Datos +1")
plt.scatter(dato_negativo[:,0:1], dato_negativo[:,1:2], c='pink', label ="Datos -1")
plt.title('EJERCICIO 2.a.1.b (vector aleatorios)')
plt.xlabel("eje_x")
plt.ylabel("eje_y")
plt.legend()
plt.show()    

#MOSTRAMOS LA MEDIA DE LAS ITERACIONES POR PANTALLA
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar hacia el ejercicio 2 apartado a subapartado 2 ---\n")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")
print("------------------------EJERCICIO 2 APARTADO A 2)--------------------------------------")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")
 
# AGREGAMOS UN 1 AL PRINCIPIO PARA QUE CONTENGA LA MISMA DIMENSION QUE W Y PODER MULTIPLICARLOS
con_ruido_datos_dimensionado = np.c_[np.ones(datos.shape[0]), datos]

#EMPEZAMOS CON EL VECTOR [0,0,0] PARA LA PRIMERA PARTE 2.a.1.a
vini=[0.0,0.0,0.0]

# calculo el vector de pesos mediante el algoritmo de perceptron
# calculamos la pendiente y el termino independiente de la recta
# respecto al vector de pesos
pendiente, independiente, iteraciones_vector_ceros = ajusta_PLA(sin_ruido_datos_dimensionado,etiquetas,1000,vini) 

 #CREAMOS LA RECTA COMO LA UNION DE DOS PUNTOS EN EL EXTREMO DEL RANGO
x = np.array([-50,50])
y = pendiente*x + independiente

# GRAFICA DEL EJERCICIO 2 APARTADO A SUBAPARTADO 2 SECCION A
plt.axis([-50,50,-50,50])  
plt.plot(x,y, 'b--', label="Recta del perceptron")
plt.scatter(ruido_dato_positivo[:,0:1], ruido_dato_positivo[:,1:2], c='orange', label="Datos +1")
plt.scatter(ruido_dato_negativo[:,0:1], ruido_dato_negativo[:,1:2], c='pink', label ="Datos -1")
plt.title('EJERCICIO 2.a.2.a (vector ceros)')
plt.xlabel("eje_x")
plt.ylabel("eje_y")
plt.legend()
plt.show()

print("Numero 'medio' de iterationes para el vector 0: ", iteraciones_vector_ceros)

input("\n--- Pulsar tecla para continuar hacia el ejercicio 2 apartado a subapartado 2 B) ---\n")


iterations2 = []
leyenda = False
for i in range(0,10):
    #AHORA EL VECTOR INICIAL ESTA FORMADO POR NUMEROS ALEATORIOS EN [0,1]
    vini = np.random.uniform(0,1,3)
    # OBTENEMOS CON LA FUNCION LOS ELEMENTOS DE LA RECTA Y EL NUMERO DE ITERACIONES REALIZADAS
    pendiente, independiente, iteraciones_aletarorias =  ajusta_PLA(con_ruido_datos_dimensionado,etiquetas,1000,vini)
    #VAMOS GUARDANDO EL TOTAL DE LAS ITERACIONES PARA CALCULAR CUAL HA SIDO LA MEDIA EN 10 ITERACIONES
    iterations2.append(iteraciones_aletarorias)
    #CREAMOS LA RECTA COMO LA UNION DE DOS PUNTOS EN EL EXTREMO DEL RANGO
    x = np.array([-50,50])
    y = pendiente*x + independiente
    
    #ESTABLECEMOS EL RANGO DE LA GRAFICA PARA LA RECTA ACTUAL OBTENIDA
    plt.axis([-50,50,-50,50])
    
    #SOLO PONEMOS EN LA LEYENDA UNA VEZ QUE SON LAS LINEAS NEGRAS
    if leyenda == False:
        plt.plot(x,y, 'k--',label='Rectas de las iteraciones del perceptron')
        leyenda = True
    #SI YA ESTA PUESTA LA LEYENDA SIMPLEMENTE AÑADIMOS LA RECTA A LA GRAFICA
    else:
        plt.plot(x,y, 'k--')
    


# GRAFICA DEL EJERCICIO 2 APARTADO A SUBAPARTADO 2 SECCION B
plt.plot(x,y, 'b--', label="Recta del perceptron")
plt.scatter(ruido_dato_positivo[:,0:1], ruido_dato_positivo[:,1:2], c='orange', label="Datos +1")
plt.scatter(ruido_dato_negativo[:,0:1], ruido_dato_negativo[:,1:2], c='pink', label ="Datos -1")
plt.title('EJERCICIO 2.a.2.b (vector aleatorios)')
plt.xlabel("eje_x")
plt.ylabel("eje_y")
plt.legend()
plt.show()    

print('Valor medio de iteraciones necesario para converger (vector inicial aleatorio): {}'.format(np.mean(np.asarray(iterations2))))


###############################################################################
###############################################################################
###############################################################################
input("\n--- Pulsar tecla para continuar hacia el ejercicio 2 apartado a subapartado 2 ---\n")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")
print("------------------------EJERCICIO 2 APARTADO B-----------------------------------------")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")


# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT
#datos_totales: conjunto total de datos sobre el que se ejecuta Regresion Logistica
#etiquetas_totales: conjunto total con las etiquetas de los datos 'datos' 
#eta: learning rate (tasa de aprendizaje)
#umbral: umbral que debe atravesar la norma de la diferencia de un w y su anterior como condicion de parada
#maxIter: total de pasos por el conjunto de datos que puede realizar el algoritmo (iteraciones maximas)
def sgdRL(datos_totales, etiquetas_totales, eta, umbral, maxIter):
    #Inicializar el vector de pesos con valores 0
    w = [0.0,0.0,0.0]
    #Guardamos el valor de w para cuando este cambie poder calcular la norma de la diferencia de w con su anterior
    w_anterior  = np.copy(w)
    #Generamos un vector que contenga todos los valores entre 0 y el numero de datos que tenemos en total
    index = np.arange(datos_totales.shape[0])
    #Inicializamos el numero de iteraciones realizadas por el conjunto de datos completo
    contador = 0
    #Creamos una condicion de parada que sera verdadero si atravesamos ese valor umbral con la norma de la diferencia entre un w y el anterior
    seguir = True    
    
    #Mientras no se haga el numero maximo de iteraciones ni hayamos encontrado la otra condicion de parada
    while  contador < maxIter and seguir:
        #Avanzamos el contador de las iteraciones por el conjunto de datos
        contador +=1
        #Cogemos el vector index y mezclamos los valores que contiene de forma aleatoria
        index = np.random.permutation(index)
        #Guardamos el valor de w para cuando este cambie poder calcular la norma de la diferencia de w con su anterior
        w_anterior = np.copy(w)        
        
        #Recorremos el vector index mezclado (como maximo hara datos_totales.shape[0] iteraciones)
        for indices in index:
            #Calculo la nueva w
            w -= eta * -(datos_totales[indices] * etiquetas_totales[indices])/(1+np.exp(etiquetas_totales[indices] * np.transpose(w).dot(datos_totales[indices])))
        
        #Si la norma de la diferencia del nuevo w con el anterior es mas pequeña que el umbral establecido, paramos el bucle
        if np.linalg.norm(w_anterior - w) < umbral:
            seguir = False
    
    #Devolvemos tanto el vector de pesos como el contador de las iteraciones realizadas para la media
    return w, contador



#Funcion del error de regresion logistica
def Funcion_error(datos_test, etiquetas_test,w):
    total = 0
    #Recorro el conjunto de datos
    for i in range(len(etiquetas_test)):
        #Ecuacion del error aportada en PRADO  (parte de la sumatoria)
        total += np.log(1 + np.exp(-etiquetas_test[i] * np.dot(np.transpose(w),datos_test[i] )))
    
    #Devolvemos el resultado de la ecuacion completa (lo anterior * 1/N)
    return total / len(datos_test)


# Obtenemos la pendiente y la variable independiente para una recta adaptada al rango en el que se encuentran los datos
#Usar un rango mayor no daria problemas pero es conveniente la adaptacion
pendiente, independiente = simula_recta([0,2])

# Generamos el conjunto de datos con simula_unif()
# N = 100, dim = 2, rango = [0,2] usando simula_unif(N, dim, rango)
x_train = simula_unif(100,2,[0,2])

#PARA CADA DATO REAGRUPAMOS SEGUN LA ETIQUETA QUE TENGA
y_train = []
datos_positivos = []
datos_negativos = []
for i in x_train:
    #Generamos una etiqueta segun la posicion del dato con respecto de la recta generada
    etiqueta = f(i[0], i[1], pendiente, independiente)    
    #Añadimos la etiqueta en el conjunto de etiquetas total
    y_train.append(etiqueta)
    
    #Si la etiqueta generada para este dato es 1
    if etiqueta == 1.0:
        #Añadimos el dato en el vector de los positivos
        datos_positivos.append(np.array([i[0],i[1]]))
    #Si la etiqueta generada para este dato es -1
    if etiqueta == -1.0:
        #Añadimos el dato en el vector de los positivos
        datos_negativos.append(np.array([i[0],i[1]]))

#CONVERTIMOS LOS VECTORES EN FLOTANTES DE 64 BITS
datos_positivos = np.array(datos_positivos, np.float64)
datos_negativos = np.array(datos_negativos, np.float64)    

#Simulamos un conjunto grande de datos test para calcular el error fuera de la muestra
x_test = simula_unif(1000,2,[0,2])

#PARA CADA DATO REAGRUPAMOS SEGUN LA ETIQUETA QUE TENGA
y_test = []
for i in x_test:
    #Generamos una etiqueta segun la posicion del dato con respecto de la recta generada
    etiqueta = f(i[0], i[1], pendiente, independiente)    
    #Añadimos la etiqueta en el conjunto de etiquetas total
    y_test.append(etiqueta)

#Redimensionamos con 1's el vector de datos test para que este tenga 3 elementos y pueda ser multiplicado por w
x_test_dimensionado = np.c_[np.ones(x_test.shape[0]), x_test]   

#Redimensionamos con 1's el vector de datos train para que este tenga 3 elementos y pueda ser multiplicado por w
x_train_dimensionado = np.c_[np.ones(x_train.shape[0]), x_train]

#Utilizamos la funcion de regresion logistica con esta muestra de datos train generada
w, iteraciones = sgdRL(x_train_dimensionado,y_train,0.01,0.01,3000)

#CREAMOS LA RECTA COMO LA UNION DE DOS PUNTOS EN EL EXTREMO DEL RANGO
x = np.array([0,2])
y = pendiente*x + independiente

#Calculamos la nueva recta despues de haber utilizado gradiente descendente con regresion para ajustar
pendiente_ajustada = -(w[0]/w[2])/(w[0]/w[1])
independiente_ajustada = -w[0]/w[2]
y1 = pendiente_ajustada*x + independiente_ajustada

# GRAFICA DEL EJERCICIO 2 APARTADO B EXPERIMENTO
plt.plot(x,y, c="k", label="Recta original")
plt.plot(x,y1, c="r", label="Recta que se alcanza")
plt.axis([0,2,0,2])  
plt.scatter(datos_positivos[:,0:1], datos_positivos[:,1:2], c='orange', label="Datos +1")
plt.scatter(datos_negativos[:,0:1], datos_negativos[:,1:2], c='pink', label ="Datos -1")
plt.xlabel("eje_x")
plt.ylabel("eje_y")
plt.legend()
plt.title('GRAFICA EJERCICIO 2 APARTADO B EXPERIMENTO')
plt.show()

#Error fuera de la muestra
print("Eout: ", Funcion_error(x_test_dimensionado,y_test,w))

input("\n--- Pulsar tecla para continuar a la repeticion del experimento 100 veces ---\n")
    
#Repetimos el experimento el 100 veces
sumatoria_errores = 0   #Variable que contendra la suma de lo errores de los 100 experimentos
iteraciones_totales = 0 #Variable que contendra la suma de las epocas necesarias en los 100 experimentos
for i in range(100):
    #Simulo un nuevo conjunto de 100 datos aleatorios
    datos_train = simula_unif(100,2,[0,2])

    #PARA CADA DATO REAGRUPAMOS SEGUN LA ETIQUETA QUE TENGA
    etiq = []
    for i in datos_train:
        #Generamos una etiqueta segun la posicion del dato con respecto de la recta generada
        etiqueta = f(i[0], i[1], pendiente, independiente)  
        #Añadimos la etiqueta en el conjunto de etiquetas total
        etiq.append(etiqueta)
        
    #Redimensionamos con 1's el vector de datos train para que este tenga 3 elementos y pueda ser multiplicado por w
    datos_train_dimensionado = np.c_[np.ones(datos_train.shape[0]), datos_train]
    
    #Utilizamos la funcion de regresion logistica con esta muestra de datos train generada
    w, iteraciones = sgdRL(datos_train_dimensionado,etiq,0.01,0.01,3000)
    
    #Sumamos las iteraciones necesarias con este conjunto de datos
    iteraciones_totales += iteraciones
    
    #Sumamos el error fuera de la muestra de este conjunto con el del resto
    sumatoria_errores += Funcion_error(x_test_dimensionado,y_test,w)

#Media de los errores de cada muestra
print("Error medio de las 100 ejecuciones:" , sumatoria_errores/100)
#Media de epocas necesarias en cada muestra
print("Epocas medias de las 100 ejecuciones:" , iteraciones_totales/100)


input("\n--- Pulsar tecla para continuar hacia el BONUS ---\n")
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos

#Funcion para calcular el error
def error_bonus(x,y,w):
    #Se cuenta el porcentaje de los datos mal calculados
    return sum(np.sign(np.dot(x,w.T)) != y)/len(y)

#Funcion para la pseudoinversa que es utilizada como modelo de Regresion Lineal 
def pseudoinverse(datos,etiquetas):
    w = np.linalg.pinv(datos)
    w = np.dot(w,etiquetas)
    return w

# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])

print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")
print("------------------------EJERCICIO BONUS APARTADO 2a------------------------------------")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")

#mostramos los datos de entrenamiento 
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

#mostramos los datos de test 
fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar al bonus apartado 2b---\n")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")
print("------------------------EJERCICIO BONUS APARTADO 2b------------------------------------")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")

#LINEAR REGRESSION FOR CLASSIFICATION 
print("\PARA REGRESION LINEAL \n")

#Llamamos a la funcion que tenemos como modelo de regresion lineal, en este caso la pseudoinversa
w = pseudoinverse(x,y)

#Calculamos el error dentro y fuera de la muestra para este modelo de regresion lineal
Error_dentro_pseudoinversa = error_bonus(x,y,w)
Error_fuera_pseudoinversa = error_bonus(x_test,y_test,w)

#Mostramos los datos por pantalla
print("Error dentro de la muestra con el modelo de Regresion Lineal pseudoinversa: ", Error_dentro_pseudoinversa)
print("Error fuera de la muestra con el modelo de Regresion Lineal pseudoinversa: ", Error_fuera_pseudoinversa)


input("\n--- Pulsar tecla para continuar al algoritmo PLA-Pocket---\n")

#POCKET ALGORITHM
print("\n\PARA PLA POCKET\n")
def pla_pocket( datos , etiquetas , vini , maxIter , umbral):
    #Inicializamos el vector de pesos al que se le ha pasado como parametro
    w = vini
    #Guardamos cual es el w con el que se obtuvo el mejor resultado hasta ahora
    mejor_w = vini
    #Guardamos el w anterior para ir comparando con el actual y obtener su norma para saber si paramos el bucle o no
    w_antiguo = vini
    #Iteraciones que llevamos a lo largo de todo el conjunto de datos (epoca)
    contador = 0
    #Inicializamos el valor mas pequeño a uno alto para que desde el primer valor, este se actualice
    menor_error = 9999.0
    #Condicion para parar que se pondra a false cuando obtengamos la norma de la diferencia entre una w y su anterior menor que el umbral
    seguir = True

    #Mientras no se llegue a un numero maximo de iteraciones o a una condicion de parada
    while contador < maxIter and seguir:
        
        #Pasamos por todos los datos del conjunto
        for i in range(len(y)):
            #Si esta mal clasificado
            if np.sign(np.dot(np.transpose(w),x[i])) != y[i]:
                #Modificamos w
                w = w + y[i] * x[i]
        
        #Si el error obtenido ahora es menor que el error almacenado, lo actualizamos y guardamos la w con la que se obtiene como un mejor resultado
        if (error_bonus(x,y,w) < menor_error):
            menor_error = error_bonus(x,y,w)
            mejor_w = w
        
        #Si la norma de la diferencia de un w con su anterior es menor que el umbral
        if np.linalg.norm(w_antiguo - w) < umbral:
            seguir = False
    
        #Avanzamos el numero de iteraciones realizadas        
        contador += 1
        #Actualizamos el valor del w anterior
        w_antiguo = np.copy(w)
     
    #Devolvemos el mejor resultado
    return mejor_w
    
#Llamamos al algoritmo de PLA-Pocket
w1 = pla_pocket(x,y,w,1000,0.01)

#Calculamos el error dentro y fuera de la muestra para este modelo de regresion lineal
Error_dentro = error_bonus(x,y,w1)
Error_fuera = error_bonus(x_test,y_test,w1)

#Mostramos los datos por pantalla
print("Error dentro de la muestra con PLA-Pocket: ", Error_dentro)
print("Error fuera de la muestra con PLA-Pocket: ", Error_fuera)


input("\n--- Pulsar tecla para continuar y ver las graficas del apartado anterior---\n")

print("GRAFICAS DE CADA ELEMENTO DEL APARTADO ANTERIOR \n")

#GRAFICA DE LOS DATOS DE ENTRENAMIENTO APLICADOS A PSEUDOINVERSA
fig, ax = plt.subplots() 
pendiente = -(w[0]/w[2])/(w[0]/w[1])
independiente = -w[0]/w[2]
x1 = np.array([0,0.5])
y1 = pendiente*x1 + independiente
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='pink', label='Numeros 4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='orange', label='Numeros 8')
plt.plot(x1,y1, "r", label="RECTA")
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='PSEUDOINVERSA CON DATOS DE ENTRENAMIENTO')
ax.set_xlim((0, 1))
plt.legend()
plt.show()


#GRAFICA DE LOS DATOS DE TEST APLICADOS A PSEUDOINVERSA
fig, ax = plt.subplots()
pendiente = -(w[0]/w[2])/(w[0]/w[1])
independiente = -w[0]/w[2]
x1 = np.array([0,0.5])
y1 = pendiente*x1 + independiente
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='pink', label='Numeros 4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='orange', label='Numeros 8')
plt.plot(x1,y1, "r", label="RECTA")
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='PSEUDOINVERSA CON DATOS DE PRUEBA')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

#GRAFICA DE LOS DATOS DE ENTRENAMIENTO APLICADOS AL ALGORITMO DE POCKET
fig, ax = plt.subplots()
pendiente = -(w1[0]/w1[2])/(w1[0]/w1[1])
independiente = -w1[0]/w1[2]
x1 = np.array([0,0.5])
y1 = pendiente*x1 + independiente
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='pink', label='Numeros 4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='orange', label='Numeros 8')
plt.plot(x1,y1, "r", label="RECTA")
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='PLA-POCKET CON DATOS DE ENTRENAMIENTO')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

#GRAFICA DE LOS DATOS DE TEST APLICADOS AL ALGORITMO DE POCKET
fig, ax = plt.subplots()
pendiente = -(w1[0]/w1[2])/(w1[0]/w1[1])
independiente = -w1[0]/w1[2]
x1 = np.array([0,0.5])
y1 = pendiente*x1 + independiente
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='pink', label='Numeros 4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='orange', label='Numeros 8')
plt.plot(x1,y1, "r", label="RECTA")
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='PLA-POCKET CON DATOS DE PRUEBA')
ax.set_xlim((0, 1))
plt.legend()
plt.show()     



input("\n--- Pulsar tecla para continuar a la parte de las cotas---\n")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")
print("------------------------EJERCICIO BONUS APARTADO 2c------------------------------------")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")

print("OBTENCION DE COTAS PARA Ein y Etest\n")

#Funcion que obtiene las cotas sobre el verdadero valor de Eout 
def cota_error(error_para_cota, tamanio, dimension, tolerancia ):
    return error_para_cota + np.sqrt((8/tamanio)*np.log((4*((2*tamanio)**dimension + 1)) / tolerancia))

#Declaramos los valores que usaremos para el calculo de la cota
tolerancia = 0.05 
dimension = 3

#Calculamos la cota sobre el error dentro de la muestra
cota_Ein = cota_error(Error_dentro_pseudoinversa, len(y) , dimension, tolerancia)
#Mostramos por pantalla la cota obtenida
print("Cota del error dentro de la muestra con pseudoinversa: ", cota_Ein)

#Calculamos la cota sobre el error fuera de la muestra
cota_Eout = cota_error(Error_fuera_pseudoinversa, len(y_test) , dimension, tolerancia)
#Mostramos por pantalla la cota obtenida
print("Cota del error fuera de la muestra con pseudoinversa: ", cota_Eout)


#Calculamos la cota sobre el error dentro de la muestra
cota_Ein = cota_error(Error_dentro, len(y) , dimension, tolerancia)
#Mostramos por pantalla la cota obtenida
print("Cota del error dentro de la muestra con PLA-POCKET: ", cota_Ein)

#Calculamos la cota sobre el error fuera de la muestra
cota_Eout = cota_error(Error_fuera, len(y_test) , dimension, tolerancia)
#Mostramos por pantalla la cota obtenida
print("Cota del error fuera de la muestra con PLA-POCKET: ", cota_Eout)

