# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: JAVIER RAMIREZ PULIDO
"""

import numpy as np
import math as mt
import matplotlib.pyplot as plt

np.random.seed(1)

print('\n')
print('----------------------------------------------------------------')
print('----------------------------------------------------------------')
print('-EJERCICIO 1 : EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS-')
print('----------------------------------------------------------------')
print('----------------------------------------------------------------')

########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
print('\nEjercicio1: Apartado 2:\n') 

#Funcion del ejercicio 1 apartado 2
def E(u,v):
    return  (u**3*np.e**(v-2)-2*v**2*np.e**(-u))**2

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return 2*(u**3*np.e**(v-2)-2*v**2*np.e**(-u))*(3*u**2*np.e**(v-2)+2*v**2*np.e**(-u))
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
    return 2*(u**3*np.e**(v-2)-2*v**2*np.e**(-u))*(u**3*np.e**(v-2)-4*v*np.e**(-u))

#Gradiente de E 
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

#EJERCICIO 1 APARTADO 1: IMPLEMENTAR GRADIENTE DESCENDENTE (PARA EL APARTADO 2)
def gradient_descent(eta, maxIter, error2get, initial_point): 
    contador = 0        #Para saber el numero de iteraciones que han sido necesarias hasta obtener el valor deseado
    seguir = True       #Booleano para parar el bucle cuando encontremos el valor deseado
    w = initial_point   #Valor desde el que partimos la busqueda 

    #Buscamos hasta obtener el valor deseado (que pondra seguir a false) o un numero de iteraciones maximo (por si no lo encuentra, que no cicle)
    while seguir and contador < maxIter: 
        contador += 1   #Contador de las iteraciones 
        w -= eta*gradE(w[0],w[1])   #Ecuacion general del gradiente descendente en regresion
        seguir = np.float64(E(w[0],w[1])) > error2get  #Si las coordenadas no dan un valor de la funcion menor al que queremos, la condicion de seguir sigue siendo verdad
    return w, contador  #Devuelve las coordenadas donde se obtiene el valor (w) y el numero de iteraciones necesarias (contador)


#Tasa de aprendizaje (0.1 para el ejercicio 1 apartado 2 )
eta = 0.1 
#Maximas iteraciones que se permiten si no encuentra el valor
maxIter = 10000000000 
#Valor del que queremos obtener uno menor 
error2get = 1e-14  
#Punto del que partimos a calcular. (1,1) para el Ejercicio1 Apartado 2
initial_point = np.array([1.0,1.0])  
#Almacenamos las coordenadas en las que se obtiene el minimo (w) y el numero de iteraciones necesarias (it) llamando a la funcion con los valores recien declarados
w, it = gradient_descent(eta,maxIter, error2get, initial_point)

#Salida por pantalla de la informacion que nos pide el ejercicio 1 apartado 2
print ('\tb) Numero de iteraciones: ', it) #Iteraciones necesarias para obtener el valor
print ('\tc) Coordenadas obtenidas: (', f"{w[0]:.2f}", ', ', f"{w[1]:.2f}",')') #Coordenadas en las que se da el valor

input("\n--- Pulsar tecla para continuar hacia el Ejercicio1: Apartado3 a)---")
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
print('\nEjercicio1: Apartado 3 a):') #

#Funcion del Ejercicio 1 apartado 3
def F(x,y):
    return  (x+2)**2+2*(y-2)**2+2*np.sin(2*mt.pi*x)*np.sin(2*mt.pi*y)

#Derivada parcial de F con respecto a x
def dFx(x,y):
    return 2*(x+2)+2*np.cos(2*mt.pi*x)*2*mt.pi*np.sin(2*mt.pi*y)
    
#Derivada parcial de F con respecto a y
def dFy(x,y):
    return 4*(y-2)+4*mt.pi*np.sin(2*mt.pi*x)*np.cos(2*mt.pi*y)

#Gradiente de F
def gradF(x,y):
    return np.array([dFx(x,y), dFy(x,y)])

#EJERCICIO 1 APARTADO 1: IMPLEMENTAR GRADIENTE DESCENDENTE (PARA EL APARTADO 3)
def gradient_descentF(eta, maxIter, initial_point):
    #Contador para el numero de iteraciones realizadas
    contador = 0
    #Punto del que partimos a calcular (pasado como parametro)
    w = initial_point 
    #Vector que contendra los valores de la funcion para cada par de coordenadas por las que avanza minimizando  
    minimos = [F(w[0],w[1])] 
    #Vector con las iteraciones (0,1,2...) para representar en la grafica tantos valores como se obtengan (uno por cada iteracion)
    iteraciones = [0] 

    #Mientras no se hagan todas las iteraciones pasadas como parametro
    while contador < maxIter:
        #Ecuacion general del gradiente descendente en regresion
        w -= eta*gradF(w[0],w[1])
        #Añadimos al vector el valor obtenido con las nuevas coordenadas
        minimos.append(F(w[0],w[1])) 
        #avanzamos el contador que lleva el numero de iteraciones realizadas
        contador += 1
        #Añadimos en el vector cada iteracion 
        iteraciones.append(contador)
    minimo_final = F(w[0],w[1])
    #Devuelve las coordenadas del minimo (w) al que se llega en el numero de iteraciones indicando (no tiene que ser el minimo de la funcion, sino al minimo al que se llega) Son las coordenadas de la ultima iteracion
    #Contador contiene el numero de iteraciones que se ha hecho, que va a ser todas las que se ha indicado que haga (no hay otra condicion de parada que hacerlas todas)
    #minimmos es el vector que contiene los valores de la funcion para las coordenadas por las que avanza (valores del eje Y de la grafica)
    #iteraciones es el vector con todas las iteraciones hechas (eje X de la grafica)
    #minimo_final es el valor que se obtiene tras todas las iteraciones que queriamos realizar
    return w, contador, minimos, iteraciones, minimo_final


#EJERCICIO 1 APARTADO 3 A)(eta=0.01) ------------------------------------------------
eta=0.01    #Tasa de acierto
maxIter = 50 #Numero maximo de iteraciones que queremos que haga 
initial_point = np.array([-1.0,1.0]) #Punto de partida del calculo
w1, it1, minimos, ejeX, minimo1= gradient_descentF(eta, maxIter, initial_point) 

print ('\na) Punto de partida: (-1,1)  Valor de eta = 0.01')
print ('\tIteraciones realizadas: ', it1)
print ('\tCoordenadas del valor minimo (ultimas): (', f"{w1[0]:.2f}", ', ', f"{w1[1]:.2f}", ')')
print ('\tValor mínimo (ultimo valor): (', f"{minimo1:.2f}", ')')

#3D ------------------------------------------------
# DISPLAY FIGURE
#GENERACION DE LO NECESARIO PARA UNA GRAFICA EN 3D (CODIGO APORTADO EN LA PLANTILLA DE PRADO)
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = F(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w1[0],w1[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], F(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.3 a)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('F(x,y)')


plt.plot(ejeX,minimos) #GRAFICO EN 3D
plt.xlabel('Iterations')
plt.ylabel('Valores')
plt.title('Ejer1.3 Empezando en (-1,1) eta = 0.01')
plt.show()

#en 2D ------------------------------------------------
plt.plot(ejeX,minimos)
plt.xlabel('Iterations')
plt.ylabel('Valores')
plt.title('Ejer1.3 Empezando en (-1,1) eta = 0.01')
plt.show()


#EJERCICIO 1 APARTADO 3 A)(eta=0.1) ------------------------------------------------
eta=0.1    #Tasa de acierto
maxIter = 50 #Numero maximo de iteraciones que queremos que haga 
initial_point = np.array([-1.0,1.0]) #Punto de partida del calculo
w11, it11, minimos11, ejeX11, minimo11 = gradient_descentF(eta, maxIter, initial_point) 

print ('\n   Punto de partida: (-1,1)  Valor de eta = 0.1')
print ('\tIteraciones realizadas: ', it11)
print ('\tCoordenadas del valor minimo (ultimas): (', f"{w11[0]:.2f}", ', ', f"{w11[1]:.2f}", ')')
print ('\tValor mínimo (ultimo valor): (', f"{minimo11:.2f}", ')')


#3D ------------------------------------------------
# DISPLAY FIGURE
#GENERACION DE LO NECESARIO PARA UNA GRAFICA EN 3D (CODIGO APORTADO EN LA PLANTILLA DE PRADO)
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = F(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1, cstride=1, cmap='jet')
min_point = np.array([w11[0],w11[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], F(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.3 b)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('F(x,y)')


plt.plot(ejeX11,minimos11) #GRAFICO EN 3D
plt.xlabel('Iterations')
plt.ylabel('Valores')
plt.title('Ejer1.3 Empezando en (-1,1) eta = 0.1')
plt.show()


#2D-------------------------------------------------------------
plt.plot(ejeX11,minimos11)
plt.xlabel('Iterations')
plt.ylabel('Valores')
plt.title('Ejer1.3 Empezando en (-1,1) eta = 0.1')
plt.show()

input("\n--- Pulsar tecla para continuar hacia el Ejercicio1:Apartado3 a)---\n")
####################################################
####################################################
####################################################
####################################################
#EJERCICIO 1 APARTADO 3 B) MISMOS EXPERIMENTOS CON OTROS PUNTOS DE PARTIDA Y COMPARANDO LOS RESULTADOS
print('Ejercicio1: Apartado 3 b)\n')

#CON ETA = 0.01 -------------------------------------------
#Iniciando en (-0.5,-0.5) 
eta=0.01
initial_point = np.array([-0.5,-0.5])
w2, it2, valores2, ejeX2, minimo2 = gradient_descentF(eta, maxIter, initial_point)

#Iniciando en (1,1)
initial_point = np.array([1.0,1.0])
w3, it3, valores3, ejeX3, minimo3 = gradient_descentF(eta, maxIter, initial_point)

#Iniciando en (2.1,-2.1)
initial_point = np.array([2.1,-2.1])
w4, it4, valores4, ejeX4, minimo4 = gradient_descentF(eta, maxIter, initial_point)

#Iniciando en (-3,3)
initial_point = np.array([-3.0,3.0])
w5, it5, valores5, ejeX5, minimo5 = gradient_descentF(eta, maxIter, initial_point)

#Iniciando en (-2,2)
initial_point = np.array([-2.0,2.0])
w6, it6, valores6, ejeX6, minimo6 = gradient_descentF(eta, maxIter, initial_point)

#AHORA LO MISMO PERO CON 0.1-------------------------------------------
#Iniciando en (-0.5,-0.5)
eta=0.1
initial_point = np.array([-0.5,-0.5])
w201, it201, valores201, ejeX201, minimo201 = gradient_descentF(eta, maxIter, initial_point)

#Iniciando en (1,1)
initial_point = np.array([1.0,1.0])
w301, it301, valores301, ejeX301, minimo301 = gradient_descentF(eta, maxIter, initial_point)

#Iniciando en (2.1,-2.1)
initial_point = np.array([2.1,-2.1])
w401, it401, valores401, ejeX401, minimo401 = gradient_descentF(eta, maxIter, initial_point)

#Iniciando en (-3,3)
initial_point = np.array([-3.0,3.0])
w501, it501, valores501, ejeX501, minimo501 = gradient_descentF(eta, maxIter, initial_point)

#Iniciando en (-2,2)
initial_point = np.array([-2.0,2.0])
w601, it601, valores601, ejeX601, minimo601 = gradient_descentF(eta, maxIter, initial_point)


####################### TABLA COMPARATIVA #######################
print('******************************************************************')
print('***************************TABLA COMPARATIVA**********************')
print('******************************************************************')
print('CON eta = 0.01 ----------')
print('Punto de partida       Coordenadas mínimo          Mínimo \n')
print(' (-0.5,-0.5)        (', f"{w2[0]:.2f}", ', ', f"{w2[1]:.2f}",')              (', f"{minimo2:.2f}", ')')
print(' ( 1.0, 1.0)        (', f"{w3[0]:.2f}", ', ', f"{w3[1]:.2f}",')                (', f"{minimo3:.2f}" ,')')
print(' ( 2.1,-2.1)        (', f"{w4[0]:.2f}", ', ', f"{w4[1]:.2f}",')               (', f"{minimo4:.2f}" ,')')
print(' (-3.0, 3.0)        (', f"{w5[0]:.2f}", ', ', f"{w5[1]:.2f}",')               (', f"{minimo5:.2f}" ,')')
print(' (-2.0, 2.0)        (', f"{w6[0]:.2f}", ', ', f"{w6[1]:.2f}",')               (', f"{minimo6:.2f}" ,')')
print('\nCON eta = 0.1 ----------')
print('Punto de partida       Coordenadas mínimo          Mínimo\n')
print(' (-0.5,-0.5)        (', f"{w201[0]:.2f}", ', ', f"{w201[1]:.2f}",')               (', f"{minimo201:.2f}", ')')
print(' ( 1.0, 1.0)        (', f"{w301[0]:.2f}", ', ', f"{w301[1]:.2f}",')               (', f"{minimo301:.2f}" ,')')
print(' ( 2.1,-2.1)        (', f"{w401[0]:.2f}", ', ', f"{w401[1]:.2f}",')               (', f"{minimo401:.2f}" ,')')
print(' (-3.0, 3.0)        (', f"{w501[0]:.2f}", ', ',f"{w501[1]:.2f}",')               (', f"{minimo501:.2f}" ,')')
print(' (-2.0, 2.0)        (', f"{w601[0]:.2f}", ', ', f"{w601[1]:.2f}",')               (', f"{minimo601:.2f}" ,')')
print('\n\n')

input("\n--- Pulsar tecla para continuar hacia el Ejercicio2: Apartado1---\n")



####################################################
####################################################
####################################################
####################################################
#EJERCICIO 2 APARTADO 1
print('----------------------------------------------------------------')
print('----------------------------------------------------------------')
print('-EJERCICIO 2 : EJERCICIO SOBRE REGRESION LINEAL-----------------')
print('----------------------------------------------------------------')
print('----------------------------------------------------------------')
print('Ejercicio 2 Apartado 1\n')

label5 = 1 #Etiquetas para el valor 5
label1 = -1 #Etiquetas para el valor 1

# Funcion para leer los datos
def readData(file_x, file_y):
    # Leemos los ficheros   
    datax = np.load(file_x)
    datay = np.load(file_y)
    y = []
    x = []  
    # Solo guardamos los datos cuya clase sea la 1 o la 5
    for i in range(0,datay.size):
          if datay[i] == 5 or datay[i] == 1:
              if datay[i] == 5:
                  y.append(label5)
              else:
                  y.append(label1)
        
              x.append(np.array([1, datax[i][0], datax[i][1]]))
            
    x = np.array(x, np.float64)
    y = np.array(y, np.float64)
    
    return x, y

# Funcion para calcular el error
#x: Conjunto de datos
#y: Etiquetas
#w: Punto inicial
def Err(x,y,w):
    #Iteraciones 
    j=0
    #Sumatoria que contendra el valor del error 
    error = 0 
    #Obtener el numero de valores que se estan sumando para hacer la media
    N=x.shape[0]
    #Bucle que recorre todo el conjunto de datos 
    for i in x:
        #Ecuacion del error
        error += (np.transpose(w).dot(i)-y[j])**2
        #Aumentamos el contador de las iteraciones
        j+=1
      
    #Dividimos la suma entre el numero total de datos que se han calculado    
    error = error / N
    #Devuelve el error
    return error

# Gradiente Descendente Estocastico
#x: Conjunto de datos
#y: Etiquetas
#eta: Tasa de aprendizaje
#umbral: Valor a partir del cual dejamos de calcular (sirve como condicion de parada)
def sgd(x,y,eta,umbral):
    #Iteraciones
    j = 0
    #Se saca una lista de 32 elementos con valores entre [0,y.size)
    indice_aleatorio = np.random.choice(y.size, 32, replace=False)
    #Sera el conjunto de datos pero solamente con los elementos que han sido introducidos en indice_aleatorio
    mb_x = x[indice_aleatorio]
    #Sera el conjunto de etiquetas correspondiente a cada x[] cogido (corresponde a los valores que han terminado en indice_aleatorio)
    mb_y = y[indice_aleatorio]
    #Creamos el punto inicial como (0,0,0). Se crea como una matriz de 3x1 que es el equivalente a trasponerla [[0][0][0]]
    #np.shape(x[0,:]) es para generar tantos ceros como columnas tenga el conjunto x (para el ejercicio que no es lineal 2.2)
    w  = np.zeros((np.shape(x[0,:])[0],1)) 
    #Bucle mientras el error sea mayor que el umbral o no se llegue a un maximo de iteraciones (200)
    while Err(mb_x,mb_y,w)>umbral and j < 200:
        #Se traspone el vector de las etiquetas de la muestra
        mb_y = mb_y.reshape(-1,1)
        #Se introduce como un vector todos los valores de la ecuacion del gradiente descendente estocastico
        total = (mb_x.dot(w)-mb_y)*mb_x
        #Hace la media de los valores 
        media = np.mean(total, axis = 0)
        media = media.reshape(-1,1)
        #Modifica el valor de w para iterar por la funcion en busqueda de minimos. w contiene coordenadas de la funcion
        w = w-eta*media
        #Aumenta el contador de las iteraciones
        j+=1

    #Pasamos las coordenadas obtenidas (que esta con formato (x,1)) a un listado de valores (1,x)
    list = [np.shape(x[0,:])[0]]
    list = [w[i][0] for i in range(np.shape(x[0,:])[0]) ]


    return list


# Pseudoinversa 
def pseudoinverse(datosEntrenamiento, label ):
    w = np.linalg.pinv(datosEntrenamiento)
    w = np.dot(w,label)
    return w


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

#Declaracion de vectores que contendran las coordenadas de cada valor correspondiente
data_label1 = []
data_label5 = []

#Contador de las iteraciones para dividir todos los datos de x segun las etiquetas
j=0

#Bucle que separa en dos vectores los datos
for i in y:
    if i == -1: #Si la etiqueta es -1, es un 1
        data_label1.append(np.array([x[j][1], x[j][2]]))
    else: #Si la etiqueta es 1, es un 5
        data_label5.append(np.array([x[j][1], x[j][2]]))
    #Aumentamos el contador de las iteraciones
    j+=1

#Pasamos los vectores a vectores que contienen datos flotantes de 64 bits
data_label1 = np.array(data_label1, np.float64)
data_label5 = np.array(data_label5, np.float64)

#llamada a la funcion de la pseudoinversa
wp = pseudoinverse(x,y)
#llamada a la funcion del gradiente descendente estocastico
w = np.array(sgd(x,y,0.1,1e-14 ), np.float64)

#Pintamos la gráfica que muestra los dos ajustes separando los datos
plt.scatter(data_label1[:,0], data_label1[:,1], c='r', s=2)
plt.scatter(data_label5[:,0], data_label5[:,1], c='b', s=2)
m = np.amax(x)
t = np.arange(0.,m+0.5,0.5)
plt.plot(t, -wp[0]/wp[2]-wp[1]/wp[2]*t, 'k', label = 'Pseudoinverse')
plt.plot(t, -w[0]/w[2]-w[1]/w[2]*t, 'green', label = 'SGD')
plt.xlabel('Intensidad')
plt.ylabel('Simetria')
plt.title('PSEUDOINVERSA Y SGD')
plt.legend()
plt.show()

#Llamamos a la funcion del error para evaluar la bondad del resultado con pseudoinversa
print ('\nBondad del resultado para pseudoinversa:')
print ("Ein: ", Err(x,y,wp))
print ("Eout: ", Err(x_test, y_test, wp))
#Llamamos a la funcion del error para evaluar la bondad del resultado con gradiente descendente estocastico
print ('\nBondad del resultado para grad. descendente estocastico:')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar hacia el Ejercicio2:Apartado2 a)---\n")
####################################################
####################################################
####################################################
#####################################################EJERCICIO 2 APARTADO 2 subapartado a)
print('Ejercicio 2 Apartado 2 a)\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
    return np.random.uniform(-size,size,(N,d))

#Obtenemos los valores en N
N = simula_unif(1000,2,1)

#GRAFICA QUE PINTA EL MAPA DE VALORES OBTENIDOS
plt.scatter(N[:,0], N[:,1], c='r', s=2)
plt.xlabel('Intensidad')
plt.ylabel('Simetria')
plt.title('EJERCICIO 2.2 a)')
plt.show()

input("\n--- Pulsar tecla para continuar hacia el Ejercicio2:Apartado2 b)---\n")
####################################################
####################################################
####################################################
####################################################
#EJERCICIO 2 APARTADO 2 subapartado b)
print('Ejercicio 2 Apartado 2 b)\n')
#Funcion que asigna etiquetas a un conjunto de datos
def asigna_etiqueta(x):
    #Vector con las etiquetas correspondiente a los valores de x
    etiquetas = []
    for i in x:
        #Utilizamos la ecuacion que nos dan en el enunciado para asignar las etiquetas
        etiquetas.append(np.sign((i[0]-0.2)**2+i[1]**2-0.6))
        
    #Pasamos los datos del vector 'etiquetas' a datos flotantes de 64 bits
    etiquetas = np.array(etiquetas, np.float64)
    #Generamos 100 numeros entre 0 y el numero de elementos que hay en x (que es el 10% de 1000)
    indice_aleatorio = np.random.randint(0, x.shape[0], size=100)
    #Barajamos los datos para reordenarlos aleatoriamente
    np.random.shuffle(indice_aleatorio)
    #Cogemos esos 100 elementos aleatorios y les cambiamos el signo para agregar ruido
    etiquetas[indice_aleatorio][0] = -etiquetas[indice_aleatorio][0]
    
    #Devuelve el vector de etiquetas
    return etiquetas

#Asignamos etiquetas al conjunto de 1000 datos que esta en N
etiquetas = asigna_etiqueta(N)

#Volvemos a crear vectores que contendran todos los datos que sean del mismo tipo
vector1 = []
vector5 = []

#Mismo fragmento de codigo previamente comentado, con este divimos segun la etiqueta el conjunto de datos en dos vectores diferentes
j = 0
for i in etiquetas:
    if i == -1:
        vector1.append(np.array([N[j][0],N[j][1]]))
    else:
        vector5.append(np.array([N[j][0],N[j][1]]))
    j +=1

#Pasamos los datos de los vectores a datos flotantes de 64 bits
vector1 = np.array(vector1, np.float64)
vector5 = np.array(vector5, np.float64)

#PINTAMOS EL MAPA PERO CON DIFERENTES COLORES PARA CADA CONJUNTO DE DATOS 
plt.scatter(vector1[:,0], vector1[:,1], c='r',s=2)
plt.scatter(vector5[:,0], vector5[:,1], c='b',s=2)
plt.xlabel('Intensidad')
plt.ylabel('Simetria')
plt.title('EJERCICIO 2.2 (B)')
plt.show()

input("\n--- Pulsar tecla para continuar hacia el Ejercicio2:Apartado2 c)---\n")
####################################################
####################################################
####################################################
####################################################
#EJERCICIO 2 APARTADO 2 subapartado c)
print('Ejercicio 2 Apartado 2 c)\n')

#Definimos el vector de caracteristicas propuesto
d = np.ones((N.shape[0],3))#Añadimos un 1 en la primera columna
d[:,1] = N[:,0] #Añadimos al vector x1
d[:,2] = N[:,1] #Añadimos al vector x2
    
#Llamada a la funcion sgd
d1 = sgd(d,etiquetas,0.01,1e-14) 

#Mostramos por pantalla el error obtenido
print ("Ein: ", Err(d,etiquetas,d1))

input("\n--- Pulsar tecla para continuar hacia el Ejercicio2:Apartado2 d)---\n")
####################################################
####################################################
####################################################
####################################################
#EJERCICIO 2 APARTADO 2 subapartado d)
print('Ejercicio 2 Apartado 2 d)\n')
#Numero de iteraciones 
t=0
#Sumatorio de los errores para hacer la media despues
suma_Ein = 0
suma_Eout = 0

#Bucle de 1000 iteraciones
while t<1000:
    
    #TRAIN
    #Se saca un conjunto de datos aleatorios por cada iteracion
    n = simula_unif(1000,2,1)
    #Se le asignan las etiquetas al conjunto aleatorio de datos
    etiquetas = asigna_etiqueta(n)
    
    #Creo la matriz añadiendo 1 en la primera columna q = [[1 ...  ...] [1 ... ...]]
    #Vector de caracteristicas 
    q = np.ones((n.shape[0],3)) 
    q[:,1] = n[:,0]
    q[:,2] = n[:,1]
    #Pasamos los datos a flotantes de 64 bits
    q = np.array(q, np.float64)

    #Llamada a la funcion sgd
    r = sgd(q,etiquetas,0.01,1e-14) 
    
    #TEST
    #Se sacan otros valores externos para el Eout
    u = simula_unif(1000,2,1)
    #Se le asignan etiquetas a este conjunto de datos sacados
    etiq2 = asigna_etiqueta(u)
    
    #Creo la matriz añadiendo 1 en la primera columna q = [[1 ...  ...] [1 ... ...]]
    e = np.ones((n.shape[0],3))
    e[:,1] = u[:,0]
    e[:,2] = u[:,1]
    #Paso los datos a flotantes de 64 bits
    e = np.array(e, np.float64)
    
    #Sumatoria de los Errores
    suma_Ein += Err(q,etiquetas,r)
    suma_Eout += Err(e, etiq2, r)
    #Aumentamos el contador de iteraciones
    t+=1
    
#Errores medios (total de la sumatoria/1000)
Ein_medio = suma_Ein/1000
Eout_medio = suma_Eout/1000

#Mostramos por pantalla los errores
print('Ein medio: ', Ein_medio)
print('Eout medio: ', Eout_medio)

input("\n--- Pulsar tecla para continuar hacia el mismo ejercicio pero con otros valores---\n")


#EJERCICIO 2 APARTADO 2 subapartado a) REPETICION CON OTROS VALORES
####################################################
####################################################
####################################################
####################################################
print('Ejercicio 2 Apartado 2 a)\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
    return np.random.uniform(-size,size,(N,d))

N = simula_unif(1000,2,1)
plt.scatter(N[:,0:1], N[:,1:2], c='r', s=2)
plt.xlabel('Intensidad')
plt.ylabel('Simetria')
plt.title('EJERCICIO 2.2 a)')
plt.show()

input("\n--- Pulsar tecla para continuar hacia el Ejercicio2:Apartado2 b)---\n")
####################################################
####################################################
####################################################
####################################################
#EJERCICIO 2 APARTADO 2 subapartado b)
print('Ejercicio 2 Apartado 2 b)\n')
def asigna_etiqueta(x):
    etiquetas = []
    for i in x:
        etiquetas.append(np.sign((i[0]-0.2)**2+i[1]**2-0.6))
        
    
    etiquetas = np.array(etiquetas, np.float64)
    indice_aleatorio = np.random.randint(0, x.shape[0], size=100)
    np.random.shuffle(indice_aleatorio)
    etiquetas[indice_aleatorio][0] = -etiquetas[indice_aleatorio][0]
    
    return etiquetas

etiquetas = asigna_etiqueta(N)

vector1 = []
vector5 = []

j = 0
for i in etiquetas:
    if i == -1:
        vector1.append(np.array([N[j][0],N[j][1]]))
    else:
        vector5.append(np.array([N[j][0],N[j][1]]))
    j +=1
    
vector1 = np.array(vector1, np.float64)
vector5 = np.array(vector5, np.float64)

plt.scatter(vector1[:,0:1], vector1[:,1:2], c='r',s=2)
plt.scatter(vector5[:,0:1], vector5[:,1:2], c='b',s=2)
plt.xlabel('Intensidad')
plt.ylabel('Simetria')
plt.title('EJERCICIO 2.2 (B)')
plt.show()
input("\n--- Pulsar tecla para continuar hacia el Ejercicio2:Apartado2 c)---\n")


#EJERCICIO 2 APARTADO 2 subapartado c)
print('Ejercicio 2 Apartado 2 c)\n')

#Añadimos un 1 en la primera columna
d = np.ones((N.shape[0],6))
d[:,1] = N[:,0]
d[:,2] = N[:,1]
d[:,3] = N[:,0]*N[:,1]
d[:,4] = N[:,0]**2
d[:,5] = N[:,1]**2
    
#Llamada a la funcion sgd
d1 = sgd(d,etiquetas,0.01,1e-14) ## Llamada a la funcion de gradientes descendente estocastico

print ("Ein: ", Err(d,etiquetas,d1))

input("\n--- Pulsar tecla para continuar hacia el Ejercicio2:Apartado2 d)---\n")
####################################################
####################################################
####################################################
####################################################
#EJERCICIO 2 APARTADO 2 subapartado d)
print('Ejercicio 2 Apartado 2 d)\n')
t=0
suma_Ein = 0
suma_Eout = 0

while t<1000:
    
    #TRAIN
    n = simula_unif(1000,2,1)
    etiquetas = asigna_etiqueta(n)
    
    #Creo la matriz añadiendo 1 en la primera columna q = [[1 ...  ...] [1 ... ...]]

    q = np.ones((n.shape[0],6))
    q[:,1] = n[:,0]
    q[:,2] = n[:,1]
    q[:,3] = n[:,0]*n[:,1]
    q[:,4] = n[:,0]**2
    q[:,5] = n[:,1]**2
    q = np.array(q, np.float64)

    #Llamada a la funcion sgd
    r = sgd(q,etiquetas,0.01,1e-14) ## Llamada a la funcion de gradientes descendente estocastico
    
    #TEST
    u = simula_unif(1000,2,1)
    etiq2 = asigna_etiqueta(u)
    
    #Creo la matriz añadiendo 1 en la primera columna q = [[1 ...  ...] [1 ... ...]]
    e = np.ones((u.shape[0],6))
    e[:,1] = u[:,0]
    e[:,2] = u[:,1]
    e[:,3] = u[:,0]*u[:,1]
    e[:,4] = u[:,0]**2
    e[:,5] = u[:,1]**2
    e = np.array(e, np.float64)
    
    #Sumatoria de los Errores
    suma_Ein += Err(q,etiquetas,r)
    suma_Eout += Err(e, etiq2, r)
    t+=1
    
#Errores medios
Ein_medio = suma_Ein/1000
Eout_medio = suma_Eout/1000

print('Ein medio: ', Ein_medio)
print('Eout medio: ', Eout_medio)

input("\n--- Pulsar tecla para continuar hacia el EjercicioBonus---\n")





# ####################################################
# ####################################################
# ####################################################
# ####################################################
#EJERCICIO BONUS
print('Ejercicio Bonus\n')

#Segundo derivada de x
def d2Fx(x,y):
    return 2-8*mt.pi**2*np.sin(2*mt.pi*x)*np.sin(2*mt.pi*y)

#segunda derivada de y
def d2Fy(x,y):
    return 4-8*mt.pi**2*np.sin(2*mt.pi*x)*np.sin(2*mt.pi*y)

#Derivada sobre x de la derivada sobre y
def dyx(x,y):
    return 8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)

#Derivada sobre y de la derivada sobre x
def dxy(x,y):
    return 8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)

#Funcion de Newton
#w -> punto inicial
#umb -> umbral para interrumpir las iteraciones
#eta -> Tasa de aprendizaje
#maxIter -> Numero maximo de iteraciones que hacemos si no cruza el umbral 
def metodo_newton(w,umb,eta,maxIter):
    #contador de las iteraciones
    it = 0.0
    #Booleano para detener el bucle. True si pasamos el umbral pasado como parametro
    salir = False
    #Matriz que contiene las iteraciones y el valor de la funcion, util para la grafica
    d = []
    
    #Mientras no superemos el numero maximo de iteraciones o no lleguemos al umbral
    while it<maxIter and salir == False:
        
        #Guardamos el valor de la funcion antes de modificar la w para ver la diferencia entre el nuevo punto y el anterior
        q = F(w[0],w[1])
        
        #Matriz para representación en gráfica
        d.append([it, F(w[0],w[1])])
        
        #Matriz heussiana
        heussiana = np.array([[d2Fx(w[0],w[0]),dxy(w[0],w[1])],[dyx(w[0],w[1]), d2Fy(w[0],w[1])]])
        
        #Coordenadas
        w = w - eta * np.dot(np.linalg.pinv(heussiana),gradF(w[0],w[1]))
        #Condición de salida
        if abs(F(w[0],w[1])-q) < umb:
            salir = True
        
        #Aumentamos el contador de las iteraciones
        it+=1.0
    
    #Devolvemos las ultimas coordenadas y la matriz para la grafica 
    return w, d


####################### BONUS: METODO DE NEWTON 3A ###############################################
#Learning rate 0.01
eta = 0.01 

#Establecemos un punto inicial
initial_point0 = np.array([-1.0,1.0])
#Establecemos un numero maximo de iteraciones
maxIter = 50
#Llamamos la funcion para el punto inicial, un umbral de 0.000001, tasa de 0.01 y 50 iteraciones maximo
w0, d0 = metodo_newton(initial_point0,0.000001,eta,maxIter)

#Sacamos los datos de la matriz para la grafica (cada columna es un eje)
primera_columna = [aux[0] for aux in d0]
segunda_columna = [aux[1] for aux in d0]

#Obtenemos el valor minimo, que es el que se encuentra en las coordenadas devueltas por Newton
minimo1 = F(w0[0],w0[1])

#Mostramos los resultados por pantalla
print ('\na) Punto de partida: (-1,1)  Valor de eta = 0.01')
print ('\tIteraciones realizadas: ', 50)
print ('\tCoordenadas del valor minimo (ultimas): (', f"{w0[0]:.2f}", ', ', f"{w0[1]:.2f}", ')')
print ('\tValor mínimo (ultimo valor): (', f"{minimo1:.2f}", ')')

#Pintamos la grafica
plt.plot(primera_columna,segunda_columna)
plt.xlabel('Iterations')
plt.ylabel('Valores')
plt.title('BONUS CON ETA 0.01')
plt.show()

#Learning rate 0.1
eta = 0.1 

#Establecemos un punto inicial
initial_point0 = np.array([-1.0,1.0])
#Establecemos un numero maximo de iteraciones
maxIter = 50
#Llamamos la funcion para el punto inicial, un umbral de 0.000001, tasa de 0.1 y 50 iteraciones maximo
w0, d0 = metodo_newton(initial_point0,0.000001,eta,maxIter)

#Sacamos los datos de la matriz para la grafica (cada columna es un eje)
primera_columna = [aux[0] for aux in d0]
segunda_columna = [aux[1] for aux in d0]

#Obtenemos el valor minimo, que es el que se encuentra en las coordenadas devueltas por Newton
minimo1 = F(w0[0],w0[1])

#Mostramos los resultados por pantalla
print ('\na) Punto de partida: (-1,1)  Valor de eta = 0.01')
print ('\tIteraciones realizadas: ', 50)
print ('\tCoordenadas del valor minimo (ultimas): (', f"{w0[0]:.2f}", ', ', f"{w0[1]:.2f}", ')')
print ('\tValor mínimo (ultimo valor): (', f"{minimo1:.2f}", ')')

#Pintamos la grafica
plt.plot(primera_columna,segunda_columna)
plt.xlabel('Iterations')
plt.ylabel('Valores')
plt.title('BONUS CON ETA 0.1')
plt.show()

####################### BONUS: METODO DE NEWTON 3B ###############################################
#Valores iniciales
initial_point1 = np.array([-0.5,-0.5])
initial_point2 = np.array([1.0,1.0])
initial_point3 = np.array([2.1,-2.1])
initial_point4 = np.array([-3.0,3.0])
initial_point5 = np.array([-2.0,2.0])


#Llamada al metodo de newton de los distintos valores iniciales
eta=0.01
w1, d1 = metodo_newton(initial_point1,0.000001,eta,maxIter)
w2, d2 = metodo_newton(initial_point2,0.000001,eta,maxIter)
w3, d3 = metodo_newton(initial_point3,0.000001,eta,maxIter)
w4, d4 = metodo_newton(initial_point4,0.000001,eta,maxIter)
w5, d5 = metodo_newton(initial_point5,0.000001,eta,maxIter)

d1 = np.array(d1,np.float64)
d2 = np.array(d2,np.float64)
d3 = np.array(d3,np.float64)
d4 = np.array(d4,np.float64)
d5 = np.array(d5,np.float64)

#Minimos
minimo1 = F(w1[0],w1[1])
minimo2 = F(w2[0],w2[1])
minimo3 = F(w3[0],w3[1])
minimo4 = F(w4[0],w4[1])
minimo5 = F(w5[0],w5[1])

#Llamada al metodo de newton de los distintos valores iniciales
eta=0.1
w11, d11 = metodo_newton(initial_point1,0.000001,eta,maxIter)
w21, d21 = metodo_newton(initial_point2,0.000001,eta,maxIter)
w31, d31 = metodo_newton(initial_point3,0.000001,eta,maxIter)
w41, d41 = metodo_newton(initial_point4,0.000001,eta,maxIter)
w51, d51 = metodo_newton(initial_point5,0.000001,eta,maxIter)

d11 = np.array(d11,np.float64)
d21 = np.array(d21,np.float64)
d31 = np.array(d31,np.float64)
d41 = np.array(d41,np.float64)
d51 = np.array(d51,np.float64)

#Minimos
minimo11 = F(w11[0],w11[1])
minimo21 = F(w21[0],w21[1])
minimo31 = F(w31[0],w31[1])
minimo41 = F(w41[0],w41[1])
minimo51 = F(w51[0],w51[1])


####################### TABLA COMPARATIVA #######################
print('\n******************************************************************')
print('***************************TABLA COMPARATIVA**********************')
print('******************************************************************')
print('CON eta = 0.01 ----------')
print('Punto de partida       Coordenadas mínimo          Mínimo \n')
print(' (-0.5,-0.5)        (', f"{w1[0]:.2f}", ', ', f"{w1[1]:.2f}",')              (', f"{minimo1:.2f}", ')')
print(' ( 1.0, 1.0)        (', f"{w2[0]:.2f}", ', ', f"{w2[1]:.2f}",')                (', f"{minimo2:.2f}" ,')')
print(' ( 2.1,-2.1)        (', f"{w3[0]:.2f}", ', ', f"{w3[1]:.2f}",')               (', f"{minimo3:.2f}" ,')')
print(' (-3.0, 3.0)        (', f"{w4[0]:.2f}", ', ', f"{w4[1]:.2f}",')               (', f"{minimo4:.2f}" ,')')
print(' (-2.0, 2.0)        (', f"{w5[0]:.2f}", ', ', f"{w5[1]:.2f}",')               (', f"{minimo5:.2f}" ,')')
print('\nCON eta = 0.1 ----------')
print('Punto de partida       Coordenadas mínimo          Mínimo\n')
print(' (-0.5,-0.5)        (', f"{w11[0]:.2f}", ', ', f"{w11[1]:.2f}",')               (', f"{minimo11:.2f}", ')')
print(' ( 1.0, 1.0)        (', f"{w21[0]:.2f}", ', ', f"{w21[1]:.2f}",')               (', f"{minimo21:.2f}" ,')')
print(' ( 2.1,-2.1)        (', f"{w31[0]:.2f}", ', ', f"{w31[1]:.2f}",')               (', f"{minimo31:.2f}" ,')')
print(' (-3.0, 3.0)        (', f"{w41[0]:.2f}", ', ',f"{w41[1]:.2f}",')               (', f"{minimo41:.2f}" ,')')
print(' (-2.0, 2.0)        (', f"{w51[0]:.2f}", ', ', f"{w51[1]:.2f}",')               (', f"{minimo51:.2f}" ,')')
print('\n\n')

# def sign(x):
#   if x >= 0:
#       return 1
#   return -1

# def f(x1, x2):
#   return sign(?) 

# #Seguir haciendo el ejercicio...



