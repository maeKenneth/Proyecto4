# Proyecto4
En este archivo .readme se describe de forma sintètica el código utilizado para la resolución de los 3 incisos asignados por el profesor. Toda la documentación respecto a las funciones anteriores se puede encontrar en el archivo P4_B45198.py o en su defecto, en el archivo P4_B45198.ipynb explicadas ampliamente por el profesor facilitador del curso:

# %%
# 4.1. - Modulación 16-QAM
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np

# Acá se modifica la función modulador para el 16_QAM:

def modulador_16qam(bits, fc, mpp):
    '''Un método que simula el esquema de 
    modulación digital 16-QAM.

    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpp: Cantidad de muestras por periodo de onda portadora
    :return: Un vector con la señal modulada
    :return: Un valor con la potencia promedio [W]
    :return: La onda portadora c(t)
    :return: La onda cuadrada moduladora (información)
    '''
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits) # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora c(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)
    portadora1 = np.sin(2*np.pi*fc*t_periodo)
    portadora2 = np.cos(2*np.pi*fc*t_periodo)
    
    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp) 
    senal_Tx = np.zeros(t_simulacion.shape)
    moduladora = np.zeros(t_simulacion.shape)  # señal de información
 
 
    # 4. Asignar las formas de onda según los bits para 
    # las señales portadora1 y portadora2 utilizando el modelo
    # 16-QAM. Para ello se edita el ciclo for para que tome en
    # cuenta los cambios en las 3 cifras significativas de la señal
    # de la siguiente manera
    cont = 1
    for i, bit in enumerate(bits):
        
        if (bit[i] == 0) and (bits[i+1] == 0):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora1 * -3
            moduladora[i*mpp : (i+1)*mpp] = -3
            cont = cont +1
        
        if (bit[i] == 0) and (bits[i+1] == 1):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora1 * -1
            moduladora[i*mpp : (i+1)*mpp] = -1
            cont = cont +1

        if (bit[i] == 1) and (bits[i+1] == 1):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora1 * 1
            moduladora[i*mpp : (i+1)*mpp] = 1
            cont = cont +1

        if (bit[i] == 1) and (bits[i+1] == 0):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora1 * 3
            moduladora[i*mpp : (i+1)*mpp] = 3
            cont = cont +1


        
   
        if (bit[i+2] == 0) and (bits[i+3] == 0):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora2 * 3
            moduladora[i*mpp : (i+1)*mpp] = -3
            cont = cont +1

        if (bit[i+2] == 0) and (bits[i+3] == 1):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora2 * 1
            moduladora[i*mpp : (i+1)*mpp] = -1
            cont = cont +1

        if (bit[i+2] == 1) and (bits[i+3] == 1):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora2 * -1
            moduladora[i*mpp : (i+1)*mpp] = -1
            cont = cont +1   

        if (bit[i+2] == 1) and (bits[i+3] == 0):
            senal_Tx[i*mpp : (i+1)*mpp] = portadora2 * -3
            moduladora[i*mpp : (i+1)*mpp] = -3
            cont = cont +1    
            
     # 5. Calcular la potencia promedio de la señal modulada
    Pm = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)
    
    portadoraX =  portadora1 +  portadora2    
  
        
    return senal_Tx, Pm, portadoraX, moduladora

# Se utiliza la función brindada para el canal:

def canal_ruidoso_16qam(senal_Tx, Pm, SNR):
    '''Un bloque que simula un medio de trans-
    misión no ideal (ruidoso) empleando ruido
    AWGN. Pide por parámetro un vector con la
    señal provieniente de un modulador y un
    valor en decibelios para la relación señal
    a ruido.

    :param senal_Tx: El vector del modulador
    :param Pm: Potencia de la señal modulada
    :param SNR: Relación señal-a-ruido en dB
    :return: La señal modulada al dejar el canal
    '''
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)
    
     # Generando ruido auditivo blanco gaussiano
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido

    return senal_Rx

# También se reutiliza el demodulador:

def demodulador_16qam(senal_Rx, portadoraX, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema 16-qam. El criterio
    de demodulación se basa en decodificación por 
    detección de energía.

    :param senal_Rx: La señal recibida del canal
    :param portadora: La onda portadora c(t)
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de bits en transmisión
    N = int(M / mpp)

    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(M)

    # Energía de un período de la portadora
    Es = np.sum(portadoraX**2)

     # Demodulación
    for i in range(N):
        # Producto interno de dos funciones
        producto = senal_Rx[i*mpp : (i+1)*mpp] * portadoraX
        senal_demodulada[i*mpp : (i+1)*mpp] = producto
        Ep = np.sum(producto) 

        # Criterio de decisión por detección de energía
        if Ep > Es*0:
            bits_Rx[i] = 1
        else:
            bits_Rx[i] = 0

    return bits_Rx.astype(int), senal_demodulada

# Se pasa la imagen de bits a RGB:

def bits_a_rgb(bits_Rx, dimensiones):
    '''Un blque que decodifica el los bits
    recuperados en el proceso de demodulación

    :param: Un vector de bits 1 x k 
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)

 # Parámetros
fc = 5000  # frecuencia de la portadora
mpp = 20   # muestras por periodo de la portadora
SNR = 5    # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bit(imagen_Tx)

# 3. Modular la cadena de bits usando el esquema BPSK
senal_Tx, Pm, portadoraX, moduladora = modulador(bits_Tx, fc, mpp)

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx, senal_demodulada = demodulador(senal_Rx, portadora, mpp)

# 6. Se visualiza la imagen recibida 
imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
Fig = plt.figure(figsize=(10,6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores = sum(abs(bits_Tx - bits_Rx))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Recuperado')
Fig.tight_layout()

plt.imshow(imagen_Rx)

# Visualizar el cambio entre las señales
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(14, 7))

# La onda cuadrada moduladora (bits de entrada)
ax1.plot(moduladora[0:600], color='r', lw=2) 
ax1.set_ylabel('$b(t)$')

# La señal modulada por BPSK
ax2.plot(senal_Tx[0:600], color='g', lw=2) 
ax2.set_ylabel('$s(t)$')

# La señal modulada al dejar el canal
ax3.plot(senal_Rx[0:600], color='b', lw=2) 
ax3.set_ylabel('$s(t) + n(t)$')

# La señal demodulada
ax4.plot(senal_demodulada[0:600], color='m', lw=2) 
ax4.set_ylabel('$b^{\prime}(t)$')
ax4.set_xlabel('$t$ / milisegundos')
fig.tight_layout()
plt.show()


# %%
# 4.2. - Estacionariedad y ergodicidad

# 1.Tiempo a muestrear 
t = np.linspace(0, 1, 50)

# 2. Funciones de tiempo x(t), específicamente 8 para este caso:
X = np.empty((8, len(t)))

# 3. Valores de A
A = [-3,-1,1,3]

# 4. Figura 
plt.figure()

# 5. Matriz con variantes de las funciones
for i in A:
    X1 =  i * np.cos(2*(np.pi)*fc*t) +  i * np.sin(2*(np.pi)*fc*t)
    X2 = -i * np.cos(2*(np.pi)*fc*t) +  i * np.sin(2*(np.pi)*fc*t) 
    X3 =  i * np.cos(2*(np.pi)*fc*t) -  i * np.sin(2*(np.pi)*fc*t) 
    X4 = -i * np.cos(2*(np.pi)*fc*t) -  i * np.sin(2*(np.pi)*fc*t) 
    X[i,:] = X1
    X[i+1,:] = X2
    plt.plot(t, X1, lw=2,color='r')
    plt.plot(t, X2, lw=2,color='g') 
    plt.plot(t, X3, lw=2,color='b') 
    plt.plot(t, X4, lw=2,color='k')      

# 6. Obtención del promedio de las realizaciones
Promedio = [np.mean(X[:,i]) for i in range(len(t))]
plt.plot(t, Promedio, lw=8,color='C0',label='Promedio de realizaciones')

# 7. Graficar el resultado teórico del valor esperado
E = np.mean(senal_Tx)*t  # Valor esperado de la señal 
plt.plot(t, E, ':', lw=4,color='r',label='Valor teórico')

# 8. Mostrar las realizaciones, y su promedio calculado y teórico
plt.title('Realizaciones del Proceso Aleatorio $X(t)$')
plt.xlabel('$t$')
plt.ylabel('$x_i(t)$')
plt.legend()
plt.show()

# %% [markdown]
# ---
# 
# ### Respecto a la Estacionariedad y Ergodicidad
# 
# #### Estacionariedad
# ##### Al observar la gráfica se puede notar que tanto el primero como el segundo momento (media y varianza) van a ser los mismos a lo largo de todo el tiempo. Esta propiedad de mantenerse constante caracteriza a un proceso con estacionariedad. En este caso, la transmisión de una señal.A
# 
# ### Ergodicidad
# #### De la figura obtenida también es posible notar que los valores teóricos y el promedio de las señales (seno y coseno) coinciden y son 0, ambos, lo que cumple la condiciòn de ergodicidad.
# 
# ---

# %%
from scipy.fft import fft

# Transformada de Fourier
senal_f = fft(senal_Tx)

# Muestras de la señal
Nm = len(senal_Tx)

# Número de símbolos (198 x 89 x 8 x 3)
Ns = Nm // mpp

# Tiempo del símbolo = periodo de la onda portadora
Tc = 1 / fc

# Tiempo entre muestras (período de muestreo)
Tm = Tc / mpp

# Tiempo de la simulación
T = Ns * Tc

# Espacio de frecuencias
f = np.linspace(0.0, 1.0/(2.0*Tm), Nm//2)

# Finalmente se obtiene la gráfica:

plt.plot(f, 2.0/Nm * np.power(np.abs(senal_f[0:Nm//2]), 2))
plt.xlim(0, 20000)
plt.grid()
plt.show()
