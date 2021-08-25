import cv2
import numpy as np
import os
import sys
import math

##########################################################################################################################################################
######################################################## Primer punto ####################################################################################
##########################################################################################################################################################
class thetaFilter:
    def __init__(self,image_gray):
        self.image_gray = image_gray
        image_gray_fft = np.fft.fft2(image_gray)
        self.image_gray_fft_shift = np.fft.fftshift(image_gray_fft)
        image_gray_fft_mag = np.absolute(self.image_gray_fft_shift)
        image_fft_view = np.log(image_gray_fft_mag + 1)
        self.image_fft_view = image_fft_view / np.max(image_fft_view)               # Especto de frecuencias de Fourier  
    
    def set_theta(self,theta,delta_theta):                                          # Método adaptable al ángulo y el cambio del ángulo 
        self.theta = theta
        self.delta_theta = delta_theta
        num_rows, num_cols = (self.image_gray.shape[0], self.image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)                      # Imagen cuadrada, col_iter = row_iter
        
        iteracion = np.zeros((num_rows,num_cols))
        low_pass_mask = np.zeros_like(self.image_gray)
        if (self.theta > 180):                                                      # Simetría de rotación para ángulos superiores a 180
            self.theta = self.theta-180
        # Límites o bandas para la creación de la máscara
        inf_1 = self.theta + 180 - self.delta_theta
        sup_1 = self.theta + 180 + self.delta_theta
        inf_2 = self.theta - self.delta_theta
        sup_2 = self.theta + self.delta_theta
        for j in range(num_rows):
            for i in range(num_cols):
                temp1 = math.degrees(math.atan2(row_iter[i][j]-num_rows/2,col_iter[i][j]-num_rows/2))
                if temp1 < 0:
                    temp1 = 360 + temp1
                iteracion[i,j] = temp1
                if(iteracion[i,j] >= inf_1 and iteracion[i,j]<=sup_1):              # Condición de la máscara para el angulo tetha en el intervalo delta_theta
                    low_pass_mask[i,j] = 1
                if(iteracion[i,j] >= inf_2 and iteracion[i,j]<=sup_2):              # Condición de la máscara para el angulo tetha en el intervalo delta_theta + 180
                    low_pass_mask[i,j] = 1
        self.mask = low_pass_mask
        # cv2.imshow("original image",self.image_gray)                                # Imagen Original (grises)
        # cv2.imshow("Spectral image",self.image_fft_view)
        # cv2.imshow("Filter frequency response", 255 * self.mask)                    # Máscara para transformar
        # cv2.waitKey(0)  
        return self.image_gray_fft_shift ,self.mask                                                        # Máscara usada para el filtrado en la transformada
        
    def filtering(self):  # can also use high or band pass mask
        fft_filtered = self.image_gray_fft_shift * self.mask                        # Filtrado en espectro según mascara de la transformada
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered = np.absolute(image_filtered)                                # Imagen filtrada
        self.image_filtered = image_filtered/np.max(image_filtered)
        cv2.imshow("Original image", self.image_gray)                               # Imagen Original (Grises)
        cv2.imshow("Mask for transformation", 255 * self.mask)                      # Máscara para transformar
        cv2.imshow("Filtered image", self.image_filtered)                           # Imagen Filtrada
        cv2.waitKey(0)
    
##########################################################################################################################################################
# Definición de la ubicación de los archivos
ruta = "C:\\Users\\User\\OneDrive - World Food Programme\\Proyectos\\Maestria\\2021_I\\Imagenes y Video\\Taller2_fourier\\images"           # Path de imagenes
#ruta = "path_images"           # Path de imagenes
files = os.listdir(ruta)                                                                                                                    # Listado de imágenes

#Interacción del usuario para definir theta y delta_theta
print("Insert theta value:")                                    
theta = int(input())                                                                # Input del usuario para el valor del ángulo theta
print("Insert delta_theta value:")
delta_theta = int(input())                                                          # input del usuario para el rango de variación del ángulo theta

# Carga de la imagen y transformación a escala de grises
image = cv2.imread(os.path.join(ruta,files[0]))                                     # Preparación de la imagen
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Transformación en especto de frecuencias
temp = thetaFilter(image_gray)                                                      # Aplicación de la clase para usar los métodos

# Filtro de frecuencias en la imagen según los valores theta y delta_theta
temp.set_theta(theta,delta_theta)                                                   # Método para obtener la máscara adaptable a los ángulos del usuario

# Imagen con filtro en frecuencias aplicado
temp.filtering()                                                                    # Método para filtrar la imagen según la máscara obtenida por los ángulos

##########################################################################################################################################################
####################################################### Segundo Punto  ####################################################################################
##########################################################################################################################################################
# Caso general, banco de datos
theta = (0,45,90,135)                                                               # Lista de ángulos para filtrar
delta_theta = 5  

#################           parte a         #######################
def bank_image(ruta,file):
    path = os.path.join(ruta,file)
    image = cv2.imread(path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    temp = thetaFilter(image_gray)
    temp_mask = np.zeros((image_gray.shape[0],image_gray.shape[1]))
    for k in theta:
        temp_mask = temp.set_theta(k, delta_theta)[1] + temp_mask                 # Iteración de los ángulos
    fft_filtered = temp.set_theta(k, delta_theta)[0] * temp_mask                       # Filtrado en espectro según mascara de la transformada
    image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
    image_filtered = np.absolute(image_filtered)                                  # Imagen filtrada
    image_filtered = image_filtered/np.max(image_filtered)
    return image_gray,temp_mask,image_filtered, file, fft_filtered

#################           parte b         #######################

# El usuario puede elegir qué index del listado files quiere visualizar 
index = 0
bank_image_filtered = bank_image(ruta,files[index])
cv2.imshow("Original_Image %s!" % bank_image_filtered[3], bank_image_filtered[0])                     # Original imagen
cv2.imshow("Mask Filter %s!" % bank_image_filtered[3], bank_image_filtered[1])                        # Máscara creada con los 4 ángulos
cv2.imshow("Filtered image %s!" % bank_image_filtered[3], bank_image_filtered[2])                     # Imagen Filtrada con los 4 ángulos
cv2.waitKey(0)


#################           parte c         #######################

def sintetize_image(ruta):
    image_filtered = np.zeros((300,300))
    for file in files:
        image_filtered = bank_image(ruta, file)[4] + image_filtered
    image_filtered = image_filtered/4
    cv2.imshow("Promedio de las 4 imagenes", image_filtered)

sintetize_image(ruta)


