# import lane_detector  otro archivo que contenga eso si es necesario
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Defininimos función para mostrar imágenes (clase)
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

# --- Leer un video ------------------------------------------------ Código del profesor
cap = cv2.VideoCapture('ruta_1.mp4')                # Abro el video
# Esto lo corremos para ver 
if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
else:
    print("El video se cargó correctamente.")
    
 
# --------------- Buscamos coordenadas de interes   
# Hacemos esto para ver las coordenadas de los puntos que nos interesan con el fin
# de quedarnos con algun polígono que represente la zona de interés
puntos = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Evento de clic izquierdo
        puntos.append((x, y))
        print(f'Coordenadas: x={x}, y={y}')

cap = cv2.VideoCapture('ruta_1.mp4')
ret, frame = cap.read()

if ret:
    cv2.namedWindow('Cuadro')   # Muestra el cuadro en una ventana llamada 'Cuadro'
    cv2.setMouseCallback('Cuadro', mouse_callback) # Espera a que se presione una tecla
    cv2.imshow('Cuadro', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # Cierra la ventana

cap.release() # Libera el recurso de video

# ----- Información del video (código de clase)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # Meta-Información del video
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
fps = int(cap.get(cv2.CAP_PROP_FPS))                # ... pero puede ser útil en otras ocasiones
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   #
ret, frame = cap.read()
cap.release() 

    
# -------- Codigo de clase  (ver si lo dejamos)

while (cap.isOpened()):                                                 # Itero, siempre y cuando el video esté abierto
    ret, frame = cap.read()                                             # Obtengo el frame
    if ret==True:                                                       # ret indica si la lectura fue exitosa (True) o no (False)
        # frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))  # Si el video es muy grande y al usar cv2.imshow() no entra en la pantalla, se lo puede escalar (solo para visualización!)
        cv2.imshow('Frame',frame)                                       # Muestro el frame
        if cv2.waitKey(25) & 0xFF == ord('q'):                          # Corto la repoducción si se presiona la tecla "q"
            break
    else:
        break                                       # Corto la reproducción si ret=False, es decir, si hubo un error o no quedán mas frames.
cap.release()                   # Cierro el video
cv2.destroyAllWindows()         # Destruyo todas las ventanas abiertas
ret, frame = cap.read()  # Lee el primer cuadro

if ret:
    cv2.imshow('Cuadro', frame)  # Muestra el cuadro en una ventana llamada 'Cuadro'
    cv2.waitKey(0)  # Espera a que se presione una tecla
    cv2.destroyAllWindows()  # Cierra la ventana

cap.release()  # Libera el recurso de video

# ----- Pintamos de negro toda el área del video que no nos sirve

# Máscara
mask = np.zeros((height, width), dtype=np.uint8)

# Definimos el trapacio 
height, width = frame.shape[:2]
puntos_2 = np.array([[120, height], [915, height], [560, 330], [400, 330] ], dtype=np.int32)

# Rellenar el triángulo en la máscara
cv2.fillPoly(mask, [puntos_2], 255)

# Aplicar la máscara a la imagen original
result = cv2.bitwise_and(frame, frame, mask=mask)

# Mostrar la imagen original y la imagen resultante
# cv2.imshow('Original', frame)
# cv2.imshow('Cortada en diagonal', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Convierto la imagen a escala de grises
img_gris = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
imshow(img_gris)

# La binarizo
_, img_binarizada = cv2.threshold(img_gris, 130, 255, cv2.THRESH_BINARY)
plt.imshow(img_binarizada, cmap='gray'), plt.show(block=False)

# Canny
edges1 = cv2.Canny(img_binarizada, 0.2*255, 0.60*255)

#Gradiente morfológico
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
f_mg = cv2.morphologyEx(edges1, cv2.MORPH_GRADIENT, kernel)
imshow(f_mg)

Rres = 1 # rho: resolución de la distancia en píxeles
Thetares = np.pi/180 # theta: resolución del ángulo en radianes
Threshold = 50 # threshold: número mínimo de intersecciones para detectar una línea
minLineLength = 100 # minLineLength: longitud mínima de la línea. Líneas más cortas que esto se descartan.
maxLineGap = 50 # maxLineGap: brecha máxima entre segmentos para tratarlos como una sola línea

# Aplicar la transformada de Hough probabilística
lines = cv2.HoughLinesP(f_mg, Rres,Thetares,Threshold,minLineLength,maxLineGap)
# lines = cv2.HoughLinesP(img_binarizada, )
# line_image = np.zeros_like(img)
final = frame.copy()
for line in lines:
    x1, y1, x2, y2 = line[0]  # Obtener los puntos extremos de la línea
    cv2.line(final, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dibujar la línea sobre la imagen original
imshow(final)



## HAY QUE SACAR LA PENDIENTE Y QUE EN CADA FRAME VAYA CALCULANDO LA PENDIENTE Y DIBUJANDO LA LINEA


while (cap.isOpened()):                                                 # Itero, siempre y cuando el video esté abierto
    ret, frame = cap.read()                                             # Obtengo el frame
    if ret==True: 
                                                              # ret indica si la lectura fue exitosa (True) o no (False)
        # frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))  # Si el video es muy grande y al usar cv2.imshow() no entra en la pantalla, se lo puede escalar (solo para visualización!)
        cv2.imshow('Frame',frame)                                       # Muestro el frame
        if cv2.waitKey(25) & 0xFF == ord('q'):                          # Corto la repoducción si se presiona la tecla "q"
            break
    else:
        break                                       # Corto la reproducción si ret=False, es decir, si hubo un error o no quedán mas frames.
cap.release()                   # Cierro el video
cv2.destroyAllWindows()         # Destruyo todas las ventanas abiertas