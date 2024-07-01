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

ret, frame = cap.read()
if not ret:
    print("Error: No se pudo leer el frame.")
else:
    print("Frame leído correctamente.")

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
# ret, frame = cap.read()
# cap.release() 

"""
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
#ret, frame = cap.read()  # Lee el primer cuadro

if ret:
    cv2.imshow('Cuadro', frame)  # Muestra el cuadro en una ventana llamada 'Cuadro'
    cv2.waitKey(0)  # Espera a que se presione una tecla
    cv2.destroyAllWindows()  # Cierra la ventana

cap.release()  # Libera el recurso de video

"""

# Procesamos un frame como una imagen, luego lo extendemos a todo el video
# ----- Pintamos de negro toda el área del video que no nos sirve

# ret, frame = cap.read()

if ret:
    height, width = frame.shape[:2]
    
    # Define los puntos del trapecio
    vertices = np.array([[
        (140, height), # punto infeior izq
        (465, 312), # punto superior izq
        (495, 312), # punto superior derecho
        (890, height) # punto ingerior derecho
    ]], dtype=np.int32)

    
    # Crea una máscara negra del mismo tamaño que el frame
    # mask = np.zeros_like(frame)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Rellena el área del trapecio con blanco
    cv2.fillPoly(mask, vertices, 255)
    
    # Convierte la máscara a tres canales
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Aplica la máscara al frame
    masked_frame = cv2.bitwise_and(frame, mask)
    
    # Muestra el frame con la máscara aplicada
    cv2.imshow('Frame', masked_frame)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

img_gris = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
imshow(img_gris)
# Binarizamos la imagen anterior para convertirla en una imagen con dos tonos 
_, img_b = cv2.threshold(img_gris, 130, 255, cv2.THRESH_BINARY)
plt.imshow(img_b, cmap='gray'), plt.show(block=False)


# ----------- Detección de bordes
# Canny
edges = cv2.Canny(img_b, 0.2*255, 0.60*255)

# # Gradiente morfológico
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
# f_mg = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
# imshow(f_mg)


# --- Pruebas 
"""
w = -1*np.ones((3,3))                   # Definimos el kernel para...
w[1,1] = 8                              # ... detectar puntos
fp = cv2.filter2D(f_mg, cv2.CV_64F, w)     # Filtramos
np.unique(fp)
imshow(fp)

fpn = abs(fp)                           # Acondicionamiento: abs()
# fpn = cv2.convertScaleAbs(fp)             # En este caso, no generaría problemas. Pero ver el ejemplo siguiente.
np.unique(fpn)                          
imshow(fpn)


fpn_colored = cv2.cvtColor(fpn.astype(np.uint8), cv2.COLOR_GRAY2BGR)  # Convertir a color
fpn_colored[np.where((fpn_colored == [255, 255, 255]).all(axis=2))] = [255, 0, 0]  # Cambiar blanco a azul

# Mostrar la imagen en color con líneas azules
plt.imshow(cv2.cvtColor(fpn_colored, cv2.COLOR_BGR2RGB))
plt.show()


# Suponiendo que fpn es la imagen binaria con las líneas detectadas
fpn_colored = np.zeros_like(frame)  # Crear una imagen en color vacía del mismo tamaño que frame

# Crear una máscara de las líneas detectadas
mask = fpn.astype(np.uint8)  # Convertir a formato uint8 si es necesario

# Aplicar el color azul a las líneas detectadas
fpn_colored[:, :, 0] = mask * 255  # Azul
fpn_colored[:, :, 1] = mask * 0    # Verde
fpn_colored[:, :, 2] = mask * 0    # Rojo

# Suponiendo que frame es la imagen original y ya está en color
# Superponer las líneas azules sobre la imagen original
overlay = cv2.addWeighted(frame, 1, fpn_colored, 1, 0)

# Mostrar la imagen final con las líneas azules superpuestas
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.show()


mask = fpn.astype(np.uint8)  # Asegúrate de que fpn esté en formato uint8

# Aplicar el color azul a las líneas detectadas
fpn_colored[mask > 0] = [255, 0, 0]  # Azul

# Superponer las líneas azules sobre la imagen original
overlay = cv2.addWeighted(frame, 1, fpn_colored, 1, 0)

# Mostrar la imagen final con las líneas azules superpuestas
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.show()



"""
# -----

# Transformada de Hough probabilística para detectar lineas rectas en la imagen 
Rres = 1 # rho: resolución de la distancia en píxeles
Thetares = np.pi/180 # theta: resolución del ángulo en radianes
Threshold = 1 # threshold: número mínimo de intersecciones para detectar una línea
minLineLength = 10 # minLineLength: longitud mínima de la línea. Líneas más cortas que esto se descartan.
maxLineGap = 20 # maxLineGap: brecha máxima entre segmentos para tratarlos como una sola línea

# Aplicar la transformada de Hough probabilística
lines = cv2.HoughLinesP(f_mg, Rres,Thetares,Threshold,minLineLength,maxLineGap)
"""
""" lines contendrá las coordenadas de inicio y fin de cada línea detectada en forma de segmentos. 
Cada línea se representa como una lista de cuatro valores (x1, y1, x2, y2), donde (x1, y1) y (x2, y2) 
son los puntos extremos de la línea en la imagen original."""
"""
# Crear una copia de la imagen original para dibujar las líneas
line_image = np.copy(frame) * 0  # Crear una imagen negra del mismo tamaño

if lines is not None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Dibujar cada línea en azul (BGR: 255, 0, 0)

# Combinar la imagen original con las líneas detectadas
combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

# Mostrar la imagen combinada
imshow(combined_image, color_img=True)
    









# ---------
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

# Binarizamos la imagen anterior para convertirla en una imagen con dos tonos 
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
Threshold = 1 # threshold: número mínimo de intersecciones para detectar una línea
minLineLength = 10 # minLineLength: longitud mínima de la línea. Líneas más cortas que esto se descartan.
maxLineGap = 20 # maxLineGap: brecha máxima entre segmentos para tratarlos como una sola línea
# Aplicar la transformada de Hough probabilística
lines = cv2.HoughLinesP(f_mg, Rres,Thetares,Threshold,minLineLength,maxLineGap)
# lines = cv2.HoughLinesP(img_binarizada, )

final = frame.copy()
for line in lines:
    x1, y1, x2, y2 = line[0]  # Obtener los puntos extremos de la línea
    cv2.line(final, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dibujar la línea sobre la imagen original
imshow(final)
# rho: resolución de la distancia en píxeles
# theta: resolución del ángulo en radianes
# threshold: número mínimo de intersecciones para detectar una línea
# minLineLength: longitud mínima de la línea. Líneas más cortas que esto se descartan.
# maxLineGap: brecha máxima entre segmentos para tratarlos como una sola línea

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
"""