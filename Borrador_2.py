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

# Define el codec y crea el objeto VideoWriter
output_video_path = 'ruta_al_video_salida.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # if not ret:
    #     print("Error: No se pudo leer el frame.")
    # else:
    #     print("Frame leído correctamente.")
    height, width = frame.shape[:2]
    
    # Define los puntos del trapecio
    vertices = np.array([[
        (152, height), # punto infeior izq
        (457, 300), # punto superior izq
        (500, 300), # punto superior derecho
        (890, height) # punto ingerior derecho
    ]], dtype=np.int32)
    
    # Crea una máscara negra del mismo tamaño que el frame
    # mask = np.zeros_like(frame)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Rellena el área del trapecio con blanco
    cv2.fillPoly(mask, vertices, 255)
    
    # Aplica la máscara al frame
    masked_frame = cv2.bitwise_and(frame, mask)
    
    # Escala de grises
    img_gris = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

    # Binarizamos la imagen anterior para convertirla en una imagen con dos tonos 
    _, img_b = cv2.threshold(img_gris, 130, 255, cv2.THRESH_BINARY)

    # ----------- Detección de bordes
    # Canny
    edges = cv2.Canny(img_b, 0.2*255, 0.60*255)

    # Gradiente morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    f_mg = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

    w = -1*np.ones((3,3))                   # Definimos el kernel para...
    w[1,1] = 8                              # ... detectar puntos
    fp = cv2.filter2D(f_mg, cv2.CV_64F, w)     # Filtramos
    np.unique(fp)

    fpn = abs(fp)                           # Acondicionamiento: abs()
    # fpn = cv2.convertScaleAbs(fp)             # En este caso, no generaría problemas. Pero ver el ejemplo siguiente.
    np.unique(fpn)                          

    fpn_colored = cv2.cvtColor(fpn.astype(np.uint8), cv2.COLOR_GRAY2BGR)  # Convertir a color
    fpn_colored[np.where((fpn_colored == [255, 255, 255]).all(axis=2))] = [255, 0, 0]  # Cambiar blanco a azul

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


    mask = fpn.astype(np.uint8)  # Asegúrate de que fpn esté en formato uint8

    # Aplicar el color azul a las líneas detectadas
    fpn_colored[mask > 0] = [255, 0, 0]  # Azul

    # Superponer las líneas azules sobre la imagen original
    overlay = cv2.addWeighted(frame, 1, fpn_colored, 1, 0)

    # Escribir el fotograma procesado en el video de salida
    out.write(overlay)

cap.release()
out.release()
cv2.destroyAllWindows()
    
# -------- Codigo de clase  (ver si lo dejamos)
"""
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

