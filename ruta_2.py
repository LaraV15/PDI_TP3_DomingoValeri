import cv2
import matplotlib.pyplot as plt
import numpy as np

# Definimos una función para mostrar imágenes
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

# Leer el video de entrada
input_video_path = 'ruta_2.mp4'
output_video_path = 'ruta_2_lineas.mp4'
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

# Obtener las propiedades del video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define el codec y crea el objeto VideoWriter para el video de salida
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

"""
# Leer el primer frame para obtener coordenadas de interés
ret, frame = cap.read()
if not ret:
    print("Error: No se pudo leer el frame inicial.")
    cap.release()
    exit()

puntos = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Evento de clic izquierdo
        puntos.append((x, y))
        print(f'Coordenadas: x={x}, y={y}')
        
cv2.namedWindow('Cuadro')   # Muestra el cuadro en una ventana llamada 'Cuadro'
cv2.setMouseCallback('Cuadro', mouse_callback) # Espera a que se presione una tecla
cv2.imshow('Cuadro', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()  # Cierra la ventana

cap.release() # Libera el recurso de video

# Reiniciar el video para leer desde el principio
cap = cv2.VideoCapture(input_video_path)

height, width = frame.shape[:2]

# Define los puntos del trapecio
vertices = np.array([[
    (90, height), # punto inferior izq
    (440, 330), # punto superior izq
    (535, 330), # punto superior derecho
    (860, height) # punto inferior derecho
]], dtype=np.int32)
# vertices = np.array([[
    #     (99, height), # punto inferior izq
    #     (480, 330), # 290 punto superior izq
    #     (500, 330), # 290 punto superior derecho
    #     (850, height) # punto inferior derecho
    # ]], dtype=np.int32)

# Crea una máscara negra del mismo tamaño que el frame
mask = np.zeros((height, width), dtype=np.uint8)

# Rellena el área del trapecio con blanco
cv2.fillPoly(mask, vertices, 255)


# Convierte la máscara a tres canales
mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Aplica la máscara al frame
masked_frame = cv2.bitwise_and(frame, mask_3ch)

# Muestra el frame con la máscara aplicada
cv2.imshow('Frame', masked_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()  # Cierra la ventana
"""

# Variables para almacenar los promedios móviles de las líneas
left_fit_average = []
right_fit_average = []


# Procesar cada fotograma
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width = frame.shape[:2]
    
    # Define los puntos del trapecio
    vertices = np.array([[
        (90, height), # punto inferior izq
        (440, 330), # punto superior izq
        (535, 330), # punto superior derecho
        (860, height) # punto inferior derecho
    ]], dtype=np.int32)
    
    # Crea una máscara negra del mismo tamaño que el frame
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Rellena el área del trapecio con blanco
    cv2.fillPoly(mask, vertices, 255)
    
    # Aplica la máscara al frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Escala de grises
    img_gris = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

    # Binarizamos la imagen anterior para convertirla en una imagen con dos tonos 
    _, img_b = cv2.threshold(img_gris, 130, 255, cv2.THRESH_BINARY)

    # Detección de bordes con Canny
    edges = cv2.Canny(img_b, 0.2*255, 0.50*255)

    # Gradiente morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    f_mg = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

    # Transformada de Hough probabilística para detectar líneas rectas
    Rres = 1 # rho: resolución de la distancia en píxeles
    Thetares = np.pi/180 # theta: resolución del ángulo en radianes
    Threshold = 50 # threshold: número mínimo de intersecciones para detectar una línea
    minLineLength = 100 # minLineLength: longitud mínima de la línea. Líneas más cortas que esto se descartan.
    maxLineGap = 110 # maxLineGap: brecha máxima entre segmentos para tratarlos como una sola línea

    # Aplicamos la transformada de Hough probabilística
    lines = cv2.HoughLinesP(f_mg, Rres, Thetares, Threshold, minLineLength, maxLineGap)
        
    # Creamos una imagen en blanco para dibujar las líneas
    line_image = np.zeros_like(frame)

    # Variables para almacenar las líneas del lado izquierdo y derecho
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Determinar si la línea está en el lado izquierdo o derecho
        if x1 < width / 2 and x2 < width / 2:
            left_lines.append((x1, y1, x2, y2))
        else:
            right_lines.append((x1, y1, x2, y2))

    # Promediamos las líneas del lado derecho para obtener una sola línea
    if right_lines:
        right_lines = np.array(right_lines)
        x_coords = np.append(right_lines[:, 0], right_lines[:, 2])
        y_coords = np.append(right_lines[:, 1], right_lines[:, 3])
        poly_right = np.polyfit(y_coords, x_coords, 1)
        right_fit_average.append(poly_right)
        if len(right_fit_average) > 5:  # Usar un promedio móvil de 5 frames
            right_fit_average.pop(0)

    if right_fit_average:
        poly_right = np.mean(right_fit_average, axis=0)
        y1_right, y2_right = height, int(height * 0.6)
        x1_right = int(np.polyval(poly_right, y1_right))
        x2_right = int(np.polyval(poly_right, y2_right))
        cv2.line(line_image, (x1_right, y1_right), (x2_right, y2_right), (255, 0, 0), 10)  # Azul y más ancho
    
    if left_lines:
        left_lines = np.array(left_lines)
        x_coords = np.append(left_lines[:, 0], left_lines[:, 2])
        y_coords = np.append(left_lines[:, 1], left_lines[:, 3])
        poly_left = np.polyfit(y_coords, x_coords, 1)
        y1_left, y2_left = height, int(height * 0.6)
        x1_left = int(np.polyval(poly_left, y1_left))
        x2_left = int(np.polyval(poly_left, y2_left))
        cv2.line(line_image, (x1_left, y1_left), (x2_left, y2_left), (255, 0, 0), 10)
    # Superponemos las líneas sobre la imagen original
    combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    # Escribimos el fotograma procesado en el video de salida
    out.write(combined_image)

# Liberar los objetos y cerrar los archivos de video
cap.release()
out.release()
cv2.destroyAllWindows()
