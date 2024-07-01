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
input_video_path = 'ruta_1.mp4'
output_video_path = 'ruta_1_lineas.mp4'
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

# Procesar cada fotograma
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width = frame.shape[:2]
    
    # Define los puntos del trapecio
    vertices = np.array([[
        (140, height), # punto infeior izq
        (465, 312), # punto superior izq
        (495, 312), # punto superior derecho
        (890, height) # punto ingerior derecho
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
    edges = cv2.Canny(img_b, 0.2*255, 0.60*255)

    # Gradiente morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    f_mg = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

    # Detectar puntos
    w = -1 * np.ones((3, 3))                   
    w[1, 1] = 8                             
    fp = cv2.filter2D(f_mg, cv2.CV_64F, w)
    fpn = abs(fp)
    fpn = np.uint8(fpn)

    # Crear una imagen en color vacía del mismo tamaño que frame
    fpn_colored = np.zeros_like(frame)

    # Crear una máscara de las líneas detectadas
    mask = fpn.astype(np.uint8)

    # Aplicar el color azul a las líneas detectadas
    fpn_colored[mask > 0] = [255, 0, 0]  # Azul

    # Superponer las líneas azules sobre la imagen original
    overlay = cv2.addWeighted(frame, 1, fpn_colored, 1, 0)

    # Escribir el fotograma procesado en el video de salida
    out.write(overlay)

# Liberar los objetos y cerrar los archivos de video
cap.release()
out.release()
cv2.destroyAllWindows()
