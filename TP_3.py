import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Definimos función para mostrar imágenes
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

# Función para obtener puntos de interés en la imagen, para definir el poligono
def get_points_from_frame(frame):
    puntos = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Evento de clic izquierdo
            puntos.append((x, y))
            print(f'Coordenadas: x={x}, y={y}')

    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', mouse_callback)
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return puntos

# Función para procesar el video y detectar líneas de carril
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return

    # Obtener meta-información del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Definir codec y crear el VideoWriter para guardar el video procesado
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Leer el primer frame para obtener puntos de interés
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer frame.")
        cap.release()
        return

    # Obtener puntos de interés para la máscara
    puntos = get_points_from_frame(frame)

    # Crear máscara de la región de interés
    vertices = np.array([puntos], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 255)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Aplicar la máscara al frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Convertir a escala de grises y binarizar
        img_gris = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        _, img_b = cv2.threshold(img_gris, 130, 255, cv2.THRESH_BINARY)

        # Detección de bordes con Canny
        edges = cv2.Canny(img_b, 0.2 * 255, 0.60 * 255)

        # Transformada de Hough probabilística para detectar líneas
        Rres = 1  # rho: resolución de la distancia en píxeles
        Thetares = np.pi / 180  # theta: resolución del ángulo en radianes
        Threshold = 1  # threshold: número mínimo de intersecciones para detectar una línea
        minLineLength = 10  # longitud mínima de la línea
        maxLineGap = 20  # brecha máxima entre segmentos para tratarlos como una sola línea

        # Aplicar la transformada de Hough probabilística
        lines = cv2.HoughLinesP(edges, Rres, Thetares, Threshold, minLineLength, maxLineGap)

        # Crear una imagen para dibujar las líneas
        line_image = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

        # Combinar la imagen original con las líneas dibujadas
        combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        output.write(combined_image)

    cap.release()
    output.release()
    cv2.destroyAllWindows()
    print("Video procesado y guardado en", output_path)

# Llamada a la función con el video de entrada y la ruta de salida
process_video('ruta_1.mp4', 'output.mp4')
