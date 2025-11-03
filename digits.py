import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model = load_model("my_model.h5")
print("Modelo cargado correctamente")

def procesar_imagen_para_modelo(imagen):
    """
    Procesa la imagen de la cámara para que sea compatible con el modelo
    """
    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    print(f"Imagen gris: {gris.shape}")
    
    # Aplicar suavizado
    suavizada = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # Binarización adaptativa
    binaria = cv2.adaptiveThreshold(suavizada, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    print(f"Encontrados {np.sum(binaria > 0)} pixeles blancos en binaria")
    
    # Operaciones morfológicas para limpiar ruido
    kernel = np.ones((3, 3), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Encontrados {len(contornos)} contornos")
    
    if contornos:
        # Tomar el contorno más grande (presumiblemente el dígito)
        contorno_principal = max(contornos, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contorno_principal)
        print(f"Contorno principal: x={x}, y={y}, w={w}, h={h}")
        
        # Solo procesar si el contorno es suficientemente grande
        if w > 30 and h > 30:
            # Agregar margen
            margen = 20
            x = max(0, x - margen)
            y = max(0, y - margen)
            w = min(binaria.shape[1] - x, w + 2 * margen)
            h = min(binaria.shape[0] - y, h + 2 * margen)
            
            # Recortar el ROI (Region of Interest)
            roi = binaria[y:y+h, x:x+w]
            print(f"ROI recortado: {roi.shape}")
            
            # Redimensionar a 28x28 manteniendo relación de aspecto
            if roi.size > 0:
                h_roi, w_roi = roi.shape
                lado_max = max(h_roi, w_roi)
                
                if lado_max > 0:
                    # Crear imagen cuadrada
                    imagen_cuadrada = np.zeros((lado_max, lado_max), dtype=np.uint8)
                    y_offset = (lado_max - h_roi) // 2
                    x_offset = (lado_max - w_roi) // 2
                    imagen_cuadrada[y_offset:y_offset+h_roi, x_offset:x_offset+w_roi] = roi
                    
                    # Redimensionar a 28x28
                    imagen_28x28 = cv2.resize(imagen_cuadrada, (28, 28), interpolation=cv2.INTER_AREA)
                    print(f"Imagen redimensionada: {imagen_28x28.shape}")
                    
                    # Normalizar
                    imagen_normalizada = imagen_28x28.astype('float32') / 255.0
                    
                    return imagen_normalizada, (x, y, w, h), roi, binaria
    
    return None, None, None, None

def predecir_digito(imagen_normalizada):
    """
    Realiza la predicción del dígito
    """
    # Reformatear para el modelo
    imagen_entrada = imagen_normalizada.reshape(1, 28, 28, 1)
    
    # Realizar predicción
    prediccion = model.predict(imagen_entrada, verbose=0)
    digito_predicho = np.argmax(prediccion)
    confianza = np.max(prediccion)
    
    print(f"Predicción: dígito {digito_predicho} con confianza {confianza:.4f}")
    
    return digito_predicho, confianza, prediccion[0]

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

# Configurar resolución de la cámara
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Sistema de diagnóstico iniciado")
print("Presiona 'd' para mostrar información de depuración")
print("Presiona 'q' para salir")

frame_count = 0

try:
    while True:
        # Capturar frame por frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: No se pudo capturar el frame")
            break
        
        # Voltear horizontalmente para efecto espejo
        frame = cv2.flip(frame, 1)
        
        # Crear una copia para dibujar
        display_frame = frame.copy()
        
        # Procesar la imagen en busca de dígitos
        imagen_procesada, coordenadas, roi, binaria = procesar_imagen_para_modelo(frame)
        
        # Dibujar área de interés recomendada
        cv2.rectangle(display_frame, (100, 100), (400, 400), (0, 255, 0), 2)
        cv2.putText(display_frame, "Zona de deteccion", (100, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Si se encontró un dígito, realizar predicción
        if imagen_procesada is not None and coordenadas is not None:
            x, y, w, h = coordenadas
            
            # Dibujar rectángulo alrededor del dígito detectado
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(display_frame, "DIGITO DETECTADO", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Realizar predicción
            digito, confianza, probabilidades = predecir_digito(imagen_procesada)
            
            # Mostrar ROI procesado en una esquina
            if roi is not None:
                roi_resized = cv2.resize(roi, (100, 100))
                # Convertir a BGR para mostrar
                roi_bgr = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2BGR)
                # Pegar en la esquina superior derecha
                display_frame[10:110, 530:630] = roi_bgr
                cv2.putText(display_frame, "ROI", (530, 125), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Mostrar predicción
            cv2.rectangle(display_frame, (10, 10), (300, 100), (0, 0, 0), -1)
            texto_prediccion = f"Digito: {digito} - Conf: {confianza:.2f}"
            cv2.putText(display_frame, texto_prediccion, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        else:
            # Mostrar que no se detectó nada
            cv2.putText(display_frame, "NO SE DETECTA DIGITO", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, "Muestra un digito en el area verde", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Mostrar el frame
        cv2.imshow('Diagnostico - Reconocimiento de Digitos', display_frame)
        
        # Mostrar también la imagen binaria para diagnóstico
        if binaria is not None:
            binaria_display = cv2.resize(binaria, (320, 240))
            cv2.imshow('Vista Binaria', binaria_display)
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            print(f"\n--- DEBUG Frame {frame_count} ---")
            if imagen_procesada is not None:
                print(f"Imagen procesada shape: {imagen_procesada.shape}")
                print(f"Valores unicos: {np.unique(imagen_procesada)}")
            print("----------------------------")
        
        frame_count += 1

except KeyboardInterrupt:
    print("\nInterrupcion por teclado")
except Exception as e:
    print(f"Error: {e}")
finally:
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Sistema terminado correctamente")