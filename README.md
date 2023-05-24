import cv2

# Carrega o classificador de faces
classificador_faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicia a captura de vídeo da câmera
captura = cv2.VideoCapture(0)

# Loop principal
while True:
    # Lê um quadro de vídeo
    ret, quadro = captura.read()

    # Converte o quadro para tons de cinza
    cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)

    # Detecta as faces no quadro
    faces = classificador_faces.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenha retângulos ao redor das faces detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(quadro, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Exibe o quadro com as faces detectadas
    cv2.imshow('Detecção de Rosto em Tempo Real', quadro)

    # Verifica se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
captura.release()
cv2.destroyAllWindows()
