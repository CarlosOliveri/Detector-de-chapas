import torch
import cv2
import numpy as np

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                        path = 'C:/Users/ceom1/Desktop/Facultad/9no Semestre/IA/DataSet_Chapas/YOLOv5/best.pt')

# Images
video = 'C:/Users/ceom1/Desktop/Facultad/9no Semestre/IA/DataSet_Chapas/YOLOv5/ch03_20230901103040.mp4'  # batch of images
cap = cv2.VideoCapture(video)

while(cap.isOpened()):
    ret,frame = cap.read()
    # Inference
    results = model(frame)

    #Mostrar FPS
    cv2.imshow('Detector de Chapas', np.zqueeze(results.render()))

    #Leemos el teclado
    t = cv2.waitKey(5)
    if t == 27:
        break
cv2.release()
cv2.destroyAllWindows()