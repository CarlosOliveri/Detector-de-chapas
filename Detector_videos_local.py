import torch
import cv2
import numpy as np

#Cargamos el Model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                        path = 'C:/Users/ceom1/Desktop/Facultad/9no Semestre/IA/DataSet_Chapas/Detector/best.pt')

# Cargamos el archivo .png, .jpg, .mp4, ...
video = 'C:/Users/ceom1/Desktop/Facultad/9no Semestre/IA/DataSet_Chapas/YOLOv5/ch03_20230901103040.mp4'  # batch of images
cap = cv2.VideoCapture(video)
#DEfinimos area de interes
y1 =250
y2 = 560
x1 = 450
x2 = 1085

# # Inference
#results = model(frame)
#Contador para evitar retardo por falta de GPU
cont = 0
mod = 1190
while(cap.isOpened()):
    #Recuperamos los Frames del video uno a uno
    cont +=1
    print('\rProcesando frame numero:'+str(cont), end="", flush=True)
    
    ret,frame = cap.read()
    img_in = frame[y1:y2,x1:x2]
    # Inference
    
    if cont>=mod:
        #mod = 13
        results = model(img_in)
    #img = cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
    try:
        crop = results.crop(save = False)
        coor = crop[0]['box']
        chapax1 = round(coor[0].detach().cpu().numpy()+0)
        chapay1 = round(coor[1].detach().cpu().numpy()+0)
        chapax2 = round(coor[2].detach().cpu().numpy()+0)
        chapay2 = round(coor[3].detach().cpu().numpy()+0)
        img_out = cv2.rectangle(frame,(x1+chapax1,y1+chapay1),(x1+chapax2,y1+chapay2),(0,255,0),2)
        print("Chapa Detectada")
    except:
        print("Chapa no Detectada")
        print('')
    #Mostrar FPS
    #cv2.imshow('Detector de Chapas', np.squeeze(results.render()))
    try:
        pass
        cv2.imshow('Detector de Chapas', img_out)
    except:
        pass
        cv2.imshow('Detector de Chapas', frame)
        #print('Error')
    #Leemos el teclado
    t = cv2.waitKey(5)
    if t == 27:
        break
cap.release()
cv2.destroyAllWindows()