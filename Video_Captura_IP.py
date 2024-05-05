import cv2
import torch
import numpy as np
import os 
#from paddleocr import PaddleOCR,draw_ocr
#ocr = PaddleOCR(use_angle_cls=True,lang='en')
#Cargamos el Model
#model = torch.hub.load('ultralytics/yolov5', 'custom',path = '/home/coliveri/best.pt')
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path = 'C:/Users/ceom1/Desktop/Facultad/9no Semestre/IA/DataSet_Chapas/Detector/best.pt')
#MASCARA PARA DETECCION DE CUERPOS EN MOVIMINETO
detection = cv2.createBackgroundSubtractorMOG2(history=10000,varThreshold=12)
#ZONA DE DETECCION DE MOVIMIENTOS   
y1_mov = 200
y2_mov = 250
x1_mov = 350
x2_mov = 750
#Area de deteccion de chapas
y1 =250
y2 = 560
x1 = 350
x2 = 1085
#Enlace al servidor RTSP de la marca Dahua
PASS = "123456hik"#input("Ingrese la Contraseña Administador del dispositivo: ")
IP = "186.0.190.133"#input("Ingrese la direccion IP: ") 
CH = "3"#input("Ingrese el numero del canal: ")
#URL = "rtsp://admin:"+PASS+"@"+IP+":554/cam/realmonitor?channel="+CH+"&subtype=0"
URL = "rtsp://admin:"+PASS+"@"+IP+":554/Streaming/Channels/"+CH+"01"
Capture = cv2.VideoCapture(URL)
while True:
    #Recuperamos los frames del video 1 a 1 en cada iteracion
    ret,frame = Capture.read()
    #DIbujar rectangulo en Area de interes
    #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
    #cv2.rectangle(frame,(x1_mov,y1_mov),(x2_mov,y2_mov),(0,255,0),2)
    w = frame.shape[0]
    h = int(w/1.79)
    frame = cv2.resize(frame, (w,h))
    #si ret = false salimos del ciclo while
    if ret == False:
        break
    #Recortamos Area de interes
    #recorte_det = frame[y1:y2,x1:x2]
    #recorte_mov = frame[y1_mov:y2_mov,x1_mov:x2_mov]
    #Aplicamos la mascara de deteccion de moviminto en el recorte
    #mascara = detection.apply(recorte_mov)
    #_,umbral = cv2.threshold(mascara,254,255,cv2.THRESH_BINARY)
    #contornos,_ = cv2.findContours(umbral,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #contornos = sorted(contornos,key=lambda x:cv2.contourArea(x),reverse= True)
    results = model(frame)
    try:
        crop = results.crop(save = False)
        print("Leemos la lectura")
        coor = crop[0]['box']
        print("Hubo lectura")
        chapax1 = round(coor[0].detach().cpu().numpy()+0)
        chapay1 = round(coor[1].detach().cpu().numpy()+0)
        chapax2 = round(coor[2].detach().cpu().numpy()+0)
        chapay2 = round(coor[3].detach().cpu().numpy()+0)
        print("Sacamos los valores")
        #Recortamos el contorno de la placa
        #placa = frame[y1 + chapay1:y1 + chapay2,x1 + chapax1:x1 + chapax2]
        #Dibujamos el contorno de la placa en el frame
        img_out = cv2.rectangle(frame,(chapax1,chapay1),(chapax2,chapay2),(0,255,0),2)
        print("Dibujamos el rectangulo")
        #Mostramos el Frame con la chapa enmarcada
        cv2.imshow('Detector de Chapas', img_out)
        print("mostramos imagen")
    except:
        #Mostramos el Frame Sin Chapa detectada
        print("Chapa no detectada\r")
        cv2.imshow('Detector de Chapas', frame)
    t = cv2.waitKey(1)
    if t == 27:
        break
Capture.release()
cv2.destroyAllWindows()