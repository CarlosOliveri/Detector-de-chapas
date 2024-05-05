import cv2
import torch
import numpy as np
import os 
#from paddleocr import PaddleOCR,draw_ocr
#ocr = PaddleOCR(use_angle_cls=True,lang='en')
#Cargamos el Model
model = torch.hub.load('ultralytics/yolov5', 'custom',path = '/home/coliveri/best.pt')

#Directorio contenedor de directorios de videos
path_dirs_videos = "/media/gpu"
#directorios de videos
dir_videos = os.listdir(path_dirs_videos)
#path de cada video
path_video = ""
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
#Ruteado a directorios de videos
for dir in dir_videos:
    path_videos = os.listdir(path_dirs_videos + "/" + dir)
    for video in path_videos:
        path_video = path_dirs_videos + "/" + dir + "/" + video
        extencion = video.split('.')
        if extencion[1] == 'mp4':
        
            #Captura de los frame del video
            capture = cv2.VideoCapture(path_video)
            #Lista de chapas cleidas consecutivas
            lecturas = []
            #Bucle de deteccion de Chapas
            while capture.isOpened():
                #leemos frame a frame
                ret,frame = capture.read()
                #si ret = false salimos del ciclo while
                if ret ==False:
                    break
                #DIbujar rectangulo en Area de interes
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                #cv2.rectangle(frame,(x1_mov,y1_mov),(x2_mov,y2_mov),(0,255,0),2)
                #Recortamos Area de interes
                recorte_det = frame[y1:y2,x1:x2]
                recorte_mov = frame[y1_mov:y2_mov,x1_mov:x2_mov]
                #Aplicamos la mascara de deteccion de moviminto en el recorte
                mascara = detection.apply(recorte_mov)
                _,umbral = cv2.threshold(mascara,254,255,cv2.THRESH_BINARY)
                contornos,_ = cv2.findContours(umbral,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                contornos = sorted(contornos,key=lambda x:cv2.contourArea(x),reverse= True)
                for contorno in contornos:
                    area = cv2.contourArea(contorno)
                    if area>5000:
                        #Aqui hacemos la deteccion
                        results = model(recorte_det)
                        try:
                            crop = results.crop(save = False)
                            coor = crop[0]['box']
                            chapax1 = round(coor[0].detach().cpu().numpy()+0)
                            chapay1 = round(coor[1].detach().cpu().numpy()+0)
                            chapax2 = round(coor[2].detach().cpu().numpy()+0)
                            chapay2 = round(coor[3].detach().cpu().numpy()+0)
                            #Recortamos el contorno de la placa
                            placa = frame[y1 + chapay1:y1 + chapay2,x1 + chapax1:x1 + chapax2]
                            #Dibujamos el contorno de la placa en el frame
                            img_out = cv2.rectangle(frame,(x1+chapax1,y1+chapay1),(x1+chapax2,y1+chapay2),(0,255,0),2)
                            #AQUI HACEMOS LA LECTURA DE LA CHAPA
                            lectura = ocr.ocr(placa,cls=True)
                            #chapa = lectura[0][0][1][0]
                            #print(chapa)
                            chapa = lectura[0][0][1]
                            #font = cv2.FONT_HERSHEY_SIMPLEX
                            #org = (50, 50)
                            #fontScale = 1
                            #color = (0,255, 0)
                            #thickness = 2
                            #Escribimos en la imagen la Chapa detectada
                            #img_out = cv2.putText(img_out, chapa, (x1+chapax2,y1+chapay1), font,fontScale, color, thickness, cv2.LINE_AA)
                            #Mostramos el Frame con la chapa enmarcada
                            #cv2.imshow('Detector de Chapas', img_out)
                            if chapa[1] <= 0.85:
                                lecturas.append(chapa)
                        except:
                            #Mostramos el Frame Sin Chapa detectada
                            cv2.imshow('Detector de Chapas', frame)
                            if len(lecturas) != 0:
                                max = ['',0]
                                for lectura in lecturas:
                                    if lectura[1]> max[0]:
                                        max = lectura 
                                file = open("Registros.txt","a")
                                file.write(max[0]+","+max[1]+","+video)      
                                file.close()
                                lecturas = []
                t = cv2.waitKey(1)
                if t == 27:
                    break
            capture.release()
            cv2.destroyAllWindows()