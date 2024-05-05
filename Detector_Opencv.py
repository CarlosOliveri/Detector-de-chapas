import cv2
import numpy as np


cap = cv2.VideoCapture('C:/Users/ceom1/Desktop/Facultad/9no Semestre/IA/DataSet_Chapas/YOLOv5/ch03_20230901103040.mp4')

Ctext = ''

#MASCARA PARA DETECCION DE CUERPOS EN MOVIMINETO
detection = cv2.createBackgroundSubtractorMOG2(history=10000,varThreshold=12)

while True:
    ret,frame = cap.read()
    if ret == False:
        break
    
    #Convertimos el frame RGB en un en Blanco y Negro
    #blancoNegro = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.rectangle(frame,(100,100),(200,150),(0,0,0),cv2.FILLED)
    cv2.putText(frame,Ctext[0:7],(130,160),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
     
    #ZONA DE DETECCION DE VEHICULO   
    y1_mov =200
    y2_mov = 250
    x1_mov = 350
    x2_mov = 750
    
    #ZONA DE DETECCION DE CHAPAS
    y1 =250
    y2 = 560
    x1 = 500
    x2 = 1070
    
    #Dibujar rectangulo de fondo negro
    cv2.rectangle(frame,(100,160),(390,200),(0,0,0),cv2.FILLED)
    #Escribir un texto en el rectangulo anterior
    cv2.putText(frame,"Procesando Placa",(105,190),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    #DIbujar rectangulo en Area de interes
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.rectangle(frame,(x1_mov,y1_mov),(x2_mov,y2_mov),(0,255,0),2)
    
    #Recortamos Area de interes
    recorte_det = frame[y1:y2,x1:x2]
    recorte_mov = frame[y1_mov:y2_mov,x1_mov:x2_mov]
    
    #Aplicamos la mascara de deteccion de moviminto en el recorte
    mascara = detection.apply(recorte_mov)
    _,umbral = cv2.threshold(mascara,254,255,cv2.THRESH_BINARY)
    contornos,_ = cv2.findContours(umbral,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #mWB = np.matrix(recorte)
    #_,umbral = cv2.threshold(mWB,80,255,cv2.THRESH_BINARY)
    #contornos,_ = cv2.findContours(umbral,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    contornos = sorted(contornos,key=lambda x:cv2.contourArea(x),reverse= True)
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area>5000:
            print("Vehiculo detectado")
            """x,y,ancho,alto =cv2.boundingReact(contorno)
            
            xpi = x + x1
            ypi = y + y1
            
            xpf = x+ ancho + x1
            ypf = y + alto + y1
            
            cv2.rectangle(blancoNegro,(xpi,ypi),(xpf,ypf),(255,255,0),2)
            
            placa = blancoNegro[ypi:ypf,xpi:xpf]
            
            alp,anp,cp = placa.shape
            
            Mva = np.zeros((alp,anp))
            
            aBp = np.matrix(placa[:,:,0])
            aGp = np.matrix(placa[:,:,1])
            aRp = np.matrix(placa[:,:,2])
            
            for col in range(0,alp):
                for fil in range(0,anp):
                    max = max(aRp[col,fil],aGp[col,fil],aBp[col,fil])
                    Mva[col,fil] = 255,Mva
            
             _,bin = cv2.threshold(Mva,150,255,cv2.THRESH_BINARY)
            bin = bin.reshape(alp,anp)
            bin = Image.fromarray(bin)
            bin = bin.convert("L") 
            
             if alp >= 1 and anp >= 1:
                pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
                
                config ="--psm 1"
                texto = pytesseract.image_to_string(bin,config = config)
                
                if len(texto)>= 7:
                    Ctext = texto """
    #Mostrar Video
    cv2.imshow("mascara",mascara)
    cv2.imshow("Detector de Placas",frame)
    t = cv2.waitKey(1)
    if t == 27:
        break
cap.release()
cv2.destroyAllWindows()
            