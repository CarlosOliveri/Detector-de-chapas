import torch
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path = 'C:/Users/ceom1/Desktop/Facultad/9no Semestre/IA/DataSet_Chapas/Detector/best.pt')
im = 'C:/Users/ceom1/Desktop/Facultad/9no Semestre/IA/DataSet_Chapas/Detector/Screem_Cam_Fiuna_modificado.png'  # file, Path, PIL.Image, OpenCV, nparray, list
img = cv2.imread(im)
results = model(im)  # inference
crop = results.crop(save=False)  # or .show(), .save(), .crop(), .pandas(), etc.
#results.show()

#results.crop()[0]['box']
coor = crop[0]['box']
chapax1 = round(coor[0].detach().cpu().numpy()+0)
chapay1 = round(coor[1].detach().cpu().numpy()+0)
chapax2 = round(coor[2].detach().cpu().numpy()+0)
chapay2 = round(coor[3].detach().cpu().numpy()+0)
img_out = cv2.rectangle(img,(chapax1,chapay1),(chapax2,chapay2),(0,255,0),2)
cv2.imshow('Detector de Chapas', img_out)
while True:
  t = cv2.waitKey(1)
  if t == 27:
    break
cv2.destroyAllWindows()