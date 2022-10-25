
import torch
import cv2
from PIL import Image
from face_search import searchFace

from facenet_pytorch import MTCNN

#Use GPU if have
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Thiet bi su dung:',device)

useMTCNN = MTCNN(keep_all=True, device=device)

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def specifyFace(image):

	#use MTCNN to detect face
   	boxes, accuracy = useMTCNN.detect(image)

   	if accuracy[0] != None:
   		print('\nPhat hien co', len(accuracy), 'guong mat voi do chinh xac:', accuracy*100)

   		for (x, y, width, height) in boxes:

   			#covert from string to int
   			x, y, width, height = int(x), int(y), int(width), int(height)
   			
   			#draw rectangle around the face detected
   			cv2.rectangle(image, (x, y), (width, height), (255, 255, 255), 1)

   			#covert numpy.ndarray to pil image
   			imagePIL = Image.fromarray(image)
   			#crop the face from the picture
   			imagePIL = imagePIL.crop((x, y, x+width, y+height))

   			cv2.putText(image, searchFace(imagePIL),
                (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (170, 170, 170), 2)

   	else:
   		print('\nKhong phat hien duoc')

   	image = ResizeWithAspectRatio(image, width=640)
   	cv2.imshow('Face Recognition', image)
