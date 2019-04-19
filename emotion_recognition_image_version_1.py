
import cv2
import numpy as np
from keras.preprocessing import image

from keras.models import load_model
model = load_model('model25.h5')
emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

face_cascade = cv2.CascadeClassifier('D:haarcascade_frontalface_default.xml')
img = cv2.imread('Shoh/7.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #transform image to gray scale
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
#print(faces)
    
		
    detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
    detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
		
    img_pixels = image.img_to_array(detected_face)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
		
    img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
		
    predictions = model.predict(img_pixels) #store probabilities of 7 expressions
		
    max_index = np.argmax(predictions[0])
		
    emotion = emotions[max_index]
    print(emotion)

    if emotion == 'Angry':
        color = (255, 0, 0)
    elif emotion == 'Sad':
        color = (0, 0, 255)
    elif emotion == 'Happy':
        color =(255, 255, 0)
    elif emotion == 'Surprise':
        color = (0, 255, 255)
    else:
        color = (0, 255, 0) 

    #draw rectangle to main image	
    cv2.rectangle(img,(x,y),(x+w,y+h),color,5) 
    #write emotion text above rectangle
    
    cv2.rectangle(img,(x,y),(x+w,y-40),color,cv2.FILLED)
    cv2.putText(img, emotion, (int(x)+int(w/4), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
     

cv2.namedWindow('img', cv2.WINDOW_FREERATIO )

cv2.imshow('img',img)
