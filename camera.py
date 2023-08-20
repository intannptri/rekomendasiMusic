import numpy as np
import cv2
from PIL import Image
from keras.models import load_model

from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array 
from keras_preprocessing import image
import datetime
from threading import Thread
#from Spotipy import *  
#import time
import pandas as pd


face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
ds_factor=0.6
emotion_model = load_model('./model/model2.h5')
cv2.ocl.setUseOpenCL(False)

emotion_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise' }
music_dist={0:"songs/angry.csv",1:"songs/disgusted.csv ",2:"songs/fearful.csv",3:"songs/happy.csv",4:"songs/neutral.csv",5:"songs/sad.csv",6:"songs/surprised.csv"}

global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1 
show_text=[0]


''' Class for calculating FPS while streaming. Used this to check performance of using another thread for video streaming '''
class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self
	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()
	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1
	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()
	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()


''' Class for using another thread for video streaming to boost performance '''
class WebcamVideoStream:
    	
		def __init__(self, src=1):
			self.stream = cv2.VideoCapture(src,cv2.CAP_DSHOW)
			(self.grabbed, self.frame) = self.stream.read()
			self.stopped = False

		def start(self):
				# start the thread to read frames from the video stream
			Thread(target=self.update, args=()).start()
			return self
			
		def update(self):
			# keep looping infinitely until the thread is stopped
			while True:
				# if the thread indicator variable is set, stop the thread
				if self.stopped:
					return
				# otherwise, read the next frame from the stream
				(self.grabbed, self.frame) = self.stream.read()

		def read(self):
			# return the frame most recently read
			return self.frame
		def stop(self):
			# indicate that the thread should be stopped
			self.stopped = True

''' Class for reading video stream, generating prediction and recommendations '''
class VideoCamera(object):
	
	def get_frame(self):
		global cap1
		global df1
		cap1 = WebcamVideoStream(src=1).start()
		image = cap1.read()
		image=cv2.resize(image,(600,500))
		gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		face_rects=face_cascade.detectMultiScale(gray,1.3,5)
		df1 = pd.read_csv(music_dist[show_text[0]])
		df1 = df1[['Name','Album','Artist']]
		df1 = df1.head(15)
		allfaces=[]
		rects=[]
		for (x,y,w,h) in face_rects:
			cv2.rectangle(image,(x,y),(x+w, y+h),(0,255,0),2)
			rol_gray_frame = gray[y:y + h, x:x + w]
			rol_gray_frame = cv2.resize(rol_gray_frame, (48,48), interpolation=cv2.INTER_AREA)
			allfaces.append(rol_gray_frame)
			rects.append((x, w, y, h))
			rol = rol_gray_frame.astype("float") / 255.0
			rol = img_to_array(rol)
			rol = np.expand_dims(rol, axis=0)
			prediction = emotion_model.predict(rol)[0]
			label = emotion_dict[prediction.argmax()]

			
			show_text[0] = prediction.argmax()
			#print("===========================================",music_dist[show_text[0]],"===========================================")
			#print(df1)
			cv2.putText(image, label, (x+20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
			df1 = music_rec()
			
		global last_frame1
		last_frame1 = image.copy()
		pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)     
		img = Image.fromarray(last_frame1)
		img = np.array(img)
		ret, jpeg = cv2.imencode('.jpg', img)
		return jpeg.tobytes(), df1

def music_rec():
	# print('---------------- Value ------------', music_dist[show_text[0]])
	df = pd.read_csv(music_dist[show_text[0]])
	df = df[['Name','Album','Artist']]
	df = df.head(10)
	return df
