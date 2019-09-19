import cv2
from imutils import face_utils
import dlib
dat="shape_predictor_68_face_landmarks.dat"
d = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dat)
cap = cv2.VideoCapture(0)
while True:
	video =cap.read()[1]
	gray = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
	rectangle= d(gray,0)
	for(i,rect)in enumerate(rectangle):
		s=predictor(gray,rect)
		s=face_utils.shape_to_np(s)
		for(x,y) in s:
		   cv2.circle(video, (x, y), 2, (255, 255, 0), -2)
	cv2.imshow("Output_Video", video)
	k=cv2.waitKey(1)
	if k==27:
		break


