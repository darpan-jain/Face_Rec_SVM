import cv2
from utils.align_custom import AlignCustom
from utils.face_feature import FaceFeature
from utils.mtcnn_detect import MTCNNDetect
from utils.tf_graph import FaceRecGraph
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import pickle
import sys
import json
import numpy as np
import glob

def main():
	camera_recog()

def camera_recog():
	c1 = cv2.getTickCount()
	for file in glob.glob('./detect/*.jpg'):	# Read images to be used for recognition
		image = cv2.imread(file)
		rects, landmarks = face_detect.detect_face(image,80) 	# Detects all faces in the given images
		aligns = []
		positions = []
		for (i, rect) in enumerate(rects):
			aligned_face, face_pos = aligner.align(160,image,landmarks[i])	# Aligns and crops all the faces and returns their positions
			if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
				aligns.append(aligned_face)
				positions.append(face_pos)
			else: 
				print("Align face failed")      
		if(len(aligns) > 0 and face_pos=='Center'):				
			features_arr = extract_feature.get_features(aligns) 	# Extracts the features and converts to embeddings
			result = findPeople(features_arr,positions)			# Function that uses SVM classifier to classify i.e. recognize the face
														
			print('\n\nRecog data: ',result[0])
			for (i,rect) in enumerate(rects):
				cv2.rectangle(image,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(255,0,0),3)	
				cv2.putText(image,result[0],(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)

		cv2.imwrite(str(result[0])+'.jpg',image)		# Resulting images with bounding boxes are saved	
		cv2.imshow("Recognition",image)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q" or "Q"):
			break
	c2 = cv2.getTickCount()
	print('Time for execution:',(c2 - c1)/cv2.getTickFrequency())

def findPeople(features_arr, positions):	# Classifier function

	f = open('./facerec_new.txt','r')
	data_set = json.loads(f.read())
	y = list(data_set.keys())

	X = []
	for i in y:
		X.append(data_set[i]['Center'])		
	X.append(features_arr)					# Adds the features of current face to X (list of features)

	X = np.reshape(X, (128,len(y)+1)).T
	
	pca = PCA(n_components=1, svd_solver='full')
	X = pca.fit_transform(X).tolist()		# Dim reduction before providing to SVM

	newX = np.reshape(X[-1],(-1,1))

	model = pickle.load(open('svm_classifier.sav', 'rb'))
	result = model.predict(newX)			# Loads the saved model and predicts the result of the face
	return result

if __name__ == '__main__':
	FRGraph = FaceRecGraph()
	aligner = AlignCustom()
	extract_feature = FaceFeature(FRGraph)
	face_detect = MTCNNDetect(FRGraph, scale_factor=3)
	main()
