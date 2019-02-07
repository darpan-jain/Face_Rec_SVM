import cv2
from utils.align_custom import AlignCustom
from utils.face_feature import FaceFeature
from utils.mtcnn_detect import MTCNNDetect
from utils.tf_graph import FaceRecGraph
import json
import numpy as np
import glob,os
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pickle

def main():
	manual_data()

def manual_data():
	for file in glob.glob('./faces/*.jpg'):
		image = cv2.imread(file)				# Reading each image in the training folder
		key = cv2.waitKey(1) & 0xFF
		f = open('./facerec_new.txt','r')
		data_set = json.loads(f.read())
		person_imgs = {"Center": []}				# Dictionary for storing the embeddings
		person_features = {"Center": []}
		rects, landmarks = face_detect.detect_face(image,80)	# Detects all the faces in the picture
		for (i, rect) in enumerate(rects):
			aligned_frame, pos = aligner.align(160,image,landmarks[i])	# Aligns and crops the faces in the image
			if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
				person_imgs[pos].append(aligned_frame)				# Appends all the faces and their positions in the image
				cv2.imshow("Captured face", aligned_frame)
				cv2.waitKey(key)
				print('Enter name/ID')		# Input from user for name/ID of the captured face
				name = input()
			
			if key == ord("q"):
				vs.release()
				cv2.destroyAllWindows()
				break

		for pos in person_imgs:		# Extracts the features of all the faces in the given images
			person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]),axis=0).tolist()]
		data_set[name] = person_features
		f = open('./facerec_new.txt', 'w')
		f.write(json.dumps(data_set))
	
	f = open('./facerec_new.txt','r')
	data_set = json.loads(f.read())

	y = list(data_set.keys())	# Reads all the input faces' names from database
	
	le = LabelEncoder()
	onehot = OneHotEncoder(sparse=False)	

	svm_y = le.fit_transform(y)
	svm_y = svm_y.reshape(len(svm_y), 1)
	onehot = onehot.fit_transform(svm_y)	# Encoding the labels to be given to the classifier

	X = []
	for i in y:
		X.append(data_set[i]['Center'])	# Reads all the features to be given to train classifier

	X = np.reshape(X, (128,len(y))).T
	pca = PCA(n_components=1, svd_solver='full')	# Dim reduction of 128D features
	X = pca.fit_transform(X).tolist()
	
	print('\n\nTraining the classifier...\n\n')
	model = SVC(probability=True,verbose=True)	
	model.fit(X,y)								# define and fit the classifier with the parameters
	
	filename = 'svm_classifier.sav'				
	pickle.dump(model, open(filename, 'wb'))	# Save the trained classifier with the faces
	print('\n\nDONE.\n\n')

if __name__ == '__main__':
	FRGraph = FaceRecGraph()
	aligner = AlignCustom()
	extract_feature = FaceFeature(FRGraph)
	face_detect = MTCNNDetect(FRGraph, scale_factor=3)
	main()