import cv2
from utils.align_custom import AlignCustom
from utils.face_feature import FaceFeature
from utils.mtcnn_detect import MTCNNDetect
from utils.tf_graph import FaceRecGraph
import argparse
import sys
import json
import numpy as np
import glob

def main():
	camera_recog()

def camera_recog():
	for file in glob.glob('./detect/*.jpg'):			# Reads all images for detection
		image = cv2.imread(file)
		rects, landmarks = face_detect.detect_face(image,80)	# Detects all faces present in the image
		aligns = []
		positions = []
		for (i, rect) in enumerate(rects):
			aligned_face, face_pos = aligner.align(160,image,landmarks[i])	# Aligns and crops the faces
			if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
				aligns.append(aligned_face)
				positions.append(face_pos)
			else: 
				print("Align face failed")      
		if(len(aligns) > 0 and face_pos=='Center'):
			features_arr = extract_feature.get_features(aligns)		# Extracts the features of th detected faces
			recog_data = findPeople(features_arr,positions)			# Calls the findPeople function to recognize the detected faces
			print(recog_data)
			for (i,rect) in enumerate(rects):						# Makes bounding boxes around the faces with the name and percent recognition
				cv2.rectangle(image,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(255,0,0),1)
				cv2.putText(image,recog_data[i][0]+" - "+str(recog_data[i][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)

		cv2.imwrite(str(recog_data[0])+'.jpg',image)				# Makes a copy of the said image with boxes 
		cv2.imshow("Recognition",image)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q" or "Q"):
			break

def findPeople(features_arr, positions, thres = 0.8, percent_thres = 80): # Function to find whether detected faces are present in database

	f = open('./facerec_new.txt','r')
	data_set = json.loads(f.read())
	returnRes = []
	for (i,features_128D) in enumerate(features_arr):
		result = "Unknown"
		smallest = sys.maxsize
		for person in data_set.keys():
			person_data = data_set[person][positions[i]]
			for data in person_data:
				distance = np.sqrt(np.sum(np.square(data-features_128D)))		# Calculates the distance of the features with the ones in the database
				if(distance < smallest):										# Assigns the smallest possible distance/similarity with the features in the database
					smallest = distance
					result = person
				print("smallest=",smallest)
		percentage =  min(100, 100 * thres / smallest)			# Calculates the % simlarity of the given face with the ones stored in the database
		if percentage <= percent_thres:							# If below threshold, person not recognized i.e. Unknown
			result = "Unknown"
		returnRes.append((result,percentage))					# Returns the name of the person recognized (or Unknown if not) and the percentage
	return returnRes

if __name__ == '__main__':
	FRGraph = FaceRecGraph()
	aligner = AlignCustom()
	extract_feature = FaceFeature(FRGraph)
	face_detect = MTCNNDetect(FRGraph, scale_factor=3)
	main()
