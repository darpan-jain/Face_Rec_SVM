import cv2
from utils.align_custom import AlignCustom
from utils.face_feature import FaceFeature
from utils.mtcnn_detect import MTCNNDetect
from utils.tf_graph import FaceRecGraph
import json
import numpy as np
import glob,os


# path = '/home/darpan/Downloads/Occipital/classification/faces/'
def main():
	manual_data()

def manual_data():
	count=1
	for file in glob.glob('./faces/*.jpg'):
		image = cv2.imread(file)
		key = cv2.waitKey(1) & 0xFF
		f = open('./facerec_new.txt','r')
		data_set = json.loads(f.read());
		person_imgs = {"Center": []}
		person_features = {"Center": []}
		rects, landmarks = face_detect.detect_face(image,80)
		for (i, rect) in enumerate(rects):
			aligned_frame, pos = aligner.align(160,image,landmarks[i])
			print('Aligned frame: \n\n\n\n',aligned_frame)
			if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
				# if(pos=="Center"):
				person_imgs[pos].append(aligned_frame)
				cv2.imshow("Captured face", aligned_frame)
				cv2.waitKey(key)
				print('Enter name/ID')
				name = input()
			
			if key == ord("q"):
				# cv2.imwrite(os.path.join('./faces',new_name+'.jpg'), aligned_frame)
				vs.release()
				cv2.destroyAllWindows()
				break

		for pos in person_imgs:
			person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]),axis=0).tolist()]
		data_set[name] = person_features;
		f = open('./facerec_new.txt', 'w')
		f.write(json.dumps(data_set))
		print('User %d added\n'%count)
		count+=1
		# print("Add new user? Y or N")
		# if input()=="n"or"N":
		# 	break
	print('%d users added'%count)

if __name__ == '__main__':
	FRGraph = FaceRecGraph()
	aligner = AlignCustom()
	extract_feature = FaceFeature(FRGraph)
	face_detect = MTCNNDetect(FRGraph, scale_factor=3)
	# __import__('ipdb').set_trace(context=5)
	main()