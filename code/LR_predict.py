import cv2
import numpy as np 
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import normalize
from sklearn.externals.joblib import load
import time

class LR(object):
	def __init__(self,model_path_LR,pred_LR_log,test_lst_path):
		self.model=load(model_path_LR)
		self.pred_LR_log=pred_LR_log
		self.test_lst_path=test_lst_path

	def read_test_img(self,fileName):
		image = cv2.imread(fileName)[:, :, 1]
		if image.shape == (720, 720):
			return image
		else:
		    image = cv2.resize(image, (720, 720))
		    return image

	def separate_img(self,img):
	    h, w = img.shape
	    
	    wmid = int(w / 2)
	    left = img[: , : wmid]
	    right = img[: , wmid :]
	    return left, right

	def get_feature(self,img, radius, METHOD, cut, dim):
		n_points = 8 * radius
		lbp_bins = range(n_points + 2)
		grey_bins = range(0, 257, int(256 / dim))
		h, w = img.shape
		cut_img = img[int(cut * h): int((1 - cut) * h), :]
		lbp = local_binary_pattern(cut_img, n_points, radius, METHOD)
		lbp_hist, _ = np.histogram(lbp, lbp_bins)
		grey_hist, _ = np.histogram(cut_img, grey_bins)
		lbp_L1_NORM = normalize(np.array([lbp_hist], dtype = np.float32), norm = 'l1')
		grey_L1_NORM = normalize(np.array([grey_hist], dtype = np.float32), norm = 'l1')
		feature = np.concatenate((lbp_L1_NORM, grey_L1_NORM), axis = 1)
		return feature

	def score(self,img, radius, METHOD, cut, dim):
		#feature_start = time.clock()
		feature = self.get_feature(img, radius, METHOD, cut, dim)
		#feature_end = time.clock()
		#print 'feature:', (feature_end - feature_start)

		# model = load(model_path_LR)
		score = self.model.decision_function(feature)
		model_end = time.clock()
		#print 'model:', (model_end - feature_end)
		return score[0]

	def process(self,fileName, radius, METHOD, cut, dim):
	    image = self.read_test_img(fileName)
	    left_img, right_img = self.separate_img(image)
	    
	    left_score = self.score(left_img, radius, METHOD, cut, dim)
	    right_score = self.score(right_img, radius, METHOD, cut, dim)
	    if left_score > right_score:
	        label = 'left'
	        # print fileName, left_score, right_score, '===>left'
	    elif left_score < right_score:
	        label = 'right'
	        # print fileName, left_score, right_score, '===>right'
	    else:
	        label = 'NaN'
	        print fileName, 'Cannot verify!!!'
	    return left_score, right_score, label

	def LR_predict(self):
		time_start = time.clock()
		names_labels=[]
	 
		# pred_file = "../prediction/Prediction_LR_log.txt"
		radius = 3
		METHOD = 'uniform'
		cut = 0.2
		dim = 8
		f2= open(self.test_lst_path)
		lines= f2.readlines()
		for line in lines:
			imgName=line.strip()
			left_score, right_score, label = self.process(imgName, radius, METHOD, cut, dim)
	  		names_labels.append([imgName,label])
	  		time_end = time.clock()
	  		with open(self.pred_LR_log, 'a') as f:
	  			f.write('   '.join([imgName, label, 'Time:', str(time_end - time_start), 'Left Score:', str('%.5f' % left_score), 'Right Score:', str('%.5f' % right_score)]) + '\n')
	  		# open(self.pred_LR_log, 'a').write('   '.join([imgName, label, 'Time:', str(time_end - time_start), 'Left Score:', str('%.5f' % left_score), 'Right Score:', str('%.5f' % right_score)]) + '\n')
	  	
	  	f2.close()	
	  	return names_labels