import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

#ShoulderDistance,EyeDistance,ShoulderEyeDistance,ShoulderSlope,EyeSlope를 이용한 자세 예측
def getPosture(SD,ED,SED,SS,ES):
	model = load_model("./postureClassModel7929.h5")
	
	input =[float(SD),float(ED),float(SED),float(SS),float(ES)]
	array=model.predict([input])

	#자세 0 1 2 중 가장 높은 예측값으로 자세 판별
	predictPosture=np.argmax(array)
	print(predictPosture)

	
if __name__ == '__main__': 
	getPosture(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
