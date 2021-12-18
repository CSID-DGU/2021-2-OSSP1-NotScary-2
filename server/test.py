import urllib.request 

from os import lseek
import cv2
import math
import imutils
import numpy as np

import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

def getPosture(SD,ED,SED,SS,ES):
	model = load_model("./postureClass68.h5")
	
	input =[float(SD),float(ED),float(SED),float(SS),float(ES)]
	array=model.predict([input])

	#자세 0 1 2 중 가장 높은 예측값으로 자세 판별
	predictPosture=np.argmax(array)
	print(predictPosture)

# 포인트 간 거리 측정
def distanceBetweenPoints(aX, aY, bX, bY):
    return math.sqrt(math.pow(aX-bX,2)+math.pow(aY-bY,2))

# 포인트 간 기울기 측정
def slopeBetweenPoints(aX, aY, bX, bY):
    if(aX-bX == 0):
        return 0
    return abs(aY-bY)/abs(aX-bX)

def output_keypoints(frame, proto_file, weights_file, threshold, model_name, BODY_PARTS):
    global points

    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # 입력 이미지의 사이즈 정의
    image_height = 400
    image_width = 400

    # 네트워크에 넣기 위한 전처리
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    # 결과 받아오기
    out = net.forward()
    out_height = out.shape[2]
    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = frame.shape[:2]

    # 포인트 리스트 초기화
    points = []

    # 0: not spotted, 1: RWrist Spotted, 2 is LWrist Spotted
    isWristSpotted = 0  

    RShoulderPoints = [0, 0]
    LShoulderPoints = [0, 0]
    REyePoints = [0, 0]
    LEyePoints = [0, 0]
    RWristPoints = [0, 0]
    LWristPoints = [0, 0]

    for i in range(len(BODY_PARTS)):

        # 신체 부위의 confidence map
        prob_map = out[0, i, :, :]

        # 최소값, 최대값, 최소값 위치, 최대값 위치
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        # 원본 이미지에 맞게 포인트 위치 조정
        x = (frame_width * point[0]) / out_width
        x = int(x)
        y = (frame_height * point[1]) / out_height
        y = int(y)

        if prob > threshold:  # [pointed]
            points.append((x, y))

            if i == 2:
                RShoulderPoints[0] = x
                RShoulderPoints[1] = y

            elif i == 5:
                LShoulderPoints[0] = x
                LShoulderPoints[1] = y

            elif i == 14:
                REyePoints[0] = x
                REyePoints[1] = y

            elif i == 15:
                LEyePoints[0] = x
                LEyePoints[1] = y

            elif i == 4:
                RWristPoints[0] = x
                RWristPoints[1] = y
                isWristSpotted = 1
            
            elif i == 7:
                LWristPoints[0] = x
                LWristPoints[1] = y
                isWristSpotted = 2

        else:  # [not pointed]
            points.append(None)
            if i == 4:
                WristEyeDistance = -1.0
            
            elif i == 7:
                WristEyeDistance = -1.0

    ShoulderDistance = distanceBetweenPoints(RShoulderPoints[0], RShoulderPoints[1], LShoulderPoints[0], LShoulderPoints[1])
    EyeDistance = distanceBetweenPoints(REyePoints[0], REyePoints[1], LEyePoints[0], LEyePoints[1])
    ShoulderEyeDistance = distanceBetweenPoints((RShoulderPoints[0]+LShoulderPoints[0])/2, (RShoulderPoints[1]+LShoulderPoints[1])/2,
                            (REyePoints[0]+LEyePoints[0])/2, (REyePoints[1]+LEyePoints[1])/2)
    ShoulderSlope = slopeBetweenPoints(RShoulderPoints[0], RShoulderPoints[1], LShoulderPoints[0], LShoulderPoints[1])
    EyeSlope = slopeBetweenPoints(REyePoints[0], REyePoints[1], LEyePoints[0], LEyePoints[1])
    if isWristSpotted == 0:
        WristEyeDistance = -1.0  # -1 is not spotted, -2 is RWrist spotted, -3 is LWrist Spotted
    elif isWristSpotted == 1:
        WristEyeDistance = distanceBetweenPoints((REyePoints[0]+LEyePoints[0])/2, (REyePoints[1]+LEyePoints[1])/2,
                            RWristPoints[0], RWristPoints[1])
    elif isWristSpotted == 2:
        WristEyeDistance = distanceBetweenPoints((REyePoints[0]+LEyePoints[0])/2, (REyePoints[1]+LEyePoints[1])/2,
                            LWristPoints[0], LWristPoints[1])
    
    getPosture(ShoulderDistance, EyeDistance, ShoulderEyeDistance, ShoulderSlope, EyeSlope)

    cv2.waitKey(0)
    return frame

BODY_PARTS_COCO = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                   5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                   10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
                   15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}

POSE_PAIRS_COCO = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [1, 8], [1, 11], [2, 3], [3, 4],
                   [5, 6], [6, 7], [8, 9], [9, 10], [12, 13], [11, 12], [14, 16], [15, 17]]


protoFile_coco = "./pose_deploy_linevec.prototxt"
weightsFile_coco = "./pose_iter_440000.caffemodel"

def test():
    man = "C:\\Users\\82109\\Downloads\\pose_capture.jpg" #다운로드된 파일 경로 (개인 pc 경로에 맞게 수정해줘야됨)
    
    points = []

    frame_coco = cv2.imread(man)
            
    frame_COCO = output_keypoints(frame=frame_coco, proto_file=protoFile_coco, weights_file=weightsFile_coco,
                                    threshold=0.2, model_name="COCO", BODY_PARTS=BODY_PARTS_COCO)    

    

if __name__ == '__main__': 
	test()
