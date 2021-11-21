from os import lseek
import cv2
import math
import imutils
import numpy as np
import time

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

    check = 0
    
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
            if (i == 2) or (i == 5) or (i == 14) or (i == 15):
                check = check + 1
        
    cv2.waitKey(0)

    # if(check == 4): 
    ## 일단 실행 잘되게하려고 2개이상 체크되면 시작하게 설정함 
    if(check > 1):
        return 1
    else:
        return 0

BODY_PARTS_COCO = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                   5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                   10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
                   15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}

POSE_PAIRS_COCO = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [1, 8], [1, 11], [2, 3], [3, 4],
                   [5, 6], [6, 7], [8, 9], [9, 10], [12, 13], [11, 12], [14, 16], [15, 17]]


protoFile_coco = "./pose_deploy_linevec.prototxt"
weightsFile_coco = "./pose_iter_440000.caffemodel"


#################### 양쪽 어깨, 눈이 5초 이상 보이면 프로그램 종료 ####################

def startCheck():
    video_capture = cv2.VideoCapture(0)

    isStart = 0
    start = time.time() + 987654321
    end = 0
        
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        frame = imutils.resize(frame, width=400)

        if ret:
            # 키포인트를 저장할 빈 리스트
            points = []

            # 이미지 읽어오기
            frame_coco = frame

            # COCO Model
            if(output_keypoints(frame=frame_coco, proto_file=protoFile_coco, weights_file=weightsFile_coco,
                                threshold=0.2, model_name="COCO", BODY_PARTS=BODY_PARTS_COCO)):
                if(not isStart):
                    start = time.time()
                    isStart = 1
            else:
                if(isStart):
                    end = time.time()
                    start = time.time() + 987654321
                    isStart = 0

        now = time.time() 
        if(now - start > 5 or end - start > 5):
            print(1)
            break
            
        # # Hit 'q' on the keyboard to quit!
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release handle to the webcam
    video_capture.release()


if __name__ == '__main__': 
	startCheck()

