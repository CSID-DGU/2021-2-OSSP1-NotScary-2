from os import lseek
import cv2
import math

def distanceBetweenPoints(aX, aY, bX, bY):
    return math.sqrt(math.pow(aX-bX,2)+math.pow(aY-bY,2))

def slopeBetweenPoints(aX, aY, bX, bY):
    return abs(aY-bY)/abs(aX-bX)

def output_keypoints(frame, proto_file, weights_file, threshold, model_name, BODY_PARTS, gtcs):
    global points

    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # 입력 이미지의 사이즈 정의
    image_height = 368
    image_width = 368

    # 네트워크에 넣기 위한 전처리
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    # 결과 받아오기
    out = net.forward()
    # The output is a 4D matrix :
    # The first dimension being the image ID ( in case you pass more than one image to the network ).
    # The second dimension indicates the index of a keypoint.
    # The model produces Confidence Maps and Part Affinity maps which are all concatenated.
    # For COCO model it consists of 57 parts – 18 keypoint confidence Maps + 1 background + 19*2 Part Affinity Maps. Similarly, for MPI, it produces 44 points.
    # We will be using only the first few points which correspond to Keypoints.
    # The third dimension is the height of the output map.
    out_height = out.shape[2]
    # The fourth dimension is the width of the output map.
    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = frame.shape[:2]

    # 포인트 리스트 초기화
    points = []

    # 출력 파일 로드
    resfile = open("./distanceslope.csv",'a')

    isWristSpotted = 0  # 0 is not spotted, 1 is RWrist Spotted, 2 is LWrist Spotted

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
           # cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
           # cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

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
#          # cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
#          # cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

            points.append(None)
            if i == 4:
                WristEyeDistance = -1.0
            
            elif i == 7:
                WristEyeDistance = -1.0
#           resfile.write(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}\n")
    print(RShoulderPoints[0], ", ", RShoulderPoints[1], " / ", LShoulderPoints[0], ", ", LShoulderPoints[1], " / ", REyePoints[0], ", ", REyePoints[1], " / ", LEyePoints[0], ", ", LEyePoints[1], "\n")

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
    # cv2.imshow("Output_Keypoints", frame)
    resfile.write(str(ShoulderDistance) + "," + str(EyeDistance) + "," + str(ShoulderEyeDistance) + "," 
    + str(ShoulderSlope) + "," + str(EyeSlope) + "," + str(WristEyeDistance) + "," + str(gtcs) + "\n")
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

# 이미지 경로
i = 1
gtcs = int(input()) # good = 0 turtle = 1 chin = 2 chin = 3
num = int(input())
while i < num:
    if gtcs == 0 :
        inputFile = ".\\good\\good (" + str(i) + ").jpg"
    elif gtcs == 1:
        inputFile = ".\\bad\\turtle\\turtle (" + str(i) + ").jpg"
    elif gtcs == 2:
        inputFile = ".\\bad\\chin\\chin (" + str(i) + ").jpg"
    elif gtcs == 3:
        inputFile = ".\\bad\\shoulder\\shoulder (" + str(i) + ").jpg"

    # 키포인트를 저장할 빈 리스트
    points = []

    # 이미지 읽어오기
    frame_coco = cv2.imread(inputFile)

    # COCO Model
    frame_COCO = output_keypoints(frame=frame_coco, proto_file=protoFile_coco, weights_file=weightsFile_coco,
                                threshold=0.2, model_name="COCO", BODY_PARTS=BODY_PARTS_COCO, gtcs=gtcs)
    #output_keypoints_with_lines(frame=frame_COCO, POSE_PAIRS=POSE_PAIRS_COCO)
    i = i + 1
