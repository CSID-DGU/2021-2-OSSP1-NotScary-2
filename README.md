# 2021-2-OSSP1-NotScary-2
2021-2 공개SW프로젝트 안무서운팀

# 캠을 이용한 비대면 학습 중 자세 교정 시스템
노트북의 웹캠을 통하여 실시간으로 자세를 판정하여 지속적인 자세 교정을 돕는 웹앱

## 팀원 구성
* 컴퓨터공학과  2018112055 박용욱
* 컴퓨터공학과  2017112138 정여준
* 컴퓨터공학과  2018112019 고가현
* 컴퓨터공학과  2019111979 이창진

## 개발 환경 및 오픈소스 활용
* Python v3.7
* Tensorflow v2.5
* OpenPose v1.7.0
* OpenCV v3.4.16
* Node js v.14.18.1
* React v.17.0.2
* Anaconda v4.10.1

## 프로젝트 구성 흐름도
<img width="60%" src="https://user-images.githubusercontent.com/45120083/146642871-b1906793-01f3-419e-b2f2-f3706ca525e9.png">
웹캠을 통해 찍은 사진의 스켈레톤 좌표값을 통해 구한 입력값을 넣은 모델의 출력값으로 자세 판정<p>
이때, 프로그램 시작 시 사용자로부터 초기 바른 자세 데이터 값을 이용한 입력값을 구함<p>
<img width="60%" src="https://user-images.githubusercontent.com/45120083/146642905-49d9a6b0-5562-47de-9d88-f7359c001caf.png"><p>
모델의 출력값으로 나온 판정 자세에 따라 사용자에게 자세 교정 알림을 보내고, 히스토리에 저장

## 프로젝트 기능
### 자세 판정 모델
<img width="60%" src="https://user-images.githubusercontent.com/45120083/146644419-7b1c2f06-433a-4c6f-8e13-34699990eecc.png">
입력값: 1)오른쪽 어깨 y 좌표값 2)왼쪽 어깨 y좌표값 3)눈 사이 거리 4)눈에서 어깨까지 거리 5)어깨 사이 거리/눈 사이 거리 6)어깨 기울기 7)눈 기울기<p>
출력값: 0)바른 자세 1)거북목 자세 2)고개-어깨 비대칭 자세 확률

### 웹앱 기능
<img width="60%" src="https://user-images.githubusercontent.com/45120083/146642973-101246b5-da35-4fa6-86de-ce801013d8ef.png">
양 쪽 눈과 양 쪽 어깨 좌표가 인식되는 바른 자세를 화면의 가이드를 참고하여 인식시킴
<img width="60%" src="https://user-images.githubusercontent.com/45120083/146642974-5ba0c6b6-31bc-41a2-a96e-f5547f62739d.png">
앞 선 바른 자세 데이터를 기준으로 자세 판정을 시작<p>
이때, 자세가 바르지 못할 경우 푸시 알림과 웹 앱의 화면 문구를 통해 사용자에게 알림
<img width="60%" src="https://user-images.githubusercontent.com/45120083/146642920-2b69cbf5-9922-470a-a17b-4d92e1ef266b.png">
사용자의 바르지 못한 자세 판별 횟수를 달력으로 확인

## 프로젝트 실행
### 프로젝트 설치
```
$ git clone https://github.com/CSID-DGU/2021-2-OSSP1-NotScary-2.git
```
###OpenPose를 위한 caffemodel 설치
```
https://drive.google.com/file/d/1L9fBnf7DU_Pk6dImTGwHl5xQs88Ny_vJ/view?usp=sharing
다운로드 후 2021-2-OSSP1-NotScary-2/server로 이동
```
### 서버 실행
```
$ cd 2021-2-OSSP1-NotScary-2/server
$ node server.js
$ node server2.js
```
### 웹앱 실행
```
$ cd 2021-2-OSSP1-NotScary-2/amsut1121
$ npm start
```

## 프로젝트 데모 영상
<img width="60%" src="https://user-images.githubusercontent.com/45120083/146645386-49c9cbbb-9cdb-43b0-a6aa-eb89741304af.gif">

## 문의
* 박용욱 pyu142857@gmail.com
* 정여준 pucoy332@naver.com
* 고가현 gahyun0527@gmail.com
* 이창진 changjini32@gmail.com
