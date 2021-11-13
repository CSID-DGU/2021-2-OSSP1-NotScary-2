#pip install Augmentor후 실행 
import Augmentor

#증강 시킬 이미지 폴더 경로(이미지 폴더 경로까지만 작성하면 ouput파일 생성 후 저장됨)
img = Augmentor.Pipeline("/Users/data/good")

#사진 80퍼센트 크롭
img.crop_random(probability=1, percentage_area=0.8, randomise_percentage_area=False)
img.process()
#사진 좌우반전
img.flip_left_right(probability=1.0)
img.process()
#사진 80퍼센트 랜덤 확대
img.zoom_random(probability=1.0, percentage_area=0.8)
img.process()
