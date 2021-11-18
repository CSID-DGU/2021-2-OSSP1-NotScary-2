import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical 
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior() 

data=pd.read_csv('./resultData.csv', encoding='latin1')#전처리한 csv 데이터 파일 불러오기

#ShoulderDistance, EyeDistance, ShoulderEyeDistance,ShoulderSlope,EyeSlope 입력을 통해 posture 결과 얻기
data_X = data[['ShoulderDistance','EyeDistance', 'ShoulderEyeDistance','ShoulderSlope','EyeSlope']].values
data_y = data['posture'].values

# 훈련 데이터와 테스트 데이터를 8:2로 나누고 랜덤으로 돌림
(X_train, X_test, y_train, y_test) = train_test_split(data_X, data_y, train_size=0.8, random_state=1)

# 원-핫 인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#임베딩층과 은닉층 입력과 인코딩 데이터
vocab_size = 10000
embedding_dim = 32
hidden_units =32

#모델링
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))#임베딩 층
model.add(SimpleRNN(hidden_units))#은닉층
#입력 뉴런 5:'ShoulderDistance','EyeDistance', 'ShoulderEyeDistance','ShoulderSlope','EyeSlope'/출력 3:0(good),1(turtle),2(shoulder+chin)
model.add(Dense(3, input_dim=5, activation='softmax'))#소프트맥스 전결합층

#컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#훈련
model.fit(X_train, y_train, epochs=200, batch_size=2, validation_data=(X_test, y_test))

model.evaluate(X_test, y_test, batch_size=2)
