import tensorflow.keras
import numpy as np
import cv2

model_filename = 'keras_model.h5'
model = tensorflow.keras.models.load_model(model_filename)
# 카메라를 제어할 수 있는 객체
capture = cv2.VideoCapture(0)
# 카메라 길이 너비 조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
# 이미지 처리하기
def preprocessing(frame):
    # 사이즈 조정 티쳐블 머신에서 사용한 이미지 사이즈로 변경해준다.
    size= (224, 224)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    # 이미지 정규화
    frame_normalized= (frame_resized.astype(np.float32) / 127.0) - 1
    # keras 모델에 전달할 올바른 모양의 배열 생성
    frame_reshaped=frame_normalized.reshape((1, 224, 224, 3))
    return frame_reshaped
# 예측용 함수
def predict(frame):
    prediction = model.predict(frame)
    return prediction

while True:
    ret, frame = capture.read()
    preprocessed = preprocessing(frame)
    prediction = predict(preprocessed)
    print(prediction)

    if prediction[0][0] > 0.5:
        print('아 입니다.')
    elif prediction[0][1] > 0.5:
        print('이 입니다.')
    elif prediction[0][2] > 0.5:
        print('우 입니다.')
    elif prediction[0][3] > 0.5:
        print('에 입니다.')
    elif prediction[0][4] > 0.5:
        print('오 입니다.')
    if cv2.waitKey(100) > 0:
        break
    cv2.imshow("VideoFrame", frame)
