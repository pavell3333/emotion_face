import numpy as np
import cv2 as cv2
import tensorflow as tf
from vidgear.gears import WriteGear


print(tf.__version__)

VIDEO_FILE      = 'videoplayback1.mp4'
HAAR_CASCADE    = 'models/haarcascade_frontalface_default.xml'
EMOT_MODEL      = 'models/emotion_classificaion_.h5'
OUTPUT_PARAMS   = {'-vcodec':'libx264', 'ctr':0, 'preset':'fast'}
CLASSES         = {'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happy': 4, 'neutral': 5, 'sad': 6, 'surprise': 7, 'uncertain': 8}

model           = tf.keras.models.load_model(EMOT_MODEL)
face_classifier = cv2.CascadeClassifier(HAAR_CASCADE)

cap             = cv2.VideoCapture(VIDEO_FILE)
out_video       = WriteGear(output_filename='output1.mp4', compression_mode=True, logging=True, )



def get_classname(dict, cl):
    return [name for name, c in dict.items() if c == cl]


while True:
    ret, test_img = cap.read()
    if not ret:
        continue

    faces_detected = face_classifier.detectMultiScale(test_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=4)
        roi = test_img[y:y + w, x:x + h]
        roi = cv2.resize(roi, (112, 112))
        img_pixels = tf.keras.preprocessing.image.img_to_array(roi)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        pred = model.predict(img_pixels)
        predicted_emotion = get_classname(CLASSES, np.argmax(pred))
        cv2.putText(test_img, predicted_emotion[0], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    cv2.imshow('Face emotion analysis ', test_img)
    out_video.write(test_img)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
out_video.release()
cv2.destroyAllWindows




