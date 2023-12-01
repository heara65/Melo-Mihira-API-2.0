# from flask import request
# , jsonify
import cv2
import numpy as np
from keras.models import model_from_json
from keras.utils import load_img, img_to_array 

def predict_emotion(image_path):
    try:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        model.load_weights('modelE.h5')

        face_cascade = cv2.CascadeClassifier('haarcascade.xml')

        # file = request.files['image']
        # img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        # image_path = './Images/ewq.jpg'  # Replace with the actual image path
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

        result = {}
        emotion =""

        if faces is None or len(faces) == 0:
            print("No faces detected in the image.")
            raise ValueError("No faces detected in the image.")

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255.0
            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            predicted_emotion = emotions[max_index]
            
            for i, emotion in enumerate(emotions):
                result[emotion] = float(predictions[0][i])
            
            result['main_emotion'] = predicted_emotion
            emotion = predicted_emotion
            
            break
        
        print(emotion)
        return emotion
    except Exception as e:
        error_message =f"{str(e)}"
        print(error_message)
        raise ValueError(error_message)
    