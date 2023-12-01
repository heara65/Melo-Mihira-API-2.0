import fastapi
import uvicorn
import ssl
import tensorflow
from fastapi import *
import os
import deepFaceFW

import PredictionModel
# import PredicNew
import emotionDetection
# import ageDetection
import uuid
# import pandas as pd

IMAGEDIR = "Images/"

app = FastAPI()


def GenderMapper(gender_):
    genders = ['m', 'f']
    if gender_ == 'female':
        return genders[1]
    else:
        return genders[0]


@app.get('/')
def hello_world():
    return "Hello World" # to test


@app.post("/demographicsImage")
async def create_upload_file(
    file: UploadFile = File(...)
    ):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()  

    # example of how you can save the file
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)

    filepath = IMAGEDIR + file.filename
    # filepath = IMAGEDIR + "abc.jpg"

    # df = pd.DataFrame(columns = ['path'])
    # Add records to dataframe using the .loc function
    # df.loc[0] = [filepath] 
    # Result = ModelLoading.finalImageOutput(df)
    try:
        gender = PredictionModel.predict_gender(filepath)
        emotion = emotionDetection.predict_emotion(filepath)
        # age = ageDetection.predict_age()
        # res =  PredicNew.finalImageOutput()
        # Result = ModelLoading.finalImageOutput(contents)
        age = deepFaceFW.predict_age(filepath)
    
        # return age
        print(gender)
        print(emotion)
        print(age[0]['age'])
        return {"Gender": gender, 
                "Age": str(age[0]['age']), 
                # "Age":"20", 
                "Emotion": emotion
                }
    except Exception as e:
        error_message =f"{str(e)}"
        print(error_message)
        return {
            "Error":error_message,
                "Gender": "None", 
                "Age": "None", 
                "Emotion": "None"
                }


# if __name__ == "__main__":
#     uvicorn.run(app, port=5000, host='192.168.1.2')




