from deepface import DeepFace



def predict_age(path):
    # path = "./Images/wsx.jpg"
    try:
        objs = DeepFace.analyze(img_path = path, actions = ['age'] )
    except Exception as e:
        error_message =f"Error: {str(e)}"
        print("error_message")
        raise ValueError("error_message")

    print(objs)
    return objs
