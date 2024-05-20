import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tkinter.filedialog import askopenfilename

loaded_model = load_model("Project_Saved_Models/model_96acc.h5")
width = 150
height = 150

def prediction(path):
    image=cv2.imread(path)
    resize_image = cv2.resize(image,(height,width))
    out_image=np.expand_dims(resize_image,axis=0)/255
    print(out_image.shape)

    my_pred = loaded_model.predict(out_image)
    print(my_pred)
    my_pred=np.argmax(my_pred,axis=1)
    my_pred = my_pred[0]
    print(my_pred)

    if my_pred ==0:
        print("Non-Damaged")
    elif my_pred ==1:
        print("Total loss")
    elif my_pred ==2:
        print("Scratch")
    elif my_pred ==3:
        print("door dent")
    elif my_pred ==4:
        print("broken headlamp")
    elif my_pred ==5:
        print("broken tail lamp")
    elif my_pred ==6:
        print("tier crack")
    elif my_pred ==7:
        print("glass shatter")
    elif my_pred ==8:
        print("bumper damage(crack-dent)")
    elif my_pred ==9:
        print("bonet dent")
    

if __name__=='__main__':
    path=askopenfilename()
    prediction(path)