from tkinter import *
from tkinter import messagebox
from PIL import ImageTk,Image
from tkinter.filedialog import askopenfilename
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import cv2

a=Tk()
a.title("Car Damage Predictor")
a.geometry("1200x600")
a.minsize(1200,600)
a.maxsize(1200,600)

loaded_model = load_model("Project_Saved_Models/model_96acc.h5")

height=150
width=150


def prediction():

    list_box.insert(1,"Loading Image")
    list_box.insert(2,"")
    list_box.insert(3,"Image Preprocessing")
    list_box.insert(4,"")
    list_box.insert(5,"Loading Trained Model")
    list_box.insert(6,"")
    list_box.insert(7,"Prediction")


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
        a="Non-Damaged"
        price=90000
    elif my_pred ==1:
        print("Total loss")
        a="Total loss"
        price=5000
    elif my_pred ==2:
        print("Scratch")
        a="Scratch"
        price=80000
    elif my_pred ==3:
        print("Door dent")
        a="Door dent"
        price=75000
    elif my_pred ==4:
        print("broken headlamp")
        a="Broken headlamp"
        price=60000
    elif my_pred ==5:
        print("broken tail lamp")
        a="Broken Tail lamp"
        price=50000
    elif my_pred ==6:
        print("tier crack")
        a="Tier crack"
        price=40000
    elif my_pred ==7:
        print("glass shatter")
        a="Glass shatter"
        price=30000
    elif my_pred ==8:
        print("bumper damage(crack-dent)")
        a="Bumper damage(crack-dent)"
        price=20000
    elif my_pred ==9:
        print("bonet dent")
        a="Bonet dent"
        price=35000

    out_label.config(text="Damage Type : "+ a)
    out_label1.config(text=("Price : "+str(price)))


def Check():
    global f
    f.pack_forget()

    f=Frame(a,bg="white")
    f.pack(side="top",fill="both",expand=True)


    
    global f1
    f1=Frame(f,bg="deepskyblue")
    f1.place(x=0,y=0,width=560,height=610)
    f1.config()
                   
    input_label=Label(f1,text="INPUT",font="arial 16",bg="deepskyblue")
    input_label.pack(padx=0,pady=20)



    upload_pic_button=Button(f1,text="Upload Picture",command=Upload,bg="pink")
    upload_pic_button.place(x=240,y=100)
    global label
    label=Label(f1,bg="deepskyblue")


    global f2
    f2=Frame(f,bg="tomato")
    f2.place(x=800,y=0,width=400,height=690)
    f2.config(pady=20)
    
    result_label=Label(f2,text="RESULT",font="arial 16",bg="tomato")
    result_label.pack(padx=0,pady=0)

    global out_label
    out_label=Label(f2,text="",bg="tomato",font="arial 16")
    out_label.pack(pady=90)
    global out_label1
    out_label1=Label(f2,text="",bg="tomato",font="arial 16")
    out_label1.pack()



    f3=Frame(f,bg="peach puff")
    f3.place(x=560,y=0,width=240,height=690)
    f3.config()

    name_label=Label(f3,text="Process",font="arial 14",bg="peach puff")
    name_label.pack(pady=20)

    global list_box
    list_box=Listbox(f3,height=12,width=31)
    list_box.pack()

    predict_button1=Button(f3,text="Predict",command=prediction,bg="deepskyblue")
    predict_button1.pack(side="top",pady=10)

  

def Upload():
    global path
    label.config(image='')
    list_box.delete(0,END)
    out_label.config(text='')
    path=askopenfilename(title='Open a file',
                         initialdir='Test_Images',
                         filetypes=(("JPG","*.jpg"),("JPEG","*.jpeg"),("PNG","*.png")))
    print("Path : ",path)
    image=Image.open(path)
    global imagename
    imagename=ImageTk.PhotoImage(image.resize((300,300)))
    label.config(image=imagename)
    label.image=imagename
    # label.pack()
    label.place(x=140,y=210)
                  


def Home():
    global f
    f.pack_forget()
    
    f=Frame(a,bg="cornflower blue")
    f.pack(side="top",fill="both",expand=True)

    front_image = Image.open("Project_Extra/car.jpeg")
    front_photo = ImageTk.PhotoImage(front_image.resize((1200,600), Image.ANTIALIAS))
    front_label = Label(f, image=front_photo)
    front_label.image = front_photo
    front_label.pack()

    home_label=Label(f,text="Car Damage Predictor",font="arial 35",bg="white")
    home_label.place(x=320,y=290)




f=Frame(a,bg="cornflower blue")
f.pack(side="top",fill="both",expand=True)

front_image = Image.open("Project_Extra/car.jpeg")
front_photo = ImageTk.PhotoImage(front_image.resize((1200,600), Image.ANTIALIAS))
front_label = Label(f, image=front_photo)
front_label.image = front_photo
front_label.pack()

home_label=Label(f,text="Car Damage Predictor",font="arial 35",bg="white")
home_label.place(x=320,y=290)

m=Menu(a)
m.add_command(label="Home",command=Home)
checkmenu=Menu(m)
m.add_command(label="Check",command=Check)
a.config(menu=m)




a.mainloop()
