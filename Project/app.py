from email.mime import image
import os
from tkinter import *
from tkvideo import tkvideo
from PIL import ImageTk, Image
from tkinter import filedialog
import pyttsx3
import time

root = Tk()
root.title("Traffic Sign recognizer")
root.geometry("900x650+600+150")
weights_path = "D:/FYP/Project/detector.pt"

voice_alerts = { 0:'Speed limit (20km/h)',1:'Speed limit (30km/h)', 
                2:'Speed limit (50km/h)',3:'Speed limit (60km/h)', 
                4:'Speed limit (70km/h)',5:'Speed limit (80km/h)', 
                6:'End of speed limit (80km/h)',7:'Speed limit (100km/h)', 
                8:'Speed limit (120km/h)',9:'No passing', 
                10:'No passing veh over 3.5 tons',11:'Right-of-way at intersection', 
                12:'Priority road',13:'Yield', 
                14:'Stop',15:'No vehicles', 
                16:'Veh > 3.5 tons prohibited',17:'No entry', 
                18:'General caution',19:'Dangerous curve left', 
                20:'Dangerous curve right',21:'Double curve', 
                22:'Bumpy road',23:'Slippery road', 
                24:'Road narrows on the right',25:'Road work', 
                26:'Traffic signals',27:'Pedestrians', 
                28:'Children crossing',29:'Bicycles crossing', 
                30:'Beware of ice/snow',31:'Wild animals crossing', 
                32:'End speed + passing limits',33:'Turn right ahead', 
                34:'Turn left ahead',35:'Ahead only', 
                36:'Go straight or right',37:'Go straight or left', 
                38:'Keep right',39:'Keep left', 
                40:'Roundabout mandatory',41:'End of no passing', 
                42:'End no passing veh > 3.5 tons' }

def upload():
    global img,input_img,input_vid,player
    root.filename = filedialog.askopenfile(initialdir="D:/FYP/Test",title="Upload a file",filetypes=[("media files","*.jpg *.png *.mp4")])
    if root.filename.name.endswith('.jpg') or root.filename.name.endswith('.png'):
        img = Image.open(root.filename.name)
        resize_img = img.resize((750,500))
        img = ImageTk.PhotoImage(resize_img)
        input_img = Label(image=img)
        input_img.pack(pady=10)
    else:
        input_vid = Label(root)
        input_vid.pack(pady=10)
        player = tkvideo(root.filename.name,input_vid,loop=1,size=(750,500))
        player.play()
    classify = Button(root,text="Classify",command=recog).pack(pady=10)

def recog():
    global res,player
    source_path = root.filename.name
    os.system('python D:/FYP/yolov5/detect_classify.py --source "'+source_path+'" --weights "'+weights_path+'" --save-crop --hide-conf')
    f = open("tmp.txt","r")
    signs = f.read().replace('\n',' ').split()
    signs = set(signs)
    if '36' in signs:
        signs=list(signs)
        signs.remove('36')
    print(signs)
    if root.filename.name.endswith('.jpg') or root.filename.name.endswith('.png'):
        input_img.pack_forget()
        exp = os.listdir("D:/FYP/yolov5/runs/detect/")
        res = Image.open("D:/FYP/yolov5/runs/detect/"+exp[-1]+"/output.jpg")
        resize_res = res.resize((750,500))
        res = ImageTk.PhotoImage(resize_res)
        output_img = Label(image=res)
        output_img.pack(pady=10)
    else:
        input_vid.pack_forget()
        output_vid = Label(root)
        output_vid.pack(pady=10)
        exp = os.listdir("D:/FYP/yolov5/runs/detect/")
        player = tkvideo("D:/FYP/yolov5/runs/detect/"+exp[-1]+"/output.mp4",output_vid,loop=1,size=(750,500))
        player.play()
    engine = pyttsx3.init()
    voice = ""
    for x in signs:
        voice+=voice_alerts[int(x)]+" "
    voice+=" ahead"
    engine.say(voice)
    engine.runAndWait()

upload = Button(root,text="Upload",command=upload).pack(pady=10)
root.mainloop()