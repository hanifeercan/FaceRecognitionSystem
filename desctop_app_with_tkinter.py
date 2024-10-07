import tkinter as tk
import threading
import numpy as np
import cv2
import face_detection as face_detection
from mtcnn import MTCNN
import requests
import time
import subprocess
import sys
import firebase_admin
from firebase_admin import credentials,storage,auth,firestore
import json
from tkinter import messagebox
import firebase_deneme as fired

event = threading.Event()

if not firebase_admin.App:
    credentialData = credentials.Certificate("firebase.json")  
    firebase_admin.initialize_app(credentialData, {
            'storageBucket': 'facerecognition-b323e.appspot.com'  # Firebase Storage bucket URL'niz
            })

db = firestore.client()
process = subprocess.Popen(["python", "backend_with_flask.py"])
time.sleep(20)

def object_detection():
    
    confidence_thresh = 0.5 
    NMS_thresh = 0.3  
    video_cap = cv2.VideoCapture(0) 

    with open("coco.names", "r") as f:
        classes = f.read().strip().split("\n")

    yolo_config = "yolov3.cfg"
    yolo_weights = "yolov3.weights"
    net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    while True:

        if event.is_set():
            event.clear()
            break
        success, frame = video_cap.read()
    
        if not success:
            break

        h = frame.shape[0]
        w = frame.shape[1]

        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255, (416, 416), swapRB=True, crop=False)
            
        net.setInput(blob)
        outputs = net.forward(output_layers)
        
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:

                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence  = scores[class_id]

                if confidence > confidence_thresh and classes[class_id] == "person":

                    box = [int(a * b) for a, b in zip(detection[0:4], [w, h, w, h])]

                    center_x, center_y, width, height = box
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, NMS_thresh)

        if(type(indices) is tuple):
            buton_stop.config(state='disabled')
            buton_stop.pack_forget()

            buton_start.config(state='normal')
            buton_start.pack(padx= 50, pady=10)

            add_text("Işık yetersiz kamera başlatılamadı!")
            return

        indices = indices.flatten()

        frame = cv2.rectangle(frame,(0,0),(1,1),(0, 255, 0),2)
        for i in indices:
            (x, y, w, h) = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_detection(frame)
        if cv2.waitKey(30) == ord("q"): 
            break
    
    buton_stop.config(state='disabled')
    buton_stop.pack_forget()

    buton_start.config(state='normal')
    buton_start.pack(padx= 50, pady=10)

    video_cap.release()
    cv2.destroyAllWindows()

def face_detection(frame):

    detector = MTCNN()
    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (160, 100, 100), 2)
        text = "face"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(160, 100, 100), 2)
        cv2.imshow('Face Recognition', frame)
        cv2.imwrite("kamera_goruntusu.jpg", frame)
        other_thread = threading.Thread(target=send_photo, args=(frame,))
        other_thread.start()
        time.sleep(10)

def send_photo(frame):
    url = 'http://localhost:5000/face_recognition'

    dosya_adi = 'C:/Users/hanif/OneDrive/Masaüstü/face_recognition/kamera_goruntusu.jpg'
    files = {'dosya': open(dosya_adi,'rb')}
    data ={
        "email" : fired.email
    }
    response=requests.post(url,data={'data': json.dumps(data)},files=files)

    if response.status_code == 200:
        add_text(response.text)
        print(response.json())
    else:
        print(response.text)

def add_text(text):
    text_area.config(state=tk.NORMAL)  
    text_area.delete("1.0",tk.END)
    text_area.insert(tk.END,text)
    text_area.config(state=tk.DISABLED)

def btn_start_clicked():
    add_text("")
    event.clear()

    other_thread = threading.Thread(target=object_detection)
    other_thread.start()

    buton_start.config(state='disabled')
    time.sleep(8)
    buton_start.pack_forget()

    buton_stop.config(state='normal')
    buton_stop.pack(padx=50, pady=10)

def btn_stop_clicked():
    add_text("Kamera durduruldu!")
    event.set()

    buton_stop.config(state='disabled')
    buton_stop.pack_forget()

    buton_start.config(state='normal')
    buton_start.pack(padx= 50, pady=10)

def cikis_yap():
    pencere.destroy()

def start_train():
    add_text("Eğitim için servis hazır hale getiriliyor!")
    add_text("Eğitim başlatıldı!")
    url = 'http://localhost:5000/fit_face_recognition_model2'
    
    other_thread = threading.Thread(target=requests.get(url))
    other_thread.start()    

def upload_train_data():
    add_text("Eğitim verileri güncelleniyor...")
    uploadFirebaseStorageData()
      
def upload_storege_data(type,personname,image_name,files):
    url = 'http://localhost:5000/upload_firebase_storage_data'

    data = {
    'type': type,
    'personname': personname,
    'image_name': image_name
    }
    response=requests.post(url,data={'data': json.dumps(data)},files=files)
    print(response.text) 

    if response.status_code == 200:
        add_text(response.text)
        print(response.json())
    else:
        print(response.text)


def uploadFirebaseStorageData():
    
    bucket = storage.bucket()
    trainUrl = f"{fired.email}/faceData/train"
    testUrl = f"{fired.email}/faceData/test"

    blobs = bucket.list_blobs(prefix=trainUrl) 
    for blob in blobs:
        if blob.name.endswith(('.png', '.jpg', '.jpeg', '.gif')):  
            personname = blob.name.split('/')[1]  
            image_name = blob.name.split('/')[-1]  
            blob.download_to_filename("C:/Users/hanif/OneDrive/Resimler/destination_file_name.jpg")
            files = {'dosya': open("C:/Users/hanif/OneDrive/Resimler/destination_file_name.jpg",'rb')}
            upload_storege_data("train",personname,image_name,files)
    
    blobs2 = bucket.list_blobs(prefix=testUrl)  
    for blob in blobs2:
        if blob.name.endswith(('.png', '.jpg', '.jpeg', '.gif')):  
            personname = blob.name.split('/')[1]  
            image_name = blob.name.split('/')[-1]  
            blob.download_to_filename("C:/Users/hanif/OneDrive/Resimler/destination_file_name.jpg")
            files = {'dosya': open("C:/Users/hanif/OneDrive/Resimler/destination_file_name.jpg",'rb')}
            upload_storege_data("test",personname,image_name,files)

    add_text("Eğitim verilerini güncelleme işlemi bitmiştir. Şimdi eğitimi başlatabilirsiniz!")

class AuthPopup:
    def __init__(self, master):
        self.master = master
        self.popup = tk.Toplevel(master)
        self.popup.title("Giriş Yap")

        tk.Label(self.popup, text="Email:").pack(pady=5)
        self.entry_email = tk.Entry(self.popup)
        self.entry_email.pack(pady=5)

        self.button_login = tk.Button(self.popup, text="Giriş Yap", command=self.login)
        self.button_login.pack(pady=20)

        self.popup.protocol("WM_DELETE_WINDOW", self.on_close)

        self.popup.transient(master)  
        self.popup.grab_set()          
        self.popup.focus_set()        

    def on_close(self):
        if messagebox.askokcancel("Çıkış", "Uygulamayı kapatmak istiyor musunuz?"):
            self.master.quit() 
    def login(self):
        email = self.entry_email.get()
    
        try:
            user = auth.get_user_by_email(email)
            docs = db.collection_group("sirket").stream()

            for doc in docs:
                if doc.id == email:
                    authority = doc.get("yetki")

                    if authority == "guvenlik":
                        fired.email = doc.get("yonetici")
                    else:
                        fired.email = email
        
            messagebox.showinfo("Başarılı", "Giriş başarılı!")
            self.popup.destroy() 

        except Exception as e:
            messagebox.showerror("Hata", str(e))

pencere = tk.Tk()
pencere.title("Yüz Tanıma Dedektörü")

menu = tk.Menu(pencere)
pencere.config(menu=menu)

dosya_menu = tk.Menu(menu, tearoff=False)
menu.add_cascade(label="Seçenekler", menu=dosya_menu)
dosya_menu.add_command(label="Eğitimi Başlat", command=start_train)
dosya_menu.add_command(label="Eğitim Verilerini Güncelle", command=upload_train_data)

text_area = tk.Text(pencere, height=10, width=40)
text_area.pack(padx=10, pady=10)
text_area.config(state=tk.DISABLED)

buton_start = tk.Button(pencere, text="Kamerayı Başlat", command=btn_start_clicked)
buton_start.pack(pady=10)

buton_stop = tk.Button(pencere, text="Kamerayı Sonlandır", command=btn_stop_clicked, state='disabled')

def on_closing():
    print("Uygulama kapatılıyor...")
    sys.exit(0) 

pencere.protocol("WM_DELETE_WINDOW", on_closing)

AuthPopup(pencere)

pencere.mainloop()
