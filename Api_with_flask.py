from flask import Flask, request, jsonify
from PIL import Image 
from numpy import asarray
from numpy import expand_dims
import cv2
import firebase_deneme as db
from mtcnn import MTCNN
import numpy as np
import pickle
import os
from keras.models import load_model 
from os import listdir
import json
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
UPLOAD_FOLDER = os.path.join(app.config['UPLOAD_FOLDER'])
DATASET_DIR = 'veriseti'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

@app.route('/get_classification_or_labes_model_files')
def get_classification_or_labes_model_files(filename):
    file_path = os.path.join('uploads' + filename)
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

@app.route('/get_facenet_model_files')
def get_facenet_model_files():
    model_path = os.path.join('uploads' + 'facenet_keras.h5')
    model_weights_path = os.path.join('uploads' +'facenet_keras_weights.h5')
    model = load_model(model_path)
    model.load_weights(model_weights_path)
    return model

@app.route('/face_recognition', methods=['POST'])
def face_recognition():

    try:

        data= json.loads(request.form['data'])
        email = data['email']

        if 'dosya' not in request.files:
            return jsonify({'Hata': 'Dosya yok'}), 400
        file = request.files['dosya']
        if file.filename == '':
            return jsonify({'Hata': "Seçili dosya yok"}),400
        if file:
            img_stream = file.stream.read()
            if not img_stream:
                return jsonify({'Hata': "Dosya boş"}), 400
        
        
            nparr = np.frombuffer(img_stream, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            detector = MTCNN()

            faces = detector.detect_faces(img)
            pixels = asarray(img) 

            kisiler = []
            for face in faces:
                realTime_face_pixels=extract_frame(pixels,face)
                realTime_face_emb=extract_embeddings(facenet_keras_model,realTime_face_pixels)
                samples = expand_dims(realTime_face_emb, axis=0)

                yhat_class = classification_model.predict(samples)
                yhat_prob = classification_model.predict_proba(samples)
                class_index = yhat_class[0]
                class_probability = yhat_prob[0,class_index] 
                predict_names = labels.inverse_transform(yhat_class)

                print(predict_names)
                print(class_probability)

                if class_probability > 0.8:
                    isStaff = db.isStaff(predict_names[0],email)
                    if isStaff:
                        db.addFirebaseLoginOrOut(predict_names[0],email)
                        kisiler.append(predict_names[0])
                    else:
                        kisiler.append("Known")
                else:
                    kisiler.append("Known")

            return jsonify({'Kisiler': kisiler}),200
    except Exception as e:
        return jsonify({'Hata': str(e)}), 400

def extract_embeddings(model,face_pixels):
    face_pixels = face_pixels.astype('float32')       
    mean, std = face_pixels.mean(), face_pixels.std() 
    face_pixels = (face_pixels - mean) / std          
    samples = expand_dims(face_pixels, axis = 0)      
    yhat = model.predict(samples)                     
    return yhat[0]

def extract_frame(pixels,face):
    if face:
        x1, y1, width, height = face['box']             
        x1, y1 = abs(x1), abs(y1)
        x2 = abs(x1 + width)
        y2 = abs(y1 + height)
        
        store_face = pixels[y1:y2, x1:x2]               
        image_face = Image.fromarray(store_face, 'RGB') 
        image_face = image_face.resize((160, 160))     
        face_array = asarray(image_face)               
        return face_array
    else:
        return None

def extract_image(image):
    img1 = Image.open(image)           
    img1 = img1.convert('RGB')         
    pixels = asarray(img1)                                        
    detector = MTCNN()                  
    faces = detector.detect_faces(pixels)

    if faces:
        x1, y1, width, height = faces[0]['box']         
        x1, y1 = abs(x1), abs(y1)
        x2 = abs(x1 + width)
        y2 = abs(y1 + height)
        
        store_face = pixels[y1:y2, x1:x2]               
        image_face = Image.fromarray(store_face, 'RGB') 
        image_face = image_face.resize((160, 160))      
        face_array = asarray(image_face)                
        return face_array
    else:
        return None

def extract_dataset_image(pixels):
    detector = MTCNN()                  
    faces = detector.detect_faces(pixels)

    if faces:
        x1, y1, width, height = faces[0]['box']         
        x1, y1 = abs(x1), abs(y1)
        x2 = abs(x1 + width)
        y2 = abs(y1 + height)
        
        store_face = pixels[y1:y2, x1:x2]               
        image_face = Image.fromarray(store_face, 'RGB') 
        image_face = image_face.resize((160, 160))      
        face_array = asarray(image_face)                
        return face_array
    else:
        return None
    
def load_faces(directory):
    faces = []
    for filename in listdir(directory):
        path = directory + filename
        face = extract_image(path)
        faces.append(face)
    return faces

def load_dataset(directory):
    x, y = [],[]
    i=1
    for subdir in listdir(directory):
      path = directory + subdir + '/' 
      faces =load_faces(path)
      if faces is None:
          continue
      labels = [subdir for _ in range(len(faces))]
      print("%d There are %d images in the class %s:"%(i,len(faces),subdir))
      x.extend(faces)
      y.extend(labels)
      i=i+1
    return asarray(x),asarray(y)  
    
@app.route('/upload_model_files', methods=['POST'])
def upload_model_files():
    uploaded_file = request.files['file']
    if uploaded_file:
        uploaded_file.save(UPLOAD_FOLDER+ uploaded_file.filename)
        return jsonify({'Result': 'Dosya başariyla yüklendi: ' + uploaded_file.filename})
    else:
        return jsonify({'hata': 'Dosya yüklenemedi.'})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/fit_face_recognition_model2')
def fit_face_recognition_model():
    
    print("\n\n\neğitim başladı\n\n\n")
    trainX, trainy = get_image_data('train')  
    print("\n\n\ntrain veriliri başarıyla yüklendi\n\n\n")
    testX, testy = get_image_data('test')
    print("\n\n\test veriliri başarıyla yüklendi\n\n\n")
    print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

    newTrainX = list()
    for train_pixels in trainX:
      embeddings = extract_embeddings(facenet_keras_model,train_pixels)
      newTrainX.append(embeddings)
    newTrainX = asarray(newTrainX)             

    newTestX = list()
    for face_pixels in testX:
        embedding = extract_embeddings(facenet_keras_model,face_pixels)
        newTestX.append(embedding)
    newTestX = asarray(newTestX)
  
    in_encode = Normalizer(norm='l2') 
    trainX = in_encode.transform(newTrainX)
    testX = in_encode.transform(newTestX)

    out_encoder = LabelEncoder() 
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
   
    classification_model = SVC(kernel='linear',probability=True) 
    classification_model.fit(trainX, trainy) 

    yhat_train = classification_model.predict(trainX)
    yhat_test = classification_model.predict(testX)
    
    score_train = accuracy_score(trainy, yhat_train)
    score_test = accuracy_score(testy, yhat_test)
    
    print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

    print("\n\n\model kayıt ediliyor\n\n\n")

    with open('svc_model.pkl', 'wb') as file:
      pickle.dump(classification_model, file)
    with open('labels.pkl', 'wb') as file:
      pickle.dump(out_encoder, file)
    upload_all_models()

def get_image_data(set_type):
    folder_path = os.path.join(DATASET_DIR, set_type)

    if not os.path.exists(folder_path):
        return jsonify({"error": "Train folder not found"}), 404
    folder_names = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    x, y = [],[]
    i=1
    
    for name in folder_names:

        folder_path = os.path.join(DATASET_DIR, set_type, name)

        print(folder_path)
        
        if not os.path.exists(folder_path):
            return jsonify({"error": "Folder not found"}), 404

        image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.jpeg')]

        faces = []
        
        for filename in image_files:
            file_path = os.path.join(folder_path, filename)
            if os.path.exists(file_path):
                img = Image.open(file_path).convert('RGB')
                pixels = asarray(img)
                face = extract_dataset_image(pixels)
                faces.append(face)
        
        if faces is None:
          continue
        labels = [name for _ in range(len(faces))]
        print("%d There are %d images in the class %s:"%(i,len(faces),name))
        x.extend(faces)
        y.extend(labels)
        i=i+1

    return asarray(x),asarray(y) 

def upload_all_models():
    facenet = 'C:/facenet_keras.h5'
    weights = 'C:/facenet_keras_weights.h5'
    classification = 'svc_model.pkl'
    labels = 'labels.pkl'

    upload(facenet)
    upload(weights)
    upload(classification)
    upload(labels)

def upload(filename):
    url = 'http://localhost:5000/upload_model_files'

    files = {'file': open(filename,'rb')}
    response=requests.post(url,files=files)

    if response.status_code == 200:
        print(response.json())
    else:
        print("Hata:", response.text)

@app.route('/upload_firebase_storage_data', methods=['POST'])
def upload_image():

    data= json.loads(request.form['data'])
    set_type = data['type']
    person_name = data['personname']
    filename = data['image_name']
    
    print(set_type)
    folder_path = os.path.join(DATASET_DIR, set_type, person_name)

    os.makedirs(folder_path, exist_ok=True)

    if 'dosya' not in request.files:
        return jsonify({'Hata': 'Dosya yok'}), 400
    file = request.files['dosya']
    if file.filename == '':
        return jsonify({'Hata': "Seçili dosya yok"}),400
    if file:
        img_stream = file.stream.read()
        if not img_stream:
            return jsonify({'Hata': "Dosya boş"}), 400
            
        nparr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if create_folder_if_not_exists(folder_path,filename):
            img.save(os.path.join(folder_path, filename))
            return jsonify({"message": "File successfully uploaded", "filename": filename}), 200
        else:
            return jsonify({"message": "File alredy uploaded", "filename": filename}), 200   

def create_folder_if_not_exists(folder_path,image_name):
    image_path = os.path.join(folder_path, image_name)
    if os.path.exists(image_path):
        return False
    else:
        return True
               
if __name__ == '__main__':
    facenet_keras_model = get_facenet_model_files()
    classification_model = get_classification_or_labes_model_files('svc_model.pkl')
    labels = get_classification_or_labes_model_files('labels.pkl')
    app.run(debug=True)
