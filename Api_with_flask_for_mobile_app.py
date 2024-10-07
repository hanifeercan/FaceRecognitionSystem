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

app = Flask(__name__)

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
        if 'dosya' not in request.files:
            return jsonify({'Hata1': 'Dosya yok'}), 400
        file = request.files['dosya']
        if file.filename == '':
            return jsonify({'Hata2': "Seçili dosya yok"}),400
        if file:
            img_stream = file.stream.read()
            if not img_stream:
                return jsonify({'Hata3': "Dosya boş"}), 400
            
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
                    isStaff = db.isStaff(predict_names[0])
                    if isStaff:
                        kisiler.append(predict_names[0])
                    else:
                        kisiler.append("Unknown")
                else:
                    kisiler.append("Unknown")

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
    
if __name__ == '__main__':
    facenet_keras_model = get_facenet_model_files()
    classification_model = get_classification_or_labes_model_files('svc_model.pkl')
    labels = get_classification_or_labes_model_files('labels.pkl')
    app.run(host='0.0.0.0', port=5000,debug=True)
