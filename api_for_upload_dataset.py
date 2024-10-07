from flask import Flask, jsonify, request
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

DATASET_DIR = 'veriseti'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_image/<set_type>/<person_name>', methods=['POST'])
def upload_image(set_type, person_name):
   
    folder_path = os.path.join(DATASET_DIR, set_type, person_name)
    os.makedirs(folder_path, exist_ok=True)
        
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
        
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if create_folder_if_not_exists(folder_path,filename):
            file.save(os.path.join(folder_path, filename))
            return jsonify({"message": "File successfully uploaded", "filename": filename}), 200
        else:
            return jsonify({"message": "File alredy uploaded", "filename": filename}), 200        
    else:
        return jsonify({"error": "File type not allowed"}), 400

def create_folder_if_not_exists(folder_path,image_name):
    image_path = os.path.join(folder_path, image_name)
    if os.path.exists(image_path):
        return False
    else:
        return True

if __name__ == '__main__':
    app.run(debug=True)
