from os import listdir
import requests

def load_dataset(type,directory):
    for subdir in listdir(directory):
      path = directory + subdir + '/' 
      load_faces(type,path,subdir)

def load_faces(type,directory,subdir):
    for filename in listdir(directory):
        path = directory + filename
        url = 'http://localhost:5000/upload_image/' + type + '/' + subdir
        print(url)
        files = {'file': open(path,'rb')}
        response=requests.post(url,files=files)
        if response.status_code == 200:
            print(response.json())
        else:
            print(f"Error uploading {filename}: {response.text}")
