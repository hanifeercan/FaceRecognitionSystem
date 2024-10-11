import firebase_admin
from firebase_admin import credentials,firestore,storage
from datetime import datetime
import pytz
import uuid

credentialData = credentials.Certificate("firebase.json")
firebase_admin.initialize_app(credentialData, {
    'storageBucket': 'facerecognition-b323e.appspot.com' 
})

db = firestore.client()
email = ""

def isStaff(name,email):
    collection_ref = db.collection("sirket").document(email).collection("personnel")
    documents = collection_ref.get()

    for document in documents:
        if(ilk_harf_kucult(document.id) == name):
            return True
            
    bucket = storage.bucket()
    date_folder = datetime.now().strftime("%Y-%m-%d") 
    blob = bucket.blob(f'{email}/unknown/{date_folder}/{str(uuid.uuid4())}')
    blob.upload_from_filename("C:/Users/hanif/OneDrive/Masaüstü/face_recognition/kamera_goruntusu.jpg")
    return False

def ilk_harf_kucult(veri):
    string_veri = str(veri)
    if string_veri:
        sonuc = string_veri[0].lower() + string_veri[1:]
    else:
        sonuc = string_veri
    return sonuc   

def isLoginOrOut2(doc_ref,name,date):
    times_ref = doc_ref.collection(date)
    id_doc_ref = times_ref.document(name)
    id_document_data = id_doc_ref.get()

    if not id_document_data.exists:
        id_doc_ref.set({})
    
    id_document_data = id_doc_ref.get()
    data = id_document_data.to_dict()
    login_list = data.get("login_list", [])
    list_size = len(login_list)
        
    if list_size == 0:
        id_doc_ref.set({
           "login_list": [],
            "out_list": []
        })
        return "login",1
    else:
        out_list = data.get("out_list", [])
        list_size = len(out_list)
        
        if list_size == 0:
            return "out",1
        else:
            id = 2
            while(1):
                print("Login List:", login_list)  
                list_size = len(login_list)
                print("login_list size: " + str(login_list))
                if list_size == id-1:
                    return "login",id
                else:
            
                    out_data = id_document_data.to_dict()
                    print("Out Data:", out_data) 
                    out_list = out_data.get("out_list", [])
                    print("Out List:", out_list)
                    print("out_list size: " + str(out_list))
                    if len(out_list) == id-1:
                        return "out",id
                    else:
                        id=id+1  

def addFirebaseLoginOrOut(name,email):
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d") 
    turkey_timezone = pytz.timezone('Europe/Istanbul')
    datetime_combined = datetime.now(tz=pytz.utc).astimezone(turkey_timezone)

    collection_ref = db.collection("sirket").document(email).collection("tracking")

    doc_ref = collection_ref.document(formatted_date)
    document_data = doc_ref.get()

    if not document_data.exists:
        doc_ref.set({})

    login_or_out,id = isLoginOrOut2(doc_ref,name,formatted_date)
    print("login or out " + login_or_out + str(id))
    times_ref = doc_ref.collection(formatted_date)
    id_doc_ref = times_ref.document(name)
    id_document_data = id_doc_ref.get()
    if login_or_out == "login":
        if id_document_data.exists:
            id_doc_ref.update({
                "login_list": firestore.ArrayUnion([datetime_combined])
            })
        else:
            id_doc_ref.set({
                "login_list": [datetime_combined],
                "out_time": []
            })
    elif login_or_out == "out":
        if id_document_data.exists:
            data = id_document_data.to_dict()
            login_list = data.get("login_list",[])
            if login_list:
                firebase_time = login_list[-1]
                local_time = datetime.now(pytz.timezone('Europe/Istanbul'))
                time_difference = local_time - firebase_time
                difference_in_minutes = time_difference.total_seconds() / 60

                if difference_in_minutes < 30:
                    print("Cikis islemi icin en az 30 dakika gecmelidir.")
                else:
                    out_times_ref = doc_ref.collection(formatted_date)
                    id_doc_ref = out_times_ref.document(name)
                    id_document_data = id_doc_ref.get()
                    if id_document_data.exists:
                        id_doc_ref.update({
                            "out_list": firestore.ArrayUnion([datetime_combined])
                        })
            else:
                print("login_list bulunamadi.")
        else:
            print("Belirtilen id_doc_ref bulunamadi.")
