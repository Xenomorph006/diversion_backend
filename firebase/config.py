import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize Firebase
cred = credentials.Certificate({
    "type": "service_account",
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
})

firebase_admin.initialize_app(cred, {
    'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET")
})

db = firestore.client()
bucket = storage.bucket()

def save_emotion_to_firebase(user_id, emotion_data, image_url=None):
    """Save emotion analysis results to Firebase"""
    try:
        doc_ref = db.collection('emotion_analysis').document()
        doc_ref.set({
            'user_id': user_id,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'emotions': emotion_data['emotions'],
            'stress_score': emotion_data['stress_score'],
            'anxiety_score': emotion_data['anxiety_score'],
            'eye_metrics': emotion_data['eye_metrics'],
            'face_metrics': emotion_data['face_metrics'],
            'image_url': image_url,
            'overall_status': emotion_data['overall_status']
        })
        return doc_ref.id
    except Exception as e:
        print(f"Firebase error: {e}")
        return None