# test_firebase.py
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

load_dotenv()

# Initialize with your .env values
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

firebase_admin.initialize_app(cred)
db = firestore.client()

# Test write
doc_ref = db.collection('test').document('connection_test')
doc_ref.set({'status': 'connected', 'timestamp': firestore.SERVER_TIMESTAMP})
print("✅ Firebase connected successfully!")

# Test read
doc = doc_ref.get()
if doc.exists:
    print(f"✅ Test document created: {doc.to_dict()}")