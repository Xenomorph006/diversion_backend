from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
from datetime import datetime
import numpy as np

from models.emotion_model import EmotionDetector
from firebase.config import save_emotion_to_firebase, bucket

# Initialize FastAPI
app = FastAPI(title="Emotion Detection API", 
              description="ML API for emotion and stress detection")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize emotion detector
emotion_detector = EmotionDetector()

# Request/Response Models
class ImageRequest(BaseModel):
    image_base64: str
    user_id: Optional[str] = "default_user"
    save_to_firebase: Optional[bool] = True

class EmotionResponse(BaseModel):
    status: str
    user_id: str
    timestamp: str
    emotions: dict
    dominant_emotion: str
    stress_score: float
    anxiety_score: float
    eye_metrics: dict
    face_metrics: dict
    overall_status: str
    firebase_id: Optional[str] = None
    message: Optional[str] = None

@app.get("/")
async def root():
    return {
        "name": "Emotion Detection API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Analyze emotion from base64 image",
            "/health": "GET - Health check",
            "/stats/{user_id}": "GET - Get user statistics"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": emotion_detector is not None
    }

@app.post("/predict", response_model=EmotionResponse)
async def predict_emotion(request: ImageRequest, background_tasks: BackgroundTasks):
    """
    Analyze emotion from base64 image
    - Returns emotion scores, stress/anxiety levels, and eye metrics
    - Optionally saves to Firebase
    """
    try:
        # Run prediction
        result = emotion_detector.predict(
            request.image_base64, 
            user_id=request.user_id
        )
        
        # Check for errors
        if result.get('overall_status') == 'ERROR':
            raise HTTPException(status_code=400, detail=result.get('error', 'Analysis failed'))
        
        # Save to Firebase in background if requested
        firebase_id = None
        if request.save_to_firebase and result.get('overall_status') != 'NO_FACE_DETECTED':
            background_tasks.add_task(
                save_to_firebase_task,
                request.user_id,
                result
            )
        
        # Prepare response
        response = EmotionResponse(
            status="success",
            user_id=request.user_id,
            timestamp=result['timestamp'],
            emotions=result['emotions'],
            dominant_emotion=result['dominant_emotion'],
            stress_score=result['stress_score'],
            anxiety_score=result['anxiety_score'],
            eye_metrics=result['eye_metrics'],
            face_metrics=result['face_metrics'],
            overall_status=result['overall_status'],
            firebase_id=firebase_id
        )
        
        # Print to console (for testing)
        print(f"\n{'='*50}")
        print(f"📊 Analysis Result for User: {request.user_id}")
        print(f"{'='*50}")
        print(f"😊 Dominant Emotion: {result['dominant_emotion']}")
        print(f"📈 Stress Score: {result['stress_score']:.1f}/100")
        print(f"😰 Anxiety Score: {result['anxiety_score']:.1f}/100")
        print(f"👁️ Eye Contact: {result['eye_metrics']['eye_contact_score']:.1f}%")
        print(f"📋 Status: {result['overall_status']}")
        print(f"{'='*50}\n")
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def save_to_firebase_task(user_id, result):
    """Background task to save to Firebase"""
    try:
        doc_id = save_emotion_to_firebase(user_id, result)
        if doc_id:
            print(f"✅ Saved to Firebase with ID: {doc_id}")
    except Exception as e:
        print(f"❌ Firebase save failed: {e}")

@app.get("/stats/{user_id}")
async def get_user_stats(user_id: str):
    """Get emotion statistics for a specific user"""
    try:
        from firebase.config import db
        
        # Query last 100 entries for user
        docs = db.collection('emotion_analysis')\
                 .where('user_id', '==', user_id)\
                 .order_by('timestamp', direction='DESCENDING')\
                 .limit(100)\
                 .stream()
        
        stats = []
        for doc in docs:
            data = doc.to_dict()
            stats.append({
                'id': doc.id,
                'timestamp': data.get('timestamp'),
                'stress_score': data.get('stress_score'),
                'anxiety_score': data.get('anxiety_score'),
                'dominant_emotion': data.get('emotions', {}),
                'overall_status': data.get('overall_status')
            })
        
        # Calculate averages
        if stats:
            avg_stress = np.mean([s['stress_score'] for s in stats if s['stress_score']])
            avg_anxiety = np.mean([s['anxiety_score'] for s in stats if s['anxiety_score']])
        else:
            avg_stress = avg_anxiety = 0
        
        return {
            'user_id': user_id,
            'total_entries': len(stats),
            'average_stress': round(avg_stress, 2),
            'average_anxiety': round(avg_anxiety, 2),
            'recent_entries': stats[:10]  # Last 10 entries
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting Emotion Detection API...")
    print("📡 Server will run on http://0.0.0.0:8000")
    print("📊 Make sure Firebase is configured in .env file")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)