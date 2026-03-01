from deepface import DeepFace
import cv2
import numpy as np
from .face_utils import FaceAnalyzer
import base64
from io import BytesIO
from PIL import Image
import json

class EmotionDetector:
    def __init__(self):
        self.face_analyzer = FaceAnalyzer()
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
    def decode_base64_image(self, base64_string):
        """Decode base64 string to image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return image
        except Exception as e:
            raise ValueError(f"Failed to decode image: {e}")
    
    def analyze_emotions(self, image):
        """Analyze emotions using DeepFace"""
        try:
            # Remove the 'silent' parameter - it doesn't exist in this version
            result = DeepFace.analyze(
                img_path=image,
                actions=['emotion', 'age', 'gender', 'race'],
                enforce_detection=False  # Keep this to prevent errors when no face detected
            )
            
            if isinstance(result, list):
                result = result[0]
            
            return result
        except Exception as e:
            print(f"DeepFace error: {e}")
            return None
    
    def calculate_stress_score(self, emotions, eye_metrics):
        """Calculate stress score based on emotions and eye metrics"""
        weights = {
            'angry': 0.3,
            'disgust': 0.2,
            'fear': 0.25,
            'sad': 0.15,
            'happy': -0.3,
            'surprise': 0.1,
            'neutral': -0.1
        }
        
        # Base stress from emotions
        stress_from_emotions = 0
        for emotion, score in emotions.items():
            stress_from_emotions += score * weights.get(emotion, 0)
        
        # Normalize to 0-100
        stress_from_emotions = (stress_from_emotions + 0.3) * 100
        
        # Eye metrics contribution
        eye_stress = (
            eye_metrics.get('blink_rate', 0) * 0.3 +
            (100 - eye_metrics.get('eye_contact_score', 50)) * 0.3 +
            eye_metrics.get('pupil_dilation', 50) * 0.2 +
            eye_metrics.get('saccadic_rate', 0) * 0.2
        )
        
        # Combined score
        total_stress = (stress_from_emotions * 0.6) + (eye_stress * 0.4)
        
        # Cap between 0-100
        total_stress = max(0, min(100, total_stress))
        return total_stress
    def calculate_anxiety_score(self, emotions, eye_metrics):
        """Calculate anxiety score (0-100)"""
        anxiety_emotions = ['fear', 'angry']
        anxiety_base = sum(emotions.get(e, 0) for e in anxiety_emotions if e in emotions) * 50
        
        # Eye indicators of anxiety
        eye_anxiety = (
            (100 - eye_metrics.get('eye_contact_score', 50)) * 0.4 +
            eye_metrics.get('blink_rate', 0) * 0.3 +
            eye_metrics.get('saccadic_rate', 0) * 0.3
        )
        
        anxiety_score = (anxiety_base * 0.5) + (eye_anxiety * 0.5)
        
        # Cap between 0-100
        anxiety_score = max(0, min(100, anxiety_score))
        return anxiety_score
    
    def get_overall_status(self, stress_score, anxiety_score, primary_emotion):
        """Determine overall emotional status"""
        if stress_score > 70 or anxiety_score > 70:
            return "HIGH_STRESS"
        elif stress_score > 50 or anxiety_score > 50:
            return "MODERATE_STRESS"
        elif primary_emotion in ['happy', 'neutral']:
            return "CALM"
        else:
            return "MONITORING"
    
    def predict(self, base64_image, user_id="default_user"):
        """Main prediction function"""
        try:
            # Decode image
            image = self.decode_base64_image(base64_image)
            
            # Analyze emotions
            emotion_result = self.analyze_emotions(image)
            
            # Analyze eye movement
            eye_metrics = self.face_analyzer.analyze_eye_movement(image)
            
            if emotion_result:
                emotions = emotion_result.get('emotion', {})
                dominant_emotion = emotion_result.get('dominant_emotion', 'neutral')
                
                # Calculate scores
                stress_score = self.calculate_stress_score(emotions, eye_metrics)
                anxiety_score = self.calculate_anxiety_score(emotions, eye_metrics)
                
                # Prepare result
                result = {
                    'user_id': user_id,
                    'timestamp': np.datetime64('now').astype(str),
                    'emotions': {
                        'angry': emotions.get('angry', 0),
                        'sad': emotions.get('sad', 0),
                        'happy': emotions.get('happy', 0),
                        'neutral': emotions.get('neutral', 0),
                        'fear': emotions.get('fear', 0),
                        'surprise': emotions.get('surprise', 0),
                        'disgust': emotions.get('disgust', 0)
                    },
                    'dominant_emotion': dominant_emotion,
                    'stress_score': round(stress_score, 2),
                    'anxiety_score': round(anxiety_score, 2),
                    'eye_metrics': {
                        k: round(v, 2) for k, v in eye_metrics.items()
                    },
                    'face_metrics': {
                        'age': emotion_result.get('age', 0),
                        'gender': emotion_result.get('dominant_gender', 'unknown'),
                        'race': emotion_result.get('dominant_race', 'unknown')
                    },
                    'overall_status': self.get_overall_status(
                        stress_score, anxiety_score, dominant_emotion
                    )
                }
                
                return result
            else:
                # Fallback if DeepFace fails
                return self.get_fallback_result(eye_metrics, user_id)
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return self.get_error_result(str(e), user_id)
    
    def get_fallback_result(self, eye_metrics, user_id):
        """Fallback when emotion detection fails"""
        return {
            'user_id': user_id,
            'timestamp': np.datetime64('now').astype(str),
            'emotions': {e: 0 for e in self.emotion_labels},
            'dominant_emotion': 'unknown',
            'stress_score': min(100, eye_metrics.get('blink_rate', 0) * 2),  # Cap at 100
            'anxiety_score': min(100, eye_metrics.get('saccadic_rate', 0) * 2),  # Cap at 100
            'eye_metrics': {k: round(v, 2) for k, v in eye_metrics.items()},
            'face_metrics': {'error': 'Face not detected clearly'},
            'overall_status': 'NO_FACE_DETECTED'
        }
    
    def get_error_result(self, error_msg, user_id):
        """Return error result"""
        return {
            'user_id': user_id,
            'timestamp': np.datetime64('now').astype(str),
            'error': error_msg,
            'overall_status': 'ERROR'
        }