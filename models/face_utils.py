import cv2
import numpy as np
import mediapipe as mp
import dlib
from scipy.spatial import distance as dist

class FaceAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # Eye landmark indices for MediaPipe
        self.LEFT_EYE_INDICES = [33, 133, 157, 158, 159, 160, 161, 173]
        self.RIGHT_EYE_INDICES = [362, 263, 387, 388, 389, 390, 391, 398]
        
    def eye_aspect_ratio(self, eye_points):
        """Calculate Eye Aspect Ratio for blink detection"""
        # Vertical eye landmarks
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        # Horizontal eye landmarks
        C = dist.euclidean(eye_points[0], eye_points[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def analyze_eye_movement(self, image):
        """Analyze eye movement patterns"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        eye_metrics = {
            'blink_rate': 0,
            'eye_contact_score': 0,
            'pupil_dilation': 0,
            'eye_movement_stability': 0,
            'saccadic_rate': 0
        }
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            # Extract eye landmarks
            left_eye = []
            right_eye = []
            
            for idx in self.LEFT_EYE_INDICES:
                point = landmarks.landmark[idx]
                left_eye.append([point.x * w, point.y * h])
            
            for idx in self.RIGHT_EYE_INDICES:
                point = landmarks.landmark[idx]
                right_eye.append([point.x * w, point.y * h])
            
            # Calculate EAR for both eyes
            left_ear = self.eye_aspect_ratio(np.array(left_eye))
            right_ear = self.eye_aspect_ratio(np.array(right_eye))
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Eye metrics calculation
            eye_metrics['blink_rate'] = self.calculate_blink_rate(avg_ear)
            eye_metrics['eye_contact_score'] = self.calculate_eye_contact(avg_ear)
            eye_metrics['pupil_dilation'] = self.estimate_pupil_dilation(image, left_eye, right_eye)
            eye_metrics['eye_movement_stability'] = self.calculate_stability(avg_ear)
            eye_metrics['saccadic_rate'] = self.calculate_saccadic_rate(avg_ear)
            
        return eye_metrics
    
    def calculate_blink_rate(self, ear):
        """Calculate blink rate (0-100)"""
        # Lower EAR indicates blink
        if ear < 0.2:
            return min(100, (0.25 - ear) * 500)
        return 0
    
    def calculate_eye_contact(self, ear):
        """Calculate eye contact score (0-100)"""
        # Higher EAR indicates open eyes, good contact
        return min(100, ear * 150)
    
    def estimate_pupil_dilation(self, image, left_eye, right_eye):
        """Estimate pupil dilation from eye region intensity"""
        try:
            # Get eye regions
            left_eye_region = self.get_eye_region(image, left_eye)
            right_eye_region = self.get_eye_region(image, right_eye)
            
            # Average intensity (darker = more dilated)
            left_intensity = np.mean(left_eye_region) if left_eye_region.size > 0 else 0
            right_intensity = np.mean(right_eye_region) if right_eye_region.size > 0 else 0
            avg_intensity = (left_intensity + right_intensity) / 2
            
            # Convert to score (0-100, higher = more dilated)
            dilation_score = max(0, min(100, 100 - (avg_intensity / 2.55)))
            return dilation_score
        except:
            return 50  # Default value
    
    def get_eye_region(self, image, eye_points):
        """Extract eye region from image"""
        points = np.array(eye_points, dtype=np.int32)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        eye_region = cv2.bitwise_and(image, image, mask=mask)
        return eye_region
    
    def calculate_stability(self, ear):
        """Calculate eye movement stability"""
        # Higher stability = less movement
        return min(100, 100 - (abs(ear - 0.25) * 200))
    
    def calculate_saccadic_rate(self, ear):
        """Calculate rapid eye movement rate"""
        # Rapid changes in EAR indicate saccades
        return abs(ear - 0.25) * 200