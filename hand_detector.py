import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque

class HandDetector:
    def __init__(self):
        # Use the legacy MediaPipe solution approach
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Minimal particle system for performance
        self.particles = []
        # Removed trails completely to eliminate lines between hands
        
        # Simple color palette
        self.colors = [
            (100, 200, 255),    # Light blue
            (255, 100, 150),    # Pink
        ]
        
        # Animation timing
        self.animation_time = 0
        
    def create_particles(self, x, y, count=3):
        """Create minimal particle effect for performance"""
        for _ in range(count):
            particle = {
                'x': x,
                'y': y,
                'vx': np.random.uniform(-2, 2),
                'vy': np.random.uniform(-2, 2),
                'life': 15,
                'max_life': 15,
                'color': self.colors[np.random.randint(0, len(self.colors))],
                'size': 2
            }
            self.particles.append(particle)
    
    def update_particles(self):
        """Update particle positions and remove dead particles"""
        for particle in self.particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['vy'] += 0.2  # Gravity
            particle['life'] -= 1
            
            if particle['life'] <= 0:
                self.particles.remove(particle)
        
        self.animation_time += 1
    
    def draw_particles(self, frame):
        """Draw simple particles with clean effects"""
        for particle in self.particles:
            alpha = particle['life'] / particle['max_life']
            size = int(particle['size'] * alpha)
            if size > 0:
                color = tuple(int(c * alpha) for c in particle['color'])
                cv2.circle(frame, (int(particle['x']), int(particle['y'])), 
                          size, color, -1)
    
    def draw_hand_effects(self, frame, hand_landmarks):
        """Draw minimal hand effects for performance"""
        h, w, _ = frame.shape
        
        # Calculate hand center
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        
        # Minimal particle effects
        if np.random.random() < 0.05:  # Very rare particles
            self.create_particles(center_x, center_y, 2)
        
        # Draw simple connections
        connections = self.mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            start_point = (int(hand_landmarks.landmark[start_idx].x * w),
                          int(hand_landmarks.landmark[start_idx].y * h))
            end_point = (int(hand_landmarks.landmark[end_idx].x * w),
                        int(hand_landmarks.landmark[end_idx].y * h))
            
            # Simple color for connections
            cv2.line(frame, start_point, end_point, (100, 200, 255), 1)
        
        # Draw simple landmarks
        for idx, landmark in enumerate(hand_landmarks.landmark):
            x, y = int(landmark.x * w), int(landmark.y * h)
            
            # Simple colors for different parts of hand
            if idx == 0:  # Wrist
                color = (255, 100, 150)
                radius = 5
            elif idx in [4, 8, 12, 16, 20]:  # Fingertips
                color = (150, 255, 150)
                radius = 3
            else:  # Other joints
                color = (255, 200, 100)
                radius = 2
            
            cv2.circle(frame, (x, y), radius, color, -1)
    
    def process_frame(self, frame):
        """Process frame and detect hands"""
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = self.hands.process(rgb_frame)
        
        # Update and draw particles
        self.update_particles()
        self.draw_particles(frame)
        
        # Draw hand landmarks and connections
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.draw_hand_effects(frame, hand_landmarks)
        
        # Remove border for cleaner look
        
        # Get frame dimensions for text positioning
        h, w, _ = frame.shape
        
        # Add simple status text
        hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        status_text = f"Hands: {hand_count}"
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
        
        # Add simple instructions
        cv2.putText(frame, "Press 'q' to quit", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (200, 200, 200), 2)
        
        return frame
    
    def cleanup(self):
        """Clean up resources"""
        self.hands.close()

def main():
    print("🎯 Hand Detector Starting...")
    print("📹 Make sure your camera is connected!")
    print("🎨 Get ready for some cool visual effects!")
    print("⚠️  Press 'q' to quit the application")
    
    # Initialize hand detector
    detector = HandDetector()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera!")
        return
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Camera initialized successfully!")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Could not read frame from camera!")
                break
            
            # Process frame and detect hands
            processed_frame = detector.process_frame(frame)
            
            # Display the result
            cv2.imshow('🤖 Cool Hand Detector 🤖', processed_frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Quitting application...")
                break
                
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user!")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()
        print("✅ Cleanup complete!")

if __name__ == "__main__":
    main()
