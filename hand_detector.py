import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque
from scipy.spatial.transform import Rotation

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
        
        # 3D cube properties
        self.cube_rotation = {'x': 0, 'y': 0, 'z': 0}
        self.previous_hand_centers = []
        self.hand_orientations = []
        self.target_rotation = {'x': 0, 'y': 0, 'z': 0}
        self.rotation_smoothing = 0.2  # Smoothing factor for rotation
        
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
    
    def create_rotation_matrix_x(self, angle):
        """Create rotation matrix for X axis"""
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    
    def create_rotation_matrix_y(self, angle):
        """Create rotation matrix for Y axis"""
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
    def create_rotation_matrix_z(self, angle):
        """Create rotation matrix for Z axis"""
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    def get_hand_orientation(self, hand_landmarks):
        """Calculate X, Y, Z rotation angles from hand landmarks"""
        h, w, _ = (480, 640, 3)  # Default frame size
        
        # Get key points for orientation
        wrist = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]
        index_mcp = hand_landmarks.landmark[5]
        middle_tip = hand_landmarks.landmark[12]
        
        # Convert to normalized coordinates
        wrist_norm = np.array([wrist.x, wrist.y, wrist.z])
        middle_norm = np.array([middle_mcp.x, middle_mcp.y, middle_mcp.z])
        index_norm = np.array([index_mcp.x, index_mcp.y, index_mcp.z])
        middle_tip_norm = np.array([middle_tip.x, middle_tip.y, middle_tip.z])
        
        # Calculate primary direction (from wrist to middle finger)
        forward = middle_norm - wrist_norm
        forward = forward / np.linalg.norm(forward)
        
        # Calculate secondary direction (from wrist to index finger)
        right = index_norm - wrist_norm
        # Make orthogonal to forward
        right = right - np.dot(right, forward) * forward
        if np.linalg.norm(right) > 0:
            right = right / np.linalg.norm(right)
        else:
            # Fallback if calculation fails
            right = np.array([forward[1], -forward[0], 0])
            right = right / np.linalg.norm(right)
        
        # Calculate up vector
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)
        
        # Extract Euler angles from rotation matrix
        rotation_matrix = np.column_stack([right, up, forward])
        
        # Ensure proper rotation matrix
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, 0] *= -1
        
        try:
            # Convert to Euler angles (X, Y, Z order)
            sy = math.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
            singular = sy < 1e-6
            
            if not singular:
                x = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
                y = math.atan2(-rotation_matrix[2,0], sy)
                z = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
            else:
                x = math.atan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                y = math.atan2(-rotation_matrix[2,0], sy)
                z = 0
            
            return {'x': x, 'y': y, 'z': z}
        except:
            # Fallback to zero rotation
            return {'x': 0, 'y': 0, 'z': 0}
    
    def project_3d_to_2d(self, point_3d, center_x, center_y, focal_length=500):
        """Project 3D point to 2D with perspective projection"""
        # Perspective projection
        z = point_3d[2] + focal_length  # Add focal length to avoid division by zero
        x_2d = center_x + (point_3d[0] * focal_length) / z
        y_2d = center_y + (point_3d[1] * focal_length) / z
        return (int(x_2d), int(y_2d))
    
    def draw_3d_cube(self, frame, center_x, center_y, size, rotation_angles):
        """Draw perfect 3D cube with X, Y, Z rotation"""
        # Define perfect cube vertices with equal sides
        s = size / 2.0  # Half size for perfect cube
        vertices = np.array([
            [-s, -s, -s],  # 0: back-bottom-left
            [s, -s, -s],   # 1: back-bottom-right
            [s, s, -s],    # 2: back-top-right
            [-s, s, -s],   # 3: back-top-left
            [-s, -s, s],   # 4: front-bottom-left
            [s, -s, s],    # 5: front-bottom-right
            [s, s, s],     # 6: front-top-right
            [-s, s, s]     # 7: front-top-left
        ])
        
        # Create combined rotation matrix
        rot_x = self.create_rotation_matrix_x(rotation_angles['x'])
        rot_y = self.create_rotation_matrix_y(rotation_angles['y'])
        rot_z = self.create_rotation_matrix_z(rotation_angles['z'])
        
        # Apply rotations in order: X, then Y, then Z
        rotation_matrix = np.dot(np.dot(rot_z, rot_y), rot_x)
        
        # Apply rotation to vertices
        rotated_vertices = np.dot(vertices, rotation_matrix.T)
        
        # Project to 2D with perspective
        points_2d = [self.project_3d_to_2d(v, center_x, center_y) for v in rotated_vertices]
        
        # Define cube faces with consistent vertex ordering
        faces = [
            ([0, 1, 2, 3], (120, 120, 120)),  # Back face - dark gray
            ([4, 7, 6, 5], (255, 255, 255)),  # Front face - white
            ([0, 4, 5, 1], (160, 160, 160)),  # Bottom face - medium gray
            ([2, 6, 7, 3], (200, 200, 200)),  # Top face - light gray
            ([0, 3, 7, 4], (180, 180, 180)),  # Left face - medium-light gray
            ([1, 5, 6, 2], (220, 220, 220)),  # Right face - very light gray
        ]
        
        # Calculate face depths for sorting (painter's algorithm)
        face_depths = []
        for face_indices, _ in faces:
            face_vertices = [rotated_vertices[i] for i in face_indices]
            avg_depth = np.mean([v[2] for v in face_vertices])
            face_depths.append((avg_depth, face_indices))
        
        # Sort faces by depth (back to front)
        face_depths.sort(key=lambda x: x[0])
        
        # Draw faces
        for depth, face_indices in face_depths:
            face_points = [points_2d[i] for i in face_indices]
            face_points_np = np.array(face_points, np.int32)
            
            # Find the corresponding color
            face_color = next(color for indices, color in faces if indices == face_indices)
            
            # Fill face
            cv2.fillPoly(frame, [face_points_np], face_color)
            
            # Draw edges
            for i in range(4):
                start = face_points[i]
                end = face_points[(i + 1) % 4]
                cv2.line(frame, start, end, (0, 0, 0), 2)
    
    def draw_neon_orb(self, frame, x, y, hand_distance, hand_orientations):
        """Draw simple white cube with size based on hand distance"""
        # Calculate cube size based on hand distance (no maximum limit)
        # When hands are close -> small cube, when hands are far -> big cube
        min_distance = 50
        min_size = 10
        
        # Only apply minimum distance constraint, no maximum
        if hand_distance < min_distance:
            hand_distance = min_distance
        
        # Direct proportional scaling (no maximum limit)
        # Scale factor: 1 pixel distance = 0.1 pixel cube size
        cube_size = min_size + (hand_distance - min_distance) * 0.1
        
        # Calculate rotation based on hand orientations
        if len(hand_orientations) == 2:
            try:
                # Average the rotations of both hands for each axis
                avg_rotation = {
                    'x': (hand_orientations[0]['x'] + hand_orientations[1]['x']) / 2,
                    'y': (hand_orientations[0]['y'] + hand_orientations[1]['y']) / 2,
                    'z': (hand_orientations[0]['z'] + hand_orientations[1]['z']) / 2
                }
                self.target_rotation = avg_rotation
            except:
                # Fallback to current rotation if averaging fails
                self.target_rotation = self.cube_rotation
        else:
            self.target_rotation = self.cube_rotation
        
        # Smooth rotation interpolation for each axis
        try:
            for axis in ['x', 'y', 'z']:
                current = self.cube_rotation[axis]
                target = self.target_rotation[axis]
                
                # Handle angle wrapping
                diff = target - current
                if diff > math.pi:
                    diff -= 2 * math.pi
                elif diff < -math.pi:
                    diff += 2 * math.pi
                
                # Smooth interpolation
                self.cube_rotation[axis] = current + diff * self.rotation_smoothing
        except:
            # Fallback to current rotation if interpolation fails
            pass
        
        # Draw the 3D cube
        self.draw_3d_cube(frame, x, y, cube_size, self.cube_rotation)
    
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
            # Draw white orb between two hands if both are detected
            if len(results.multi_hand_landmarks) == 2:
                # Calculate centers and orientations of both hands
                hand_centers = []
                hand_orientations = []
                
                for hand_landmarks in results.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                    center_x = int(np.mean(x_coords))
                    center_y = int(np.mean(y_coords))
                    hand_centers.append((center_x, center_y))
                    
                    # Get hand orientation
                    orientation = self.get_hand_orientation(hand_landmarks)
                    hand_orientations.append(orientation)
                
                # Calculate midpoint and distance between hands
                mid_x = (hand_centers[0][0] + hand_centers[1][0]) // 2
                mid_y = (hand_centers[0][1] + hand_centers[1][1]) // 2
                
                # Calculate distance between hand centers
                dx = hand_centers[1][0] - hand_centers[0][0]
                dy = hand_centers[1][1] - hand_centers[0][1]
                hand_distance = math.sqrt(dx*dx + dy*dy)
                
                # Draw dynamic 3D cube with rotation
                self.draw_neon_orb(frame, mid_x, mid_y, hand_distance, hand_orientations)
            
            # Draw hand effects for each hand
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
