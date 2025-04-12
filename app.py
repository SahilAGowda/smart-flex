from flask import Flask, request, jsonify, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import json
import time
import threading
import os
import sys

app = Flask(__name__, static_folder='public', template_folder='public')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Global variables to store workout data
workout_data = {
    "repCount": 0,
    "duration": 0,
    "calories": 0,
    "stage": None
}

# Flag to control the workout monitoring thread
is_workout_running = False
workout_thread = None
start_time = 0

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def detect_exercise(landmarks, workout_type):
    """Detect exercise based on pose landmarks and workout type."""
    if workout_type == "bicep_curl":
        # Get coordinates for bicep curl
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        # Calculate angle
        angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # Count rep when angle is less than 30 degrees (arm is curled)
        return angle < 30
    
    elif workout_type == "squat":
        # Get coordinates for squat
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        # Calculate angle
        angle = calculate_angle(left_hip, left_knee, left_ankle)
        
        # Count rep when angle is less than 100 degrees (squat position)
        return angle < 100
    
    elif workout_type == "pushup":
        # Get coordinates for pushup
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        # Calculate angle
        angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # Count rep when angle is less than 90 degrees (pushup position)
        return angle < 90
    
    return False

def workout_monitoring(workout_type):
    """Monitor workout using OpenCV and MediaPipe."""
    global workout_data, is_workout_running, start_time
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Initialize variables
    rep_count = 0
    stage = None
    start_time = time.time()
    
    while is_workout_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Get landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Detect exercise
            is_rep = detect_exercise(landmarks, workout_type)
            
            # Count reps
            if is_rep and stage == None:
                stage = "down"
            elif not is_rep and stage == "down":
                stage = "up"
                rep_count += 1
                
            # Calculate duration and calories
            duration = int(time.time() - start_time)
            calories = int(duration * 0.1)  # Rough estimate: 0.1 calories per second
            
            # Update workout data
            workout_data = {
                "repCount": rep_count,
                "duration": duration,
                "calories": calories,
                "stage": stage
            }
            
            # Draw rep count on frame
            cv2.putText(frame, f'Reps: {rep_count}', (10,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        # Display the frame
        cv2.imshow('Workout Monitoring', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('startworkout.html')

@app.route('/api/workout/start', methods=['POST'])
def start_workout():
    global is_workout_running, workout_thread, workout_data
    
    # Reset workout data
    workout_data = {
        "repCount": 0,
        "duration": 0,
        "calories": 0,
        "stage": None
    }
    
    # Get workout type from request
    data = request.json
    workout_type = data.get('workoutType', 'squat')
    
    # Start workout monitoring in a separate thread
    is_workout_running = True
    workout_thread = threading.Thread(target=workout_monitoring, args=(workout_type,))
    workout_thread.daemon = True
    workout_thread.start()
    
    return jsonify({"status": "success", "message": "Workout started"})

@app.route('/api/workout/data', methods=['GET'])
def get_workout_data():
    return jsonify(workout_data)

@app.route('/api/workout/end', methods=['POST'])
def end_workout():
    global is_workout_running, workout_data
    
    # Stop workout monitoring
    is_workout_running = False
    
    # Wait for thread to finish
    if workout_thread:
        workout_thread.join(timeout=1)
    
    # Return final workout data
    return jsonify(workout_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 