import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
from collections import deque
from scipy import signal
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ==========================================
# 1. SCIENTIFIC SIGNAL PROCESSING (For Heart Rate)
# ==========================================

def calculate_angle(a, b, c):
    """Calculates angle between 3 joints (Shoulder-Elbow-Wrist)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def process_pulse_signal(green_data, fps=30):
    """
    Analyzes face color changes to find heart beat using SciPy.
    """
    if len(green_data) < 60: return 0
    
    # 1. Prepare Data
    raw = np.array(green_data)
    
    # 2. Detrend (Remove lighting changes)
    detrended = signal.detrend(raw)
    
    # 3. Normalize
    normalized = (detrended - np.mean(detrended)) / (np.std(detrended) + 1e-5)
    
    # 4. Smooth the signal
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(normalized, kernel, mode='same')
    
    # 5. Find Peaks (Heart beats)
    peaks, _ = signal.find_peaks(smoothed, height=0.5, distance=fps*0.5)
    
    # 6. Calculate BPM
    if len(peaks) < 2: return 0
    gaps = np.diff(peaks)
    avg_gap = np.mean(gaps)
    bpm = (60 * fps) / avg_gap
    return int(bpm)

# ==========================================
# 2. AI MOVEMENT ENGINE
# ==========================================

class BioMotionProcessor(VideoTransformerBase):
    def __init__(self):
        # Setup MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        
        # --- BUFFERS ---
        # For Heart Rate
        self.green_history = deque(maxlen=300) # Store 300 frames of color data
        self.current_bpm = 0
        self.bpm_buffer = deque(maxlen=5) 
        
        # For Movement Visualization (Ghost Line)
        self.wrist_path = deque(maxlen=50) # Longer trail
        
        # Performance
        self.prev_frame_time = 0
        self.fps = 0

    def draw_hud(self, img, angle, bpm, posture_ok):
        """Draws the Advanced Interface."""
        h, w, _ = img.shape
        
        # Dark Side Panel
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (250, h), (10, 10, 10), -1)
        alpha = 0.7
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        # 1. PULSE DISPLAY
        cv2.putText(img, "BIO-METRICS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
        
        # Dynamic Color for BPM
        bpm_color = (0, 255, 0) # Green
        if bpm > 100: bpm_color = (0, 0, 255) # Red if high
        
        cv2.putText(img, "HEART RATE", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        val_text = f"{bpm}" if bpm > 0 else "--"
        cv2.putText(img, f"{val_text} BPM", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, bpm_color, 2)

        # 2. MOTION DISPLAY
        cv2.putText(img, "JOINT MOTION", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, f"Angle: {int(angle)}", (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 3. VISUAL FEEDBACK BAR
        # A bar that moves up and down as you move your arm
        bar_h = int(np.interp(angle, [0, 180], [300, 240]))
        cv2.rectangle(img, (20, 240), (200, 250), (50, 50, 50), -1) # Track
        # Using a circle as a slider
        slider_x = int(np.interp(angle, [0, 180], [20, 200]))
        cv2.circle(img, (slider_x, 245), 8, (255, 200, 0), -1)

        # 4. POSTURE ALERT
        if not posture_ok:
            cv2.rectangle(img, (20, h-60), (230, h-20), (0, 0, 255), -1)
            cv2.putText(img, "BAD POSTURE", (40, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return img

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # FPS Calculation
        curr_time = time.time()
        self.fps = 1 / (curr_time - self.prev_frame_time) if self.prev_frame_time > 0 else 30
        self.prev_frame_time = curr_time
        
        results = self.pose.process(img_rgb)
        
        angle = 0
        posture_ok = True
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # --- MOTION TRACKING ---
            l_sh = [lm[11].x, lm[11].y]
            l_el = [lm[13].x, lm[13].y]
            l_wr = [lm[15].x, lm[15].y]
            r_sh = [lm[12].x, lm[12].y] # For posture symmetry
            
            angle = calculate_angle(l_sh, l_el, l_wr)
            
            # Check Posture (Shoulder Tilt)
            if abs(l_sh[1] - r_sh[1]) > 0.05:
                posture_ok = False
            
            # Ghost Line (Wrist Trajectory)
            wrist_px = (int(l_wr[0]*w), int(l_wr[1]*h))
            self.wrist_path.append(wrist_px)
            
            for i in range(1, len(self.wrist_path)):
                # Fade effect
                alpha = int(255 * (i / len(self.wrist_path)))
                color = (255, 255, 0) # Cyan
                cv2.line(img, self.wrist_path[i-1], self.wrist_path[i], color, 2)
            
            # Draw Skeleton
            # Dynamic color: Red if arm bent, Green if straight
            skel_color = (0, 0, 255) if angle < 90 else (0, 255, 0)
            
            self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                 self.mp_drawing.DrawingSpec(color=skel_color, thickness=2, circle_radius=2),
                 self.mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1))
            
            # --- PULSE TRACKING ---
            nose = lm[0]
            nx, ny = int(nose.x * w), int(nose.y * h)
            face_w = int(abs(lm[7].x - lm[8].x) * w) # Width of face
            
            # ROI (Forehead)
            fx = nx - int(face_w * 0.25)
            fy = ny - int(face_w * 0.8)
            fw = int(face_w * 0.5)
            fh = int(face_w * 0.2)
            
            if 0 < fx < w and 0 < fy < h:
                # Tech Target Brackets (Corner Only)
                l_len = 10
                # Top Left
                cv2.line(img, (fx, fy), (fx + l_len, fy), (0,255,255), 2)
                cv2.line(img, (fx, fy), (fx, fy + l_len), (0,255,255), 2)
                # Bottom Right
                cv2.line(img, (fx+fw, fy+fh), (fx+fw - l_len, fy+fh), (0,255,255), 2)
                cv2.line(img, (fx+fw, fy+fh), (fx+fw, fy+fh - l_len), (0,255,255), 2)

                roi = img[fy:fy+fh, fx:fx+fw]
                if roi.size > 0:
                    avg_g = np.mean(roi[:,:,1])
                    self.green_history.append(avg_g)
                    
                    if len(self.green_history) % 30 == 0:
                        calc = process_pulse_signal(self.green_history, int(self.fps))
                        if 40 < calc < 190:
                            self.bpm_buffer.append(calc)
                            self.current_bpm = int(np.mean(self.bpm_buffer))

        img = self.draw_hud(img, angle, self.current_bpm, posture_ok)
        return img

# ==========================================
# 3. UI
# ==========================================
st.set_page_config(page_title="Bio-Monitor", layout="wide")
st.markdown("""
<style>
.main { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)

st.title("Live Bio-Metric & Motion Analysis")

webrtc_streamer(
    key="bio-monitor", 
    mode=WebRtcMode.SENDRECV, 
    video_transformer_factory=BioMotionProcessor,
    async_processing=True,
)
