import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import math

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

@dataclass
class Landmark3D:
    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)
    
@dataclass
class FingerAngles:
    mcp: Optional[float] = None
    dip: Optional[float] = None
    pip: Optional[float] = None

def safe_norm(v: np.ndarray, epsilon: float = 1e-7) -> float:
    n = np.linalg.norm(v)
    return n if n > epsilon else epsilon

def angle_between_vectors_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    v1n = v1 / safe_norm(v1)
    v2n = v2 / safe_norm(v2)
    dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
    return math.degrees(math.acos(dot))

def flexion_angle_deg(p_prox: np.ndarray, p_joint: np.ndarray, p_dist: np.ndarray) -> float:
    v1 = p_prox - p_joint
    v2 = p_dist - p_joint
    interior = angle_between_vectors_deg(v1, v2)
    return 180.0 - interior

class MediapipeHandTracker:
    def __init__(self, model_path: str, num_hands: int = 1, min_hand_detection_confidence: float = 0.5,
               min_hand_presence_confidence: float = 0.5, min_tracking_confidence: float = 0.5) -> None:
        self.model_path = str(model_path)
            
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"hand landmarker model file not found: {self.model_path}\n"
            )

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        LandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = LandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.landmarker = HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self.landmarker.close()

    def process_bgr_frame(self, frame_bgr: np.ndarray, timestamp_ms: int):
        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return result
    
class MediapipeAngleEstimator:
    #thumb omitted due to laziness
    WRIST = 0

    FINGER_LANDMARKS = {
        "index": {"mcp": 5, "pip": 6, "dip": 7, "tip": 8},
        "middle": {"mcp": 9, "pip": 10, "dip": 11, "tip": 12},
        "ring": {"mcp": 13, "pip": 14, "dip": 15, "tip": 16},
        "pinky": {"mcp": 17, "pip": 18, "dip": 19, "tip": 20},
    }    
    
    def extract_world_landmarks(self, result, hand_index: int,) -> Optional[Dict[int, Landmark3D]]:
        if not result.hand_world_landmarks:
            return None
        
        if hand_index >= len(result.hand_world_landmarks):
            return None

        lms = result.hand_world_landmarks[hand_index]
        return {
            i: Landmark3D(lm.x, lm.y, lm.z)
            for i, lm in enumerate(lms)
        }
    
    def compute_finger_angles(self, result, hand_index: int = 0) -> Optional[Dict[int, FingerAngles]]:
        landmarks = self.extract_world_landmarks(result, hand_index)
        if landmarks is None:
            return None

        wrist = landmarks[self.WRIST].as_array()
        out: Dict[str, FingerAngles] = {}

        for finger_name, index in self.FINGER_LANDMARKS.items():
            mcp = landmarks[index["mcp"]].as_array()
            pip = landmarks[index["pip"]].as_array()
            dip = landmarks[index["dip"]].as_array()
            tip = landmarks[index["tip"]].as_array()

            mcp_angle = flexion_angle_deg(wrist, mcp, pip)
            pip_angle = flexion_angle_deg(mcp, pip, dip)
            dip_angle = flexion_angle_deg(pip, dip, tip)
            
            out[finger_name] = FingerAngles(
                mcp=mcp_angle,
                pip=pip_angle,
                dip=dip_angle,
            )

        return out
            

#drawing helper
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), 
    (13, 17), (17, 18), (18, 19), (19, 20),
]

def draw_hand_landmarks(frame: np.ndarray, result) -> None:
    h, w = frame.shape[:2]

    if not result.hand_landmarks:
        return

    for hand_landmarks in result.hand_landmarks:
        pts = []
        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            pts.append((x, y))
            cv.circle(frame, (x, y), 3, (0, 0, 255), -1)

        for i0, i1  in HAND_CONNECTIONS:
            x0, y0 = pts[i0]
            x1, y1 = pts[i1]
            cv.line(frame, (x0, y0), (x1, y1), (42, 205, 54), 2)
    
def draw_angle_text(frame: np.ndarray, finger_angles: Dict[str, FingerAngles], 
                        hand_index: int = 0, x0: int = 10, y0: int = 30) -> None:
    y = y0 + hand_index * 120
    cv.putText(frame, f"hand {hand_index}", (x0, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv.LINE_AA)
    y += 25

    for finger_name, angles in finger_angles.items():
        text = (
            f"{finger_name.capitalize():6s} | "
            f"MCP {angles.mcp:6.1f} "
            f"PIP {angles.pip:6.1f} "
            f"DIP {angles.dip:6.1f} "
        )

        cv.putText(frame, text, (x0, y), cv.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv.LINE_AA)
        y += 22
        

def main() -> None:
    model_path = "hand_landmarker.task"

    tracker = MediapipeHandTracker(
        model_path = model_path,
        num_hands = 1, 
        min_hand_detection_confidence = 0.6, 
        min_hand_presence_confidence = 0.6,
        min_tracking_confidence = 0.6
    )

    angle_estimator = MediapipeAngleEstimator()
    
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        tracker.close()
        raise RuntimeError("could not open source 0")
    
    t0 = time.perf_counter()
    prev_t = t0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.flip(frame, 1)

            timestamp_ms = int((time.perf_counter() - t0) * 1000.0)
            result = tracker.process_bgr_frame(frame, timestamp_ms)
            
            draw_hand_landmarks(frame, result)
            
            if result.hand_world_landmarks:
                for hand_index in range(len(result.hand_world_landmarks)):
                    finger_angles = angle_estimator.compute_finger_angles(result, hand_index)
                    if finger_angles is not None:
                        draw_angle_text(frame, finger_angles, hand_index=hand_index, x0=10, y0=30)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now

            cv.putText(frame, f"fps: {fps:.1f}", (10, frame.shape[0] - 12), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv.LINE_AA)
            cv.imshow("mediapipe tasks hand angles", frame)
            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
