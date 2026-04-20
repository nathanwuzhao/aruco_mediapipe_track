import os
import csv
import time
import glob
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

BG        = "#1e1e2e"
PANEL     = "#46465d"
ACCENT    = "#3d63ec"
ACCENT_DK = "#2d78c3"
SUCCESS   = "#5ee07e"
DANGER    = "#e74a42"
WARNING   = "#e8d54a"
FG        = "#f2f2f7"
FG_DIM    = "#8e8e93"
MONO      = ("Courier New", 10)
SANS      = ("Consolas", 10)


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_sign_consistency(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    if np.dot(q, q_ref) < 0:
        return -q
    return q


def rvec_to_quaternion(rvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rvec)
    if theta < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = rvec.flatten() / theta
    qw = np.cos(theta / 2.0)
    qxyz = axis * np.sin(theta / 2.0)
    return np.array([qw, qxyz[0], qxyz[1], qxyz[2]], dtype=np.float64)


def quaternion_to_rvec(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    qw, qx, qy, qz = q
    angle = 2.0 * np.arccos(np.clip(qw, -1.0, 1.0))
    s = np.sqrt(max(1e-12, 1.0 - qw * qw))
    if s < 1e-8:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = np.array([qx, qy, qz], dtype=np.float64) / s
    return axis * angle


class QuaternionKF:
    def __init__(self, q_init: np.ndarray):
        self.x = normalize_quaternion(q_init.astype(np.float64))
        self.P = np.eye(4) * 0.01
        self.F = np.eye(4)
        self.H = np.eye(4)
        self.Q = np.eye(4) * 0.0005
        self.R = np.eye(4) * 0.067
        self.I = np.eye(4)

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x = normalize_quaternion(self.x)

    def update(self, z: np.ndarray) -> None:
        z = quat_sign_consistency(z, self.x)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
        self.x = normalize_quaternion(self.x)

    def get_quaternion(self) -> np.ndarray:
        return self.x


def calculate_joint_angle_orient_axis(
    rvec_prox: np.ndarray,
    rvec_dist: np.ndarray,
    axis: str = "x",
    reference_vector: Optional[np.ndarray] = None,
) -> float:
    R1, _ = cv.Rodrigues(rvec_prox)
    R2, _ = cv.Rodrigues(rvec_dist)

    axis = axis.lower()
    if axis == "x":
        v1, v2 = R1[:, 1], R2[:, 1]
        default_ref = np.array([0.0, 0.0, 1.0])
    elif axis == "y":
        v1, v2 = R1[:, 0], R2[:, 0]
        default_ref = np.array([0.0, 0.0, 1.0])
    elif axis == "z":
        v1, v2 = R1[:, 2], R2[:, 2]
        default_ref = np.array([0.0, 1.0, 0.0])
    else:
        raise ValueError("axis must be one of: x, y, z")

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    if reference_vector is None:
        reference_vector = default_ref
    reference_vector = reference_vector / np.linalg.norm(reference_vector)

    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_magnitude = np.arccos(dot_product)
    cross_product = np.cross(v1, v2)
    sign = np.sign(np.dot(cross_product, reference_vector))
    return float(sign * np.degrees(angle_magnitude))



@dataclass
class CalibrationData:
    K: np.ndarray
    D: np.ndarray


def save_calibration_yaml(path: str, K: np.ndarray, D: np.ndarray) -> None:
    fs = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
    fs.write("K", K)
    fs.write("D", D)
    fs.release()


def load_calibration_yaml(path: str) -> CalibrationData:
    fs = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open calibration file: {path}")
    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat()
    fs.release()
    if K is None or D is None:
        raise ValueError("Calibration file missing K and/or D.")
    return CalibrationData(K=K, D=D)


class ArucoToolkit:
    def __init__(
        self,
        aruco_dict_name: int = cv.aruco.DICT_4X4_50,
        marker_length_mm: float = 9.0,
        calibration: Optional[CalibrationData] = None,
    ):
        self.aruco_dict_name = aruco_dict_name
        self.marker_length_mm = marker_length_mm
        self.aruco_dict = cv.aruco.getPredefinedDictionary(aruco_dict_name)
        self.params = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(self.aruco_dict, self.params)
        self.calibration = calibration

    def set_calibration(self, calibration: CalibrationData) -> None:
        self.calibration = calibration

    def estimate_pose_ippe_square(self, corner: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.calibration is None:
            raise ValueError("Calibration must be loaded before pose estimation.")

        m = self.marker_length_mm
        obj_points = np.array([
            [-m / 2,  m / 2, 0],
            [ m / 2,  m / 2, 0],
            [ m / 2, -m / 2, 0],
            [-m / 2, -m / 2, 0],
        ], dtype=np.float32)

        img_points = corner.reshape(-1, 2).astype(np.float32)
        success, rvec, tvec = cv.solvePnP(
            objectPoints=obj_points,
            imagePoints=img_points,
            cameraMatrix=self.calibration.K,
            distCoeffs=self.calibration.D,
            flags=cv.SOLVEPNP_IPPE_SQUARE,
        )
        if success:
            return rvec, tvec
        return None, None

    def detect_markers(self, frame: np.ndarray):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        return corners, ids, rejected

    def annotate_frame(self, frame: np.ndarray, draw_axes: bool = True):
        corners, ids, _ = self.detect_markers(frame)
        pose_dict = {}
        out = frame.copy()

        if ids is not None and len(ids) > 0:
            cv.aruco.drawDetectedMarkers(out, corners, ids)
            if draw_axes and self.calibration is not None:
                for i, marker_id in enumerate(ids):
                    marker_id_int = int(marker_id[0])
                    rvec, tvec = self.estimate_pose_ippe_square(corners[i])
                    if rvec is not None:
                        pose_dict[marker_id_int] = (rvec, tvec)
                        cv.drawFrameAxes(
                            out,
                            self.calibration.K,
                            self.calibration.D,
                            rvec,
                            tvec,
                            self.marker_length_mm * 0.5,
                        )
        return out, pose_dict, ids


def calibrate_charuco_from_folder(
    image_dir: str,
    output_yaml: str,
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
    dict_name: int,
    preview: bool = True,
    status_cb=None,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    image_paths += sorted(glob.glob(os.path.join(image_dir, "*.png")))
    image_paths = list(dict.fromkeys(image_paths))

    if not image_paths:
        raise FileNotFoundError(f"No .jpg or .png images found in: {image_dir}")

    aruco_dict = cv.aruco.getPredefinedDictionary(dict_name)
    board = cv.aruco.CharucoBoard(
        size=(squares_x, squares_y),
        squareLength=float(square_length_mm),
        markerLength=float(marker_length_mm),
        dictionary=aruco_dict,
    )
    detector = cv.aruco.ArucoDetector(aruco_dict, cv.aruco.DetectorParameters())

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None
    used_images = 0

    for image_file in image_paths:
        image = cv.imread(image_file)
        if image is None:
            continue

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)

        if marker_ids is None or len(marker_ids) == 0:
            if status_cb:
                status_cb(f"Skipped (no markers): {os.path.basename(image_file)}")
            continue

        vis = image.copy()
        cv.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
        ret, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
        )

        if charuco_ids is not None and charuco_corners is not None and len(charuco_corners) > 3:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            used_images += 1
            if status_cb:
                status_cb(f"Accepted: {os.path.basename(image_file)} ({len(charuco_ids)} corners)")
        else:
            if status_cb:
                status_cb(f"Rejected (not enough ChArUco corners): {os.path.basename(image_file)}")

        if preview:
            cv.imshow("charuco_calibration_preview", vis)
            cv.waitKey(250)

    cv.destroyWindow("charuco_calibration_preview") if preview else None

    if not all_charuco_corners:
        raise RuntimeError("No valid ChArUco detections were found for calibration.")
    if image_size is None:
        raise RuntimeError("Could not determine image size.")

    ret, camera_matrix, dist_coeffs, _, _ = cv.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )

    save_calibration_yaml(output_yaml, camera_matrix, dist_coeffs)
    return camera_matrix, dist_coeffs, float(ret), used_images

def generate_charuco_board_image(
    output_path: str,
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
    dict_name: int,
    dpi: int = 300,
    margin_px: int = 40,
    status_cb=None,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    if square_length_mm <= 0 or marker_length_mm <= 0:
        raise ValueError("Square length and marker length must be positive.")
    if marker_length_mm >= square_length_mm:
        raise ValueError("Marker length must be smaller than square length.")
    if dpi <= 0:
        raise ValueError("DPI must be positive.")

    aruco_dict = cv.aruco.getPredefinedDictionary(dict_name)
    board = cv.aruco.CharucoBoard(
        size=(squares_x, squares_y),
        squareLength=float(square_length_mm),
        markerLength=float(marker_length_mm),
        dictionary=aruco_dict,
    )

    px_per_mm = dpi / 25.4
    board_w = max(1, int(round(squares_x * square_length_mm * px_per_mm)))
    board_h = max(1, int(round(squares_y * square_length_mm * px_per_mm)))
    out_size = (board_w, board_h)
    img = board.generateImage(out_size, marginSize=int(margin_px), borderBits=1)

    ok = cv.imwrite(output_path, img)
    if not ok:
        raise IOError(f"Failed to write ChArUco board image to: {output_path}")

    if status_cb:
        status_cb(f"Saved board image -> {output_path} ({board_w}x{board_h} px at {dpi} DPI)")
    return img, out_size


class CalibrationImageCapture:
    def __init__(
        self,
        camera_id: int,
        output_dir: str,
        prefix: str = "charuco",
        annotate: bool = True,
        toolkit: Optional[ArucoToolkit] = None,
        countdown_seconds: float = 0.0,
        max_images: int = 0,
    ):
        self.camera_id = camera_id
        self.output_dir = output_dir
        self.prefix = prefix
        self.annotate = annotate
        self.toolkit = toolkit
        self.countdown_seconds = max(0.0, float(countdown_seconds))
        self.max_images = max(0, int(max_images))
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True

    def run(self, status_cb=None):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        capture = cv.VideoCapture(self.camera_id)
        if not capture.isOpened():
            raise IOError(f"Cannot open camera {self.camera_id}")

        saved_count = 0
        last_capture_time = 0.0
        pending_auto_capture = False

        if status_cb:
            status_cb("Calibration capture started. Keys: s=save, a=toggle auto-capture, q=quit.")

        while not self.stop_flag:
            ret, frame = capture.read()
            if not ret:
                break

            display_frame = frame.copy()
            if self.annotate and self.toolkit is not None:
                display_frame, _, ids = self.toolkit.annotate_frame(frame, draw_axes=False)
                detected_count = 0 if ids is None else len(ids)
            else:
                corners = ids = None
                if self.toolkit is not None:
                    corners, ids, _ = self.toolkit.detect_markers(frame)
                    if ids is not None and len(ids) > 0:
                        cv.aruco.drawDetectedMarkers(display_frame, corners, ids)
                detected_count = 0 if ids is None else len(ids)

            now = time.time()
            wait_left = max(0.0, self.countdown_seconds - (now - last_capture_time)) if pending_auto_capture else 0.0
            ready_for_auto = pending_auto_capture and wait_left <= 1e-6

            status_line = f"saved: {saved_count} | markers: {detected_count}"
            if pending_auto_capture:
                status_line += f" | auto in {wait_left:.1f}s" if not ready_for_auto else " | auto capture ready"

            cv.putText(display_frame, status_line, (20, 35), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
            cv.putText(display_frame, "s=save  a=auto  q=quit", (20, 68), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
            cv.imshow("charuco_capture", display_frame)

            should_save = False
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                should_save = True
            elif key == ord("a"):
                pending_auto_capture = not pending_auto_capture
                last_capture_time = now
                if status_cb:
                    status_cb(f"Auto-capture {'enabled' if pending_auto_capture else 'disabled'}.")

            if ready_for_auto:
                should_save = True

            if should_save:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{self.prefix}_{timestamp}_{saved_count+1:03d}.jpg"
                path = str(Path(self.output_dir) / filename)
                ok = cv.imwrite(path, frame)
                if not ok:
                    raise IOError(f"Failed to save image: {path}")
                saved_count += 1
                last_capture_time = time.time()
                if status_cb:
                    status_cb(f"Saved calibration image -> {path}")
                if self.max_images > 0 and saved_count >= self.max_images:
                    if status_cb:
                        status_cb(f"Reached max image count: {self.max_images}")
                    break

        capture.release()
        cv.destroyWindow("charuco_capture")
        if status_cb:
            status_cb(f"Calibration capture stopped. Saved {saved_count} image(s).")

class LiveRecorder:
    def __init__(self, camera_id: int, output_path: str, annotate: bool, toolkit: Optional[ArucoToolkit], fps_override: int = 0):
        self.camera_id = camera_id
        self.output_path = output_path
        self.annotate = annotate
        self.toolkit = toolkit
        self.fps_override = fps_override
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True

    def run(self, status_cb=None):
        capture = cv.VideoCapture(self.camera_id)
        if not capture.isOpened():
            raise IOError(f"Cannot open camera {self.camera_id}")

        frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        framerate = int(capture.get(cv.CAP_PROP_FPS))
        if self.fps_override > 0:
            framerate = self.fps_override
        if framerate <= 0:
            framerate = 30

        writer = cv.VideoWriter(
            self.output_path,
            cv.VideoWriter_fourcc(*"mp4v"),
            framerate,
            (frame_width, frame_height),
        )

        if status_cb:
            status_cb(f"Live recording started -> {self.output_path}")

        while not self.stop_flag:
            ret, frame = capture.read()
            if not ret:
                break

            display_frame = frame
            if self.annotate and self.toolkit is not None:
                display_frame, _, _ = self.toolkit.annotate_frame(frame, draw_axes=True)

            writer.write(display_frame)
            cv.imshow("live_recording", display_frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        capture.release()
        writer.release()
        cv.destroyWindow("live_recording")
        if status_cb:
            status_cb("Live recording stopped.")


class VideoPlayer:
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True

    def run(self, status_cb=None):
        capture = cv.VideoCapture(self.input_path)
        if not capture.isOpened():
            raise IOError(f"Cannot open video file: {self.input_path}")

        fps = capture.get(cv.CAP_PROP_FPS)
        delay_ms = 1 if fps <= 0 else max(1, int(1000 / fps))

        if status_cb:
            status_cb(f"Playing video: {self.input_path}")

        while not self.stop_flag:
            ret, frame = capture.read()
            if not ret:
                break
            cv.imshow("video_player", frame)
            key = cv.waitKey(delay_ms) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                while True:
                    key2 = cv.waitKey(30) & 0xFF
                    if key2 == ord(" "):
                        break
                    if key2 == ord("q"):
                        self.stop_flag = True
                        break
        capture.release()
        cv.destroyWindow("video_player")
        if status_cb:
            status_cb("Playback finished.")


class VideoProcessor:
    def __init__(
        self,
        input_path: str,
        output_video_path: str,
        output_csv_path: str,
        toolkit: ArucoToolkit,
        marker_id_proximal: int,
        marker_id_distal: int,
        joint_axis: str = "x",
        use_quaternion_filter: bool = True,
    ):
        self.input_path = input_path
        self.output_video_path = output_video_path
        self.output_csv_path = output_csv_path
        self.toolkit = toolkit
        self.marker_id_proximal = marker_id_proximal
        self.marker_id_distal = marker_id_distal
        self.joint_axis = joint_axis
        self.use_quaternion_filter = use_quaternion_filter
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True

    def run(self, status_cb=None):
        capture = cv.VideoCapture(self.input_path)
        if not capture.isOpened():
            raise IOError(f"Cannot open file: {self.input_path}")

        frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        framerate = int(capture.get(cv.CAP_PROP_FPS))
        if framerate <= 0:
            framerate = 30

        writer = cv.VideoWriter(
            self.output_video_path,
            cv.VideoWriter_fourcc(*"mp4v"),
            framerate,
            (frame_width, frame_height),
        )

        csv_file = open(self.output_csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame",
            "prox_rvec_x", "prox_rvec_y", "prox_rvec_z",
            "prox_tvec_x", "prox_tvec_y", "prox_tvec_z",
            "dist_rvec_x", "dist_rvec_y", "dist_rvec_z",
            "dist_tvec_x", "dist_tvec_y", "dist_tvec_z",
            "joint_angle",
            "filtered_angle",
        ])

        kf_prox = None
        kf_dist = None
        frame_count = 0

        if status_cb:
            status_cb(f"Processing started -> {os.path.basename(self.input_path)}")

        while not self.stop_flag:
            ret, frame = capture.read()
            if not ret:
                break
            frame_count += 1

            annotated_frame, pose_dict, ids = self.toolkit.annotate_frame(frame, draw_axes=True)

            angle = None
            filtered_angle = None
            prox_rvec = ["", "", ""]
            prox_tvec = ["", "", ""]
            dist_rvec = ["", "", ""]
            dist_tvec = ["", "", ""]

            q_prox_filt = None
            q_dist_filt = None

            rvec_prx, tvec_prx = pose_dict.get(self.marker_id_proximal, (None, None))
            rvec_dst, tvec_dst = pose_dict.get(self.marker_id_distal, (None, None))

            if self.use_quaternion_filter and kf_prox is not None:
                kf_prox.predict()
            if self.use_quaternion_filter and kf_dist is not None:
                kf_dist.predict()

            if rvec_prx is not None:
                prox_rvec = [float(x) for x in rvec_prx.flatten()]
                prox_tvec = [float(x) for x in tvec_prx.flatten()]
                if self.use_quaternion_filter:
                    q_meas = rvec_to_quaternion(rvec_prx)
                    if kf_prox is None:
                        kf_prox = QuaternionKF(q_meas)
                    else:
                        kf_prox.update(q_meas)
                    q_prox_filt = kf_prox.get_quaternion()

            if rvec_dst is not None:
                dist_rvec = [float(x) for x in rvec_dst.flatten()]
                dist_tvec = [float(x) for x in tvec_dst.flatten()]
                if self.use_quaternion_filter:
                    q_meas = rvec_to_quaternion(rvec_dst)
                    if kf_dist is None:
                        kf_dist = QuaternionKF(q_meas)
                    else:
                        kf_dist.update(q_meas)
                    q_dist_filt = kf_dist.get_quaternion()

            if (rvec_prx is not None) and (rvec_dst is not None):
                angle = abs(calculate_joint_angle_orient_axis(rvec_prx, rvec_dst, axis=self.joint_axis))

            if self.use_quaternion_filter and (q_prox_filt is not None) and (q_dist_filt is not None):
                q_prox_rvec = quaternion_to_rvec(q_prox_filt)
                q_dist_rvec = quaternion_to_rvec(q_dist_filt)
                filtered_angle = abs(calculate_joint_angle_orient_axis(q_prox_rvec, q_dist_rvec, axis=self.joint_axis))

            text_value = filtered_angle if filtered_angle is not None else angle
            if text_value is not None:
                cv.putText(
                    annotated_frame,
                    f"angle: {text_value:.1f} deg",
                    (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )

            csv_writer.writerow([
                frame_count,
                *prox_rvec, *prox_tvec,
                *dist_rvec, *dist_tvec,
                angle, filtered_angle,
            ])

            writer.write(annotated_frame)
            cv.imshow("mp4_processor", annotated_frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if status_cb and frame_count % 30 == 0:
                status_cb(f"Processed frame {frame_count}")

        capture.release()
        writer.release()
        csv_file.close()
        cv.destroyWindow("mp4_processor")
        if status_cb:
            status_cb(f"Processing complete -> video: {self.output_video_path}, csv: {self.output_csv_path}")

#gui
class CameraUtilsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("camera_utils")
        self.geometry("1100x760")
        self.configure(bg=BG)

        self.current_worker = None
        self.current_controller = None
        self._build_style()
        self._build_vars()
        self._build_ui()

    def _build_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL, foreground=FG, padding=(14, 8), font=SANS)
        style.map("TNotebook.Tab", background=[("selected", ACCENT)], foreground=[("selected", FG)])

        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG, font=SANS)
        style.configure("Header.TLabel", background=BG, foreground=FG, font=("Consolas", 12, "bold"))
        style.configure("Panel.TLabel", background=PANEL, foreground=FG, font=SANS)
        style.configure("TButton", background=ACCENT, foreground=FG, font=SANS, padding=6)
        style.map("TButton", background=[("active", ACCENT_DK)])
        style.configure("Danger.TButton", background=DANGER, foreground=FG)
        style.map("Danger.TButton", background=[("active", "#c73b35")])
        style.configure("Success.TButton", background=SUCCESS, foreground="#111111")
        style.map("Success.TButton", background=[("active", "#45c964")])
        style.configure("TEntry", fieldbackground=PANEL, foreground=FG, insertcolor=FG)
        style.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=FG)
        style.configure("TCheckbutton", background=BG, foreground=FG, font=SANS)

    def _build_vars(self):
        # calibration
        self.calib_dir_var = tk.StringVar(value="calibration_images")
        self.calib_out_var = tk.StringVar(value="calibration_charuco.yaml")
        self.squares_x_var = tk.IntVar(value=11)
        self.squares_y_var = tk.IntVar(value=9)
        self.square_length_var = tk.DoubleVar(value=20.0)
        self.charuco_marker_length_var = tk.DoubleVar(value=15.0)
        self.calib_preview_var = tk.BooleanVar(value=True)
        self.board_output_var = tk.StringVar(value="charuco_board.png")
        self.board_dpi_var = tk.IntVar(value=300)
        self.board_margin_var = tk.IntVar(value=40)

        # calibration image capture
        self.capture_dir_var = tk.StringVar(value="calibration_images")
        self.capture_prefix_var = tk.StringVar(value="charuco")
        self.capture_camera_id_var = tk.IntVar(value=0)
        self.capture_countdown_var = tk.DoubleVar(value=2.0)
        self.capture_max_images_var = tk.IntVar(value=0)
        self.capture_annotate_var = tk.BooleanVar(value=True)

        # aruco config
        self.dict_name_var = tk.StringVar(value="DICT_4X4_50")
        self.pose_marker_length_var = tk.DoubleVar(value=9.0)
        self.calib_yaml_var = tk.StringVar(value="calibration_charuco.yaml")

        # live recording
        self.camera_id_var = tk.IntVar(value=0)
        self.live_output_var = tk.StringVar(value="live_capture.mp4")
        self.live_annotate_var = tk.BooleanVar(value=True)
        self.live_fps_override_var = tk.IntVar(value=0)

        # player
        self.play_path_var = tk.StringVar(value="test.mp4")

        # processor
        self.proc_input_var = tk.StringVar(value="test.mp4")
        self.proc_output_video_var = tk.StringVar(value="annotated_output.mp4")
        self.proc_output_csv_var = tk.StringVar(value="pose_log.csv")
        self.marker_id_prox_var = tk.IntVar(value=10)
        self.marker_id_dist_var = tk.IntVar(value=5)
        self.joint_axis_var = tk.StringVar(value="x")
        self.use_filter_var = tk.BooleanVar(value=True)

    def _build_ui(self):
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=12, pady=12)

        title = tk.Label(root, text="camera_utils", bg=BG, fg=FG, font=("Consolas", 16, "bold"))
        title.pack(anchor="w", pady=(0, 10))

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)

        self.tab_calib = ttk.Frame(notebook)
        self.tab_capture = ttk.Frame(notebook)
        self.tab_live = ttk.Frame(notebook)
        self.tab_play = ttk.Frame(notebook)
        self.tab_proc = ttk.Frame(notebook)

        notebook.add(self.tab_calib, text="ChArUco Calibration")
        notebook.add(self.tab_capture, text="Capture Calibration Images")
        notebook.add(self.tab_live, text="Live Recording")
        notebook.add(self.tab_play, text="MP4 Playback")
        notebook.add(self.tab_proc, text="MP4 ArUco + CSV")

        self._build_calib_tab(self.tab_calib)
        self._build_capture_tab(self.tab_capture)
        self._build_live_tab(self.tab_live)
        self._build_play_tab(self.tab_play)
        self._build_proc_tab(self.tab_proc)

        log_frame = ttk.Frame(root, style="Panel.TFrame")
        log_frame.pack(fill="both", expand=False, pady=(10, 0))

        log_header = ttk.Label(log_frame, text="Status", style="Panel.TLabel")
        log_header.pack(anchor="w", padx=8, pady=(8, 4))

        self.log_text = tk.Text(
            log_frame,
            bg=BG,
            fg=FG,
            insertbackground=FG,
            height=10,
            relief="flat",
            font=MONO,
            wrap="word",
        )
        self.log_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _row(self, parent, r, label, var, browse_cmd=None, width=48):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=8, pady=6)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=r, column=1, sticky="ew", padx=8, pady=6)
        if browse_cmd is not None:
            ttk.Button(parent, text="Browse", command=browse_cmd).grid(row=r, column=2, sticky="ew", padx=8, pady=6)
        return entry

    def _build_calib_tab(self, parent):
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Generate ChArUco camera calibration YAML from an image folder.", style="Header.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(10, 14))
        self._row(parent, 1, "Image folder", self.calib_dir_var, self._browse_calib_dir)
        self._row(parent, 2, "Output YAML", self.calib_out_var, self._browse_calib_out)
        self._dict_dropdown(parent, 3)
        self._entry_num(parent, 4, "Squares X", self.squares_x_var)
        self._entry_num(parent, 5, "Squares Y", self.squares_y_var)
        self._entry_num(parent, 6, "Square length (mm)", self.square_length_var)
        self._entry_num(parent, 7, "Marker length (mm)", self.charuco_marker_length_var)
        ttk.Checkbutton(parent, text="Preview detections while calibrating", variable=self.calib_preview_var).grid(row=8, column=1, sticky="w", padx=8, pady=6)
        ttk.Separator(parent, orient="horizontal").grid(row=9, column=0, columnspan=3, sticky="ew", padx=8, pady=(10, 10))
        ttk.Label(parent, text="Board image output").grid(row=10, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(parent, textvariable=self.board_output_var, width=48).grid(row=10, column=1, sticky="ew", padx=8, pady=6)
        ttk.Button(parent, text="Browse", command=self._browse_board_output).grid(row=10, column=2, sticky="ew", padx=8, pady=6)
        self._entry_num(parent, 11, "Board DPI", self.board_dpi_var)
        self._entry_num(parent, 12, "Board margin (px)", self.board_margin_var)
        btns = ttk.Frame(parent)
        btns.grid(row=13, column=1, sticky="w", padx=8, pady=14)
        ttk.Button(btns, text="Run Calibration", style="Success.TButton", command=self.start_calibration).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Generate Board Image", command=self.generate_board_image).pack(side="left")

    def _build_capture_tab(self, parent):
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Capture and save ChArUco calibration photos from the live camera.", style="Header.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(10, 14))
        self._row(parent, 1, "Output folder", self.capture_dir_var, self._browse_capture_dir)
        self._row(parent, 2, "Calibration YAML (optional, for axes)", self.calib_yaml_var, self._browse_calib_yaml)
        self._dict_dropdown(parent, 3)
        self._entry_num(parent, 4, "Pose marker length (mm)", self.pose_marker_length_var)
        self._entry_num(parent, 5, "Camera ID", self.capture_camera_id_var)
        self._entry_num(parent, 6, "Countdown / auto interval (s)", self.capture_countdown_var)
        self._entry_num(parent, 7, "Max images (0 = unlimited)", self.capture_max_images_var)
        ttk.Label(parent, text="Filename prefix").grid(row=8, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(parent, textvariable=self.capture_prefix_var, width=18).grid(row=8, column=1, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(parent, text="Annotate detected markers during capture", variable=self.capture_annotate_var).grid(row=9, column=1, sticky="w", padx=8, pady=6)
        note = "In the OpenCV window: s=save one image, a=toggle timed auto-capture, q=quit."
        ttk.Label(parent, text=note).grid(row=10, column=0, columnspan=3, sticky="w", padx=8, pady=(4, 8))
        btns = ttk.Frame(parent)
        btns.grid(row=11, column=1, sticky="w", padx=8, pady=14)
        ttk.Button(btns, text="Start Capture", style="Success.TButton", command=self.start_capture_images).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Stop", style="Danger.TButton", command=self.stop_current_task).pack(side="left")

    def _build_live_tab(self, parent):
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Record from a live camera to MP4, optionally with ArUco overlays.", style="Header.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(10, 14))
        self._row(parent, 1, "Calibration YAML", self.calib_yaml_var, self._browse_calib_yaml)
        self._dict_dropdown(parent, 2)
        self._entry_num(parent, 3, "Pose marker length (mm)", self.pose_marker_length_var)
        self._entry_num(parent, 4, "Camera ID", self.camera_id_var)
        self._entry_num(parent, 5, "FPS override (0 = camera/default)", self.live_fps_override_var)
        self._row(parent, 6, "Output MP4", self.live_output_var, self._browse_live_output)
        ttk.Checkbutton(parent, text="Annotate ArUco markers during recording", variable=self.live_annotate_var).grid(row=7, column=1, sticky="w", padx=8, pady=6)
        btns = ttk.Frame(parent)
        btns.grid(row=8, column=1, sticky="w", padx=8, pady=14)
        ttk.Button(btns, text="Start Recording", style="Success.TButton", command=self.start_live_recording).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Stop", style="Danger.TButton", command=self.stop_current_task).pack(side="left")

    def _build_play_tab(self, parent):
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Open and play an MP4. Press space to pause/resume, q to quit.", style="Header.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(10, 14))
        self._row(parent, 1, "Input video", self.play_path_var, self._browse_play_path)
        btns = ttk.Frame(parent)
        btns.grid(row=2, column=1, sticky="w", padx=8, pady=14)
        ttk.Button(btns, text="Play", style="Success.TButton", command=self.start_playback).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Stop", style="Danger.TButton", command=self.stop_current_task).pack(side="left")

    def _build_proc_tab(self, parent):
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Annotate an MP4 with ArUco detections and export a pose/angle CSV.", style="Header.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(10, 14))
        self._row(parent, 1, "Calibration YAML", self.calib_yaml_var, self._browse_calib_yaml)
        self._dict_dropdown(parent, 2)
        self._entry_num(parent, 3, "Pose marker length (mm)", self.pose_marker_length_var)
        self._row(parent, 4, "Input video", self.proc_input_var, self._browse_proc_input)
        self._row(parent, 5, "Output annotated MP4", self.proc_output_video_var, self._browse_proc_output_video)
        self._row(parent, 6, "Output CSV", self.proc_output_csv_var, self._browse_proc_output_csv)
        self._entry_num(parent, 7, "Proximal marker ID", self.marker_id_prox_var)
        self._entry_num(parent, 8, "Distal marker ID", self.marker_id_dist_var)
        ttk.Label(parent, text="Joint axis").grid(row=9, column=0, sticky="w", padx=8, pady=6)
        axis_combo = ttk.Combobox(parent, textvariable=self.joint_axis_var, values=["x", "y", "z"], width=8, state="readonly")
        axis_combo.grid(row=9, column=1, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(parent, text="Use quaternion Kalman filtering", variable=self.use_filter_var).grid(row=10, column=1, sticky="w", padx=8, pady=6)
        btns = ttk.Frame(parent)
        btns.grid(row=11, column=1, sticky="w", padx=8, pady=14)
        ttk.Button(btns, text="Run Processor", style="Success.TButton", command=self.start_processing).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Stop", style="Danger.TButton", command=self.stop_current_task).pack(side="left")

    def _entry_num(self, parent, row, label, var):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(parent, textvariable=var, width=18).grid(row=row, column=1, sticky="w", padx=8, pady=6)

    def _dict_dropdown(self, parent, row):
        ttk.Label(parent, text="ArUco dictionary").grid(row=row, column=0, sticky="w", padx=8, pady=6)
        values = [
            "DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_5X5_100",
            "DICT_6X6_50", "DICT_6X6_100", "DICT_7X7_50", "DICT_7X7_100",
        ]
        combo = ttk.Combobox(parent, textvariable=self.dict_name_var, values=values, width=22, state="readonly")
        combo.grid(row=row, column=1, sticky="w", padx=8, pady=6)

    def _dict_code(self) -> int:
        name = self.dict_name_var.get().strip()
        if not hasattr(cv.aruco, name):
            raise ValueError(f"Unknown ArUco dictionary: {name}")
        return getattr(cv.aruco, name)

    def log(self, msg: str):
        def _append():
            ts = time.strftime("%H:%M:%S")
            self.log_text.insert("end", f"[{ts}] {msg}\n")
            self.log_text.see("end")
        self.after(0, _append)

    def _run_in_thread(self, target, controller=None):
        if self.current_worker is not None and self.current_worker.is_alive():
            messagebox.showwarning("Busy", "Another task is already running. Stop it first.")
            return
        self.current_controller = controller
        self.current_worker = threading.Thread(target=target, daemon=True)
        self.current_worker.start()

    def stop_current_task(self):
        if self.current_controller is not None:
            try:
                self.current_controller.stop()
                self.log("Stop requested.")
            except Exception as exc:
                self.log(f"Stop failed: {exc}")
        else:
            self.log("No active task to stop.")

    def _build_toolkit(self) -> ArucoToolkit:
        calib = load_calibration_yaml(self.calib_yaml_var.get().strip())
        return ArucoToolkit(
            aruco_dict_name=self._dict_code(),
            marker_length_mm=float(self.pose_marker_length_var.get()),
            calibration=calib,
        )

    def start_calibration(self):
        def job():
            try:
                K, D, rms, used = calibrate_charuco_from_folder(
                    image_dir=self.calib_dir_var.get().strip(),
                    output_yaml=self.calib_out_var.get().strip(),
                    squares_x=int(self.squares_x_var.get()),
                    squares_y=int(self.squares_y_var.get()),
                    square_length_mm=float(self.square_length_var.get()),
                    marker_length_mm=float(self.charuco_marker_length_var.get()),
                    dict_name=self._dict_code(),
                    preview=bool(self.calib_preview_var.get()),
                    status_cb=self.log,
                )
                self.log(f"Calibration saved to {self.calib_out_var.get().strip()}")
                self.log(f"Used images: {used} | RMS reprojection error: {rms:.6f}")
                self.log(f"K =\n{K}")
                self.log(f"D =\n{D}")
            except Exception as exc:
                self.log(f"Calibration failed: {exc}")
                messagebox.showerror("Calibration Error", str(exc))
        self._run_in_thread(job)

    def start_live_recording(self):
        try:
            toolkit = self._build_toolkit() if self.live_annotate_var.get() else None
            controller = LiveRecorder(
                camera_id=int(self.camera_id_var.get()),
                output_path=self.live_output_var.get().strip(),
                annotate=bool(self.live_annotate_var.get()),
                toolkit=toolkit,
                fps_override=int(self.live_fps_override_var.get()),
            )
        except Exception as exc:
            messagebox.showerror("Setup Error", str(exc))
            return

        def job():
            try:
                controller.run(status_cb=self.log)
            except Exception as exc:
                self.log(f"Live recording failed: {exc}")
                messagebox.showerror("Live Recording Error", str(exc))

        self._run_in_thread(job, controller=controller)

    def start_playback(self):
        controller = VideoPlayer(self.play_path_var.get().strip())

        def job():
            try:
                controller.run(status_cb=self.log)
            except Exception as exc:
                self.log(f"Playback failed: {exc}")
                messagebox.showerror("Playback Error", str(exc))

        self._run_in_thread(job, controller=controller)

    def start_processing(self):
        try:
            toolkit = self._build_toolkit()
            controller = VideoProcessor(
                input_path=self.proc_input_var.get().strip(),
                output_video_path=self.proc_output_video_var.get().strip(),
                output_csv_path=self.proc_output_csv_var.get().strip(),
                toolkit=toolkit,
                marker_id_proximal=int(self.marker_id_prox_var.get()),
                marker_id_distal=int(self.marker_id_dist_var.get()),
                joint_axis=self.joint_axis_var.get().strip(),
                use_quaternion_filter=bool(self.use_filter_var.get()),
            )
        except Exception as exc:
            messagebox.showerror("Setup Error", str(exc))
            return

        def job():
            try:
                controller.run(status_cb=self.log)
            except Exception as exc:
                self.log(f"Processing failed: {exc}")
                messagebox.showerror("Processing Error", str(exc))

        self._run_in_thread(job, controller=controller)

    def generate_board_image(self):
        def job():
            try:
                _, size = generate_charuco_board_image(
                    output_path=self.board_output_var.get().strip(),
                    squares_x=int(self.squares_x_var.get()),
                    squares_y=int(self.squares_y_var.get()),
                    square_length_mm=float(self.square_length_var.get()),
                    marker_length_mm=float(self.charuco_marker_length_var.get()),
                    dict_name=self._dict_code(),
                    dpi=int(self.board_dpi_var.get()),
                    margin_px=int(self.board_margin_var.get()),
                    status_cb=self.log,
                )
                self.log(f"Board image dimensions: {size[0]} x {size[1]} px")
            except Exception as exc:
                self.log(f"Board generation failed: {exc}")
                messagebox.showerror("Board Generation Error", str(exc))
        self._run_in_thread(job)

    def start_capture_images(self):
        try:
            toolkit = None
            if self.capture_annotate_var.get():
                try:
                    toolkit = self._build_toolkit()
                except Exception:
                    toolkit = ArucoToolkit(
                        aruco_dict_name=self._dict_code(),
                        marker_length_mm=float(self.pose_marker_length_var.get()),
                        calibration=None,
                    )
            else:
                toolkit = ArucoToolkit(
                    aruco_dict_name=self._dict_code(),
                    marker_length_mm=float(self.pose_marker_length_var.get()),
                    calibration=None,
                )

            controller = CalibrationImageCapture(
                camera_id=int(self.capture_camera_id_var.get()),
                output_dir=self.capture_dir_var.get().strip(),
                prefix=self.capture_prefix_var.get().strip(),
                annotate=bool(self.capture_annotate_var.get()),
                toolkit=toolkit,
                countdown_seconds=float(self.capture_countdown_var.get()),
                max_images=int(self.capture_max_images_var.get()),
            )
        except Exception as exc:
            messagebox.showerror("Setup Error", str(exc))
            return

        def job():
            try:
                controller.run(status_cb=self.log)
            except Exception as exc:
                self.log(f"Calibration image capture failed: {exc}")
                messagebox.showerror("Capture Error", str(exc))

        self._run_in_thread(job, controller=controller)

    # -------------------------
    # Browse helpers
    # -------------------------
    def _browse_board_output(self):
        p = filedialog.asksaveasfilename(title="Save board image", defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if p:
            self.board_output_var.set(p)

    def _browse_capture_dir(self):
        p = filedialog.askdirectory(title="Select output folder for calibration images")
        if p:
            self.capture_dir_var.set(p)
            self.calib_dir_var.set(p)

    def _browse_calib_dir(self):
        p = filedialog.askdirectory(title="Select calibration image folder")
        if p:
            self.calib_dir_var.set(p)

    def _browse_calib_out(self):
        p = filedialog.asksaveasfilename(title="Save calibration YAML", defaultextension=".yaml", filetypes=[("YAML", "*.yaml")])
        if p:
            self.calib_out_var.set(p)
            self.calib_yaml_var.set(p)

    def _browse_calib_yaml(self):
        p = filedialog.askopenfilename(title="Open calibration YAML", filetypes=[("YAML", "*.yaml")])
        if p:
            self.calib_yaml_var.set(p)

    def _browse_live_output(self):
        p = filedialog.asksaveasfilename(title="Save MP4", defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if p:
            self.live_output_var.set(p)

    def _browse_play_path(self):
        p = filedialog.askopenfilename(title="Open video", filetypes=[("Video", "*.mp4;*.avi;*.mov"), ("All files", "*.*")])
        if p:
            self.play_path_var.set(p)

    def _browse_proc_input(self):
        p = filedialog.askopenfilename(title="Open video", filetypes=[("Video", "*.mp4;*.avi;*.mov"), ("All files", "*.*")])
        if p:
            self.proc_input_var.set(p)

    def _browse_proc_output_video(self):
        p = filedialog.asksaveasfilename(title="Save annotated MP4", defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if p:
            self.proc_output_video_var.set(p)

    def _browse_proc_output_csv(self):
        p = filedialog.asksaveasfilename(title="Save CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            self.proc_output_csv_var.set(p)

if __name__ == "__main__":
    app = CameraUtilsApp()
    app.mainloop()