import os
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import easyocr
import threading
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

try:
    from firebase_manager import FirebaseManager
except ImportError:
    FirebaseManager = None

class SITACore:
    def __init__(self, model_path="yolov8n.pt", n_skip=3):
        print("[SITA Core] Initializing Smart Intelligent Traffic Analyzer...")
        self.n_skip = n_skip
        
        # 1. Model Initialization & ONNX Optimization for AMD Ryzen
        onnx_path = model_path.replace(".pt", ".onnx")
        if not os.path.exists(onnx_path):
            print(f"[SITA Core] Exporting model to ONNX for AMD optimization: {onnx_path}")
            model_pt = YOLO(model_path)
            model_pt.export(format="onnx")
            
        print(f"[SITA Core] Loading YOLO model via ONNXRuntime: {onnx_path}")
        self.model = YOLO(onnx_path, task='detect')
        
        # 2. EasyOCR
        print("[SITA Core] Initializing EasyOCR...")
        # using CPU explicitly for Ryzen environment without dedicated CUDA GPU
        self.reader = easyocr.Reader(['en'], gpu=False)
        
        # 3. Firebase setup (Dual-write)
        if FirebaseManager:
            self.firebase = FirebaseManager()
        else:
            print("[SITA Core] WARNING: firebase_manager not found. DB writes skipped.")
            self.firebase = None
        
        # 4. Background Thread Pool for Async heavy tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # To avoid processing the same vehicle ID multiple times
        self.processed_ids = set()
        self.ocr_attempts = {}
        
    def _init_firebase(self):
        pass

    def get_vehicle_color(self, crop_img, check_hood=True):
        """
        Extract the dominant color using HSV histogram, with a center mask
        to ignore the road/background inside the bounding box.
        """
        # Convert to HSV
        hsv_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        h, w = hsv_img.shape[:2]
        
        # Create a mask for the center region (assume vehicle occupies center)
        # We'll mask out the outer 20% of the edges where road/background likely exists
        mask = np.zeros((h, w), dtype=np.uint8)
        y1, y2 = int(h * 0.2), int(h * 0.8)
        x1, x2 = int(w * 0.2), int(w * 0.8)
        mask[y1:y2, x1:x2] = 255
        
        hist_h = cv2.calcHist([hsv_img], [0], mask, [180], [0, 180])
        hist_s = cv2.calcHist([hsv_img], [1], mask, [256], [0, 256])
        
        # Calculate the total number of non-zero mask pixels
        total_pixels = cv2.countNonZero(mask)
        
        dominant_hue = int(np.argmax(hist_h))
        max_hue_val = hist_h[dominant_hue][0]
        
        dominant_sat = int(np.argmax(hist_s))
        
        # Check Confidence Weight: ignore if dominant hue is less than 15% of masked area
        if total_pixels > 0 and (max_hue_val / total_pixels) < 0.15:
            color = "Unknown (Low Confidence)"
        elif dominant_sat < 40:
            color = "Gray/Silver/Black"
            # If windshield glare caused it, check a small 10% hood crop
            if check_hood and h > 20 and w > 20:
                hood_crop = crop_img[int(h*0.7):int(h*0.9), int(w*0.3):int(w*0.7)]
                if hood_crop.size > 0:
                    return self.get_vehicle_color(hood_crop, check_hood=False)
        else:
            if dominant_hue < 10 or dominant_hue > 170: color = "Red"
            elif 10 <= dominant_hue < 35: color = "Yellow/Orange"
            elif 35 <= dominant_hue < 85: color = "Green"
            elif 85 <= dominant_hue < 130: color = "Blue"
            elif 130 <= dominant_hue < 170: color = "Purple/Pink"
            else: color = "Unknown"
        
        return color

    def recognize_plate(self, crop_img):
        """
        Bilateral filtering and Grayscale + EasyOCR. Returns (plate_text, conf)
        """
        # 1. Grayscale
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        # 2. "Super-Resolution" / Sharpening via Bilateral Filter
        # Bilateral filter reduces noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # EasyOCR Inference (Optimized for CPU: paragraph=False, detail=0)
        results = self.reader.readtext(filtered, paragraph=False, detail=0)
        
        if not results:
            return None, 0.0
            
        # Since detail=0 only returns strings, we'll assign a mock confidence
        # Since the architecture relies heavily on string extraction speed here
        best_plate = " ".join(results)
        return best_plate, 1.0 # Bypassing original conf threshold due to user's 'detail=0' request

    def async_heavy_task(self, vehicle_id, crop_img, class_name):
        """ Runs in a background thread """
        if vehicle_id in self.processed_ids:
            return
        
        self.processed_ids.add(vehicle_id)
        
        print(f"[Thread] Processing Vehicle ID: {vehicle_id} ({class_name})")
        # 1. Color detection
        color = self.get_vehicle_color(crop_img)
        
        # 2. LPR (Optional: you might want to detect plate region first. 
        # Without a plate detector, passing the whole crop to OCR is heavy but functional).
        plate_text, conf = self.recognize_plate(crop_img)
        
        # 3. OCR Buffer Mechanism (Adapted for detail=0 return)
        if (not plate_text or len(plate_text.strip()) < 2) and self.ocr_attempts.get(vehicle_id, 0) < 2:
            self.ocr_attempts[vehicle_id] = self.ocr_attempts.get(vehicle_id, 0) + 1
            print(f"[Thread] Retry OCR ID: {vehicle_id} (Attempt {self.ocr_attempts[vehicle_id]}/2).")
            return # Skip adding to processed_ids to try again
            
        self.processed_ids.add(vehicle_id)
        if not plate_text:
            plate_text = "UNKNOWN"
            
        print(f"[Thread] Result ID: {vehicle_id} | Color: {color} | Plate: {plate_text} (Conf: {conf:.2f})")
        
        # 4. Firebase Upload
        if self.firebase:
            self.firebase.upload_vehicle_data(vehicle_id, class_name, plate_text, color)

    def process_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[SITA Core] Error openning video: {video_path}")
            return
            
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[SITA Core] Video Input: {width}x{height} @ {actual_fps}fps")
        
        # Output writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, actual_fps, (width, height))

        # 1. Tracker Init: adjusted frame_rate to 1/3 source FPS
        tracker_fps = int(actual_fps // self.n_skip)
        if tracker_fps < 1: tracker_fps = 1
        byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=tracker_fps)
        
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # Center "Processing Zone" ROI
        roi_y1 = int(height * 0.4)
        roi_y2 = int(height * 0.6)
        
        annotated_frame = None
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            
            # Frame skipping
            if frame_idx % self.n_skip != 0:
                if writer and annotated_frame is not None:
                    writer.write(annotated_frame) # Write previous annotated frame to keep smooth video output
                continue
                
            # Inference (Main Thread)
            results = self.model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # Filter detections for vehicles (usually car=2, mtb=3, bus=5, truck=7 in COCO)
            # Assuming basic YOLO labeling, filtering ids 2, 3, 5, 7
            vehicle_class_ids = [2, 3, 5, 7]
            detections = detections[np.isin(detections.class_id, vehicle_class_ids)]
            
            # Tracking
            detections = byte_tracker.update_with_detections(detections)
            
            annotated_frame = frame.copy()
            
            # Draw ROI lines
            cv2.line(annotated_frame, (0, roi_y1), (width, roi_y1), (0, 0, 255), 2)
            cv2.line(annotated_frame, (0, roi_y2), (width, roi_y2), (0, 0, 255), 2)
            
            labels = []
            
            for i, (xyxy, mask, conf, class_id, tracker_id, data) in enumerate(detections):
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                class_name = self.model.names[class_id]
                labels.append(f"#{tracker_id} {class_name}")
                
                # Check if center point is in Processing Zone
                if roi_y1 <= cy <= roi_y2:
                    if tracker_id not in self.processed_ids:
                        # Extract crop
                        crop_img = frame[y1:y2, x1:x2].copy()
                        if crop_img.size > 0:
                            # Submit to thread pool
                            self.executor.submit(self.async_heavy_task, tracker_id, crop_img, class_name)
            
            # Render annotations
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            if writer:
                writer.write(annotated_frame)
                
            # Optional: cv2.imshow for local dev if desired
            # cv2.imshow("SITA Pipeline", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("[SITA Core] Processing Complete.")

# To test locally:
# if __name__ == "__main__":
#     core = SITACore()
#     core.process_video("path/to/test.mp4", "output.mp4")
