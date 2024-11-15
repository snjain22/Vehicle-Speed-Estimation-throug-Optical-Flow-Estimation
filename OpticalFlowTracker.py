import cv2 as cv
import numpy as np
from collections import defaultdict, deque
import time
from sklearn.cluster import DBSCAN

class CornerDetectionMetrics:
    def __init__(self):
        self.total_corners = 0
        self.stable_corners = 0
        self.processing_times = []
        self.frame_count = 0
        self.vehicle_counts = []
        self.corner_counts = []
        
    def update(self, corners_count, process_time, vehicle_count):
        self.frame_count += 1
        self.corner_counts.append(corners_count)
        self.processing_times.append(process_time)
        self.vehicle_counts.append(vehicle_count)
        
    def get_summary(self):
        avg_corners = np.mean(self.corner_counts) if self.corner_counts else 0
        avg_time = np.mean(self.processing_times) if self.processing_times else 0
        # avg_vehicles = np.mean(self.vehicle_counts) if self.vehicle_counts else 0
        
        return {
            'average_corners': avg_corners,
            'average_processing_time': avg_time * 1000,  # Convert to ms
            # 'average_vehicles': avg_vehicles,
            'total_frames': self.frame_count
        }

class VehicleSpeedTracker:
    def __init__(self, detector_type, feature_params):
        self.detector_type = detector_type
        self.feature_params = feature_params
        
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(25, 25),
            maxLevel=3,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
            minEigThreshold=1e-4
        )
        
        # Tracking parameters
        self.track_length = 15
        self.detect_interval = 5
        self.tracks = []
        self.frame_idx = 0
        self.track_colors = {}
        
        # Speed estimation parameters
        self.pixel_to_meter_ratio = 0.1
        self.fps = 30
        self.min_speed_threshold = 5
        self.max_speed_threshold = 120
        
        # Vehicle tracking parameters
        self.vehicle_clusters = {}
        self.cluster_radius = 50
        self.next_vehicle_id = 1
        self.vehicle_speeds = defaultdict(lambda: deque(maxlen=15))
        self.vehicle_positions = defaultdict(lambda: deque(maxlen=15))
        self.min_detection_frames = 5
        self.vehicle_lifetime = defaultdict(int)
        self.vehicle_active = defaultdict(bool)
        self.inactive_frames = defaultdict(int)
        self.max_inactive_frames = 10
        
        # Performance optimization
        self.processing_width = 960
        self.scale_factor = None
        self.prev_gray = None
        
        # DBSCAN parameters
        self.dbscan_eps = 30
        self.dbscan_min_samples = 5

    def resize_frame(self, frame):
        if self.scale_factor is None:
            self.scale_factor = self.processing_width / frame.shape[1]
        return cv.resize(frame, (self.processing_width, int(frame.shape[0] * self.scale_factor)))

    def calculate_speed(self, positions):
        if len(positions) < 3:
            return 0
        positions = np.array(list(positions))
        speeds = []
        for i in range(len(positions)-3):
            displacement = np.linalg.norm(positions[i+3] - positions[i])
            time_diff = 3 / self.fps
            speed = (displacement * self.pixel_to_meter_ratio / time_diff) * 3.6
            speeds.append(speed)
        return np.median(speeds) if speeds else 0

    def smooth_speed(self, speeds):
        if not speeds:
            return 0
        speeds_array = np.array(list(speeds))
        mean = np.mean(speeds_array)
        std = np.std(speeds_array)
        valid_speeds = speeds_array[abs(speeds_array - mean) < 1.5 * std]
        if len(valid_speeds) < 3:
            return 0
        return np.median(valid_speeds)

    def is_vehicle_moving(self, positions, speeds):
        if len(positions) < 3:
            return False
        recent_positions = np.array(list(positions))
        total_movement = np.linalg.norm(recent_positions[-1] - recent_positions[0])
        min_movement = 25
        recent_speeds = list(speeds)[-5:] if len(speeds) >= 5 else list(speeds)
        avg_speed = np.median(recent_speeds) if recent_speeds else 0
        return (total_movement > min_movement and 
                self.min_speed_threshold < avg_speed < self.max_speed_threshold and
                len(recent_speeds) >= 3)

    def update_vehicle_tracking(self, tracks):
        try:
            points = np.array([tr[-1] for tr in tracks])
            if len(points) < self.dbscan_min_samples:
                return
            
            clustering = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples
            ).fit(points)
            
            new_clusters = {}
            for cluster_id in set(clustering.labels_):
                if cluster_id == -1:
                    continue
                    
                cluster_points = points[clustering.labels_ == cluster_id]
                center = np.mean(cluster_points, axis=0)
                
                vehicle_id = None
                min_dist = float('inf')
                
                for vid in self.vehicle_clusters:
                    if vid in self.vehicle_positions:
                        last_pos = self.vehicle_positions[vid][-1]
                        dist = np.linalg.norm(center - last_pos)
                        if dist < self.cluster_radius and dist < min_dist:
                            vehicle_id = vid
                            min_dist = dist
                
                if vehicle_id is None:
                    vehicle_id = self.next_vehicle_id
                    self.next_vehicle_id += 1
                
                new_clusters[vehicle_id] = {
                    'center': center,
                    'points': cluster_points
                }
                
                self.vehicle_positions[vehicle_id].append(center)
                self.vehicle_lifetime[vehicle_id] += 1
                
                speed = self.calculate_speed(self.vehicle_positions[vehicle_id])
                if self.min_speed_threshold < speed < self.max_speed_threshold:
                    self.vehicle_speeds[vehicle_id].append(speed)
                    self.vehicle_active[vehicle_id] = True
                    self.inactive_frames[vehicle_id] = 0
                else:
                    self.inactive_frames[vehicle_id] += 1
                    if self.inactive_frames[vehicle_id] > self.max_inactive_frames:
                        self.vehicle_active[vehicle_id] = False
            
            for vid in list(self.vehicle_positions.keys()):
                if vid not in new_clusters:
                    self.remove_vehicle(vid)
            
            self.vehicle_clusters = new_clusters
            
        except Exception as e:
            print(f"Error in update_vehicle_tracking: {e}")

    def remove_vehicle(self, vehicle_id):
        self.vehicle_positions.pop(vehicle_id, None)
        self.vehicle_speeds.pop(vehicle_id, None)
        self.vehicle_lifetime.pop(vehicle_id, None)
        self.track_colors.pop(vehicle_id, None)
        self.vehicle_active.pop(vehicle_id, None)
        self.inactive_frames.pop(vehicle_id, None)

    def process_frame(self, frame):
        try:
            small_frame = self.resize_frame(frame)
            frame_gray = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)
            vis = small_frame.copy()
            corners = []

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, status, _ = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
                p0r, status, _ = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1

                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_length:
                        del tr[0]
                    new_tracks.append(tr)
                    corners.append((x, y))
                    if len(tr) > 1:
                        cv.polylines(vis, [np.int32(tr)], False, (0, 255, 0), 1)

                self.tracks = new_tracks
                self.update_vehicle_tracking(new_tracks)

                for vehicle_id, cluster in self.vehicle_clusters.items():
                    if (self.vehicle_lifetime[vehicle_id] >= self.min_detection_frames and 
                        self.vehicle_active[vehicle_id]):
                        if vehicle_id not in self.track_colors:
                            self.track_colors[vehicle_id] = np.random.randint(0, 255, 3).tolist()
                        color = self.track_colors[vehicle_id]
                        center = np.int32(cluster['center'])
                        
                        cv.circle(vis, center, 8, color, -1)
                        
                        if len(self.vehicle_speeds[vehicle_id]) >= 3:
                            avg_speed = self.smooth_speed(self.vehicle_speeds[vehicle_id])
                            if self.min_speed_threshold < avg_speed < self.max_speed_threshold:
                                text = f'{avg_speed:.0f} km/h'
                                font = cv.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.6
                                thickness = 2
                                (text_width, text_height), _ = cv.getTextSize(
                                    text, font, font_scale, thickness)
                                cv.rectangle(vis, 
                                        (center[0] - 5, center[1] - text_height - 15),
                                        (center[0] + text_width + 5, center[1] - 5),
                                        (0, 0, 0), -1)
                                cv.putText(vis, text, (center[0], center[1] - 10),
                                        font, font_scale, color, thickness)

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                features = cv.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
                if features is not None:
                    for x, y in np.float32(features).reshape(-1, 2):
                        self.tracks.append([(x, y)])
                        corners.append((x, y))

            self.frame_idx += 1
            self.prev_gray = frame_gray
            
            return vis, np.array(corners)
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            return frame, np.array([])

    def validate_vehicle(self, cluster_points, speed):
        hull = cv.convexHull(np.float32(cluster_points).reshape(-1, 1, 2))
        area = cv.contourArea(hull)
        min_area = 100
        max_area = 5000
        speed_valid = self.min_speed_threshold < speed < self.max_speed_threshold
        density = len(cluster_points) / area if area > 0 else 0
        min_density = 0.01
        return min_area < area < max_area and speed_valid and density > min_density

def main():
    cap = cv.VideoCapture("FinalProject/aerial.mp4")  # Replace with your video path
    
    # Initialize both detectors with their respective parameters
    shi_tomasi_params = dict(
        maxCorners=200,
        qualityLevel=0.2,
        minDistance=20,
        blockSize=15,
        k=0.04,
        useHarrisDetector=False
    )
    
    harris_params = dict(
        maxCorners=200,
        qualityLevel=0.2,
        minDistance=20,
        blockSize=15,
        k=0.04,
        useHarrisDetector=True
    )
    
    # Initialize trackers
    shi_tomasi_tracker = VehicleSpeedTracker("Shi-Tomasi", shi_tomasi_params)
    harris_tracker = VehicleSpeedTracker("Harris", harris_params)
    
    # Set FPS for both trackers
    fps = cap.get(cv.CAP_PROP_FPS)  
    shi_tomasi_tracker.fps = fps
    harris_tracker.fps = fps
    
    # Create window
    cv.namedWindow('Corner Detection Comparison', cv.WINDOW_NORMAL)
    
    # Initialize metrics storage
    metrics = {
        'Shi-Tomasi': {
            'corners': [],
            'processing_times': [],
            # 'vehicles_tracked': [],
            'fps': []
        },
        'Harris': {
            'corners': [],
            'processing_times': [],
            # 'vehicles_tracked': [],
            'fps': []
        }
    }
    
    frame_times = []
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process with both detectors
        shi_tomasi_start = time.time()
        shi_tomasi_result, shi_tomasi_corners = shi_tomasi_tracker.process_frame(frame.copy())
        shi_tomasi_time = time.time() - shi_tomasi_start
        
        harris_start = time.time()
        harris_result, harris_corners = harris_tracker.process_frame(frame.copy())
        harris_time = time.time() - harris_start
        
        # Update metrics
        metrics['Shi-Tomasi']['corners'].append(len(shi_tomasi_corners))
        metrics['Shi-Tomasi']['processing_times'].append(shi_tomasi_time)
        # metrics['Shi-Tomasi']['vehicles_tracked'].append(len(shi_tomasi_tracker.vehicle_clusters))
        
        metrics['Harris']['corners'].append(len(harris_corners))
        metrics['Harris']['processing_times'].append(harris_time)
        # metrics['Harris']['vehicles_tracked'].append(len(harris_tracker.vehicle_clusters))
        
        # Combine results side by side
        combined_result = np.hstack((shi_tomasi_result, harris_result))
        
        # Calculate and display FPS
        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / np.mean(frame_times)
        
        # Calculate text positions for Harris metrics
        harris_x_offset = shi_tomasi_result.shape[1]  # Width of first frame
        
        # Add labels and metrics
        # Shi-Tomasi metrics (left side)
        cv.putText(combined_result, "Shi-Tomasi", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(combined_result, f"FPS: {fps:.1f}", (10, 60), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.putText(combined_result, f"Corners: {len(shi_tomasi_corners)}", (10, 90), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # cv.putText(combined_result, f"Vehicles: {len(shi_tomasi_tracker.vehicle_clusters)}", 
                #   (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.putText(combined_result, f"Process Time: {shi_tomasi_time*1000:.1f}ms", 
                  (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Harris metrics (right side)
        cv.putText(combined_result, "Harris", (harris_x_offset + 10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(combined_result, f"FPS: {fps:.1f}", (harris_x_offset + 10, 60), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.putText(combined_result, f"Corners: {len(harris_corners)}", 
                  (harris_x_offset + 10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # cv.putText(combined_result, f"Vehicles: {len(harris_tracker.vehicle_clusters)}", 
                #   (harris_x_offset + 10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.putText(combined_result, f"Process Time: {harris_time*1000:.1f}ms", 
                  (harris_x_offset + 10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv.imshow('Corner Detection Comparison', combined_result)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Print final comparison metrics
    print("\nFinal Performance Metrics:")
    print("=" * 50)
    
    for detector in ['Shi-Tomasi', 'Harris']:
        avg_corners = np.mean(metrics[detector]['corners'])
        avg_time = np.mean(metrics[detector]['processing_times']) * 1000  # Convert to ms
        # avg_vehicles = np.mean(metrics[detector]['vehicles_tracked'])
        
        print(f"\n{detector} Detector:")
        print(f"Average corners detected: {avg_corners:.1f}")
        print(f"Average processing time: {avg_time:.2f} ms")
        # print(f"Average vehicles tracked: {avg_vehicles:.1f}")
        print(f"Total frames processed: {len(metrics[detector]['corners'])}")
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()