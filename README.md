# Vehicle-Speed-Estimation-through-Optical-Flow-Estimation
![demo](https://github.com/user-attachments/assets/a68e77d2-d804-44ee-af66-9e17fc5bdbc4)

# Methodology
 The system follows a modular pipeline architecture consisting of five main components:
 1) Frame preprocessing and feature detection
 2) Optical flow computation
 3) Vehicle clustering and tracking
 4) Speed estimation
 5) Performance optimization

## Frame preprocessing and feature detection
We implement and compare two prominent corner detection algorithms:
 1) Shi-Tomasi Corner Detector
 2) Harris Corner Detector

## Optical Flow Computation
The system employs the Lucas-Kanade optical flow algorithm with pyramidal implementation.
![pyramid_lk](https://github.com/user-attachments/assets/38c98b89-cff0-44c8-a327-c8bed96cfc4a)
Pyramidal Lucas-Kanade Implementation showing the multi-level processing approach. The algorithm processes frames at different resolution levels (Level 2 to Level 0), calculating optical flow at each level and propagating the results to obtain the final flow estimation. Frame i and Frame i+1 represent consecutive video frames being analyzed.

## Vehicle Clustering and Tracking
Our vehicle clustering and tracking system employs a sophisticated DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm, specifically optimized for aerial vehicle tracking scenarios. The clustering process operates on tracked feature points to identify and isolate individual vehicles within the frame. The DBSCAN algorithm was chosen for its ability to handle clusters of arbitrary shapes and its robustness to noise, making it particularly suitable for real-world traffic scenarios where vehicles may appear in various orientations and sizes.

## Speed Estimation Algorithm
The speed estimation module implements a sophisticated multi-step process that converts pixel-space measurements to real-world velocities. Our approach accounts for perspective distortion, camera calibration parameters, and temporal characteristics of the video stream to provide accurate speed measurements.

## Performance Optimization
 • Adaptive thresholding for varying conditions
 • Early termination of invalid tracks
 • Efficient data structures for track management

# Results
Our experimental evaluation demonstrates comprehensive performance analysis of both Shi-Tomasi and Harris corner detectors in the context of vehicle tracking and speed estimation.

The experimental results highlight several important observations:
 • Corner Detection: Shi-Tomasi detector consistently identified more corners (499.7 vs 117.9), providing more tracking points but requiring additional computational resources.
 • Processing Efficiency: Harris detector demonstrated superior processing speed, requiring only 23.24ms compared to Shi-Tomasi’s 42.49ms, representing a 45.3% reduction in processing time.
 • Tracking Accuracy: Despite the significant difference in corner counts, both methods maintained comparable tracking accuracy, as evidenced by the similar speed measurements shown in Figure 3.
 • Real-time Performance: Both methods achieved real time processing capabilities, with frame rates exceeding 20 FPS (frames per second)

 ![demo](https://github.com/user-attachments/assets/a68e77d2-d804-44ee-af66-9e17fc5bdbc4)
 Side-by-side comparison of Shi-Tomasi (left) and Harris (right) corner detectors in real-time vehicle tracking. The image shows simultaneous tracking of multiple vehicles with their respective speeds (km/h). Note the difference in corner detection counts (380 vs 136) and processing times (36.5ms vs 19.7ms) between the two methods while maintaining similar tracking accuracy.
