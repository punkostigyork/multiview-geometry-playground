import cv2
import matplotlib.pyplot as plt
from src.camera import PinholeCamera
from src.features import detect_and_match
from src.geometry import estimate_fundamental_matrix_robust, triangulate_points, compute_reprojection_error
from src.visualization import draw_epipolar_lines

def run_pipeline():
    print("🚀 Starting Multiview Geometry Pipeline...")
    
    # 1. Load Data
    img1 = cv2.imread('data/temple/temple0001.png')
    img2 = cv2.imread('data/temple/temple0002.png')
    metadata = 'data/temple/temple_par.txt'
    
    cam1 = PinholeCamera.from_middlebury(metadata, 1)
    cam2 = PinholeCamera.from_middlebury(metadata, 2)
    
    # 2. Match Features
    print("🔍 Detecting SIFT features...")
    pts1, pts2, kp1, kp2, matches = detect_and_match(img1, img2)
    
    # 3. Robust Estimation (RANSAC)
    print("🛠️ Estimating Fundamental Matrix (RANSAC)...")
    F, in1, in2 = estimate_fundamental_matrix_robust(pts1, pts2)
    
    # 4. Triangulation
    print("📐 Triangulating 3D points...")
    pts3d = triangulate_points(in1, in2, cam1.P, cam2.P)
    
    # 5. Validation
    err, _ = compute_reprojection_error(in1, pts3d, cam1.P)
    print(f"✅ Success! Mean Reprojection Error: {err:.4f} pixels")
    print(f"📦 Reconstructed {len(pts3d)} 3D points.")

    # 6. Quick Viz
    print("🖼️ Displaying Epipolar Geometry...")
    v1, v2 = draw_epipolar_lines(img1, img2, F, in1, in2)
    plt.subplot(121), plt.imshow(cv2.cvtColor(v1, cv2.COLOR_BGR2RGB))
    plt.subplot(122), plt.imshow(cv2.cvtColor(v2, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == "__main__":
    run_pipeline()