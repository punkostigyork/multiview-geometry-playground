import numpy as np
import pytest
import cv2
from src.camera import PinholeCamera
from src.geometry import estimate_fundamental_matrix_robust

def test_camera_projection():
    # 1. Create a simple Identity Camera
    K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]])
    R = np.eye(3)
    t = np.zeros((3, 1))
    cam = PinholeCamera(K, R, t)
    
    # 2. Define a 3D point exactly 2 units in front of the camera
    point_3d = np.array([[0, 0, 2]])
    
    # 3. Project it
    # Math: (1000 * 0 / 2) + 500 = 500
    pts_2d = cam.project(point_3d)
    
    assert np.allclose(pts_2d[0], [500, 500]), "Projection math is incorrect!"

def test_ray_direction_unit_norm():
    # Ensure generated rays are always unit vectors (length = 1)
    K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]])
    cam = PinholeCamera(K, np.eye(3), np.zeros((3,1)))
    
    rays = cam.get_ray_directions(np.array([[100, 100]]))
    norm = np.linalg.norm(rays)
    
    assert np.isclose(norm, 1.0), "Rays are not normalized!"

def test_ransac_outlier_rejection():
    # 1. Create 20 perfect points matching an identity relationship
    # (Simplified: just points shifted horizontally)
    pts1 = np.random.rand(20, 2) * 100
    pts2 = pts1.copy()
    pts2[:, 0] += 50  # Shifted by 50 pixels
    
    # 2. Add 5 "Zombies" (total garbage outliers)
    outliers1 = np.random.rand(5, 2) * 500
    outliers2 = np.random.rand(5, 2) * 500
    
    dirty_pts1 = np.vstack([pts1, outliers1])
    dirty_pts2 = np.vstack([pts2, outliers2])
    
    # 3. Run your robust estimator
    F, in1, in2 = estimate_fundamental_matrix_robust(dirty_pts1, dirty_pts2)
    
    # 4. Assertions
    assert len(in1) >= 20, "RANSAC failed to keep the good points!"
    assert len(in1) < 25, "RANSAC failed to filter out the outliers!"
    print(f"\n✅ RANSAC filtered {25 - len(in1)} outliers.")