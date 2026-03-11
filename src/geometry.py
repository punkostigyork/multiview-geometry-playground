import numpy as np
import cv2

def normalize_points(pts):
    centroid = np.mean(pts, axis=0)
    shifted_pts = pts - centroid
    scale = np.sqrt(2) / np.mean(np.linalg.norm(shifted_pts, axis=1))
    T = np.array([[scale, 0, -scale*centroid[0]], [0, scale, -scale*centroid[1]], [0, 0, 1]])
    pts_hom = np.hstack((pts, np.ones((len(pts), 1))))
    pts_norm = (T @ pts_hom.T).T
    return pts_norm[:, :2], T

def estimate_fundamental_matrix(pts1, pts2):
    n_pts1, T1 = normalize_points(pts1)
    n_pts2, T2 = normalize_points(pts2)

    A = np.zeros((len(pts1), 9))
    for i in range(len(pts1)):
        u1, v1 = n_pts1[i]
        u2, v2 = n_pts2[i]
        A[i] = [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1]

    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt

    return T2.T @ F @ T1


def triangulate_points(pts1, pts2, P1, P2):
    """
    Triangulates 2D matches into 3D world points using DLT.
    
    Args:
        pts1, pts2: (N, 2) matched pixel coordinates
        P1, P2: (3, 4) Projection matrices for Cam 1 and Cam 2
    Returns:
        points_3d: (N, 3) reconstructed world coordinates
    """
    points_3d = []
    
    for i in range(len(pts1)):
        u1, v1 = pts1[i]
        u2, v2 = pts2[i]
        
        # Construct matrix A for AX = 0
        # From x cross (PX) = 0
        A = np.array([
            u1 * P1[2, :] - P1[0, :],
            v1 * P1[2, :] - P1[1, :],
            u2 * P2[2, :] - P2[0, :],
            v2 * P2[2, :] - P2[1, :]
        ])
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        
        # Convert from homogeneous (x, y, z, w) to 3D (x, y, z)
        X = X[:3] / X[3]
        points_3d.append(X)
        
    return np.array(points_3d)


def compute_reprojection_error(pts2d, pts3d, P):
    """
    Calculates the average distance between original 2D points 
    and 3D points projected back into the image.
    """
    # Convert 3D points to homogeneous (N, 4)
    n = pts3d.shape[0]
    pts3d_hom = np.hstack((pts3d, np.ones((n, 1))))
    
    # Project back to 2D
    projected_hom = (P @ pts3d_hom.T).T
    projected_2d = projected_hom[:, :2] / projected_hom[:, 2:3]
    
    # Calculate Euclidean distance
    errors = np.linalg.norm(pts2d - projected_2d, axis=1)
    return np.mean(errors), errors


def estimate_fundamental_matrix_robust(pts1, pts2):
    """
    Estimates F-matrix using RANSAC to handle outliers.
    Returns the F-matrix and the filtered inlier points.
    """
    # cv2.FM_RANSAC applies Random Sample Consensus to find the best F
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    
    # We only keep points where the mask is 1 - the 'inliers'
    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]
    
    return F, inliers1, inliers2