import numpy as np

class PinholeCamera:
    def __init__(self, K: np.ndarray, R: np.ndarray, t: np.ndarray):
        self.K = K.astype(float)
        self.R = R.astype(float)
        self.t = t.reshape(3, 1).astype(float)
        self.Rt = np.hstack((self.R, self.t))
        self.P = self.K @ self.Rt

    @classmethod
    def from_middlebury(cls, filepath: str, view_index: int):
        K, R, t = cls._parse_middlebury(filepath, view_index)
        if K is None: raise ValueError(f"View {view_index} not found")
        return cls(K, R, t)

    @staticmethod
    def _parse_middlebury(filepath: str, view_index: int):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.split()
                if f"{view_index:04d}" in parts[0]:
                    K = np.array(parts[1:10], dtype=float).reshape(3, 3)
                    R = np.array(parts[10:19], dtype=float).reshape(3, 3)
                    t = np.array(parts[19:22], dtype=float).reshape(3, 1)
                    return K, R, t
        return None, None, None

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        points_3d_hom = np.hstack((points_3d, np.ones((len(points_3d), 1))))
        points_2d_hom = (self.P @ points_3d_hom.T).T
        return points_2d_hom[:, :2] / (points_2d_hom[:, 2:3] + 1e-10)

    @property
    def center(self) -> np.ndarray:
        """Returns the camera optical center in world coordinates."""
        # The formula: C = -R^T * t
        # This transforms the camera origin (0,0,0) back to world coordinates
        return -self.R.T @ self.t

    def get_ray_directions(self, pixel_coords: np.ndarray) -> np.ndarray:
        """
        Converts 2D pixels to 3D unit ray directions in World Space.
        Used in NeRF and Ray-Tracing.
        """
        # 1. Pixel to Camera Space: K^-1 * [u, v, 1]
        K_inv = np.linalg.inv(self.K)
        p_hom = np.hstack((pixel_coords, np.ones((len(pixel_coords), 1))))
        dirs_cam = (K_inv @ p_hom.T).T
        
        # 2. Camera Space to World Space: R^T * dirs_cam
        dirs_world = (self.R.T @ dirs_cam.T).T
        
        # 3. Normalize to unit vectors
        norms = np.linalg.norm(dirs_world, axis=1, keepdims=True)
        return dirs_world / norms