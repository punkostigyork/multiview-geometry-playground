import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_epipolar_lines(img1, img2, F, pts1, pts2, num_lines=10):
    """
    Draws epipolar lines on img2 corresponding to points in img1.
    """
    # Select a subset of points to keep the plot clean
    indices = np.random.choice(len(pts1), num_lines, replace=False)
    pts1_sub = pts1[indices]
    pts2_sub = pts2[indices]

    # Find epilines corresponding to points in img1: l = Fx
    # pts1_sub needs to be (N, 1, 2) for cv2.computeCorrespondEpilines
    lines = cv2.computeCorrespondEpilines(pts1_sub.reshape(-1, 1, 2), 1, F)
    lines = lines.reshape(-1, 3)

    img1_out = img1.copy()
    img2_out = img2.copy()

    w = img2.shape[1]
    
    for r, pt1, pt2 in zip(lines, pts1_sub, pts2_sub):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Line equation: ax + by + c = 0
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [w, -(r[2] + r[0]*w)/r[1]])
        
        # Draw the line on the second image
        img2_out = cv2.line(img2_out, (x0, y0), (x1, y1), color, 2)
        # Draw the points
        img1_out = cv2.circle(img1_out, tuple(pt1.astype(int)), 5, color, -1)
        img2_out = cv2.circle(img2_out, tuple(pt2.astype(int)), 5, color, -1)

    return img1_out, img2_out


def export_to_ply(filename, points, colors=None):
    """
    Exports 3D points to a PLY file for viewing in MeshLab/Blender.
    """
    header = """ply
format ascii 1.0
element vertex {0}
property float x
property float y
property float z
{1}
end_header
"""
    color_prop = "property uchar red\nproperty uchar green\nproperty uchar blue" if colors is not None else ""
    
    with open(filename, 'w') as f:
        f.write(header.format(len(points), color_prop))
        for i in range(len(points)):
            line = f"{points[i,0]} {points[i,1]} {points[i,2]}"
            if colors is not None:
                line += f" {int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}"
            f.write(line + "\n")