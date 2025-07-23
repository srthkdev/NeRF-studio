import tempfile
import numpy as np
from app.ml.nerf import colmap_utils

def test_extract_camera_poses_real():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a valid cameras.txt
        cameras_txt = f"{tmpdir}/cameras.txt"
        with open(cameras_txt, "w") as f:
            f.write("""# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
1 PINHOLE 800 600 1000 1000 400 300
""")
        # Write a valid images.txt
        images_txt = f"{tmpdir}/images.txt"
        with open(images_txt, "w") as f:
            f.write("""# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
1 1 0 0 0 0 0 0 1 img1.png
0 0 0 0
""")
        K, poses = colmap_utils.extract_camera_poses(tmpdir)
        assert K.shape == (3, 3)
        assert poses.shape[1:] == (4, 4)
        assert poses.shape[0] == 1 