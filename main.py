import os
import cv2
import numpy as np
from glob import glob


# --- Setup ---
ds = 2  # 0: KITTI, 1: Malaga, 2: Parking, 3: Own Dataset

# Define dataset paths
# (Set these variables before running)
kitti_path = "/path/to/kitti"
malaga_path = "/path/to/malaga"
parking_path = "/path/to/parking"
own_dataset_path = "/path/to/own_dataset"



if ds == 0:
  
    kitti_path = r"./datasets/kitti05"
    ground_truth = np.loadtxt(os.path.join(kitti_path, 'poses', '05.txt'))
    ground_truth = ground_truth[:, [-9, -1]]  # same as MATLAB(:, [end-8 end])
    last_frame = 4540
    K = np.array([
        [7.18856e+02, 0, 6.071928e+02],
        [0, 7.18856e+02, 1.852157e+02],
        [0, 0, 1]
    ])
elif ds == 1:
  
    malaga_path = r"./datasets/malaga"
    img_dir = os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images')
    left_images = sorted(glob(os.path.join(img_dir, '*.png')))
    last_frame = len(left_images)
    K = np.array([
        [621.18428, 0, 404.0076],
        [0, 621.18428, 309.05989],
        [0, 0, 1]
    ])
elif ds == 2:
    parking_path = r"./datasets/parking"
    last_frame = 598
    K = np.loadtxt(os.path.join(parking_path, 'K.txt'), delimiter=',', usecols=(0, 1, 2))
    ground_truth = np.loadtxt(os.path.join(parking_path, 'poses.txt'))
    ground_truth = ground_truth[:, [-9, -1]]
elif ds == 3:
    # Own Dataset
    # TODO: define your own dataset and load K obtained from calibration of own camera
    assert 'own_dataset_path' in locals(), "You must define own_dataset_path"

else:
    raise ValueError("Invalid dataset index")



# --- PARAMETERS ---
#KLT PARAMETERS - !!!!!! tune them !!!!! (may be necessary to tune them fro each dataset)
klt_params=dict(
    winSize=(21,21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01)
)
#GOODFEATURESTOTRACK PARAMETERS - !!!!!! tune them !!!!! (may be necessary to tune them fro each dataset)
max_num_corners=1000
quality_level=0.01
min_distance=2

#PNP RANSAC PARAMETERS
rep_error = 3.0 
iter_count = 200
confidence = 0.9999

#Functions that transform keypoints from 2xN shape to Nx1x2 shape for KLT
def P2xN_to_klt(P):return P.T.astype(np.float32).reshape(-1,1,2) 
def klt_to_P2xN(Pklt):return Pklt.reshape(-1,2).T.astype(np.float32)


# --- Bootstrap ---
bootstrap_frames = [0, 1]  # example: you must set actual bootstrap indices

if ds == 0:
    img0 = cv2.imread(os.path.join(kitti_path, '05', 'image_0', f"{bootstrap_frames[0]:06d}.png"), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(os.path.join(kitti_path, '05', 'image_0', f"{bootstrap_frames[1]:06d}.png"), cv2.IMREAD_GRAYSCALE)
elif ds == 1:
    img0 = cv2.imread(left_images[bootstrap_frames[0]], cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(left_images[bootstrap_frames[1]], cv2.IMREAD_GRAYSCALE)
elif ds == 2:
    img0 = cv2.imread(os.path.join(parking_path, 'images', f"img_{bootstrap_frames[0]:05d}.png"), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(os.path.join(parking_path, 'images', f"img_{bootstrap_frames[1]:05d}.png"), cv2.IMREAD_GRAYSCALE)
elif ds == 3:
    # Load images from own dataset
    img0 = cv2.imread(os.path.join(own_dataset_path, f"{bootstrap_frames[0]:06d}.png"), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(os.path.join(own_dataset_path, f"{bootstrap_frames[1]:06d}.png"), cv2.IMREAD_GRAYSCALE)
else:
    raise ValueError("Invalid dataset index")

S = {
    # Localization
    "P": np.zeros((2, 0), dtype=float),  # 2xN
    "X": np.zeros((3, 0), dtype=float),  # 3xN

    # Triangulation candidates
    "C": np.zeros((2, 0), dtype=float),  # 2xM
    "F": np.zeros((2, 0), dtype=float),  # 2xM
    "T": np.zeros((12, 0), dtype=float)  # 12xM (pose at first obs)
}

prev_img = img1
# --- Continuous operation ---
for i in range(bootstrap_frames[1] + 1, last_frame + 1):
    print(f"\n\nProcessing frame {i}\n=====================")

   
    # LOAD IMAGE 
   
    if ds == 0:
        image_path = os.path.join(kitti_path, '05', 'image_0', f"{i:06d}.png")
    elif ds == 1:
        image_path = left_images[i]
    elif ds == 2:
        image_path = os.path.join(parking_path, 'images', f"img_{i:05d}.png")
    elif ds == 3:
        image_path = os.path.join(own_dataset_path, f"{i:06d}.png")
    else:
        raise ValueError("Invalid dataset index")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: could not read {image_path}")
        continue

    
    # SHOW IMAGE 
    cv2.imshow("Input image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

   
    # 1) TRACK ACTIVE KEYPOINTS P (KLT)
    P_prev =  P2xN_to_klt(S["P"])  # Nx1x2
    X_prev = S["X"].T.astype(np.float32)                    # Nx3

    P_tr, st, _ = cv2.calcOpticalFlowPyrLK(prev_img, img, P_prev, None, **klt_params)
    

    st = st.reshape(-1).astype(bool)
    P_tr = P_tr.reshape(-1, 2)[st]  # Nx2
    X_tr = X_prev[st]               # Nx3

    

    # 2) PnP + RANSAC (2D-3D) -> Pose
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=X_tr,
        imagePoints=P_tr,
        cameraMatrix=K,
        distCoeffs=None,
        reprojectionError=rep_error, #when we consider a point inlier or not
        iterationsCount=iter_count, #number of random subsets
        confidence=confidence,
        flags=cv2.SOLVEPNP_ITERATIVE
    )


    inliers = inliers.reshape(-1)
    P_in = P_tr[inliers]  # nin x2
    X_in = X_tr[inliers]  # nin x3

    R_cw, _ = cv2.Rodrigues(rvec)
    T_cw = np.hstack([R_cw,tvec])
    T_cw=np.vstack([T_cw,[0,0,0,1]])
    T_wc = np.linalg.inv(T_cw)
    T_wc=T_wc.flatten()

    # aggiorna stato localization in formato 2xN, 3xN
    S["P"] = P_in.T
    S["X"] = X_in.T

    #nello stato abbiamo inserito i PUNTI 2D e 3D relativi a img (che useremo come prev_img al prossimo loop)


    # 3) TRACK CANDIDATES C (KLT)
    if S["C"].shape[1] > 0:
        C_prev =  P2xN_to_klt(S["C"])  # Mx1x2

        C_tr, stc, _ = cv2.calcOpticalFlowPyrLK(prev_img, img, C_prev, None, *klt_params)
    
        stc = stc.reshape(-1).astype(bool)
        C_tr = C_tr.reshape(-1, 2)[stc]       # Mc x2, teniamo solo i candidate points che sono stati trackati con successo 
        F_tr = S["F"].T[stc]                  # Mc x2, shrinkiamo F
        T_tr = S["T"][:, stc]                 # 12 x Mc, shrinkiamo T

        S["C"] = C_tr.T                       #aggiorniamo lo stato (C, F, T)
        S["F"] = F_tr.T
        S["T"] = T_tr

   
   