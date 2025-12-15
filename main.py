
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# --- VISUALIZATION FLAGS ---
SHOW_TRAJECTORY = True
SHOW_OPENCV_VIS = True

# Flags
DRAW_TRACKS_GREEN = True    # Keypoints usati per PnP
DRAW_CANDS_RED = False       # Candidates in attesa
DRAW_NEW_TRI_BLUE = False    # Appena triangolati
DRAW_NEW_CANDS_CYAN = False  # Nuovi candidates
DRAW_REPROJ_MAGENTA = True  # Verifica riproiezione 3D->2D

# --- Setup ---
ds = 2 # 0: KITTI, 1: Malaga, 2: Parking, 3: Own Dataset

# Define dataset paths
# (Set these variables before running)
# kitti_path = "/path/to/kitti"
# malaga_path = "/path/to/malaga"
# parking_path = "/path/to/parking"
# own_dataset_path = "/path/to/own_dataset"

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
    assert 'own_dataset_path' in locals(), "You must define own_dataset_path"
else:
    raise ValueError("Invalid dataset index")

# --- PARAMETERS ---
#threshold for bearing angle function
ANGLE_THRESHOLD = np.deg2rad(5.72)  # minimum bearing angle for triangulation
#KLT PARAMETERS - !!!!!! tune them !!!!! (may be necessary to tune them fro each dataset)
klt_params=dict(
    winSize=(21,21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01)
)
#goodFeaturesToTrack PARAMETERS - !!!!!! tune them !!!!! (may be necessary to tune them fro each dataset)
max_num_corners=1000
quality_level=0.01
min_distance=2
#findEssentialMat PARAMETERS - !!!!!! tune them !!!!! (may be necessary to tune them fro each dataset)
prob_essent_mat=0.999
thresh_essent_mat=1.0
#PNP RANSAC PARAMETERS
rep_error = 3.0 
iter_count = 2000
confidence = 0.99

# --- Helper Functions ---
def P2xN_to_klt(P): return P.T.astype(np.float32).reshape(-1, 1, 2)
def klt_to_P2xN(Pklt): return Pklt.reshape(-1, 2).T.astype(np.float32)

def bearing_angle_over_threshold(K, f, c, T_cw0, T_cw):
    T_cw_h = np.vstack([T_cw, [0, 0, 0, 1]])
    T_wc_h = np.linalg.inv(T_cw_h)
    T_wc = T_wc_h[:3, :]

    T_cw0_h = np.vstack([T_cw0, [0, 0, 0, 1]])
    T_wc0_h = np.linalg.inv(T_cw0_h)
    T_wc0 = T_wc0_h[:3, :]

    f_h = np.array([f[0], f[1], 1.0])
    c_h = np.array([c[0], c[1], 1.0])

    v0_cam = np.linalg.inv(K) @ f_h
    v1_cam = np.linalg.inv(K) @ c_h

    v0_cam /= np.linalg.norm(v0_cam)
    v1_cam /= np.linalg.norm(v1_cam)

    R0 = T_wc0[:3, :3]
    R1 = T_wc[:3, :3]

    v0_w = R0 @ v0_cam
    v1_w = R1 @ v1_cam

    angle = np.arccos(np.clip(np.dot(v0_w, v1_w), -1.0, 1.0))
    return angle > ANGLE_THRESHOLD

# --- State ---
S = {
    # Localization
    "P": np.zeros((2, 0), dtype=float),  # 2xN - image coordinates of tracked features
    "X": np.zeros((3, 0), dtype=float),  # 3xN - world coordinates of tracked features

    # Triangulation candidates
    "C": np.zeros((2, 0), dtype=float),  # 2xM - position of candidate features in the current frame (image coordinates)
    "F": np.zeros((2, 0), dtype=float),  # 2xM - position of candidate features in the first frame they were observed (image coordinates)
    "T": np.zeros((12, 0), dtype=float)  # 12xM - pose of the frame at which candidate features were observed firstly observed
}

# --- Bootstrap ---
bootstrap_frames = [0, 2]  # example: you must set actual bootstrap indices

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

# 1) - harris to detect keypoints in first keyframe (img0)
pts1=cv2.goodFeaturesToTrack(img0,max_num_corners,quality_level,min_distance) #Nx1x2
n=0 if pts1 is None else pts1.shape[0]
print("Number of detected features in keyframe 1: ", n)
pts1=pts1.astype(np.float32)

# 2) - KLT to track the keypoints to the second keyframe (img1)
pts2,status,err=cv2.calcOpticalFlowPyrLK(img0,img1,pts1,None,**klt_params)
#Filter valid tracks
status=status.reshape(-1)
pts1_tracked=pts1[status==1]
pts2_tracked=pts2[status==1]
#reshape to 2xN
keypoints1=klt_to_P2xN(pts1_tracked)
keypoints2=klt_to_P2xN(pts2_tracked)

# 3) - now we have the 2D-2D point correspondences: we can apply 8-point RANSAC to retrieve the pose of the second keyframe
#origin of world frame is assumed to coincide with the pose of the first keyframe
#findEssentialMat wants Nx2
corr1=keypoints1.T.astype(np.float32)
corr2=keypoints2.T.astype(np.float32)

E,maskE=cv2.findEssentialMat(corr1,corr2,K,method=cv2.RANSAC,prob=prob_essent_mat,threshold=thresh_essent_mat)
maskE=maskE.reshape(-1).astype(bool)
#we take only the inlier correspondences
corr1_inliers=corr1[maskE]
corr2_inliers=corr2[maskE]
#decompose E into R and t
_,R,t,maskPose=cv2.recoverPose(E,corr1_inliers,corr2_inliers,K)
#filter again good correspondences
maskPose=maskPose.reshape(-1).astype(bool)
corr1_final=corr1_inliers[maskPose]
corr2_final=corr2_inliers[maskPose]

t=t.reshape(3,1)
T_CW2=np.hstack((R,t)).astype(np.float64) #3x4

# 4) - finally, we can perform triangulation, and thus construnct the first point cloud
#compute the projection matrices
P1=K@np.hstack((np.eye(3),np.zeros((3,1))))
P2=K@np.hstack((R,t))
#triangulatePoints wants 2xN
points1=corr1_final.T
points2=corr2_final.T
X_homogeneous=cv2.triangulatePoints(P1,P2,points1,points2)
X=(X_homogeneous[:3,:]/X_homogeneous[3:4,:]).astype(np.float32)

# 5) - set up of the state
S["P"]=points2
S["X"]=X
#to create the candidates set C, we must detect new features, and check that they are not already in P
cand=cv2.goodFeaturesToTrack(img1,max_num_corners,quality_level,min_distance)
cand=klt_to_P2xN(cand)
#to ensures points in C are not redundant with ones in P, we perform a minimum distance check
diff=cand[:,:,None]-points2[:,None,:]
dist_sq=np.sum(diff**2,axis=0) #distance of each candidate to all points in P
min_dist_sq=np.min(dist_sq,axis=1) #distance of each candidate to the closest point in P
#Keep only candidates farther than min_distance
mask=min_dist_sq>(min_distance**2)
C=cand[:,mask]
T=np.repeat(T_CW2.reshape(12,1),C.shape[1],axis=1)
S["C"]=C
S["F"]=C.copy()
S["T"]=T

# --- MATPLOTLIB INIT (TRAJECTORY) ---
if SHOW_TRAJECTORY:
    plt.ion()
    fig_traj = plt.figure(figsize=(7, 7))
    ax_traj = fig_traj.add_subplot(111)
    
    # Ground Truth
    if 'ground_truth' in locals() and ground_truth is not None:
        ax_traj.plot(ground_truth[:, 0], ground_truth[:, 1], 'k--', label="Ground Truth", alpha=0.5)

    est_traj_x = []
    est_traj_z = []
    line_est, = ax_traj.plot([], [], 'b.-', label="Estimated PnP")
    arrow_curr = None
    
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Z (m)")
    ax_traj.set_title("Trajectory (Top-Down View)")
    ax_traj.legend()
    ax_traj.grid(True)
    ax_traj.axis('equal')

prev_img = img1

# --- Continuous operation ---
for i in range(bootstrap_frames[1] + 1, last_frame + 1):
    print(f"\nProcessing frame {i}")

    # --- 0. LOAD IMAGE ---
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

    # Prepare Visualization Image
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    n_triangulated = 0
    n_new_candidates = 0

    # --- 1. TRACK KEYPOINTS (LANDMARKS) ---
    P_prev = P2xN_to_klt(S["P"])
    X_prev = S["X"].T.astype(np.float32)
    
    P_tr, st, _ = cv2.calcOpticalFlowPyrLK(prev_img, img, P_prev, None, **klt_params)
    st = st.reshape(-1).astype(bool)
    
    P_tr_valid = P_tr.reshape(-1, 2)[st] #contiene i punti 2d trackati in curr_img
    X_tr = X_prev[st]
    P_prev_valid = P_prev[st].reshape(-1, 2) #contiene i punti 2d i prev_img che sono riuscito a trackare
    print(f"NUM KEYPOINTS: {len(X_tr)}")

    # --- 2. LOCALIZATION (PnP + RANSAC) ---
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=X_tr,
        imagePoints=P_tr_valid,
        cameraMatrix=K,
        distCoeffs=None,
        reprojectionError=rep_error,
        iterationsCount=iter_count,
        confidence=confidence,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if (not ok) or (inliers is None) or (len(inliers) < 4):
        print("PnP failed")
        prev_img = img
        continue
    print(f"NUM INLIERS: {len(inliers)}")
    print (f"Percentage: {len(inliers)/len(X_tr)}")
    inliers = inliers.reshape(-1)
    P_in = P_tr_valid[inliers]
    X_in = X_tr[inliers]
    P_prev_in = P_prev_valid[inliers]

    R_cw, _ = cv2.Rodrigues(rvec)
    T_cw = np.hstack([R_cw, tvec])

    S["P"] = P_in.T
    S["X"] = X_in.T

    # --- PLOT: TRAJECTORY UPDATE (Matplotlib) ---
    if SHOW_TRAJECTORY:
        # Camera pose in World frame: T_wc = inv(T_cw)
        R_wc = R_cw.T
        t_wc = -R_wc @ tvec
        
        est_traj_x.append(t_wc[0, 0])
        est_traj_z.append(t_wc[2, 0])
        
        # Update plot every frame (or utilize modulus for speed)
        line_est.set_data(est_traj_x, est_traj_z)
        
        if arrow_curr: arrow_curr.remove()
        
        # Forward vector (Z-axis of camera in world frame) is the 3rd column of R_wc
        forward_x = R_wc[0, 2]
        forward_z = R_wc[2, 2]
        arrow_curr = ax_traj.arrow(t_wc[0, 0], t_wc[2, 0], forward_x*2, forward_z*2, 
                                   head_width=1.0, head_length=1.0, fc='r', ec='r')
        
        ax_traj.relim()
        ax_traj.autoscale_view()
        fig_traj.canvas.draw()
        fig_traj.canvas.flush_events()

    # --- PLOT: OPENCV VISUALIZATION ---
    if SHOW_OPENCV_VIS:
        
        # A. Keypoints (GREEN)
        if DRAW_TRACKS_GREEN:
            for prev_pt, curr_pt in zip(P_prev_in, P_in):
                cv2.line(vis_img, (int(prev_pt[0]), int(prev_pt[1])), (int(curr_pt[0]), int(curr_pt[1])), (0, 255, 0), 1)
                cv2.circle(vis_img, (int(curr_pt[0]), int(curr_pt[1])), 3, (0, 255, 0), -1)

        # B. Reprojection Check (MAGENTA)
        if DRAW_REPROJ_MAGENTA:
            points_reproj, _ = cv2.projectPoints(X_in, rvec, tvec, K, None)
            points_reproj = points_reproj.reshape(-1, 2)
            for pt in points_reproj:
                cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 2, (255, 0, 255), -1)

    # --- 3. CANDIDATE TRACKING & TRIANGULATION ---
    if S["C"].shape[1] > 0:
        C_prev = P2xN_to_klt(S["C"])
        C_tr, stc, _ = cv2.calcOpticalFlowPyrLK(prev_img, img, C_prev, None, **klt_params)
        C_tr = klt_to_P2xN(C_tr)

        if stc is not None:
            stc = stc.reshape(-1).astype(bool)
            C_tr = C_tr[:, stc]
            C_prev_valid = C_prev[stc].reshape(-1, 2) # For plotting

            # C. Candidates (RED)
            if SHOW_OPENCV_VIS and DRAW_CANDS_RED:
                for prev_pt, curr_pt in zip(C_prev_valid, C_tr.T):
                    cv2.line(vis_img, (int(prev_pt[0]), int(prev_pt[1])), (int(curr_pt[0]), int(curr_pt[1])), (0, 0, 255), 1)
                    cv2.circle(vis_img, (int(curr_pt[0]), int(curr_pt[1])), 2, (0, 0, 255), -1)

            F_tr = S["F"][:, stc]
            T_tr = S["T"][:, stc]

            new_P = []
            new_X = []
            promoted_idx = []

            for idx in range(C_tr.shape[1]):
                c = C_tr[:, idx]
                f = F_tr[:, idx]
                T_CW_fvec = T_tr[:, idx]
                
                T_cw0 = T_CW_fvec.reshape(3, 4)

                if not bearing_angle_over_threshold(K, f, c, T_cw0, T_cw):
                    continue

                P0 = K @ T_cw0
                P1 = K @ T_cw
                X_h = cv2.triangulatePoints(P0, P1, f.reshape(2, 1), c.reshape(2, 1))
                X = X_h[:3] / X_h[3]


                new_P.append(c)
                new_X.append(X.flatten())
                promoted_idx.append(idx)

            if len(new_P) > 0:
                new_P_arr = np.array(new_P).T
                new_X_arr = np.array(new_X).T
                S["P"] = np.hstack([S["P"], new_P_arr])
                S["X"] = np.hstack([S["X"], new_X_arr])
                n_triangulated = new_P_arr.shape[1]
                
                # D. New Triangulations (BLUE)
                if SHOW_OPENCV_VIS and DRAW_NEW_TRI_BLUE:
                    for pt in new_P_arr.T:
                        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), 2)

            if C_tr.shape[1] > 0:
                keep_mask = np.ones(C_tr.shape[1], dtype=bool)
                keep_mask[promoted_idx] = False
                S["C"] = C_tr[:, keep_mask]
                S["F"] = F_tr[:, keep_mask]
                S["T"] = T_tr[:, keep_mask]

    # --- 4. ADD NEW CANDIDATES ---
    new_corners = cv2.goodFeaturesToTrack(img, max_num_corners, quality_level, min_distance)
    if new_corners is not None:
        cand = klt_to_P2xN(new_corners.astype(np.float32))
        mask = np.ones(cand.shape[1], dtype=bool)

        if S["P"].shape[1] > 0:
            diffP = cand[:, :, None] - S["P"][:, None, :]
            mask &= (np.min(np.sum(diffP**2, axis=0), axis=1) > min_distance**2)

        if S["C"].shape[1] > 0:
            diffC = cand[:, :, None] - S["C"][:, None, :]
            mask &= (np.min(np.sum(diffC**2, axis=0), axis=1) > min_distance**2)

        C_new = cand[:, mask]

        if C_new.shape[1] > 0:
            T12 = T_cw.reshape(12, 1)
            T_new = np.repeat(T12, C_new.shape[1], axis=1)

            S["C"] = np.hstack([S["C"], C_new])
            S["F"] = np.hstack([S["F"], C_new.copy()])
            S["T"] = np.hstack([S["T"], T_new])
            n_new_candidates = C_new.shape[1]
            
            # E. New Candidates (CYAN)
            if SHOW_OPENCV_VIS and DRAW_NEW_CANDS_CYAN:
                 for pt in C_new.T:
                    cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 2, (255, 255, 0), -1)

    # --- FINAL VISUALIZATION (LEGEND & SHOW) ---
    if SHOW_OPENCV_VIS:
        x_leg, y_leg, gap = 10, 20, 20
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5
        
        cv2.putText(vis_img, f"Tracked (P): {S['P'].shape[1]}", (x_leg, y_leg), font, scale, (0, 255, 0), 1, cv2.LINE_AA)
        if DRAW_REPROJ_MAGENTA:
            cv2.putText(vis_img, f"Reprojected", (x_leg, y_leg+gap), font, scale, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(vis_img, f"Candidates (C): {S['C'].shape[1]}", (x_leg, y_leg+2*gap), font, scale, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(vis_img, f"Triangulated: +{n_triangulated}", (x_leg, y_leg+3*gap), font, scale, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(vis_img, f"New Cands: +{n_new_candidates}", (x_leg, y_leg+4*gap), font, scale, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(vis_img, f"Frame: {i}", (x_leg, y_leg+5*gap+5), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Visual Odometry", vis_img)
    
    prev_img = img 
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break

cv2.destroyAllWindows()
plt.ioff()
plt.show()
