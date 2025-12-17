import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from plotting_utils import init_live_plots, update_traj, update_world, update_frame_with_points, cam_center_from_Tcw


# --- Setup ---
ds = 0  # 0: KITTI, 1: Malaga, 2: Parking, 3: Own Dataset
visualize_frames = False

# Define dataset paths
# (Set these variables before running)
# kitti_path = "/path/to/kitti"
# malaga_path = "/path/to/malaga"
# parking_path = "/path/to/parking"
# own_dataset_path = "/path/to/own_dataset"

if ds == 0:
    #threshold for bearing angle function
    ANGLE_THRESHOLD = np.deg2rad(0.1)
    bootstrap_frames = [0, 2] #frames betweem which bootstrap is performed
    kitti_path = r"./datasets/kitti"
    ground_truth = np.loadtxt(os.path.join(kitti_path, 'poses', '05.txt'))
    ground_truth = ground_truth[:, [-9, -1]]  # same as MATLAB(:, [end-8 end])
    last_frame = 400
    K = np.array([
        [7.18856e+02, 0, 6.071928e+02],
        [0, 7.18856e+02, 1.852157e+02],
        [0, 0, 1]
    ])
    HAS_GT=True
    poses=np.loadtxt(os.path.join(kitti_path,'poses','05.txt')) #N x 12
    gt_x=poses[:,3]
    gt_z=poses[:,11]
elif ds == 1:
    ANGLE_THRESHOLD = np.deg2rad(0.02)
    bootstrap_frames = [0, 2] #frames betweem which bootstrap is performed
    malaga_path = r"./datasets/malaga-urban-dataset-extract-07"
    img_dir = os.path.join(
        malaga_path,
        "malaga-urban-dataset-extract-07_rectified_800x600_Images"
    )
    left_images = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.endswith("_left.jpg")
    ])
    last_frame = len(left_images) - 1
    K = np.array([
        [621.18428, 0, 404.0076],
        [0, 621.18428, 309.05989],
        [0, 0, 1]
    ])
    HAS_GT=False
    gt_x=gt_z=None
elif ds == 2:
    #threshold for bearing angle function
    ANGLE_THRESHOLD = np.deg2rad(5.72)
    bootstrap_frames = [0, 2] #frames betweem which bootstrap is performed
    parking_path = r"./datasets/parking"
    last_frame = 598
    K = np.loadtxt(os.path.join(parking_path, 'K.txt'), delimiter=',', usecols=(0, 1, 2))
    ground_truth = np.loadtxt(os.path.join(parking_path, 'poses.txt'))
    ground_truth = ground_truth[:, [-9, -1]]
    HAS_GT=True
    poses=np.loadtxt(os.path.join(parking_path,'poses.txt'))  #N x 12
    gt_x=poses[:,3]
    gt_z=poses[:,11]
elif ds == 3:
    bootstrap_frames = [0, 15] #frames betweem which bootstrap is performed
    HAS_GT=False
    gt_x=gt_z=None
    ANGLE_THRESHOLD = np.deg2rad(0.1)
    last_frame = 1740
    own_dataset_path = r"./datasets/our_dataset7"
    K = np.array([
        [1109.7, 0, 637.5062],
        [0, 1113.5, 357.1623],
        [0, 0, 1]
    ])
else:
    raise ValueError("Invalid dataset index")

K_inv = np.linalg.inv(K)

# --- PARAMETERS ---
#KLT PARAMETERS
klt_params=dict(
    winSize=(21,21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01)
)
#goodFeaturesToTrack PARAMETERS
max_num_corners=1000
quality_level=0.01
min_distance=2
#findEssentialMat PARAMETERS
prob_essent_mat=0.999
thresh_essent_mat=1.0
#PNP RANSAC PARAMETERS
rep_error = 3.0 
iter_count = 2000
confidence = 0.99

# --- Helper Functions ---
#functions that transform keypoints from 2xN shape to Nx1x2 shape for KLT, and viceversa
def P2xN_to_klt(P):return P.T.astype(np.float32).reshape(-1,1,2) 
def klt_to_P2xN(Pklt):return Pklt.reshape(-1,2).T.astype(np.float32)


def bearing_angle_over_threshold(K, K_inv, f, c, T_cw0, T_cw):
    T_cw_h = np.vstack([T_cw, [0, 0, 0, 1]])
    T_wc_h = np.linalg.inv(T_cw_h)
    T_wc = T_wc_h[:3, :]

    T_cw0_h = np.vstack([T_cw0, [0, 0, 0, 1]])
    T_wc0_h = np.linalg.inv(T_cw0_h)
    T_wc0 = T_wc0_h[:3, :]

    f_h = np.array([f[0], f[1], 1.0]) #f in homogeneous coordinates
    c_h = np.array([c[0], c[1], 1.0]) #c in homogeneous coordinates

    v0_cam = K_inv @ f_h
    v1_cam = K_inv @ c_h

    v0_cam /= np.linalg.norm(v0_cam) #normalized vector from f 
    v1_cam /= np.linalg.norm(v1_cam) #normalized vector from c

    R0 = T_wc0[:3, :3]
    R1 = T_wc[:3, :3]

    v0_w = R0 @ v0_cam
    v1_w = R1 @ v1_cam

    angle = np.arccos(np.clip(np.dot(v0_w, v1_w), -1.0, 1.0))   

    return angle>ANGLE_THRESHOLD

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
    img0 = cv2.imread(os.path.join(own_dataset_path, 'Images', f"img_{bootstrap_frames[0]:05d}.png"), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(os.path.join(own_dataset_path,'Images',  f"img_{bootstrap_frames[1]:05d}.png"), cv2.IMREAD_GRAYSCALE)
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


prev_img = img1
# --- Continuous operation ---
gt=None
if HAS_GT:
    gt=(gt_x,gt_z)
if visualize_frames:
    plots=init_live_plots(gt=gt)
traj=[]
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
        image_path = os.path.join(own_dataset_path, 'Images', f"img_{i:05d}.png")
    else:
        raise ValueError("Invalid dataset index")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: could not read {image_path}")
        continue
    
    P_prev_state=S["P"].copy() #2xN_prev (positions in prev_img)

    # 1) - track keypoints from previous frame, that are already associated to a landmark
    P_prev =  P2xN_to_klt(S["P"]) # Nx1x2
    X_prev = S["X"].T.astype(np.float32) # Nx3
    #track with klt
    P_tr, st, _ = cv2.calcOpticalFlowPyrLK(prev_img, img, P_prev, None, **klt_params)
    st = st.reshape(-1).astype(bool)
    #filter out keypoints for which tracking fails, and also corresponding landmarks
    P_tr = P_tr.reshape(-1, 2)[st] # Nx2
    X_tr = X_prev[st] # Nx3

    print(f"Tracked keypoints: {P_tr.shape[0]}")

    # 2) - LOCALIZATION: exploiting the now established 3D-2D correspondences between landmarks and
    #keypoints in the current frame, with PnP + RANSAC we retrieve the pose of the current frame wrt the world
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=X_tr,
        imagePoints=P_tr,
        cameraMatrix=K,
        distCoeffs=None,
        reprojectionError=rep_error, #defines how far from the model points start to be considered outliers
        iterationsCount=iter_count, #max number of iteration of RANSAC
        confidence=confidence,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if (not ok) or (inliers is None) or (len(inliers) < 4):
        print(f"PnP failed / inliers too few: {0 if inliers is None else len(inliers)}")
        prev_img = img
        continue

    inliers = inliers.reshape(-1)
    #filter out outliers
    print(f"PnP inliers: {len(inliers)} ")
    P_in = P_tr[inliers] # Nx2
    X_in = X_tr[inliers] # Nx3
    P_prev_valid=P_prev.reshape(-1,2)[st] #prev positions Nx2
    P_prev_in=P_prev_valid[inliers] #prev positions of inlier tracks Nx2

    R_cw, _ = cv2.Rodrigues(rvec)
    T_cw = np.hstack([R_cw,tvec])

    # update 2D keypoints and 3D landmarks of the state, to the current frame
    S["P"] = P_in.T
    S["X"] = X_in.T
    P_prev_for_plot=P_prev_in.T #2xN_inliers

    traj.append(cam_center_from_Tcw(T_cw))

    if visualize_frames:
        update_traj(plots,traj)
        update_world(plots,T_cw,S["X"])
        update_frame_with_points(plots,img,S["P"],P_prev_for_plot,frame_idx=i)

        plots["fig"].canvas.draw()
        plots["fig"].canvas.flush_events()
        plt.pause(0.001)

    # 3) - 3D MAP COUNTINUOUS UPDATE: in this section we analyze each element of C, which is the set of candidates
    #keypoints. If they satisfy approrpiate conditions, they are triangulated and moved from C to P, and added to X
    if S["C"].shape[1] > 0:
        C_prev =  P2xN_to_klt(S["C"]) # Mx1x2
        #track the candidates of the previous frame to the current one
        C_tr, stc, _ = cv2.calcOpticalFlowPyrLK(prev_img, img, C_prev, None, **klt_params)
        C_tr = klt_to_P2xN(C_tr) # 2xM

        if stc is not None:
            stc = stc.reshape(-1).astype(bool)
            C_tr = C_tr[:,stc]

            F_tr = S["F"][:, stc]
            T_tr = S["T"][:, stc]

            new_P = []
            new_X = []
            promoted_idx=[]
            #now we loop over all elements of C and, if it's appropriate, triangulate them
            #stiamo consideranod T_CW_f
            for idx in range(C_tr.shape[1]):
                c = C_tr[:, idx]
                f = F_tr[:, idx]
                T_CW_fvec = T_tr[:, idx]
                        
                T_cw0 = T_CW_fvec.reshape(3, 4)
    
                if not bearing_angle_over_threshold(K, K_inv, f, c, T_cw0, T_cw):
                    continue         
              
                #compute projection matrices
                P0 = K @ T_cw0
                P1 = K @ T_cw
                X_h = cv2.triangulatePoints(P0, P1,f.reshape(2,1),c.reshape(2,1))
                X = X_h[:3] / X_h[3]
                #now we want to verify that no points triangulated are behind the camera
                T_cw0_h = np.vstack([T_cw0, [0, 0, 0, 1]])
                T_cw_h  = np.vstack([T_cw,  [0, 0, 0, 1]])

                X_c0 = T_cw0_h @ np.vstack([X, 1.0])
                X_c1 = T_cw_h  @ np.vstack([X, 1.0])

                if X_c0[2] <= 0 or X_c1[2] <= 0 or X_c0[2]>300 or X_c1[2]>300:
                    continue

                new_P.append(c)
                new_X.append(X.flatten())
                promoted_idx.append(idx)

            if len(new_P) > 0:
                new_P = np.array(new_P).T # 2xN
                new_X = np.array(new_X).T # 3xN
                S["P"] = np.hstack([S["P"], new_P])
                S["X"] = np.hstack([S["X"], new_X])
                #remove triangulated points from C, F and T
            if C_tr.shape[1] > 0:
                keep_mask = np.ones(C_tr.shape[1], dtype=bool)
                keep_mask[promoted_idx] = False
                S["C"] = C_tr[:, keep_mask]
                S["F"] = F_tr[:, keep_mask]
                S["T"] = T_tr[:, keep_mask]

    # 4) - CANDIDATES SET UPDATE
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

    prev_img = img

if not visualize_frames:
    if len(traj) == 0:
        print("No trajectory to plot.")
    else:
        traj_arr = np.array(traj)
        
        # Creazione figura con lo stile della funzione init_live_plots
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(1, 1, 1)
        
        # --- Estetica dello stile richiesto ---
        ax.set_title("Estimated Trajectory (Final Result)", fontsize=14, fontweight='bold')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.axis("equal")
        
        # Griglia specifica: tratteggiata, sottile e semitrasparente
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        
        # --- Plot Ground Truth (se presente) ---
        if HAS_GT and (gt_x is not None) and (gt_z is not None):
            ax.plot(
                gt_x, gt_z, 
                linestyle='--', 
                color='black', 
                linewidth=1.0, 
                alpha=0.7, 
                label='Ground Truth'
            )
            
        # --- Plot Traiettoria Stimata ---
        # Usiamo il colore rosso e lo spessore 2.0 come nel tuo stile live
        ax.plot(
            traj_arr[:, 0], traj_arr[:, 2], 
            color='red', 
            linewidth=2.0, 
            label='Estimated Trajectory'
        )
        
        # Legenda e layout
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        
        plt.show()