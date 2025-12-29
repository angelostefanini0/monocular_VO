import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from utils.plotting_utils import init_live_plots, update_traj, update_world, update_frame_with_points, cam_center_from_Tcw, plot_trajectory
from utils.utils import compute_all_angles, P2xN_to_klt, klt_to_P2xN,run_ba
import time

# --- Setup ---
ds = 0  # 0: KITTI, 1: Malaga, 2: Parking, 3: Own Dataset
visualize_frames = False
use_BA=True

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
    kitti_path = r"./datasets/kitti05"
    ground_truth = np.loadtxt(os.path.join(kitti_path, 'poses', '05.txt'))
    ground_truth = ground_truth[:, [-9, -1]]  # same as MATLAB(:, [end-8 end])
    last_frame = 2670
    last_frame = 1000    #TEST
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
    own_dataset_path = r"./datasets/our_dataset8"
    K = np.array([
        [1109.7, 0, 637.5062],
        [0, 1113.5, 357.1623],
        [0, 0, 1]
    ])
else:
    raise ValueError("Invalid dataset index")


start_time = time.time()

K_inv = np.linalg.inv(K)

# --- PARAMETERS ---
#KLT PARAMETERS
klt_params=dict(
    winSize=(21,21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01)
)
#goodFeaturesToTrack PARAMETERS
max_num_corners_bootstrap=1000
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
#Bundle Adjustment PARAMETERS
buffer_dim=20                    #5,15 Huber,max n_fev=100 ok 15 Hz         20,10 Huber,max n_fev=100 ok 13 Hz            
update_freq=5                  #20,50 soft_l1,max n_fev=100 12.2 Hz           5,15 Huber,no max n_fev 12 Hz
buffer=[]



# --- State ---
S = {
    # Localization
    "P": np.zeros((2, 0), dtype=float),  # 2xN - image coordinates of tracked features
    "X": np.zeros((3, 0), dtype=float),  # 3xN - world coordinates of tracked features

    # Triangulation candidates
    "C": np.zeros((2, 0), dtype=float),  # 2xM - position of candidate features in the current frame (image coordinates)
    "F": np.zeros((2, 0), dtype=float),  # 2xM - position of candidate features in the first frame they were observed (image coordinates)
    "T": np.zeros((12, 0), dtype=float),  # 12xM - pose of the frame at which candidate features were observed firstly observed
    "ids": np.zeros((0),dtype=int)  #to track the id of every X point
}
next_landmark_id = 0

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
pts1=cv2.goodFeaturesToTrack(img0,max_num_corners_bootstrap,quality_level,min_distance) #Nx1x2
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
S["ids"] = np.arange(next_landmark_id, next_landmark_id + X.shape[1])
next_landmark_id += X.shape[1]
#to create the candidates set C, we must detect new features, and check that they are not already in P
cand=cv2.goodFeaturesToTrack(img1,max_num_corners_bootstrap,quality_level,min_distance)
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
S["frame_index"] = np.zeros(C.shape[1], dtype = int)
assert S["frame_index"].shape[0] == S["C"].shape[1], f"point 1, {S['frame_index'].shape[0]}"


SAVINGS_T_wc = []

R_cw = T_CW2[:3, :3]
t_cw = T_CW2[:3, 3]
R_wc = R_cw.T                    
t_wc = -R_wc @ t_cw              
T_wc = np.hstack([R_wc, t_wc.reshape(3, 1)])
SAVINGS_T_wc.append(T_wc)


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
    ids_tr = S["ids"][st] #tracks indexes for BA

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
    ids_in=ids_tr[inliers]
    P_prev_valid=P_prev.reshape(-1,2)[st] #prev positions Nx2
    P_prev_in=P_prev_valid[inliers] #prev positions of inlier tracks Nx2

    R_cw, _ = cv2.Rodrigues(rvec)
    T_cw = np.hstack([R_cw,tvec])

    # update 2D keypoints and 3D landmarks of the state, to the current frame
    S["P"] = P_in.T
    S["X"] = X_in.T
    S["ids"] = ids_in
    P_prev_for_plot=P_prev_in.T #2xN_inliers

    current_obs = {}
    for j, lid in enumerate(ids_in):
        current_obs[lid] = P_in[j] 
        
    buffer.append({
        'pose': T_cw.copy(), 
        'obs': current_obs })
    if len(buffer)>buffer_dim:
        buffer.pop(0)

    # traj.append(cam_center_from_Tcw(T_cw))

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
            frame_index = S["frame_index"][stc]

            new_P = []
            new_X = []
            promoted_idx=[]
            #now we loop over all elements of C and, if it's appropriate, triangulate them
            #stiamo consideranod T_CW_f

            all_angles = compute_all_angles(K_inv, F_tr, C_tr, frame_index, T_cw, SAVINGS_T_wc)

            mask_angle = all_angles > ANGLE_THRESHOLD

            for idx in range(C_tr.shape[1]):
                c = C_tr[:, idx]
                f = F_tr[:, idx]
                T_CW_fvec = T_tr[:, idx]
                        
                T_cw0 = T_CW_fvec.reshape(3, 4)
                angle = all_angles[idx]
    
                if not angle>ANGLE_THRESHOLD:
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
                new_ids = np.arange(next_landmark_id, next_landmark_id + new_X.shape[1])
                S["ids"] = np.hstack([S["ids"], new_ids])
                next_landmark_id += new_X.shape[1]
                #remove triangulated points from C, F and T
            if C_tr.shape[1] > 0:
                keep_mask = np.ones(C_tr.shape[1], dtype=bool)
                keep_mask[promoted_idx] = False
                S["C"] = C_tr[:, keep_mask]
                S["F"] = F_tr[:, keep_mask]
                S["T"] = T_tr[:, keep_mask]
                S["frame_index"] = frame_index[keep_mask]

    # 4) - CANDIDATES SET UPDATE

    # Crea una maschera bianca (tutti 1) grande quanto l'immagine
    mask_cv = np.ones(img.shape, dtype=np.uint8) * 255

    # Per ogni punto già esistente in S["P"] e S["C"], disegna un cerchio nero (0)
    for pt in S["P"].T:
        cv2.circle(mask_cv, tuple(pt.astype(int)), min_distance + 1, 0, -1)
    for pt in S["C"].T:
        cv2.circle(mask_cv, tuple(pt.astype(int)), min_distance + 1, 0, -1)

    # Chiedi a OpenCV di trovare i punti solo dove la maschera è bianca (255)
    new_corners = cv2.goodFeaturesToTrack(img, max_num_corners, quality_level, min_distance, mask=mask_cv)
    # new_corners = cv2.goodFeaturesToTrack(img, max_num_corners, quality_level, min_distance)
    if new_corners is not None:
        C_new = klt_to_P2xN(new_corners.astype(np.float32))

        if C_new.shape[1] > 0:
            T12 = T_cw.reshape(12, 1)
            T_new = np.repeat(T12, C_new.shape[1], axis=1)
            frame_index_new = np.ones(C_new.shape[1], dtype=int) * (i - bootstrap_frames[1])
            S["C"] = np.hstack([S["C"], C_new])
            S["F"] = np.hstack([S["F"], C_new.copy()])
            S["T"] = np.hstack([S["T"], T_new])
            S["frame_index"] = np.hstack([S["frame_index"], frame_index_new])
            n_new_candidates = C_new.shape[1]

    # 5) BUNDLE ADJUSTMENT
    if use_BA:
        UPDATE_THRESHOLD=i%update_freq==0
        if i >= bootstrap_frames[1] and UPDATE_THRESHOLD and len(buffer)==buffer_dim:
            S = run_ba(buffer, S, K, buffer_dim)               
            if buffer_dim - 1 > 0:
                if len(traj) >= buffer_dim - 1:
                    #cut the last frames
                    traj = traj[:-(buffer_dim - 1)]
                else: #if traj is too short than buffer_dim overwrite it
                    traj = []
            #add optimized poses
            for pose_data in buffer:
                T_optimized = pose_data['pose']
                center = cam_center_from_Tcw(T_optimized)
                traj.append(center)
            T_cw = buffer[-1]['pose']

    prev_img = img


if not visualize_frames:
    end_time = time.time()
    Hz = last_frame/(end_time - start_time)
    print(f"Processed {last_frame} in {end_time - start_time} sec")
    print(f"Frame rate: {Hz} Hz")
    plot_trajectory(traj, HAS_GT, gt_x, gt_z)



