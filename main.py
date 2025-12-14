import os
import cv2
import numpy as np
from glob import glob

# --- Setup ---
ds = 2  # 0: KITTI, 1: Malaga, 2: Parking, 3: Own Dataset

# Define dataset paths
# (Set these variables before running)
# kitti_path = "/path/to/kitti"
# malaga_path = "/path/to/malaga"
# parking_path = "/path/to/parking"
# own_dataset_path = "/path/to/own_dataset"

if ds == 0:
    assert 'kitti_path' in locals(), "You must define kitti_path"
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
    assert 'malaga_path' in locals(), "You must define malaga_path"
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
    assert 'parking_path' in locals(), "You must define parking_path"
    parking_path = r"./datasets/parking"
    last_frame = 598
    K = np.loadtxt(os.path.join(parking_path, 'K.txt'))
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
#gooìdFeaturesToTrack PARAMETERS - !!!!!! tune them !!!!! (may be necessary to tune them fro each dataset)
max_num_corners=1000
quality_level=0.01
min_distance=2
#findEssentialMat PARAMETERS - !!!!!! tune them !!!!! (may be necessary to tune them fro each dataset)
prob_essent_mat=0.999
thresh_essent_mat=1.0

# --- Helper Functions ---
#functions that transform keypoints from 2xN shape to Nx1x2 shape for KLT
def P2xN_to_klt(P):return P.T.astype(np.float32).reshape(-1,1,2) 
def klt_to_P2xN(Pklt):return Pklt.reshape(-1,2).T.astype(np.float32)

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
pts1=cv2.goodFeaturesToTrack(img0,max_num_corners,quality_level,min_distance)
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
T_WC1=np.eye(4,dtype=np.float64)
T_WC2=np.eye(4,dtype=np.float64)
T_WC2[:3,:3]=R
T_WC2[:3,3:4]=t

# 4) - finally, we can perform triangulation, and thus construnct the first point cloud
#compute the projection matrices
P1=K@np.hstack((np.eye(3),np.zeros((3,1))))
P2=K@np.hstack((R,t))
#triangulatePoints wants 2xN
points1=corr1_final.T
points2=corr2_final.T

X_homogeneous=cv2.triangulatePoints(P1,P2,points1,points2)
X=(X_homogeneous[:3,:]/X_homogeneous[3:4,:]).astype(np.float32)






# --- Continuous operation ---
for i in range(bootstrap_frames[1] + 1, last_frame + 1):
    print(f"\n\nProcessing frame {i}\n=====================")
    
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
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Warning: could not read {image_path}")
        continue
    
    # Simulate 'pause(0.01)'
    cv2.waitKey(10)
    
    prev_img = image