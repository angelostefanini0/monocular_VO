import os
import cv2
import numpy as np
from plotting_utils import init_live_plots, update_traj, update_frame_with_points, update_world, plot_trajectory
import matplotlib.pyplot as plt
import time
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares



class VO():
    def __init__(self, ds = 0, use_ba = True, visualize_frames = False, args = {}):
        self.ds = ds
        self.use_ba = use_ba
        self.visualize_frames = visualize_frames

        #KLT PARAMETERS
        self.klt_params=dict(
            winSize=(21,21), 
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01)
        )

        #goodFeaturesToTrack PARAMETERS
        self.max_num_corners_bootstrap = args.get("max_num_corners_bootstrap", 1000)
        self.max_num_corners = args.get("max_num_corners", 1000)
        self.quality_level = args.get("quality_level", 0.01)
        self.min_distance = args.get("min_distance", 2)

        #findEssentialMat PARAMETERS
        self.prob_essent_mat = args.get("prob_essent_mat", 0.99)
        self.thresh_essent_mat = args.get("thresh_essent_mat", 1.0)

        #PNP RANSAC PARAMETERS
        self.rep_error = args.get("rep_error", 3.0)
        self.iter_count = args.get("iter_count", 2000)
        self.confidence = args.get("confidence", 0.99)

        #Bundle Adjustment PARAMETERS
        if self.use_ba:
            self.buffer_dim = args.get("buffer_dim", 8)
            self.update_freq = args.get("update_freq", 7)
            self.n_fix_ba = args.get("n_fix_ba", 1)
            self.min_frame_count = args.get("min_frame_count", 3)
            self.max_num_ba_points = args.get("max_num_ba_points", 5000)
            self.z_threshold_ba = args.get("z_threshold_ba", [1.0, 100.0])
            self.ba_tol = args.get("ba_tol", 1e-3)
            self.max_nfev = args.get("max_nfev", 1e-3)
            self.buffer = []

        # --- State ---
        self.S = {
            # Localization
            "P": np.zeros((2, 0), dtype=float),  # 2xN - image coordinates of tracked features
            "X": np.zeros((3, 0), dtype=float),  # 3xN - world coordinates of tracked features
            "ids": np.zeros((0),dtype=int),  # 1xN to track the id of every X point
            "count": np.zeros((0), dtype= int),

            # Triangulation candidates
            "C": np.zeros((2, 0), dtype=float),  # 2xM - position of candidate features in the current frame (image coordinates)
            "F": np.zeros((2, 0), dtype=float),  # 2xM - position of candidate features in the first frame they were observed (image coordinates)
            # "T": np.zeros((12, 0), dtype=float),  # 12xM - pose of the frame at which candidate features were observed firstly observed
            "frame_id": np.zeros((0), dtype=int),   #1xM To track id of the frame (instead of T)
            
        }

        self.next_landmark_id = 0

    def klt_to_P2xN(self, Pklt):
        return Pklt.reshape(-1,2).T.astype(np.float32)
    def P2xN_to_klt(self, P):
        return P.T.astype(np.float32).reshape(-1,1,2) 
    
    def cam_center_from_Tcw(self, T_cw):
        R=T_cw[:,:3]
        t=T_cw[:,3:4]
        C=-R.T@t
        return C.reshape(3)

    def HomogMatrix2twist(self, T): 
        R=T[:3,:3]
        t=T[:3,3]
        rvec,_=cv2.Rodrigues(R)
        return np.hstack((rvec.flatten(),t.flatten()))
    
    def Twist2HomogMatrix(self, twist): #from exercise 8
        rvec = twist[:3]
        t = twist[3:]
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    def run(self):
        start_time = time.time()
        self.initialize()
        self.bootstrap()
                # --- Continuous operation ---
        self.gt=None
        if self.HAS_GT:
            self.gt=(self.gt_x, self.gt_z)
        if self.visualize_frames:
            self.plots=init_live_plots(gt=self.gt)
        self.traj = np.zeros((self.last_frame + 1, 3))
        for i in range(self.bootstrap_frames[1] + 1, self.last_frame + 1):
            self.continuos_operation(i)

        if not self.visualize_frames:
            end_time = time.time()
            Hz = self.last_frame/(end_time - start_time)
            print(f"Processed {self.last_frame} frames in {end_time - start_time} sec")
            print(f"Frame rate: {Hz} Hz")
            plot_trajectory(self.traj, self.HAS_GT, self.gt_x, self.gt_z)

        


    def initialize(self):
        if self.ds == 0:
            self.ANGLE_THRESHOLD = np.deg2rad(0.1)
            self.bootstrap_frames = [0, 2] #frames betweem which bootstrap is performed
            kitti_path = r"./datasets/kitti"
            self.dataset_path = kitti_path
            ground_truth = np.loadtxt(os.path.join(kitti_path, 'poses', '05.txt'))
            ground_truth = ground_truth[:, [-9, -1]]  # same as MATLAB(:, [end-8 end])
            self.last_frame = 2670
            #self.last_frame = 1000 #TEST
            self.K = np.array([
                [7.18856e+02, 0, 6.071928e+02],
                [0, 7.18856e+02, 1.852157e+02],
                [0, 0, 1]
            ])
            self.HAS_GT=True
            poses=np.loadtxt(os.path.join(kitti_path,'poses','05.txt')) #N x 12
            self.gt_x=poses[:,3]
            self.gt_z=poses[:,11]
            if self.use_ba:
                self.ba_tol = 1e-3
        elif self.ds == 1:
            self.ANGLE_THRESHOLD = np.deg2rad(0.02)
            self.bootstrap_frames = [0, 2] #frames betweem which bootstrap is performed
            malaga_path = r"./datasets/malaga-urban-dataset-extract-07"
            self.dataset_path = malaga_path
            img_dir = os.path.join(
                malaga_path,
                "malaga-urban-dataset-extract-07_rectified_800x600_Images"
            )
            self.left_images = sorted([
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if f.endswith("_left.jpg")
            ])
            self.last_frame = len(self.left_images)-1
            self.K = np.array([
                [621.18428, 0, 404.0076],
                [0, 621.18428, 309.05989],
                [0, 0, 1]
            ])
            self.HAS_GT=False
            self.gt_x=self.gt_z=None
        elif self.ds == 2:
            #threshold for bearing angle function
            self.ANGLE_THRESHOLD = np.deg2rad(5.72)
            self.bootstrap_frames = [0, 2] #frames betweem which bootstrap is performed
            parking_path = r"./datasets/parking"
            self.dataset_path = parking_path
            self.last_frame = 598
            self.K = np.loadtxt(os.path.join(parking_path, 'K.txt'), delimiter=',', usecols=(0, 1, 2))
            ground_truth = np.loadtxt(os.path.join(parking_path, 'poses.txt'))
            ground_truth = ground_truth[:, [-9, -1]]
            self.HAS_GT=True
            poses=np.loadtxt(os.path.join(parking_path,'poses.txt'))  #N x 12
            self.gt_x=poses[:,3]
            self.gt_z=poses[:,11]
            if self.use_ba:
                self.ba_tol = 1e-3
        elif self.ds == 3:
            self.ANGLE_THRESHOLD = np.deg2rad(0.1)
            self.bootstrap_frames = [0, 15] #frames betweem which bootstrap is performed
            self.HAS_GT=False
            self.gt_x=self.gt_z=None
            self.last_frame = 602
            self.dataset_path = r"./datasets/our_dataset8"
            self.K = np.array([
                [1109.7, 0, 637.5062],
                [0, 1113.5, 357.1623],
                [0, 0, 1]
            ])
        else:
            raise ValueError("Invalid dataset index")
        
        self.K_inv = np.linalg.inv(self.K)
        self.T_cw_array = np.zeros((self.last_frame + 1, 3, 4), dtype=np.float64)
        self.T_wc_array = np.zeros((self.last_frame + 1, 3, 4), dtype=np.float64)
        self.pose_valid = np.zeros(self.last_frame + 1, dtype=bool) #pose_valid[i] = True when the pose has been estimated

    def bootstrap(self):
        # --- Bootstrap ---
        if self.ds == 0:
            
            img0 = cv2.imread(os.path.join(self.dataset_path, '05', 'image_0', f"{self.bootstrap_frames[0]:06d}.png"), cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(os.path.join(self.dataset_path, '05', 'image_0', f"{self.bootstrap_frames[1]:06d}.png"), cv2.IMREAD_GRAYSCALE)
        elif self.ds == 1:
            img_dir = os.path.join(
                self.dataset_path,
                "malaga-urban-dataset-extract-07_rectified_800x600_Images"
            )
            left_images = sorted([
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if f.endswith("_left.jpg")
            ])
            img0 = cv2.imread(left_images[self.bootstrap_frames[0]], cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(left_images[self.bootstrap_frames[1]], cv2.IMREAD_GRAYSCALE)
        elif self.ds == 2:
            img0 = cv2.imread(os.path.join(self.dataset_path, 'images', f"img_{self.bootstrap_frames[0]:05d}.png"), cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(os.path.join(self.dataset_path, 'images', f"img_{self.bootstrap_frames[1]:05d}.png"), cv2.IMREAD_GRAYSCALE)
        elif self.ds == 3:
            # Load images from own dataset
            img0 = cv2.imread(os.path.join(self.dataset_path, 'Images', f"img_{self.bootstrap_frames[0]:05d}.png"), cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(os.path.join(self.dataset_path,'Images',  f"img_{self.bootstrap_frames[1]:05d}.png"), cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError("Invalid dataset index")


        # 1) - harris to detect keypoints in first keyframe (img0)
        pts1=cv2.goodFeaturesToTrack(img0,self.max_num_corners_bootstrap,self.quality_level,self.min_distance) #Nx1x2
        n=0 if pts1 is None else pts1.shape[0]
        print("Number of detected features in keyframe 1: ", n)
        pts1=pts1.astype(np.float32)

        # 2) - KLT to track the keypoints to the second keyframe (img1)
        pts2,status,err=cv2.calcOpticalFlowPyrLK(img0,img1,pts1,None,**self.klt_params)
        #Filter valid tracks
        status=status.reshape(-1)
        pts1_tracked=pts1[status==1]
        pts2_tracked=pts2[status==1]
        #reshape to 2xN
        keypoints1=self.klt_to_P2xN(pts1_tracked)
        keypoints2=self.klt_to_P2xN(pts2_tracked)

        # 3) - now we have the 2D-2D point correspondences
        # origin of world frame is assumed to coincide with the pose of the first keyframe
        
        corr1 = keypoints1.T.astype(np.float32)
        corr2 = keypoints2.T.astype(np.float32)

        _, mask_initial = cv2.findFundamentalMat(
            corr1, 
            corr2, 
            method=cv2.FM_RANSAC, 
            ransacReprojThreshold=self.thresh_essent_mat, 
            confidence=self.prob_essent_mat
        )
        
        if mask_initial is None or np.sum(mask_initial) < 8:
            print("RANSAC found too few inliers for 8-point algorithm.")
            return None # O gestisci come preferisci

        mask_initial = mask_initial.ravel().astype(bool)
        pts1_in = corr1[mask_initial]
        pts2_in = corr2[mask_initial]

        F_final, _ = cv2.findFundamentalMat(pts1_in, pts2_in, method=cv2.FM_8POINT)

        E_approx = self.K.T @ F_final @ self.K

        U, S, Vt = np.linalg.svd(E_approx)
        S_new = np.diag([1, 1, 0]) 
        E = U @ S_new @ Vt

        _, R, t, maskPose = cv2.recoverPose(E, pts1_in, pts2_in, self.K)
        
        # filter again good correspondences (Cheirality check elimina punti dietro la camera)
        maskPose = maskPose.reshape(-1).astype(bool)
        corr1_final = pts1_in[maskPose]
        corr2_final = pts2_in[maskPose]

        t = t.reshape(3, 1)
        T_CW2 = np.hstack((R, t)).astype(np.float64) # 3x4

        # 4) - finally, we can perform triangulation, and thus construnct the first point cloud
        #compute the projection matrices
        P1=self.K@np.hstack((np.eye(3),np.zeros((3,1))))
        P2=self.K@np.hstack((R,t))
        #triangulatePoints wants 2xN
        points1=corr1_final.T
        points2=corr2_final.T
        X_homogeneous=cv2.triangulatePoints(P1,P2,points1,points2)
        X=(X_homogeneous[:3,:]/X_homogeneous[3:4,:]).astype(np.float32)

        # 5) - set up of the state
        self.S["P"]=points2
        self.S["X"]=X
        self.S["ids"] = np.arange(self.next_landmark_id, self.next_landmark_id + X.shape[1])
        self.S["count"] = np.ones(X.shape[1])

        self.next_landmark_id += X.shape[1]
        #to create the candidates set C, we must detect new features, and check that they are not already in P
        cand=cv2.goodFeaturesToTrack(img1,self.max_num_corners_bootstrap,self.quality_level,self.min_distance)
        cand=self.klt_to_P2xN(cand)
        #to ensures points in C are not redundant with ones in P, we perform a minimum distance check
        diff=cand[:,:,None]-points2[:,None,:]
        dist_sq=np.sum(diff**2,axis=0) #distance of each candidate to all points in P
        min_dist_sq=np.min(dist_sq,axis=1) #distance of each candidate to the closest point in P
        #Keep only candidates farther than min_distance
        mask=min_dist_sq>(self.min_distance**2)
        C=cand[:,mask]
        # T=np.repeat(T_CW2.reshape(12,1),C.shape[1],axis=1)
        self.S["C"]=C
        self.S["F"]=C.copy()
        # self.S["T"]=T
        frame_i = self.bootstrap_frames[1]
        self.S["frame_id"] = np.full(C.shape[1], frame_i, dtype=int)

        assert self.S["frame_id"].shape[0] == self.S["C"].shape[1], f"point 1, {self.S['frame_id'].shape[0]}"


        R_cw = T_CW2[:3, :3]
        t_cw = T_CW2[:3, 3]
        R_wc = R_cw.T                    
        t_wc = -R_wc @ t_cw              
        T_wc = np.hstack([R_wc, t_wc.reshape(3, 1)])

        f1 = self.bootstrap_frames[1]
        self.T_cw_array[f1] = T_CW2
        self.T_wc_array[f1] = T_wc
        self.pose_valid[f1] = True

        f0 = self.bootstrap_frames[0]
        self.T_cw_array[f0] = np.hstack([np.eye(3), np.zeros((3,1))])
        self.T_wc_array[f0] = np.hstack([np.eye(3), np.zeros((3,1))])
        self.pose_valid[f0] = True

        self.prev_img = img1


    def continuos_operation(self, i):
        print(f"\n\nProcessing frame {i}\n=====================")
        assert (self.S["C"].shape[1] == self.S["frame_id"].shape[0])
        assert (self.S["X"].shape[1] == self.S["count"].shape[0]), f'{self.S["C"].shape[1]} e {self.S["count"].shape[0]}'
        # LOAD IMAGE
        if self.ds == 0:
            image_path = os.path.join(self.dataset_path, '05', 'image_0', f"{i:06d}.png")
        elif self.ds == 1:
            image_path = self.left_images[i]
        elif self.ds == 2:
            image_path = os.path.join(self.dataset_path, 'images', f"img_{i:05d}.png")
        elif self.ds == 3:
            image_path = os.path.join(self.dataset_path, 'Images', f"img_{i:05d}.png")
        else:
            raise ValueError("Invalid dataset index")

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: could not read {image_path}")
            return
        
        # 1) - track keypoints from previous frame, that are already associated to a landmark
        P_prev =  self.P2xN_to_klt(self.S["P"]) # Nx1x2
        X_prev = self.S["X"].T.astype(np.float32) # Nx3
        # track with klt
        P_tr, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_img, img, P_prev, None, **self.klt_params)
        st = st.reshape(-1).astype(bool)
        # filter out keypoints for which tracking fails, and also corresponding landmarks
        P_tr = P_tr.reshape(-1, 2)[st] # Nx2
        X_tr = X_prev[st] # Nx3
        ids_tr = self.S["ids"][st] #tracks indexes for BA

        print(f"Tracked keypoints: {P_tr.shape[0]}")

        # 2) - LOCALIZATION: exploiting the now established 3D-2D correspondences between landmarks and
        #keypoints in the current frame, with PnP + RANSAC we retrieve the pose of the current frame wrt the world
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=X_tr,
            imagePoints=P_tr,
            cameraMatrix=self.K,
            distCoeffs=None,
            reprojectionError=self.rep_error, #defines how far from the model points start to be considered outliers
            iterationsCount=self.iter_count, #max number of iteration of RANSAC
            confidence=self.confidence,
            flags=cv2.SOLVEPNP_P3P
        )

        if (not ok) or (inliers is None) or (len(inliers) < 4):
            print(f"PnP failed / inliers too few: {0 if inliers is None else len(inliers)}")
            self.prev_img = img
            return

        inliers = inliers.reshape(-1)
        #filter out outliers
        print(f"PnP inliers: {len(inliers)} ")
        P_in = P_tr[inliers] # Nx2
        X_in = X_tr[inliers] # Nx3
        ids_in=ids_tr[inliers]
        P_prev_valid=P_prev.reshape(-1,2)[st] #prev positions Nx2
        P_prev_in=P_prev_valid[inliers] #prev positions of inlier tracks Nx2

        # T_cw
        R_cw, _ = cv2.Rodrigues(rvec)
        T_cw = np.hstack([R_cw,tvec])

        # T_wc
        R_cw = T_cw[:3, :3]
        R_wc = R_cw.T                    
        t_wc = -R_wc @ T_cw[:3, 3]              
        T_wc = np.hstack([R_wc, t_wc.reshape(3, 1)])

        self.T_cw_array[i] = T_cw
        self.T_wc_array[i] = T_wc
        self.pose_valid[i] = True

        # update 2D keypoints and 3D landmarks of the state, to the current frame
        self.S["P"] = P_in.T
        self.S["X"] = X_in.T
        self.S["ids"] = ids_in
        self.S["count"] = self.S["count"][inliers]
        self.S["count"] += 1
        P_prev_for_plot=P_prev_in.T #2xN_inliers

        current_obs = {}
        for j, lid in enumerate(ids_in):
            current_obs[lid] = P_in[j] 
        
        if self.use_ba:
            self.buffer.append({
                'pose': T_cw.copy(), 
                'obs': current_obs,
                'frame_id': i })
            if len(self.buffer)>self.buffer_dim:
                self.buffer.pop(0)



        self.traj[i] = self.cam_center_from_Tcw(T_cw)
        
        # 3) BUNDLE ADJUSTMENT
        if self.use_ba:
            UPDATE_THRESHOLD= i % self.update_freq == 0
            if i >= self.bootstrap_frames[1] and UPDATE_THRESHOLD:
                self.run_ba()               
                for pose_data in self.buffer:
                    frame_id = pose_data["frame_id"]
                    T_optimized = pose_data['pose']
                    center = self.cam_center_from_Tcw(T_optimized)
                    self.traj[frame_id] = center
                T_cw = self.buffer[-1]['pose']

   

        if self.visualize_frames:
            update_traj(self.plots,self.traj[:i+1])
            update_world(self.plots,T_cw,self.S["X"])
            update_frame_with_points(self.plots,img,self.S["P"],P_prev_for_plot,frame_idx=i)

            self.plots["fig"].canvas.draw()
            self.plots["fig"].canvas.flush_events()
            plt.pause(0.0001)



        # 4) - 3D MAP COUNTINUOUS UPDATE: in this section we analyze each element of C, which is the set of candidates
        #keypoints. If they satisfy approrpiate conditions, they are triangulated and moved from C to P, and added to X
        if self.S["C"].shape[1] > 0:
            C_prev =  self.P2xN_to_klt(self.S["C"]) # Mx1x2
            #track the candidates of the previous frame to the current one
            C_tr, stc, _ = cv2.calcOpticalFlowPyrLK(self.prev_img, img, C_prev, None, **self.klt_params)
            C_tr = self.klt_to_P2xN(C_tr) # 2xM

            if stc is not None:
                stc = stc.reshape(-1).astype(bool)
                C_tr = C_tr[:,stc]

                F_tr = self.S["F"][:, stc]
                # T_tr = self.S["T"][:, stc]
                frame_id_tr = self.S["frame_id"][stc]

                new_P = []
                new_X = []
                promoted_idx=[]
                #now we loop over all elements of C and, if it's appropriate, triangulate them
                #stiamo consideranod T_CW_f

                all_angles = self.compute_all_angles(F_tr, C_tr, frame_id_tr, T_cw)

                for idx in range(C_tr.shape[1]):
                    c = C_tr[:, idx]
                    f = F_tr[:, idx]
                    # T_CW_fvec = T_tr[:, idx]                    
                            
                    T_cw0 = self.T_cw_array[frame_id_tr[idx]]
                    angle = all_angles[idx]
        
                    if not angle>self.ANGLE_THRESHOLD:
                        continue         
                    #compute projection matrices
                    P0 = self.K @ T_cw0
                    P1 = self.K @ T_cw
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
                    self.S["P"] = np.hstack([self.S["P"], new_P])
                    self.S["X"] = np.hstack([self.S["X"], new_X])
                    new_ids = np.arange(self.next_landmark_id, self.next_landmark_id + new_X.shape[1])
                    self.S["ids"] = np.hstack([self.S["ids"], new_ids])
                    self.S["count"] = np.hstack([self.S["count"], np.ones(new_X.shape[1])])
                    self.next_landmark_id += new_X.shape[1]
                    print(f"Added {new_P.shape[1]} new points")

                if C_tr.shape[1] > 0:
                    #remove triangulated points from C, F and T
                    keep_mask = np.ones(C_tr.shape[1], dtype=bool)
                    keep_mask[promoted_idx] = False
                    self.S["C"] = C_tr[:, keep_mask]
                    self.S["F"] = F_tr[:, keep_mask]
                    # S["T"] = T_tr[:, keep_mask]
                    self.S["frame_id"] = frame_id_tr[keep_mask]
                

        # 5) - CANDIDATES SET UPDATE

        # we create a mask so that Shi-Tomasi looks for corners distant enough from features already found
        mask_cv = np.ones(img.shape, dtype=np.uint8) * 255
        # For every point in S["P"] and S["C"] we draw a black circle with radius min_distance + 1
        for pt in self.S["P"].T:
            cv2.circle(mask_cv, tuple(pt.astype(int)), self.min_distance + 1, 0, -1)
        for pt in self.S["C"].T:
            cv2.circle(mask_cv, tuple(pt.astype(int)), self.min_distance + 1, 0, -1)

        new_corners = cv2.goodFeaturesToTrack(img, self.max_num_corners, self.quality_level, self.min_distance, mask=mask_cv)
        if new_corners is not None:
            C_new = self.klt_to_P2xN(new_corners.astype(np.float32))

            if C_new.shape[1] > 0:
                frame_index_new = np.full(C_new.shape[1], i, dtype=int)
                self.S["C"] = np.hstack([self.S["C"], C_new])
                self.S["F"] = np.hstack([self.S["F"], C_new.copy()])
                self.S["frame_id"] = np.hstack([self.S["frame_id"], frame_index_new])


        self.prev_img = img



    def compute_all_angles(self, F_tr, C_tr, frame_index, T_cw):
        assert np.all(self.pose_valid[frame_index]), "Using uninitialized poses"
        N = C_tr.shape[1]
        if N == 0: return np.array([])
        R_cw = T_cw[:3, :3]
        R_wc = R_cw.T  

        # Homogeneous coordinates
        F_h = np.vstack([F_tr, np.ones(N)])
        C_h = np.vstack([C_tr, np.ones(N)])
        
        v0_cam = self.K_inv @ F_h
        v1_cam = self.K_inv @ C_h

        # Normalization
        v0_cam /= np.linalg.norm(v0_cam, axis=0, keepdims=True)
        v1_cam /= np.linalg.norm(v1_cam, axis=0, keepdims=True)

        v1_w = R_wc @ v1_cam 
        all_R0 = self.T_wc_array[frame_index][:, :3, :3]
        v0_cam_batch = v0_cam.T.reshape(N, 3, 1)
        v0_w_batch = all_R0 @ v0_cam_batch
        v0_w = v0_w_batch.squeeze(-1).T

        # Angles computation
        cos_angles = np.sum(v0_w * v1_w, axis=0)
        return np.arccos(np.clip(cos_angles, -1.0, 1.0))
    
    
    def run_ba(self):
        """
        Sliding-window BA with gauge fixing:
        - first self.n_fix_ba poses in the window are FIXED (not optimized)
        - remaining poses + 3D points are optimized
        Updates:
        - self.buffer[*]['pose'] (3x4) for optimized frames (fixed frames unchanged)
        - self.T_cw_array / self.T_wc_array for optimized frames
        - self.S["X"] for optimized points
        """
        max_points = self.max_num_ba_points
        z_threshold = self.z_threshold_ba

        n_fix = int(getattr(self, "n_fix_ba", 2))  # e.g. 2
        if len(self.buffer) < max(n_fix + 1, 2):
            return

        start_idx = max(0, len(self.buffer) - self.buffer_dim)
        window_frames = self.buffer[start_idx:]
        n_frames = len(window_frames)

        if n_frames <= n_fix:
            return

        # ---- Collect valid landmark ids observed in the window ----
        observed_ids = set().union(*(f['obs'].keys() for f in window_frames))
        global_id_map = {idd: i for i, idd in enumerate(self.S["ids"])}
        valid_ids = sorted([idd for idd in observed_ids if idd in global_id_map])

        if not valid_ids:
            print("No valid ID")
            return

        # ---- Filter points by depth in last pose (must be in front and not too far) ----
        candidate_global_indices = [global_id_map[id] for id in valid_ids]
        points_3d_world = self.S["X"][:, candidate_global_indices]  # 3 x N

        last_pose = window_frames[-1]['pose']  
        R_cw_last = last_pose[:3, :3]
        t_cw_last = last_pose[:3, 3]
        depth = R_cw_last[2, :] @ points_3d_world + t_cw_last[2]


        preliminary_indices = np.where((depth > z_threshold[0]) & (depth < z_threshold[1]) & (self.S["count"] >= self.min_frame_count))[0]
        
        if preliminary_indices.size == 0:
            print("NO close enough indices")
            return

        if preliminary_indices.size > max_points:
            random_selection = np.random.choice(preliminary_indices.size, size=max_points, replace=False)
            final_indices = preliminary_indices[random_selection]
        else:
            final_indices = preliminary_indices

        valid_ids = np.array(valid_ids)[final_indices].tolist()

        n_points = len(valid_ids)
        id_to_local_idx = {idd: i for i, idd in enumerate(valid_ids)}

        # ---- Build observation lists ----
        camera_indices_global = []
        point_indices = []
        observations = []

        x0_cameras = np.zeros((n_frames, 6))
        for i, frame in enumerate(window_frames):
            x0_cameras[i] = self.HomogMatrix2twist(frame['pose'])
            for lm_id, uv in frame['obs'].items():
                if lm_id in id_to_local_idx:
                    camera_indices_global.append(i)  # 0..n_frames-1 (includes fixed)
                    point_indices.append(id_to_local_idx[lm_id])
                    observations.append(uv)

        if len(camera_indices_global) == 0:
            print("NO camera indices")
            return

        camera_indices_global = np.asarray(camera_indices_global, dtype=int)

        point_indices = np.asarray(point_indices, dtype=int)
        observations = np.asarray(observations, dtype=float)

        # ---- Pack parameters: (free camera params) + (3D points) ----
        global_indices = [global_id_map[id] for id in valid_ids]
        x0_points = self.S["X"][:, global_indices].T.flatten()  # (n_points*3,)

        x0_cameras_fixed = x0_cameras[:n_fix].copy()            # (n_fix, 6)
        x0_cameras_free = x0_cameras[n_fix:].copy()             # (n_frames-n_fix, 6)

        x0 = np.hstack((x0_cameras_free.flatten(), x0_points))

        # ---- Precompute weights using initial poses (fixed+free) ----
        # Build Rs, ts for ALL cameras from x0 (fixed + initial free)
        Rs_init = np.zeros((n_frames, 3, 3))
        ts_init = np.zeros((n_frames, 3))

        for i in range(n_fix):
            rvec = x0_cameras_fixed[i, :3]
            Rs_init[i], _ = cv2.Rodrigues(rvec)
            ts_init[i] = x0_cameras_fixed[i, 3:]

        for i in range(n_fix, n_frames):
            rvec = x0_cameras_free[i - n_fix, :3]
            Rs_init[i], _ = cv2.Rodrigues(rvec)
            ts_init[i] = x0_cameras_free[i - n_fix, 3:]

        pts_3d_init = x0_points.reshape((n_points, 3))
        R_obs_init = Rs_init[camera_indices_global]
        t_obs_init = ts_init[camera_indices_global]
        p_obs_init = pts_3d_init[point_indices]

        z_obs_values = np.einsum('ij,ij->i', R_obs_init[:, 2, :], p_obs_init) + t_obs_init[:, 2]
        z_obs_values = np.maximum(z_obs_values, 0.5)  # avoid huge weights
        obs_weights = 1.0 / z_obs_values 
        # an alternative would be: obs_weights = np.ones(len(observations))



        # ---- Sparsity (only FREE cameras are variables) ----
        n_poses_free = n_frames - n_fix

        # sparsity is built only for residual rows that depend on free cameras
        # residuals for fixed-camera observations will only depend on points
        # we build a custom sparsity below for all residuals

        m = camera_indices_global.size * 2
        n = n_poses_free * 6 + n_points * 3
        sparse_matrix = lil_matrix((m, n), dtype=int)

        obs_i = np.arange(camera_indices_global.size)

        # Camera part: only if observation uses a free camera
        free_obs = (camera_indices_global >= n_fix)
        obs_i_free = obs_i[free_obs]
        cam_i_free = (camera_indices_global[free_obs] - n_fix)

        for k in range(6):
            sparse_matrix[2 * obs_i_free, cam_i_free * 6 + k] = 1
            sparse_matrix[2 * obs_i_free + 1, cam_i_free * 6 + k] = 1

        # Point part: always
        for k in range(3):
            col = n_poses_free * 6 + point_indices * 3 + k
            sparse_matrix[2 * obs_i, col] = 1
            sparse_matrix[2 * obs_i + 1, col] = 1

        # ---- Residual function that uses fixed + free cameras ----
        def residual_function_fixed_gauge(params):
            # unpack
            cam_free = params[:n_poses_free * 6].reshape((n_poses_free, 6))
            pts = params[n_poses_free * 6:].reshape((n_points, 3))

            # assemble all camera params
            cam_all = np.zeros((n_frames, 6))
            cam_all[:n_fix] = x0_cameras_fixed
            cam_all[n_fix:] = cam_free

            rvecs = cam_all[:, :3]
            tvecs = cam_all[:, 3:]

            # Rodrigues for all
            Rs = np.zeros((n_frames, 3, 3))
            for i in range(n_frames):
                Rs[i], _ = cv2.Rodrigues(rvecs[i])

            R_obs = Rs[camera_indices_global]
            t_obs = tvecs[camera_indices_global]
            p_obs = pts[point_indices]

            P_cam = np.einsum('kij,kj->ki', R_obs, p_obs) + t_obs

            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]

            z = P_cam[:, 2]
            z = np.where(z > 1e-6, z, 1e-6) 

            u = fx * (P_cam[:, 0] / z) + cx
            v = fy * (P_cam[:, 1] / z) + cy

            proj = np.column_stack([u, v])
            res = (proj - observations) * obs_weights[:, None]
            return res.ravel()

        f0 = residual_function_fixed_gauge(x0)
        cost0 = 0.5 * np.dot(f0, f0)
        print(f"Running BA. Initial cost: {cost0:.5f}")


        # ---- Solve ----
        res = least_squares(
            residual_function_fixed_gauge, x0,
            jac_sparsity=sparse_matrix,
            method='trf', x_scale='jac',
            ftol=self.ba_tol, xtol=self.ba_tol, gtol=self.ba_tol,
            verbose=0, loss='huber', f_scale=1.5, max_nfev=self.max_nfev
        )
        x_opt = res.x

        opt_cam_free = x_opt[:n_poses_free * 6].reshape((n_poses_free, 6))
        opt_points = x_opt[n_poses_free * 6:].reshape((n_points, 3))

        # ---- Write back optimized FREE poses to buffer + global arrays ----
        for k_free in range(n_poses_free):
            k_global = n_fix + k_free
            entry = window_frames[k_global]
            frame_idx = entry['frame_id']  # must exist in buffer entries

            T_cw_opt = self.Twist2HomogMatrix(opt_cam_free[k_free])[:3, :]
            entry['pose'] = T_cw_opt
            self.T_cw_array[frame_idx] = T_cw_opt

            R = T_cw_opt[:, :3]
            t = T_cw_opt[:, 3]
            T_wc_opt = np.hstack([R.T, (-R.T @ t).reshape(3, 1)])
            self.T_wc_array[frame_idx] = T_wc_opt
            self.pose_valid[frame_idx] = True

        # Fixed poses remain unchanged; ensure arrays are consistent for them too
        for k in range(min(n_fix, n_frames)):
            entry = window_frames[k]
            frame_idx = entry['frame_id']
            T_cw_fix = entry['pose'][:3, :]
            self.T_cw_array[frame_idx] = T_cw_fix
            R = T_cw_fix[:, :3]
            t = T_cw_fix[:, 3]
            self.T_wc_array[frame_idx] = np.hstack([R.T, (-R.T @ t).reshape(3, 1)])
            self.pose_valid[frame_idx] = True

        # ---- Update 3D points in state ----
        self.S["X"][:, global_indices] = opt_points.T

        print(f"BA(fixed {n_fix} points): {n_frames} frames, {n_points} pts. nfev: {res.nfev} Final cost: {res.cost:.5f}")
        return


    def build_sparsity(self, n_poses,n_points, camera_indices, point_indices):
        m=camera_indices.size*2
        n=n_points*3+n_poses*6

        sparse_matrix=lil_matrix((m,n),dtype=int)

        i=np.arange(camera_indices.size)
        for k in range(6):
            sparse_matrix[2*i,camera_indices*6+k]=1
            sparse_matrix[2*i+1,camera_indices*6+k]=1
        for k in range (3):
            sparse_matrix[2*i,n_poses*6+point_indices*3+k]=1
            sparse_matrix[2*i+1,n_poses*6+point_indices*3+k]=1
        return sparse_matrix
    
 