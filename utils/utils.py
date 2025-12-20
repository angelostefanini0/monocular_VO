import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import cv2


# --- Helper Functions ---
#functions that transform keypoints from 2xN shape to Nx1x2 shape for KLT, and viceversa
def P2xN_to_klt(P):return P.T.astype(np.float32).reshape(-1,1,2) 
def klt_to_P2xN(Pklt):return Pklt.reshape(-1,2).T.astype(np.float32)

def compute_all_angles(K_inv, F_tr, C_tr, frame_index, T_cw, SAVINGS_T_wc):
    N = C_tr.shape[1]
    if N == 0: return np.array([])

    # 1. T_wc (Current)
    R_cw = T_cw[:3, :3]
    R_wc = R_cw.T                    
    t_wc = -R_wc @ T_cw[:3, 3]              
    T_wc = np.hstack([R_wc, t_wc.reshape(3, 1)])
    SAVINGS_T_wc.append(T_wc)

    # 2. Back-projection
    F_h = np.vstack([F_tr, np.ones(N)])
    C_h = np.vstack([C_tr, np.ones(N)])
    
    v0_cam = K_inv @ F_h
    v1_cam = K_inv @ C_h

    # Normalizzazione lungo l'asse delle coordinate (asse 0, ovvero le righe x,y,z)
    v0_cam /= np.linalg.norm(v0_cam, axis=0, keepdims=True)
    v1_cam /= np.linalg.norm(v1_cam, axis=0, keepdims=True)

    # 3. Rotazione World
    # Frame corrente (v1_w)
    v1_w = R_wc @ v1_cam  # (3, 3) @ (3, N) -> (3, N)

    # Frame di origine (v0_w)
    # Estraiamo R0 per ogni punto
    all_R0 = np.array([SAVINGS_T_wc[int(idx)][:3, :3] for idx in frame_index]) # (N, 3, 3)
    
    # Prepariamo v0_cam per la moltiplicazione batch: (N, 3, 1)
    v0_cam_batch = v0_cam.T.reshape(N, 3, 1)
    
    # Moltiplicazione batch: (N, 3, 3) @ (N, 3, 1) -> (N, 3, 1)
    v0_w_batch = all_R0 @ v0_cam_batch
    
    # Riportiamo a (3, N)
    v0_w = v0_w_batch.squeeze(-1).T

    # 4. Angoli
    cos_angles = np.sum(v0_w * v1_w, axis=0)
    return np.arccos(np.clip(cos_angles, -1.0, 1.0))

def build_sparsity(n_poses,n_points, camera_indices, point_indices):
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

def project_points(K,R,t,pts):
    P_camera = pts @ R.T + t.reshape(1, 3)
    pts_2d = P_camera @ K.T
    projected = pts_2d[:, :2] / pts_2d[:, 2:3]
    return projected


def residual_function(params, n_frames, n_points, camera_indices, point_indices, observed_points, K):
    #Divide params and points
    camera_params = params[:n_frames * 6].reshape((n_frames, 6))
    points_3d = params[n_frames * 6:].reshape((n_points, 3))
   
    projected = np.zeros_like(observed_points)

    Rs = []
    ts = []
    for i in range(n_frames):
        rvec = camera_params[i, :3]
        t = camera_params[i, 3:]
        R, _ = cv2.Rodrigues(rvec)
        Rs.append(R)
        ts.append(t)
    for i in range(n_frames):
        mask = (camera_indices == i)
        if np.sum(mask) > 0:
            pts_3d_frame = points_3d[point_indices[mask]]
            proj_frame = project_points(K, Rs[i], ts[i], pts_3d_frame)
            projected[mask] = proj_frame


    return (projected - observed_points).ravel()

def run_ba(buffer_frames,S,K,buffer_dim):
    if len(buffer_frames) < 2: return S

    
    start_idx=max(0,len(buffer_frames)-buffer_dim)
    window_frames=buffer_frames[start_idx:]
    n_frames=len(window_frames)

    #Map of Landmark IDs
    observed_ids=set()
    for frame in window_frames:
        observed_ids.update(frame['obs'].keys())
    id_map = {idd: i for i, idd in enumerate(S["ids"])}
    valid_ids = sorted([idd for idd in observed_ids if idd in id_map])
    id_to_local_idx = {idd: i for i, idd in enumerate(valid_ids)}
    n_points = len(valid_ids)
    if n_points == 0:
        return S
    
    #Construct Optimization arrays and initialize them
    camera_indices= []
    point_indices = []
    observations = []

    x0_cameras = np.zeros(n_frames * 6)
    x0_points = np.zeros(n_points * 3)

    for i, frame in enumerate(window_frames):
        x0_cameras[i*6 : (i+1)*6] = HomogMatrix2twist(frame['pose'])

        for id, uv in frame['obs'].items():
            if id in id_to_local_idx:
                camera_indices.append(i)
                point_indices.append(id_to_local_idx[id])
                observations.append(uv)
    for i, id in enumerate(valid_ids):
        global_idx = id_map[id]
        x0_points[i*3 : (i+1)*3] = S["X"][:, global_idx]

    x0 = np.hstack((x0_cameras, x0_points))
    
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    observations = np.array(observations)

    #Build the Sparse Jacobian and do Levenberg-Marquardt
    sparse_matrix=build_sparsity(n_frames,n_points,camera_indices,point_indices)
    res=least_squares(residual_function,x0,jac_sparsity=sparse_matrix,method='trf',x_scale='jac',args=(n_frames,n_points,camera_indices,point_indices,observations,K), max_nfev=10,  ftol=1e-3,
  xtol=1e-3,
  gtol=1e-3,
  verbose=0)
    x_opt=res.x

    opt_poses=x_opt[:n_frames*6].reshape((n_frames,6))
    for i in range (n_frames):
        buffer_frames[start_idx+i]['pose']=Twist2HomogMatrix(opt_poses[i])
    opt_points=x_opt[n_frames*6:].reshape((n_points,3)) 
    
    for i,id in enumerate(valid_ids):
        global_idx=id_map[id]
        S["X"][:,global_idx]=opt_points[i]

    print(f"BA applied on {n_frames} frames. Final Cost: {res.cost:.2f}")
    return S

def HomogMatrix2twist(T): 
    R=T[:3,:3]
    t=T[:3,3]
    rvec,_=cv2.Rodrigues(R)
    return np.hstack((rvec.flatten(),t.flatten()))

def Twist2HomogMatrix(twist): #from exercise 8
    rvec = twist[:3]
    t = twist[3:]
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

