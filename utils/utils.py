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


def residual_function(params, n_frames, n_points, camera_indices, point_indices, observed_points, K, weights):
    # Recover camera and points parameters
    camera_params = params[:n_frames * 6].reshape((n_frames, 6))
    points_3d = params[n_frames * 6:].reshape((n_points, 3))
    
    rvecs = camera_params[:, :3]
    tvecs = camera_params[:, 3:]

    # Extraxt R
    Rs = np.zeros((n_frames, 3, 3))
    for i in range(n_frames):
        Rs[i], _ = cv2.Rodrigues(rvecs[i])

    R_obs = Rs[camera_indices] 
    t_obs = tvecs[camera_indices] 
    p_3d_obs = points_3d[point_indices] 
    
    #World -> Camera
    P_camera = np.einsum('kij,kj->ki', R_obs, p_3d_obs) + t_obs
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    z_inv = 1.0 / (P_camera[:, 2] + 1e-6) # Numerical stability
    u = fx * (P_camera[:, 0] * z_inv) + cx
    v = fy * (P_camera[:, 1] * z_inv) + cy
    
    projected = np.column_stack([u, v])
    
    # Weighted residuals
    residuals = (projected - observed_points) * weights[:, np.newaxis]
    
    return residuals.ravel()

def run_ba(buffer_frames, S, K, buffer_dim, max_points=500, z_threshold=100.0):
    if len(buffer_frames) < 2: return S
    
    start_idx = max(0, len(buffer_frames) - buffer_dim)
    window_frames = buffer_frames[start_idx:]
    n_frames = len(window_frames)

    # ID mapping
    observed_ids = set().union(*(f['obs'].keys() for f in window_frames))
    
    global_id_map = {idd: i for i, idd in enumerate(S["ids"])}
    valid_ids = sorted([idd for idd in observed_ids if idd in global_id_map])
    
    if not valid_ids:
        return S

    candidate_global_indices = [global_id_map[id] for id in valid_ids]
    points_3d_world = S["X"][:, candidate_global_indices] 
    
    last_pose = window_frames[-1]['pose']
    R_cw_last = last_pose[:3, :3]
    t_cw_last = last_pose[:3, 3]
    
    # Calculate depth
    depth = R_cw_last[2, :] @ points_3d_world + t_cw_last[2]

    close_enough_indices = np.where(depth < z_threshold)[0]
    
    if close_enough_indices.size == 0:
        return S

    if close_enough_indices.size > max_points:
        surviving_z = depth[close_enough_indices]
        top_k_local = np.argsort(surviving_z)[:max_points]
        final_indices = close_enough_indices[top_k_local]
    else:
        final_indices = close_enough_indices

    valid_ids_np = np.array(valid_ids)
    valid_ids = valid_ids_np[final_indices].tolist()

    n_points = len(valid_ids)
    id_to_local_idx = {idd: i for i, idd in enumerate(valid_ids)}
    
    camera_indices = []
    point_indices = []
    observations = []

    x0_cameras = np.zeros((n_frames, 6))

    for i, frame in enumerate(window_frames):
        x0_cameras[i] = HomogMatrix2twist(frame['pose'])
        f_ids = list(frame['obs'].keys())
        f_uvs = list(frame['obs'].values())
        
        for j, id in enumerate(f_ids):
            if id in id_to_local_idx:
                camera_indices.append(i)
                point_indices.append(id_to_local_idx[id])
                observations.append(f_uvs[j])

    if not camera_indices:
        return S

    camera_indices = np.array(camera_indices, dtype=int)
    point_indices = np.array(point_indices, dtype=int)
    observations = np.array(observations)

    global_indices = [global_id_map[id] for id in valid_ids]
    x0_points = S["X"][:, global_indices].T.flatten()
    
    x0 = np.hstack((x0_cameras.flatten(), x0_points))



    Rs_init = np.zeros((n_frames, 3, 3))
    for i in range(n_frames):
        rvec_init = x0_cameras[i, :3]
        Rs_init[i], _ = cv2.Rodrigues(rvec_init)
    
    ts_init = x0_cameras[:, 3:]
    
    pts_3d_init = x0_points.reshape((n_points, 3))

    R_obs_init = Rs_init[camera_indices]
    t_obs_init = ts_init[camera_indices]
    p_obs_init = pts_3d_init[point_indices]

    z_obs_values = np.einsum('ij,ij->i', R_obs_init[:, 2, :], p_obs_init) + t_obs_init[:, 2]
    
    # Clip z for stability
    z_obs_values = np.maximum(z_obs_values, 0.1)
    
    # WEIGHTS
    obs_weights = 1.0 / z_obs_values

    sparse_matrix = build_sparsity(n_frames, n_points, camera_indices, point_indices)
    
    res = least_squares(
        residual_function, x0, 
        jac_sparsity=sparse_matrix, 
        method='trf', 
        x_scale='jac',
        args=(n_frames, n_points, camera_indices, point_indices, observations, K, obs_weights),
        ftol=1e-3, xtol=1e-3, gtol=1e-3, 
        verbose=0, loss='huber', f_scale=1.5, max_nfev=50
    )

    x_opt = res.x
    opt_poses = x_opt[:n_frames*6].reshape((n_frames, 6))
    opt_points = x_opt[n_frames*6:].reshape((n_points, 3))

    for i in range(n_frames):
        buffer_frames[start_idx+i]['pose'] = Twist2HomogMatrix(opt_poses[i])

    S["X"][:, global_indices] = opt_points.T
    print(f"BA: {n_frames} frames, {n_points} pts. Cost: {res.cost:.2f}")
    
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


