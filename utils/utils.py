import numpy as np


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

