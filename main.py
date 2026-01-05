import numpy as np
from VO import VO



def main():
    ds = 1
    use_ba = False
    visualize_frames = False

    args = {
        # goodFeaturesToTrack
        "max_num_corners_bootstrap": 1000,
        "max_num_corners": 300,
        "quality_level": 0.01,
        "min_distance": 6,

        # findEssentialMat
        "prob_essent_mat": 0.99,
        "thresh_essent_mat": 1.0,

        # PNP RANSAC
        "rep_error": 1.0,
        "iter_count": 2000,
        "confidence": 0.99,

        # Bundle Adjustment
        "buffer_dim": 5,
        "update_freq": 1,
        "n_fix_ba": 1,
        "min_frame_count": 0,
        "max_num_ba_points": 100,
        "z_threshold_ba": [1.0, 100.0],
        "ba_tol": 1e-2,
        "max_nfev": 50
    }

    if ds == 0:
        if use_ba == False:
            pass
    elif ds == 1:
        if use_ba == True:
            args["z_threshold_ba"] = [0.1, 100.0]
            args["n_fix_ba"] = 2

    elif ds == 2:
        pass
    elif ds == 3:
        if use_ba == True:
            args["ba_tol"] = 5e-2
            args["max_nfev"] = 20

    np.random.seed(42)
    vo = VO(ds = ds, use_ba=use_ba, visualize_frames= visualize_frames, args=args)
    vo.run()

if __name__ == "__main__":
    main()

