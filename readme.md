

# Real time Monocular Visual Odometry (VO) with Bundle Adjustment


## How to Use
### 1. Prerequisites

The required Python environment and all dependencies are defined in the `environment.yml` file.  
Make sure you have **Conda** (or Miniconda) installed, then create and activate the environment by running:

```bash
conda env create -f environment.yml
conda activate vo
```

### 2. Dataset Setup

The code expects datasets to be located in a `./datasets/` folder.

* **KITTI**: Place in `./datasets/kitti/`
* **Malaga**: Place in `./datasets/malaga-urban-dataset-extract-07/`
* **Parking**: Place in `./datasets/parking/`
* **Our**: Place in `./datasets/our_dataset8/`

### 3. Configuration

Open `main.py` to select your dataset and toggle features:

```python
def main():
    ds = 0             # 0: KITTI, 1: Malaga, 2: Parking, 3: Custom
    use_ba = True      # Set to True to enable Bundle Adjustment
    visualize_frames = False # Set to True for live plotting

```

### 4. Running the Pipeline

Simply execute the main script:

```bash
python main.py

```
## Output

* **Live View**: If `visualize_frames` is True, you will see a real-time update of the camera view, the 3D landmarks, and the estimated trajectory.
* **Final trajectory plot**: If `visualize_frames` is False, the final estimated trajectory is saved as `traj.png`.
* **Performance statistics**: Upon completion, the script prints the total processing time and the average frame rate (Hz).  
  *(Note: enabling live visualization significantly degrades performance.)*
* **Per-frame diagnostic output**: For each processed frame, the script prints diagnostic information to the console, including the number of tracked keypoints, PnP inliers, Bundle Adjustment statistics (when enabled), and the number of newly added 3D points.



---
--- 
## Hyperparameters

You can modify the `args` dictionary in `main.py` to tune the algorithm.

Below is a detailed breakdown of the parameters found in `main.py`:

### Feature Detection & Tracking
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `max_num_corners` | 300| Max new features to add per frame. |
| `max_num_corners_bootstrap` | 1000 | Number of features used for the initial two-frame alignment. |
| `quality_level` | 0.01 | Threshold for Shi-Tomasi corner detection (higher = stricter). |
| `min_distance` | 6 | Minimum pixel distance between detected features. |

### Localization (PnP RANSAC)
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `rep_error` | 1.0 | Maximum allowed reprojection error (pixels) for RANSAC inliers. |
| `iter_count` | 2000 | Maximum number of RANSAC iterations for pose estimation. |
| `confidence` | 0.99 | Probability that the RANSAC estimate is correct. |

### Bundle Adjustment (BA)
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `buffer_dim` | 5 | Number of recent frames included in the sliding window. |
| `update_freq` | 1 | How many frames to skip between BA optimizations. |
| `n_fix_ba` | 1 | Number of "fixed" poses (gauge fixing) to anchor the map. |
| `max_num_ba_points` | 100 | Max number of 3D points to optimize (for performance). |
| `z_threshold_ba` | [1.0, 100.0] | Valid depth range for points to be included in BA. |
| `ba_tol` | 1e-2 | Tolerance for the least_squares optimizer convergence. |
| `max_nfev` | 50 | Maximum number of function evaluations for the optimizer. |
| `min_frame_count` | 0 | Minimum number of frames a point must be seen in before optimization (otherwise excluded). |

### Mapping & Triangulation
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `ANGLE_THRESHOLD` | ~0.1 rad | Minimum bearing angle (parallax) required to triangulate a candidate point. |

---

---

