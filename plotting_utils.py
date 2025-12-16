import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

def cam_center_from_Tcw(T_cw):
    R=T_cw[:,:3]
    t=T_cw[:,3:4]
    C=-R.T@t
    return C.reshape(3)

def Rwc_from_Tcw(T_cw):
    Rcw=T_cw[:,:3]
    return Rcw.T

def init_live_plots(gt=None):
    plt.ion()

    fig=plt.figure(figsize=(14,8))
    gs=fig.add_gridspec(2,2,height_ratios=[1,1.2])

    # ---------- TRAJECTORY ----------
    ax_traj=fig.add_subplot(gs[0,0])
    ax_traj.set_title("Ground Truth and Estimated Trajectory")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("z")
    ax_traj.axis("equal")
    ax_traj.grid(True,which="both",linestyle="--",linewidth=0.5,alpha=0.6)

    if gt is not None:
        gt_x,gt_z=gt
        ax_traj.plot(
            gt_x,
            gt_z,
            linestyle='--',
            color='black',
            linewidth=1.0,
            alpha=0.7,
            label='Ground Truth'
        )


    traj_line,=ax_traj.plot(
        [],
        [],
        color='red',
        linewidth=2.0,
        label='Estimated Trajectory'
    )
    ax_traj.legend(loc="best")


    # ---------- WORLD ----------
    ax_world=fig.add_subplot(gs[0,1])
    ax_world.set_title("Camera Frame and Triangulated Landmarks")
    ax_world.set_xlabel("x")
    ax_world.set_ylabel("z")
    ax_world.axis("equal")
    ax_world.grid(True,which="both",linestyle="--",linewidth=0.4,alpha=0.4)

    lm_scatter=ax_world.scatter([],[],s=2,alpha=0.7)
    cam_point,=ax_world.plot([],[],'o',color='blue')

    cam_xaxis=FancyArrowPatch(
        (0,0),(0,0),
        arrowstyle='-|>',
        linewidth=1.5,
        color='black',
        mutation_scale=10
    )
    ax_world.add_patch(cam_xaxis)

    cam_zaxis=FancyArrowPatch(
        (0,0),(0,0),
        arrowstyle='-|>',
        linewidth=1.5,
        color='red',
        mutation_scale=10
    )
    ax_world.add_patch(cam_zaxis)
    legend_handles=[
    Line2D([0],[0],color='red',linewidth=2,label='Camera z-axis'),
    Line2D([0],[0],color='black',linewidth=2,label='Camera x-axis')
    ]
    ax_world.legend(
        handles=legend_handles,
        loc='best'
    )



    # ---------- IMAGE ----------
    ax_img=fig.add_subplot(gs[1,:])
    ax_img.set_title("Current Frame with Tracked Inliers")
    ax_img.axis("off")

    frame_text=ax_img.text(
        0.01,0.99,               # x = left, y = top (axes coords)
        "",
        transform=ax_img.transAxes,
        color="white",
        fontsize=12,
        va="top",                # anchor text to its top
        ha="left",
        bbox=dict(facecolor="black",alpha=0.4,edgecolor="none",pad=2)
    )


    img_artist=ax_img.imshow(
        np.zeros((10,10),dtype=np.uint8),
        cmap="gray",
        vmin=0,vmax=255,
        extent=[0,10,10,0]
    )

    pts_scatter=ax_img.scatter(
        [],[],
        marker='+',
        s=25,
        linewidths=0.8,
        c='magenta'
    )

    fig.tight_layout()

    flow_lc=LineCollection([],linewidths=0.8,colors='lime',alpha=0.8)
    ax_img.add_collection(flow_lc)


    return {
        "fig":fig,

        "ax_traj":ax_traj,
        "traj_line":traj_line,

        "ax_world":ax_world,
        "lm_scatter":lm_scatter,
        "cam_point":cam_point,
        "cam_xaxis":cam_xaxis,
        "cam_zaxis":cam_zaxis,

        "ax_img":ax_img,
        "img_artist":img_artist,
        "pts_scatter":pts_scatter,
        "flow_lc":flow_lc,
        "frame_text":frame_text
    }

def update_traj(plot_state,traj_Cw):
    if len(traj_Cw)==0:
        return
    C=np.array(traj_Cw,dtype=float)
    plot_state["traj_line"].set_data(C[:,0],C[:,2])

    #Optional autoscale
    ax=plot_state["ax_traj"]
    ax.relim()
    ax.autoscale_view()

def update_world(
    plot_state,
    T_cw,
    X_w,
    frame_scale=0.08,
    p_near=90,
    min_window=1.0,
    max_window=50.0,
    smooth=0.9
):
    ax=plot_state["ax_world"]

    C=cam_center_from_Tcw(T_cw)
    Rwc=Rwc_from_Tcw(T_cw)

    if X_w is not None and X_w.size>0:
        xs=X_w[0,:]
        zs=X_w[2,:]
        plot_state["lm_scatter"].set_offsets(np.vstack([xs,zs]).T)

        dx=xs-C[0]
        dz=zs-C[2]
        r=np.sqrt(dx*dx+dz*dz)
        r=r[np.isfinite(r)]
        w=float(np.percentile(r,p_near)) if r.size>10 else min_window
    else:
        plot_state["lm_scatter"].set_offsets(np.zeros((0,2)))
        w=min_window

    w=float(np.clip(w,min_window,max_window))
    if "world_window" not in plot_state:
        plot_state["world_window"]=w
    else:
        plot_state["world_window"]=smooth*plot_state["world_window"]+(1.0-smooth)*w
    w=plot_state["world_window"]

    ax.set_xlim(C[0]-w,C[0]+w)
    ax.set_ylim(C[2]-w,C[2]+w)
    ax.set_aspect("equal",adjustable="box")

    axis_len=frame_scale*(2.0*w)
    x_axis_end=C+axis_len*Rwc[:,0]
    z_axis_end=C+axis_len*Rwc[:,2]

    plot_state["cam_point"].set_data([C[0]],[C[2]])
    plot_state["cam_xaxis"].set_positions(
    (C[0],C[2]),
    (x_axis_end[0],x_axis_end[2])
    )

    plot_state["cam_zaxis"].set_positions(
        (C[0],C[2]),
        (z_axis_end[0],z_axis_end[2])
    )

def update_frame_with_points(plot_state,img_gray,P_2xN,Pprev_2xN=None,frame_idx=None):
    h,w=img_gray.shape[:2]

    img_artist=plot_state["img_artist"]
    ax=plot_state["ax_img"]

    img_artist.set_data(img_gray)
    img_artist.set_clim(0,255)
    img_artist.set_extent([0,w,h,0])

    ax.set_xlim(0,w)
    ax.set_ylim(h,0)
    ax.set_aspect("equal",adjustable="box")

    #Current points (green crosses)
    if P_2xN is not None and P_2xN.size>0:
        plot_state["pts_scatter"].set_offsets(P_2xN.T.astype(float))
    else:
        plot_state["pts_scatter"].set_offsets(np.zeros((0,2)))

    #Flow lines: prev -> current
    if (Pprev_2xN is not None) and (P_2xN is not None) and (Pprev_2xN.size>0) and (P_2xN.size>0):
        #Assume correspondence by column order, so sizes must match
        n=min(Pprev_2xN.shape[1],P_2xN.shape[1])
        p0=Pprev_2xN[:,:n].T.astype(float)  #Nx2
        p1=P_2xN[:,:n].T.astype(float)      #Nx2
        segs=np.stack([p0,p1],axis=1)       #Nx2x2
        plot_state["flow_lc"].set_segments(segs)
    else:
        plot_state["flow_lc"].set_segments([])

    if frame_idx is not None:
        plot_state["frame_text"].set_text(f"Frame: {frame_idx}")
    else:
        plot_state["frame_text"].set_text("")



