import numpy as np
import matplotlib
# Force Agg backend immediately for HPC stability
import sys
if 'hpc' in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import os

# ==========================================
# 1. SCIENTIFIC PARAMETERS & GEOMETRY
# ==========================================

# -- Dimensions --
R_CELL = 12.18         # Radius (um)
H_CELL = 1.2           # Height (um)

# -- Grid --
NR = 15               # Radial steps
NTH = 20               # Angular steps
NZ = 3                 # Axial steps

# -- Physics Constants --
D_VALS = {'S': 15.0, 'I1': 15.0, 'I2': 12.0, 'P': 15.0}

# Kinetics (Updated Kcat values)
KCATS = {'C': 1.6/60, 'A': 11.2/60, 'B': 1.7/60} 
KM_VAL = 20.0          # uM
TOTAL_ENZ_MASS = 2000.0 

# -- Time Stepping (CFL Stability) --
dr = R_CELL / NR
dth = 2 * np.pi / NTH
dz = H_CELL / NZ

min_dx = min(dr, (dr/2)*dth) 
DT = 0.1 * (min_dx**2) / (2 * max(D_VALS.values()))
DURATION = 250.0       # Seconds (Long simulation time)
FRAMES = 50            # Total frames in GIF

# ==========================================
# 2. SOLVER ENGINE (3D Cylindrical PDE)
# ==========================================

r = np.linspace(dr/2, R_CELL - dr/2, NR)
th = np.linspace(0, 2*np.pi, NTH, endpoint=False)
z = np.linspace(dz/2, H_CELL - dz/2, NZ)
R, TH, Z = np.meshgrid(r, th, z, indexing='ij')

def get_laplacian(C):
    """Computes ∇²C in cylindrical coordinates (r, θ, z) using NumPy vectorization"""
    # Radial (r) terms
    dC_dr_f = (np.roll(C, -1, axis=0) - C) / dr
    dC_dr_b = (C - np.roll(C, 1, axis=0)) / dr
    
    # Enforce Neumann BC at R_CELL boundary (r=R_CELL) and handle center
    dC_dr_f[-1,:,:] = 0
    dC_dr_b[0,:,:] = 0 
    
    r_p = R + dr/2; r_m = R - dr/2
    term_r = (1/R) * ((r_p * dC_dr_f - r_m * dC_dr_b) / dr)
    
    # Angular (theta) term
    d2C_dth2 = (np.roll(C, -1, axis=1) - 2*C + np.roll(C, 1, axis=1)) / (dth**2)
    term_th = (1/(R**2)) * d2C_dth2

    # Axial (z) term (Neumann BCs enforced)
    d2C_dz2 = (np.roll(C, -1, axis=2) - 2*C + np.roll(C, 1, axis=2)) / (dz**2)
    d2C_dz2[:,:,0] = 0; d2C_dz2[:,:,-1] = 0 

    return term_r + term_th + d2C_dz2

def init_enzymes(mode):
    # E1 (DszC): Always Central (Symmetric Background)
    E1 = np.exp(-(R/(0.4*R_CELL))**2)
    
    if mode == 'dispersed':
        np.random.seed(42)
        E2 = np.random.rand(*R.shape) + 0.5
        np.random.seed(101)
        E3 = np.random.rand(*R.shape) + 0.3
        
    else: 
        # UNSYMMETRIC CLUSTER LOGIC
        shift_r, shift_th = 0.5 * R_CELL, np.pi / 2
        shift_r2, shift_th2 = 0.6 * R_CELL, np.pi / 2 + 0.3
        
        dist_sq_1 = R**2 + shift_r**2 - 2*R*shift_r*np.cos(TH - shift_th)
        dist_sq_2 = R**2 + shift_r2**2 - 2*R*shift_r2*np.cos(TH - shift_th2)
        
        raw_blob = 1.0 * np.exp(-dist_sq_1 / (2.2**2)) + \
                   0.6 * np.exp(-dist_sq_2 / (1.5**2))
        
        E2 = raw_blob; E3 = raw_blob 

    E1 = E1 / np.sum(E1) * TOTAL_ENZ_MASS
    E2 = E2 / np.sum(E2) * TOTAL_ENZ_MASS
    E3 = E3 / np.sum(E3) * TOTAL_ENZ_MASS
    return E1, E2, E3

# ==========================================
# 3. SIMULATION RUNNER
# ==========================================

S_d, I1_d, I2_d, P_d = [np.zeros_like(R, dtype=np.float64) for _ in range(4)]
S_c, I1_c, I2_c, P_c = [np.zeros_like(R, dtype=np.float64) for _ in range(4)]
S_d[:] = 100.0; S_c[:] = 100.0 

E1, E2_d, E3_d = init_enzymes('dispersed') 
_,  E2_c, E3_c = init_enzymes('clustered') 

hist_d, hist_c, time_arr = [], [], []
steps_per_frame = int((DURATION / FRAMES) / DT)
max_y_seen = 1.0 
PLOT_FLAG = 'interactive' # Default mode

if 'hpc' in sys.argv:
    PLOT_FLAG = 'hpc'
    print("Running in HPC mode. Output will be saved to final_simulation_dashboard.gif")

# ==========================================
# 4. VISUALIZATION LAYOUT
# ==========================================

fig = plt.figure(figsize=(16, 12), constrained_layout=True)
gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 2, 0.8])

ax_disp = fig.add_subplot(gs[0:2, 0], projection='polar')
ax_clus = fig.add_subplot(gs[0:2, 1], projection='polar')
ax_graph = fig.add_subplot(gs[0:2, 2])
ax_info = fig.add_subplot(gs[2, :])
ax_info.axis('off')

# --- TEXT DASHBOARD (Robust Formatting) ---
kcat_c_val = f"{KCATS['C']:.4f}"
kcat_a_val = f"{KCATS['A']:.4f}"
kcat_b_val = f"{KCATS['B']:.4f}"

# Note: Changed \bf to \mathbf and removed fragile chars like '&' inside math mode
col1_text = (
    r"$\mathbf{Governing\ Eqns:}$" + "\n"
    r"$\frac{\partial C}{\partial t} = D \nabla^2 C + v_{rxn}$" + "\n"
    r"$v = \frac{k_{cat} [E] [S]}{K_M + [S]}$" + "\n\n"
    r"$\mathbf{Geometry:}$" + "\n"
    rf"$R={R_CELL}\,\mu m, H={H_CELL}\,\mu m$"
)

col2_text = (
    r"$\mathbf{Kinetics\ and\ Mass:}$" + "\n"
    rf"Mass $[E_{{1,2,3}}] \approx {int(TOTAL_ENZ_MASS)}$ (a.u.)" + "\n"
    rf"$K_M = {KM_VAL}\,\mu M$" + "\n"
    r"$\mathbf{k_{cat}\ (s^{-1}):}$" + "\n"
    rf" C={kcat_c_val}, A={kcat_a_val}, B={kcat_b_val}"
)

col3_text = (
    r"$\mathbf{Diffusion\ (\mu m^2/s):}$" + "\n"
    rf"$D_S={D_VALS['S']}, D_{{I1}}={D_VALS['I1']}$" + "\n"
    rf"$D_{{I2}}={D_VALS['I2']}, D_P={D_VALS['P']}$" + "\n\n"
    r"$\mathbf{Configuration:}$" + "\n"
    r"$\it{Disp:}$ Random Distribution" + "\n"
    r"$\it{Clus:}$ Asymmetric Co-localized"
)

bbox_props = dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.5)
ax_info.text(0.05, 0.95, col1_text, fontsize=12, va='top', bbox=bbox_props)
ax_info.text(0.35, 0.95, col2_text, fontsize=12, va='top')
ax_info.text(0.70, 0.95, col3_text, fontsize=12, va='top', color='darkblue')

# ==========================================
# 5. ANIMATION LOOP
# ==========================================

pbar = tqdm(total=FRAMES, desc="Simulating") 

def update(frame):
    global S_d, I1_d, I2_d, P_d, S_c, I1_c, I2_c, P_c, max_y_seen

    for _ in range(steps_per_frame):
        # --- DISPERSED KINETICS ---
        v1d = KCATS['C'] * E1 * S_d / (KM_VAL + S_d + 1e-6)
        v2d = KCATS['A'] * E2_d * I1_d / (KM_VAL + I1_d + 1e-6) 
        v3d = KCATS['B'] * E3_d * I2_d / (KM_VAL + I2_d + 1e-6) 
        
        S_d  += DT * (D_VALS['S']*get_laplacian(S_d) - v1d)
        I1_d += DT * (D_VALS['I1']*get_laplacian(I1_d) + v1d - v2d)
        I2_d += DT * (D_VALS['I2']*get_laplacian(I2_d) + v2d - v3d)
        P_d  += DT * (D_VALS['P']*get_laplacian(P_d) + v3d)

        # --- CLUSTERED KINETICS ---
        v1c = KCATS['C'] * E1 * S_c / (KM_VAL + S_c + 1e-6)
        v2c = KCATS['A'] * E2_c * I1_c / (KM_VAL + I1_c + 1e-6) 
        v3c = KCATS['B'] * E3_c * I2_c / (KM_VAL + I2_c + 1e-6) 

        S_c  += DT * (D_VALS['S']*get_laplacian(S_c) - v1c)
        I1_c += DT * (D_VALS['I1']*get_laplacian(I1_c) + v1c - v2c)
        I2_c += DT * (D_VALS['I2']*get_laplacian(I2_c) + v2c - v3c)
        P_c  += DT * (D_VALS['P']*get_laplacian(P_c) + v3c)
        
        # Boundary Conditions and Clipping
        S_d[-1,:,:] = 100.0; S_c[-1,:,:] = 100.0 # Constant concentration at boundary
        for arr in [S_d, I1_d, I2_d, P_d, S_c, I1_c, I2_c, P_c]:
            np.clip(arr, 0, None, out=arr)

    # Data collection
    hist_d.append(np.sum(P_d))
    hist_c.append(np.sum(P_c))
    time_arr.append(frame * (DURATION/FRAMES))

    # --- PLOTTING ---
    ax_disp.clear(); ax_clus.clear(); ax_graph.clear()
    
    # Polar plot data
    P_d_vis = np.concatenate((np.max(P_d, axis=2), np.max(P_d, axis=2)[:, :1]), axis=1)
    P_c_vis = np.concatenate((np.max(P_c, axis=2), np.max(P_c, axis=2)[:, :1]), axis=1)
    
    th_closed = np.concatenate((th, [2*np.pi]))
    TH_closed, R_closed = np.meshgrid(th_closed, r)
    vmax = max(np.max(P_c), np.max(P_d), 1e-3)
    
    # Polar 1
    ax_disp.pcolormesh(TH_closed, R_closed, P_d_vis, cmap='magma', vmin=0, vmax=vmax, shading='gouraud')
    ax_disp.set_title(f"Dispersed E3\nProduct: {hist_d[-1]:.1f} (a.u.)", fontsize=14, fontweight='bold')
    ax_disp.grid(False); ax_disp.set_xticks([]); ax_disp.set_yticks([])

    # Polar 2
    ax_clus.pcolormesh(TH_closed, R_closed, P_c_vis, cmap='magma', vmin=0, vmax=vmax, shading='gouraud')
    ax_clus.set_title(f"Clustered E3\nProduct: {hist_c[-1]:.1f} (a.u.)", fontsize=14, fontweight='bold')
    ax_clus.grid(False); ax_clus.set_xticks([]); ax_clus.set_yticks([])

    # Efficiency Graph (Stabilized)
    ax_graph.grid(True, linestyle=':', alpha=0.6, zorder=0)
    ax_graph.plot(time_arr, hist_c, 'r-', linewidth=3, label='Clustered', zorder=5)
    ax_graph.plot(time_arr, hist_d, 'b--', linewidth=3, label='Dispersed', zorder=5)
    ax_graph.fill_between(time_arr, hist_d, hist_c, color='red', alpha=0.1, zorder=2) 
    
    # Axis handling (Y-axis only expands)
    curr_max = max(hist_c[-1], hist_d[-1]) if len(hist_c) > 0 else 1
    if curr_max > max_y_seen: max_y_seen = curr_max
        
    ax_graph.set_xlim(0, DURATION)
    ax_graph.set_ylim(0, max_y_seen * 1.1)
    ax_graph.set_xlabel("Time (s)", fontsize=12)
    ax_graph.set_ylabel("Total Product (a.u.)", fontsize=12)
    ax_graph.set_title("Efficiency Comparison", fontsize=14, fontweight='bold')
    ax_graph.legend(fontsize=12, loc='upper left', frameon=True, facecolor='white', framealpha=1)
    
    pbar.update(1)
    return ax_disp, ax_clus, ax_graph

# ==========================================
# 6. RUN AND SAVE
# ==========================================

if PLOT_FLAG == 'hpc':
    # HPC Mode: Use non-interactive saving method
    # NOTE: Using 'pillow' writer is safest for GIFs without ffmpeg
    Writer = animation.writers['pillow']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    
    output_filename = 'final_simulation_dashboard.gif'
    
    # Save animation manually frame by frame to avoid issues
    with writer.saving(fig, output_filename, dpi=100):
        for frame in range(FRAMES):
            update(frame)
            writer.grab_frame()
            
    pbar.close()
    plt.close(fig)
    print(f"Simulation Complete. Output saved to {output_filename}")
    
else:
    # Interactive Mode
    try:
        from IPython.display import Image as ImgDisplay
        ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=150, repeat=False)
        output_filename = 'final_simulation_dashboard.gif'
        ani.save(output_filename, writer='pillow', fps=10)
        pbar.close()
        print("Simulation Complete.")
        ImgDisplay(filename=output_filename)
    except ImportError:
        pbar.close()
        print("Interactive plotting failed. Run with 'python run_simulation.py hpc'")
