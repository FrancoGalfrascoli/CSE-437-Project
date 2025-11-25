import numpy as np
import matplotlib.pyplot as plt

# ---- Load the saved file ----
data = np.load("Same_good_run/mppi_partial_3.npz", allow_pickle=True)

trajectory = data["trajectory"].tolist()


print("Loaded MPPI run:")
print(f"- {len(trajectory)} steps")
print()

# ---- Extract arrays for easier plotting ----
steps   = np.array([item["t"] for item in trajectory])
states  = np.array([item["state"] for item in trajectory])      # shape (T, 4)
IPFs    = np.array([item["IPFs"] for item in trajectory])       # shape (T, 16)
costs   = np.array([item["cost"] for item in trajectory])
boundaries = [item["boundary"] for item in trajectory]          # list
psis       = [item["psi"] for item in trajectory]               # list

# ---- Quick summary ----
print("Trajectory summary:")
print("- States shape:    ", states.shape)
print("- IPFs shape:      ", IPFs.shape)
print("- Costs shape:     ", costs.shape)
print()

# ---- Plot cost over time ----
plt.figure(figsize=(6,4))
plt.plot(steps, costs, '-o')
plt.xlabel("Time Step")
plt.ylabel("Cost")
plt.title("MPPI Cost Evolution")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Plot state evolution (R0, a, kappa, delta) ----
labels = ["R0", "a", "kappa", "delta"]
plt.figure(figsize=(6,6))
for i in range(4):
    plt.plot(steps, states[:, i], label=labels[i])
plt.xlabel("Time Step")
plt.ylabel("State Value")
plt.title("State Evolution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

traj = data["trajectory"].tolist()

# Extract boundaries, steps, costs
boundaries = [step["boundary"] for step in traj]  # each: (2, N)
steps      = [step["t"] for step in traj]
costs      = [step["cost"] for step in traj]

# Safety check
if len(boundaries) < 2:
    raise ValueError("Need at least 2 trajectory steps to show initial and final boundaries.")

# Initial and final boundaries
boundary_init  = boundaries[0]      # initial condition
boundary_final = boundaries[-1]     # "target" / final state

# ---- Setup figure ----
fig, ax = plt.subplots(figsize=(5, 6))

# Global axis limits from all boundaries so view doesn't jump
all_R = np.concatenate([b[0, :] for b in boundaries])
all_Z = np.concatenate([b[1, :] for b in boundaries])
margin_R = 0.05 * (all_R.max() - all_R.min())
margin_Z = 0.05 * (all_Z.max() - all_Z.min())

ax.set_xlim(all_R.min() - margin_R, all_R.max() + margin_R)
ax.set_ylim(all_Z.min() - margin_Z, all_Z.max() + margin_Z)
ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("R [m]")
ax.set_ylabel("Z [m]")
ax.grid(True)

# ---- Static lines: initial (blue dashed) and final (red dashed) ----
init_line,  = ax.plot(
    boundary_init[0, :], boundary_init[1, :],
    "--b", lw=1.5, label="Initial LCFS"
)
final_line, = ax.plot(
    boundary_final[0, :], boundary_final[1, :],
    "--r", lw=1.5, label="Final LCFS"
)

# ---- Animated line: current LCFS (solid) ----
current_line, = ax.plot([], [], "k-", lw=2.0, label="Current LCFS")
title = ax.set_title("")

ax.legend(loc="best")

# ---- Animation update function ----
def update(frame_idx):
    boundary = boundaries[frame_idx]  # shape (2, N)
    R = boundary[0, :]
    Z = boundary[1, :]

    current_line.set_data(R, Z)
    title.set_text(
        f"Step {steps[frame_idx]} | Cost = {costs[frame_idx]:.2e}"
    )
    # Only the animated elements need to be returned
    return current_line, title

# ---- Create animation ----
anim = FuncAnimation(
    fig,
    update,
    frames=len(boundaries),
    interval=100,      # ms per frame (10 fps)
    blit=False
)

plt.show()

# Optional: save to file
# anim.save("mppi_lcfs.mp4", fps=10, dpi=150)
# anim.save("mppi_lcfs.gif", fps=10)
