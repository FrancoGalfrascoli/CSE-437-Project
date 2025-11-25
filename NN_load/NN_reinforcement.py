from data_io import load_tok_data_struct
from models import Net
from plasma_utils import Mesh_generation, infer_params, plasma_model, plot_boundary, plot_flux
from MPPI import MPPIController
import torch, numpy as np
from data_io import load_norm_and_datasets
import matplotlib.pyplot as plt

_, _, _, norm_values = load_norm_and_datasets("Data_generation")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Build geometry
tok = load_tok_data_struct("east_obj_2016_6565.mat")
Mesh = Mesh_generation(tok)

# Model
net_PINN = Net().to(device)
net_PINN.load_state_dict(torch.load("NN_pth/NN_dir.pth", map_location=device))
net_PINN.eval()

# Problem setup

#   Ip     = 320e3 : 500e3;   % A
#   beta_p = 0.2   : 0.8;
#   R0     = 1.82  : 1.90;    % m
#   a      = 0.40  : 0.45;    % m
#   kappa  = 1.2   : 1.8;
#   delta  = 0.0   : 0.4;

# init_state = [350e3, 0.2, 1.87, 0.4, 1.2, 0.1]
init_state = [350e3, 0.2, 1.88, 0.43, 1.6, 0]

IPFs, Ip, beta_p = plasma_model(*init_state, Mesh)  # (16,)
obj = [1.85767, 0.39879, 1.18224, 0.08319]
init_state[0] = Ip
init_state[1] = beta_p
IpBeta = np.array([Ip, beta_p], dtype=float)
IpBeta_t = torch.tensor(IpBeta[None,:], dtype=torch.float32, device=device)

mppi = MPPIController(
    H=5, N_seq=500, λ=1, sigma=0.15,
    net_PINN=net_PINN, Mesh=Mesh, obj=obj,
    init_state=init_state, IPFs=IPFs, device=device,
    noise_mask=[1]*14 + [0,0],   # example: freeze last 2 coils
)

state = init_state
steps = 1000
trajectory = []


trajectory = []   # list of dicts is more flexible than list of lists

for t in range(steps):
    dIPFs = mppi.action(state)
    IPFs = IPFs + dIPFs

    IPFs_t = torch.tensor(IPFs[None,:], dtype=torch.float32, device=device)
    psi = net_PINN(IPFs_t, IpBeta_t).detach().squeeze().numpy()

    state_next, cost, done, valid, boundary = mppi._plasma_step(IPFs, state)

    if cost < 1:
        print('stop')

    trajectory.append({
        "t": t,
        "state": state.copy(),
        "IPFs": IPFs.copy(),
        "cost": float(cost),
        "boundary": boundary.copy(),
        "psi": psi.copy(),
    })

    if not valid:
        print("Invalid state reached, stopping simulation.")
        break

    state = state_next

    if t % 5 == 0:
        np.savez_compressed("mppi_partial.npz",     
        trajectory=trajectory,
        target=obj,
        init_state=init_state,
        )
        
        print(t, np.round(state[2:], 5), f"Cost: {cost:.2e}")
        
    if done:
        print("Terminal state reached.")
        print(t, np.round(state[2:], 5), f"Cost: {cost:.2e}")
        break

# ---- Save data after simulation ----
np.savez_compressed(
    "mppi_run.npz",
    trajectory=trajectory,
    target=obj,
    init_state=init_state,
)
print("Saved run to mppi_run.npz")
