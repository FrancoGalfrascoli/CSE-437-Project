
import numpy as np
from scipy.io import loadmat
import torch
from models import Net
from plasma_utils import Mesh_generation, infer_params
from data_io import load_tok_data_struct
import time
import matplotlib.pyplot as plt

net_PINN = Net()
net_PINN.load_state_dict(torch.load("NN_pth/NN_dir.pth", map_location="cpu"))
net_PINN.eval()

tok = load_tok_data_struct("east_obj_2016_6565.mat")
Mesh = Mesh_generation(tok)

class MPPIController:
    def __init__(self, *, H, N_seq, λ, sigma,
                 net_PINN, Mesh, obj,
                 init_state, IPFs,
                 device="cpu", gamma=0.98,
                 invalid_penalty=1e10, term_tol=1e-1,
                 noise_mask=None):
        """
        H: horizon
        N_seq: number of sampled sequences
        λ: temperature (entropy regularization)
        sigma: stddev of control noise (scalar or (16,))
        net_PINN: torch model (IPFs, [Ip,beta_p]) -> psi(R,Z)
        parameters: object exposing .infer_params(flux)->(state6, valid)
        obj: (R0*, a*, kappa*, delta*)
        init_state: (Ip, beta_p, R0, a, kappa, delta)
        init_IPFs: (16,) initial mean action
        noise_mask: optional (16,) bool/0-1 array to freeze some coils (e.g., last 2)
        """
        self.H, self.N = H, N_seq
        self.lmbda = λ
        self.sigma = np.asarray(sigma, dtype=float) if np.isscalar(sigma) else np.array(sigma, float)
        self.net = net_PINN
        self.Mesh = Mesh
        self.obj = obj
        self.device = device
        self.gamma = gamma
        self.invalid_penalty = invalid_penalty
        self.term_tol = term_tol
        self.cost_scale = 1.0

        self.U_mean = np.tile(np.asarray(IPFs, float), (H, 1))  # (H,16)
        self.state = np.asarray(init_state, float)
        self.noise_mask = np.ones_like(self.U_mean, dtype=float)
        if noise_mask is not None:
            m = np.asarray(noise_mask, float).reshape(1, -1)  # (1,16)
            self.noise_mask[:] = m

    # ------------------ Embedded environment glue ------------------

    def _plasma_step(self, IPFs16, state6):
        Ip_beta = torch.tensor(state6[:2], dtype=torch.float32, device=self.device).unsqueeze(0)
        u_t     = torch.tensor(IPFs16, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            psi = self.net(u_t, Ip_beta).squeeze().detach().cpu().numpy()


        state4, boundary, valid = infer_params(self.Mesh, psi)   # (R0, a, kappa, delta)

        if valid and not any(np.isnan(state4)):
            R0, a, k, d = state4
        else:
            # fallback to current geometry
            _, _, R0, a, k, d = state6
            valid = False

        # next 6-state: keep Ip,beta_p, update geometry in correct order
        s_next6 = np.array(state6, dtype=float)
        s_next6[2:] = (R0, a, k, d)

        # cost on (R0,a,kappa,delta) with scales
        R0o, ao, ko, do = self.obj
        diff = np.array([(R0-R0o)/0.01, (a-ao)/0.01, (k-ko)/0.1, (d-do)/0.1], float)
        cost = float(diff @ diff)
        if not valid:
            cost += self.invalid_penalty
        done = (cost < self.term_tol) and valid 
        return s_next6, cost, done, valid, boundary


    def _rollout(self, state6, U_seq):
        """Evaluate a full sequence; return (traj, total_cost)."""
        H = U_seq.shape[0]
        s = np.asarray(state6, float).copy()
        total = 0.0
        disc = 1.0

        for t in range(H):
            s, c, done, _, _ = self._plasma_step(U_seq[t], s)
            total += disc * c
            disc *= self.gamma
            if done:
                break
        return total

    # ------------------ MPPI update ------------------

    def action(self, state6=None):
        """
        Sample sequences around U_mean, compute path costs, update mean (MPPI),
        return the first control (both 14 & 16 forms if you need).
        """
        if state6 is None:
            state6 = self.state
        H, A = self.U_mean.shape
        # noise ~ N(0, sigma^2), optionally mask some channels
        if np.isscalar(self.sigma):
            eps = np.random.randn(self.N, H, A) * self.sigma * self.cost_scale
        else:
            eps = np.random.randn(self.N, H, A) * self.sigma.reshape(1,1,-1) * self.cost_scale
        eps *= self.noise_mask  # freeze channels if mask==0

        costs = np.zeros(self.N, dtype=float)
        # Evaluate sequences

        for n in range(self.N):
            U = self.U_mean + eps[n]
            costs[n] = self._rollout(state6, U)

        Jmin = np.min(costs)
        self.cost_scale = min(1.0, Jmin**2)  # adapt cost scale
            
        self.cost_scale = 1
        # Scale by temperature λ
        beta = (costs - Jmin) / self.lmbda

        w = np.exp(-beta)
        w_sum = np.sum(w) + 1e-12
        weights = w / w_sum

        dU = np.tensordot(weights, eps, axes=(0, 0))

        self.U_mean = self.U_mean + dU
 
        self.U_mean[:-1] = self.U_mean[1:]
        self.U_mean[-1] = self.U_mean[-2]  # or zeros

        return dU[0]