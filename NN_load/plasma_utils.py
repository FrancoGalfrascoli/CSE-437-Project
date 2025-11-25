
import numpy as np
from pyparsing import Optional
from scipy import ndimage
import scipy
from skimage import measure
from scipy.ndimage import map_coordinates
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from data_io import load_norm_and_datasets
from matplotlib.path import Path
from scipy.interpolate import RectBivariateSpline

__all__ = [
    "Mesh_generation",
    "compute_psi_sep",
    "smooth_boundary_spline",
    "params",
    "plot_boundary",
    "find_p",
    "plasma_model"
]

def Mesh_generation(tok_data_struct):
    """
    Python port of MATLAB Mesh_generation.m

    tok_data_struct must provide:
      - 'rg': 1D array of R coordinates (length >= 2)
      - 'zg': 1D array of Z coordinates (length >= 2)
      - 'nr': integer, number of R cells   (MATLAB uses nr-1 intervals)
      - 'nz': integer, number of Z cells   (MATLAB uses nz-1 intervals)

    Returns:
      Mesh: dict with keys mirroring the MATLAB struct:
        XStart, XEnd, Nr, XStep, R_interm (2D grid), R_Core (1D vector)
        YStart, YEnd, Nz, YStep, Z_interm (2D grid)
        X0, Y0 (meshgrid for field points, same shape as flux)
        R_vec, Z_vec (1D coordinate vectors used to build the grids)
        limdata (limiter polygon as (2,M))
    """
    rg = np.array(tok_data_struct["rg"][0, 0]).squeeze()
    zg = np.array(tok_data_struct["zg"][0, 0]).squeeze()
    nr = int(tok_data_struct["nr"][0, 0])
    nz = int(tok_data_struct["nz"][0, 0])
    limdata = tok_data_struct["limdata"][0, 0]  # shape (2, M)

    Mesh = {}
    # ---- X (R) axis
    Mesh["XStart"] = float(rg[0])
    Mesh["XEnd"] = float(rg[-1])
    Mesh["Nr"] = nr - 1  # intervals
    Mesh["XStep"] = abs(Mesh["XEnd"] - Mesh["XStart"]) / Mesh["Nr"]
    # Core 1D vectors (include both ends, Nr+1 points)
    R_vec = np.linspace(Mesh["XStart"], Mesh["XEnd"], Mesh["Nr"] + 1)
    Mesh["R_vec"] = R_vec.copy()
    Mesh["R_Core"] = R_vec[1:-1].copy()

    # ---- Y (Z) axis
    Mesh["YStart"] = float(zg[0])
    Mesh["YEnd"] = float(zg[-1])
    Mesh["Nz"] = nz - 1  # intervals
    Mesh["YStep"] = abs(Mesh["YEnd"] - Mesh["YStart"]) / Mesh["Nz"]
    Z_vec = np.linspace(Mesh["YStart"], Mesh["YEnd"], Mesh["Nz"] + 1)
    Mesh["Z_vec"] = Z_vec.copy()

    # ---- Full grids (field points)
    X0, Y0 = np.meshgrid(R_vec, Z_vec, indexing="xy")
    Mesh["X0"] = X0
    Mesh["Y0"] = Y0
    Mesh["R_interm"] = X0
    Mesh["Z_interm"] = Y0

    Mesh["limdata"] = limdata

    return Mesh

def compute_psi_sep(flux, R_grid, Z_grid, limiter_poly, max_iter=30, tol=1e-6):
    """
    Compute ψ_sep (flux at separatrix) using limiter-masked domain,
    equivalent to MATLAB lcfs_fast.m but returns only psi_sep.
    """
    flux = np.asarray(flux)
    R_grid = np.asarray(R_grid)
    Z_grid = np.asarray(Z_grid)
    assert flux.shape == R_grid.shape == Z_grid.shape

    # Refine grid with cubic spline interpolation
    Rv = R_grid[0, :]      # x-axis (R)
    Zv = Z_grid[:, 0]      # y-axis (Z)

    # ---- refinement factor ----
    refine = 3
    nR_fine = refine * len(Rv)
    nZ_fine = refine * len(Zv)

    R_fine_vec = np.linspace(Rv.min(), Rv.max(), nR_fine)
    Z_fine_vec = np.linspace(Zv.min(), Zv.max(), nZ_fine)

    R_fine, Z_fine = np.meshgrid(R_fine_vec, Z_fine_vec)

    # ---- cubic spline interpolation (MATLAB 'spline') ----
    spline = RectBivariateSpline(Zv, Rv, flux, kx=3, ky=3)   # cubic-cubic
    flux_fine = spline(Z_fine_vec, R_fine_vec)

    # Replace old grids
    R_grid = R_fine
    Z_grid = Z_fine
    flux   = flux_fine

    H, W = flux.shape

    # --- 0) Build limiter mask and its 1-cell edge ---
    poly = np.asarray(limiter_poly)
    if poly.shape[0] == 2 and poly.shape[1] != 2:
        poly = poly.T  # (2,N) -> (N,2)
    path = Path(poly)
    pts = np.c_[R_grid.ravel(), Z_grid.ravel()]
    in_lim = path.contains_points(pts).reshape(H, W)
    if not np.any(in_lim):
        raise ValueError("Limiter polygon has no overlap with grid. Check [R,Z] order/units.")

    valid = np.isfinite(flux)
    mask_domain = in_lim & valid

    # 1-pixel limiter edge (8-connectivity)
    structure3 = np.ones((3, 3), dtype=bool)
    inner = ndimage.binary_erosion(in_lim, structure=structure3, border_value=0)
    edge_lim = in_lim & (~inner)

    # --- 1) Magnetic axis inside limiter ---
    flux_masked = np.where(mask_domain, flux, np.nan)
    val_max = np.nanmax(flux_masked)
    val_min = np.nanmin(flux_masked)
    med_in = np.nanmedian(flux_masked[mask_domain])

    if np.isfinite(val_max) and val_max >= med_in:
        ax_flat = np.nanargmax(flux_masked)
        inside_high = True   # ψ decreases outward
    else:
        ax_flat = np.nanargmin(flux_masked)
        inside_high = False  # ψ increases outward

    ax_r, ax_c = np.unravel_index(ax_flat, flux.shape)
    psi_axis = flux[ax_r, ax_c]

    if not in_lim[ax_r, ax_c]:
        raise ValueError("Magnetic axis is not inside limiter.")

    # --- 2) Bracket ψ_sep ---
    if inside_high:
        psi_outer = np.nanpercentile(flux[mask_domain], 5)
        lo, hi = psi_outer, psi_axis
    else:
        psi_outer = np.nanpercentile(flux[mask_domain], 95)
        lo, hi = psi_axis, psi_outer
    if lo > hi:
        lo, hi = hi, lo

    structure8 = ndimage.generate_binary_structure(2, 2)  # 8-connectivity

    def is_open_at_level(tau):
        mask_psi = (flux >= tau) if inside_high else (flux <= tau)
        mask_psi &= in_lim
        labeled, _ = ndimage.label(mask_psi, structure=structure8)
        lab_ax = labeled[ax_r, ax_c]
        if lab_ax == 0:
            return True
        comp = (labeled == lab_ax)
        return np.any(comp & edge_lim)

    # --- 3) Bisection for ψ_sep ---
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        if is_open_at_level(mid):
            if inside_high:
                lo = mid
            else:
                hi = mid
        else:
            if inside_high:
                hi = mid
            else:
                lo = mid
        if abs(hi - lo) <= tol * max(1.0, abs(psi_axis)):
            break

    psi_sep = 0.5 * lo 
    return float(psi_sep)


def smooth_boundary_spline(R, Z, n_out=500, smooth=6e-3):
    """
    R, Z: 1D arrays of boundary points in order (closed or open).
    n_out: number of output samples on the smooth curve.
    smooth: spline smoothing parameter 's' (0 = interpolate exactly; increase to denoise).

    Returns: R_s, Z_s (smoothed, uniformly-parameterized, closed)
    """
    R = np.asarray(R).ravel()
    Z = np.asarray(Z).ravel()

    # ensure closed (first==last) to help periodic fit
    if R[0] != R[-1] or Z[0] != Z[-1]:
        R = np.r_[R, R[0]]
        Z = np.r_[Z, Z[0]]

    # chord-length parameter for numerical stability
    d = np.hypot(np.diff(R), np.diff(Z))
    s_cum = np.r_[0.0, np.cumsum(d)]
    if s_cum[-1] == 0:
        raise ValueError("Degenerate boundary (all points identical).")
    u = s_cum / s_cum[-1]

    # periodic parametric spline fit
    tck, _ = splprep([R, Z], u=u, s=smooth, per=True, k=3)

    # resample uniformly in parameter
    u_new = np.linspace(0, 1, n_out, endpoint=False)
    R_s, Z_s = splev(u_new, tck)

    return np.array(R_s), np.array(Z_s)


def infer_params(Mesh, flux, max_iter=30, tol=1e-6):
    """
    Mesh: dict produced by Mesh_generation(...)
    flux: 2D array with same shape as Mesh["X0"]/["Y0"]
    """
    Rg = Mesh["X0"]   # 2D mesh of R
    Zg = Mesh["Y0"]   # 2D mesh of Z
    lim = [Mesh["limdata"][1,:], Mesh["limdata"][0,:]]  # [Z; R] -> [R; Z]

    # 1) compute psi_sep with 2D grids
    psi_sep = compute_psi_sep(flux, Rg, Zg, lim, max_iter, tol)
    psi_sep = psi_sep+1e-5  

    # 2) sanity-check the level is inside data range
    fmin, fmax = float(np.nanmin(flux)), float(np.nanmax(flux))
    if not (fmin <= psi_sep <= fmax):
        print(f"[warn] psi_sep={psi_sep:.6g} outside data range [{fmin:.6g}, {fmax:.6g}]")

    # 4) extract contours numerically (index space is fine)
    contours = measure.find_contours(flux, level=float(psi_sep))

    if not contours:
        return (float("nan"),) * 4, np.zeros((2, 0)), False

    # 5) choose a closed contour if possible, map to (R,Z)
    boundary_R = boundary_Z = None
    valid = False
    for c in contours:
        rows, cols = c[:, 0], c[:, 1]
        if np.hypot(*(c[0, :] - c[-1, :])) < 1e-2:
            Rc = map_coordinates(Rg, [rows, cols], order=1, mode="nearest")
            Zc = map_coordinates(Zg, [rows, cols], order=1, mode="nearest")
            boundary_R, boundary_Z = Rc, Zc
            valid = True
            break
    if boundary_R is None:
        rows, cols = contours[0][:, 0], contours[0][:, 1]
        boundary_R = map_coordinates(Rg, [rows, cols], order=1, mode="nearest")
        boundary_Z = map_coordinates(Zg, [rows, cols], order=1, mode="nearest")

    boundary = np.vstack((boundary_R, boundary_Z))

    # 6) smooth + parameters
    R_s, Z_s = smooth_boundary_spline(boundary[0, :], boundary[1, :])
    rmin_idx, rmax_idx = np.argmin(R_s), np.argmax(R_s)
    zmax_idx = np.argmax(Z_s)
    a = (R_s[rmax_idx] - R_s[rmin_idx]) / 2.0
    R0 = R_s[rmin_idx] + a
    delta = (R0 - R_s[zmax_idx]) / a if a != 0 else 0.0
    kappa = (Z_s[zmax_idx]) / a if a != 0 else 0.0

    state = float(R0), float(a), float(kappa), float(delta)
    return state, boundary, valid


def plot_flux(flux, Mesh, levels=30, title=None, ylim=(-1, 1)):
    plt.contour(Mesh["X0"], Mesh["Y0"], flux, levels=levels)
    plt.xlabel('R (m)')
    plt.ylabel('Z (m)')
    plt.axis('equal')
    plt.ylim(*ylim)
    if title:
        plt.title(title)

def plot_boundary(boundaries, labels=None, line_styles=None, ylim=(-1, 1), title=None):
    # ---- removed plt.cla() so it does NOT clear previous contour ----
    if not isinstance(boundaries, (list, tuple)):
        boundaries = [boundaries]
    if labels is None:
        labels = [f'Boundary {i+1}' for i in range(len(boundaries))]
    if line_styles is None:
        line_styles = ['--' for _ in range(len(boundaries))]

    for b, lbl, ls in zip(boundaries, labels, line_styles):
        plt.plot(b[0, :], b[1, :], ls, lw=2, label=lbl)

    plt.xlabel('R (m)')
    plt.ylabel('Z (m)')
    plt.axis('equal')
    plt.ylim(*ylim)
    if title:
        plt.title(title)
    plt.legend()


def find_p(Ip_val, beta_p_val, R0_val, a_val, kappa_val, delta_val):
    import numpy as np

    # Grids (must match MATLAB exactly!)
    Ip     = np.arange(320e3, 500e3 + 1e-9, 30e3)
    beta_p = np.arange(0.2,   0.8   + 1e-12, 0.05)
    R0     = np.arange(1.82,  1.90  + 1e-12, 0.04)
    a      = np.arange(0.40,  0.45  + 1e-12, 0.05)
    kappa  = np.arange(1.2,   1.8   + 1e-12, 0.1)
    delta  = np.arange(0.0,   0.4   + 1e-12, 0.1)

    # Nearest indices (0-based in Python)
    i = int(np.argmin(abs(Ip    - Ip_val   )))
    j = int(np.argmin(abs(beta_p- beta_p_val)))
    k = int(np.argmin(abs(R0    - R0_val   )))
    l = int(np.argmin(abs(a     - a_val    )))
    m = int(np.argmin(abs(kappa - kappa_val)))
    n = int(np.argmin(abs(delta - delta_val)))

    # Sizes
    NI, NB, NR = len(Ip), len(beta_p), len(R0)
    NA, NK, ND = len(a), len(kappa), len(delta)

    # Compute flattened MATLAB-like index p (1-based)
    p = (
        i * NB * NR * NA * NK * ND +
        j * NR * NA * NK * ND +
        k * NA * NK * ND +
        l * NK * ND +
        m * ND +
        n +
        1
    )

    return p

def plasma_model(Ip, beta_p, R0, a, kappa, delta, Mesh):
    _, _, _, norm_values = load_norm_and_datasets("Data_generation")

    p = find_p(Ip, beta_p, R0, a, kappa, delta)
    # load matab .m  Data_generation/Data_NN/dsnx_p.mat


    dsnx_p = scipy.io.loadmat(f'Data_generation/Data_NN/dsnx_{p}.mat', struct_as_record=False, squeeze_me=True)

    E = dsnx_p["EAST_output"]   # becomes a MATLAB-like object with attributes

    state, _, _ = infer_params(Mesh, E.Psi_map)
    R0, a, kappa, delta = state
    # Scalars
    Ip     = float(np.asarray(E.Ip).squeeze())
    beta_p = float(np.asarray(E.beta_p).squeeze())

    IPFs = E.IPFs

    print(f'Initial plasma parameters loaded from p={p}:')
    print(f'  Ip = {Ip} A')
    print(f'  beta_p = {beta_p} A')
    print(f'  R0 = {R0} m')
    print(f'  a = {a} m')
    print(f'  kappa = {kappa}')
    print(f'  delta = {delta}')

    Ip = (Ip - norm_values['Ip_mean']) / norm_values['Ip_std']
    beta_p = (beta_p - norm_values['beta_p_mean']) / norm_values['beta_p_std']
    IPFs[0:14] = (IPFs[0:14] - norm_values['IPFs_mean']) / norm_values['IPFs_std']
    IPFs = IPFs.squeeze()

    return IPFs, Ip, beta_p
