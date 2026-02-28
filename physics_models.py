import math
import numpy as np
from scipy.integrate import solve_ivp

# The parameters of Jones, you can define based on your data
A_JONES_FIT = 1.0000e+05
B_JONES_FIT = 1.0000e+02
K_BLUNT_JONES_FIT = 2.8830e-06
K_EROSION_JONES_FIT = 1.4703e-08


def calculate_forrestal(w_val, d_val, l_val, v_val, f_val, q_val, crh_val):
    d_m = d_val * 0.001
    l_m = l_val * 0.001
    f_pa = f_val * 1e6

    if (
            d_m <= 1e-9
            or q_val <= 1e-9
            or v_val <= 1e-9
            or f_pa <= 1e-9
            or l_m <= 1e-9
            or np.isnan(w_val)
            or np.isnan(d_val)
            or np.isnan(l_val)
            or np.isnan(v_val)
            or np.isnan(f_val)
            or np.isnan(q_val)
            or np.isnan(crh_val)
    ):
        return 0.0

    u = l_m / d_m if d_m > 1e-9 else 0.0
    if u <= 0:
        return 0.0

    n_geom = (8 * u - 1) / (24 * u * u) if u > 0.125 else 0.72
    n_val = max(n_geom, 0.1)
    s_val = 82.6 * pow(max(f_val, 1.0), -0.544)
    if s_val <= 1e-9:
        return 0.0

    term_sf_pa = np.pi * d_m ** 3 * s_val * f_pa
    denom_v12 = 2 * w_val + np.pi * d_m ** 3 * n_val * q_val
    if denom_v12 <= 1e-9:
        return 2 * d_m

    v12_sq_num = 2 * w_val * v_val ** 2 - term_sf_pa
    if v12_sq_num < 0:
        return 2 * d_m

    v12_sq = v12_sq_num / denom_v12
    if v12_sq < 0:
        v12_sq = 0.0

    log_arg = 1 + (q_val * n_val * v12_sq) / (s_val * f_pa + 1e-9)
    if log_arg <= 1e-9:
        return 2 * d_m

    try:
        log_term = math.log(log_arg)
    except (ValueError, OverflowError):
        return 2 * d_m

    term1_h_den = np.pi * d_m ** 2 * q_val * n_val
    if term1_h_den <= 1e-9:
        return 2 * d_m

    term1_h = (2 * w_val) / term1_h_den
    h_val = term1_h * log_term + 2 * d_m
    return max(h_val, 0.0)


def jones_ode_system(x, y, d_p_m, a_jones, b_jones, k_blunt_jones, k_erosion_jones):
    v_val, m_p, r_nose_eff = y
    if v_val <= 1e-6 or m_p <= 1e-6:
        return [0.0, 0.0, 0.0]

    projectile_area = np.pi / 4 * d_p_m ** 2
    blunting_factor = 1 + (r_nose_eff / (d_p_m + 1e-9)) ** 2
    r_total = (a_jones * blunting_factor + b_jones * blunting_factor * v_val ** 2) * projectile_area
    dvdx = -r_total / m_p
    dm_pdx = -k_erosion_jones * r_total
    dr_nose_eff_dx = k_blunt_jones
    return [dvdx, dm_pdx, dr_nose_eff_dx]


def calculate_jones(w_val, d_val, l_val, v_val, f_val, q_val, crh_val,
                    a_jones=A_JONES_FIT, b_jones=B_JONES_FIT,
                    k_blunt_jones=K_BLUNT_JONES_FIT, k_erosion_jones=K_EROSION_JONES_FIT):

    if any(np.isnan([w_val, d_val, v_val, a_jones, b_jones, k_blunt_jones, k_erosion_jones])):
        return 0.0

    d_p_m = d_val * 0.001
    m_p0 = w_val
    v0 = v_val
    initial_r_nose_eff = crh_val * d_p_m / 2.0 if crh_val > 0.01 else d_p_m / 2.0

    if v0 <= 1e-6 or m_p0 <= 1e-6 or d_p_m <= 1e-6 or initial_r_nose_eff <= 1e-6:
        return 0.0

    y0 = [v0, m_p0, initial_r_nose_eff]
    max_penetration_est = d_p_m * 200
    x_span_ode = [0.0, max_penetration_est]
    ode_args = (d_p_m, a_jones, b_jones, k_blunt_jones, k_erosion_jones)

    def event_velocity_zero(x, y, *args):
        return y[0] - 1e-6

    event_velocity_zero.terminal = True
    event_velocity_zero.direction = -1

    def event_mass_zero(x, y, *args):
        return y[1] - 1e-6

    event_mass_zero.terminal = True
    event_mass_zero.direction = -1

    try:
        sol = solve_ivp(
            jones_ode_system,
            x_span_ode,
            y0,
            method="RK45",
            args=ode_args,
            events=[event_velocity_zero, event_mass_zero],
            dense_output=False,
            rtol=1e-5,
            atol=1e-8,
            max_step=max_penetration_est / 5000,
        )
    except (ValueError, OverflowError, RuntimeError):
        return 0.0

    final_depth = 0.0
    if sol.status in [0, 1] and sol.t is not None and len(sol.t) > 0:
        final_depth = sol.t[-1]
        if np.isnan(final_depth):
            final_depth = 0.0

    return max(final_depth, 0.0)


def precompute_physics_predictions(x_orig_np, q_list, feature_index):
    n_samples = x_orig_np.shape[0]
    physics_preds = np.zeros(n_samples, dtype=np.float32)
    physics_masks = np.zeros(n_samples, dtype=bool)

    print(f"Precomputing physics predictions for {n_samples} samples...")

    for i in range(n_samples):
        try:
            w_val = x_orig_np[i, feature_index["弹体重量"]]
            d_val = x_orig_np[i, feature_index["弹体直径"]]
            l_val = x_orig_np[i, feature_index["弹体长度"]]
            v_val = x_orig_np[i, feature_index["侵彻速度"]]
            f_val = x_orig_np[i, feature_index["靶体抗压强度"]]
            crh_val = x_orig_np[i, feature_index["弹体形状系数"]]
            q_val = float(q_list[i])
        except (IndexError, ValueError, TypeError):
            physics_preds[i] = 0.0
            physics_masks[i] = False
            continue

        if any(np.isnan([w_val, d_val, l_val, v_val, f_val, crh_val, q_val])):
            physics_preds[i] = 0.0
            physics_masks[i] = False
            continue

        if v_val < 800:
            pred = calculate_forrestal(w_val, d_val, l_val, v_val, f_val, q_val, crh_val)
            physics_masks[i] = True
        elif 800 <= v_val < 1500:
            pred = calculate_jones(w_val, d_val, l_val, v_val, f_val, q_val, crh_val)
            physics_masks[i] = True
        else:
            pred = 0.0
            physics_masks[i] = False

        if pred is None or np.isnan(pred):
            pred = 0.0
            physics_masks[i] = False

        physics_preds[i] = pred

    print(f"Precomputation complete. Valid physics predictions: {physics_masks.sum()}/{n_samples}")
    return physics_preds, physics_masks
