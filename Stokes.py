import numpy as np
import sky


def rotation_angles(theta_obs, phi_obs, theta_lua, phi_lua):
    global alpha_out, alpha_in

    gamma = sky.func_gamma(theta_obs, phi_obs, theta_lua, phi_lua)

    if (0 <= (phi_obs - phi_lua) <= np.pi and 0 <= phi_lua <= np.pi) or (
            0 <= (phi_obs - phi_lua) <= np.pi or 0 <= phi_obs <= (phi_lua + np.pi) % (2 * np.pi)):
        alpha_in = np.arccos(
            (-np.cos(theta_obs) + np.cos(theta_lua) * np.cos(gamma)) / (-np.sin(gamma) * np.sin(theta_lua)))
        alpha_out = np.arccos(
            (-np.cos(theta_lua) + np.cos(theta_obs) * np.cos(gamma)) / (-np.sin(gamma) * np.sin(theta_obs)))
    else:
        alpha_in = np.arccos(
            (-np.cos(theta_obs) + np.cos(theta_lua) * np.cos(gamma)) / (np.sin(gamma) * np.sin(theta_lua)))
        alpha_out = np.arccos(
            (-np.cos(theta_lua) + np.cos(theta_obs) * np.cos(gamma)) / (np.sin(gamma) * np.sin(theta_obs)))

    if np.sin(theta_obs) == 0:
        alpha_in = np.arccos(-np.cos(theta_lua) * np.cos(phi_obs - phi_lua))
        alpha_out = np.arccos(np.cos(theta_lua))
    if np.sin(theta_lua) == 0:
        alpha_in = np.arccos(np.cos(theta_obs))
        alpha_out = np.arccos(-np.cos(theta_obs) * np.cos(phi_obs - phi_lua))

    if gamma == 0 or gamma == np.pi:
        alpha_in = 0
        alpha_out = 0

    if theta_obs < 0:
        if 0 <= phi_lua <= np.pi:
            if 0 <= (phi_obs - phi_lua) <= np.pi:
                alpha_out = phi_obs - phi_lua
            else:
                alpha_out = phi_lua - phi_obs
        else:
            if (0 <= (phi_obs - phi_lua) <= np.pi) or (0 <= phi_obs <= (phi_lua + np.pi) % (2 * np.pi)):
                alpha_out = phi_obs - phi_lua
            else:
                alpha_out = phi_lua - phi_obs
    else:
        pass

    alpha = [alpha_in, alpha_out]

    return alpha

