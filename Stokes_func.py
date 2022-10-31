import numpy as np
import pandas as pd
import coordinates
import lmfit
import matplotlib.pyplot as plt
import astropy.units as u
import albedo
import field_functions
import moon_functions
from re import search
from tqdm import tqdm
import matplotlib as mpl


def reverse_colourmap(comap, name='my_cmap_r'):
    reverse = []
    k = []

    for key in comap._segmentdata:
        k.append(key)
        channel = comap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1 - t[0], t[2], t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k, reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r


def func_gamma(theta_obs, phi_obs, theta_lua, phi_lua):
    gamma = np.arccos(
        np.sin(theta_lua) * np.sin(theta_obs) * np.cos(phi_obs - phi_lua) + np.cos(theta_lua) * np.cos(theta_obs))

    return gamma


def rot_angles(theta_obs, phi_obs, theta_lua, phi_lua):
    gamma = func_gamma(theta_obs, phi_obs, theta_lua, phi_lua)

    if 0 <= phi_lua <= np.pi:
        if 0 <= (phi_obs - phi_lua) <= np.pi:
            alpha_IN = np.arccos(
                (-np.cos(theta_obs) + np.cos(theta_lua) * np.cos(gamma)) / (np.sin(gamma) * np.sin(theta_lua)))
            alpha_OUT = np.arccos(
                (-np.cos(theta_lua) + np.cos(theta_obs) * np.cos(gamma)) / (np.sin(gamma) * np.sin(theta_obs)))
        else:
            alpha_IN = np.arccos(
                (-np.cos(theta_obs) + np.cos(theta_lua) * np.cos(gamma)) / (-np.sin(gamma) * np.sin(theta_lua)))
            alpha_OUT = np.arccos(
                (-np.cos(theta_lua) + np.cos(theta_obs) * np.cos(gamma)) / (-np.sin(gamma) * np.sin(theta_obs)))
    else:
        if (0 <= (phi_obs - phi_lua) <= np.pi) or (0 <= phi_obs <= (phi_lua + np.pi) % (2 * np.pi)):
            alpha_IN = np.arccos(
                (-np.cos(theta_obs) + np.cos(theta_lua) * np.cos(gamma)) / (np.sin(gamma) * np.sin(theta_lua)))
            alpha_OUT = np.arccos(
                (-np.cos(theta_lua) + np.cos(theta_obs) * np.cos(gamma)) / (np.sin(gamma) * np.sin(theta_obs)))
        else:
            alpha_IN = np.arccos(
                (-np.cos(theta_obs) + np.cos(theta_lua) * np.cos(gamma)) / (-np.sin(gamma) * np.sin(theta_lua)))
            alpha_OUT = np.arccos(
                (-np.cos(theta_lua) + np.cos(theta_obs) * np.cos(gamma)) / (-np.sin(gamma) * np.sin(theta_obs)))

    if gamma < 10 ** -4:
        alpha_OUT = 0
        alpha_IN = 0
    else:
        pass

    if theta_obs < 0:
        if 0 <= phi_lua <= np.pi:
            if 0 <= (phi_obs - phi_lua) <= np.pi:
                alpha_OUT = phi_obs - phi_lua
            else:
                alpha_OUT = phi_lua - phi_obs
        else:
            if (0 <= (phi_obs - phi_lua) <= np.pi) or (0 <= phi_obs <= (phi_lua + np.pi) % (2 * np.pi)):
                alpha_OUT = phi_obs - phi_lua
            else:
                alpha_OUT = phi_lua - phi_obs
    else:
        pass

    alpha = [alpha_IN, alpha_OUT]

    return alpha


def angle_in(a, b, c, d):
    # theta_obs, phi_obs, theta_lua, phi_lua
    matrix = np.column_stack((a, b, c, d))
    alpha = []

    for i in range(len(matrix)):
        theta_obs = matrix[i][0]
        phi_obs = matrix[i][1]
        theta_lua = matrix[i][2]
        phi_lua = matrix[i][3]

        gamma = func_gamma(theta_obs, phi_obs, theta_lua, phi_lua)

        if np.sin(theta_obs) == 0:
            alpha_IN = np.arccos(-np.cos(theta_lua) * np.cos(phi_obs - phi_lua))
        elif np.sin(theta_lua) == 0:
            alpha_IN = np.arccos(np.cos(theta_obs))
        elif 0 <= (phi_obs - phi_lua) <= np.pi or (0 <= phi_obs <= (phi_lua + np.pi) % (2 * np.pi)):
            alpha_IN = np.arccos(
                (np.cos(theta_obs) - np.cos(theta_lua) * np.cos(gamma)) / (np.sin(gamma) * np.sin(theta_lua)))
        else:
            alpha_IN = np.arccos(
                (-np.cos(theta_obs) + np.cos(theta_lua) * np.cos(gamma)) / (np.sin(gamma) * np.sin(theta_lua)))

        alpha.append(alpha_IN)

    return alpha


def angle_out(theta_obs, phi_obs, theta_lua, phi_lua):
    # theta_obs, phi_obs, theta_lua, phi_lua
    gamma = func_gamma(theta_obs, phi_obs, theta_lua, phi_lua)

    if np.sin(theta_obs) == 0:
        alpha_OUT = np.arccos(np.cos(theta_lua))
    elif np.sin(theta_lua) == 0:
        alpha_OUT = np.arccos(-np.cos(theta_obs) * np.cos(phi_obs - phi_lua))
    elif 0 <= (phi_obs - phi_lua) <= np.pi or (0 <= phi_obs <= (phi_lua + np.pi) % (2 * np.pi)):
        alpha_OUT = np.arccos(
            (np.cos(theta_lua) - np.cos(theta_obs) * np.cos(gamma)) / (np.sin(gamma) * np.sin(theta_obs)))
    else:
        alpha_OUT = np.arccos(
            (-np.cos(theta_lua) + np.cos(theta_obs) * np.cos(gamma)) / (np.sin(gamma) * np.sin(theta_obs)))

    return alpha_OUT


def rotation_angles(theta_obs, phi_obs, theta_lua, phi_lua):
    global alpha_out, alpha_in

    gamma = func_gamma(theta_obs, phi_obs, theta_lua, phi_lua)

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


def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

        # return string
    return str1


# ----------------------------------------------------------------------------------------


def func_reg_DOP(allvars, par):
    crds1, crds2, cLUA1, cLUA2 = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4))
        r_out = np.zeros((4, 4))

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4))

        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha = rotation_angles(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha_IN = - alpha[0]
        alpha_OUT = np.pi - alpha[1]

        # matriz de rotação
        r_out[0, :] = [1, 0, 0, 0]
        r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
        r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
        r_out[3, :] = [0, 0, 0, 1]

        r_in[0, :] = [1, 0, 0, 0]
        r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
        r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
        r_in[3, :] = [0, 0, 0, 1]

        # matriz reduzida para Rayleigh scattering
        m[0, :] = [1, -np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 0, 0]
        m[1, :] = [-np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 1, 0, 0]
        m[2, :] = [0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2), 0]
        m[3, :] = [0, 0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2)]

        # matriz de fase
        # P = R_out * M * R_in
        r_out = np.dot(r_out, m)

        p = np.dot(r_out, r_in)

        # parâmetros de Stokes
        # S_out = [I,Q,U,V]^T
        I = np.dot(p[0, :], s_in)
        Q = np.dot(p[1, :], s_in)
        U = np.dot(p[2, :], s_in)
        # V = np.dot(p[3, :], s_in)

        DOP[i] = par * np.sqrt(Q ** 2 + U ** 2) / I

    return DOP


def func_dep_DOP(allvars, P):
    crds1, crds2, cLUA1, cLUA2 = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4))
        r_out = np.zeros((4, 4))

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4))

        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha = rotation_angles(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha_IN = - alpha[0]
        alpha_OUT = np.pi - alpha[1]

        # matriz de rotação
        r_out[0, :] = [1, 0, 0, 0]
        r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
        r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
        r_out[3, :] = [0, 0, 0, 1]

        r_in[0, :] = [1, 0, 0, 0]
        r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
        r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
        r_in[3, :] = [0, 0, 0, 1]

        A = (3 / 2) * ((1 - P) / (1 + P / 2))

        # matriz reduzida para Rayleigh scattering
        m[0, :] = [0.5 * A * (((1 + P) / (1 - P)) + np.cos(gamma) ** 2), -0.5 * A * (np.sin(gamma) ** 2), 0, 0]
        m[1, :] = [-0.5 * A * np.sin(gamma) ** 2, 0.5 * A * (1 + np.cos(gamma) ** 2), 0, 0]
        m[2, :] = [0, 0, A * np.cos(gamma), 0]
        m[3, :] = [0, 0, 0, ((1 - 2 * P) / (1 - P)) * A * np.cos(gamma)]

        # matriz de fase
        # P = R_out * M * R_in
        r_out = np.dot(r_out, m)

        p = np.dot(r_out, r_in)

        # parâmetros de Stokes
        # S_out = [I,Q,U,V]^T
        I = np.dot(p[0, :], s_in)
        Q = np.dot(p[1, :], s_in)
        U = np.dot(p[2, :], s_in)
        # V = np.dot(p[3, :], s_in)

        DOP[i] = np.sqrt(Q ** 2 + U ** 2) / I

    return DOP


def func_depo_DOP(allvars, P):
    crds1, crds2, cLUA1, cLUA2, banda, par_wave = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4))
        r_out = np.zeros((4, 4))

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4))

        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha = rotation_angles(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha_IN = - alpha[0]
        alpha_OUT = np.pi - alpha[1]

        # matriz de rotação
        r_out[0, :] = [1, 0, 0, 0]
        r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
        r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
        r_out[3, :] = [0, 0, 0, 1]

        r_in[0, :] = [1, 0, 0, 0]
        r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
        r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
        r_in[3, :] = [0, 0, 0, 1]

        A = (3 / 2) * ((1 - P) / (1 + P / 2))

        # matriz reduzida para Rayleigh scattering
        m[0, :] = [0.5 * A * (((1 + P) / (1 - P)) + np.cos(gamma) ** 2), -0.5 * A * (np.sin(gamma) ** 2), 0, 0]
        m[1, :] = [-0.5 * A * np.sin(gamma) ** 2, 0.5 * A * (1 + np.cos(gamma) ** 2), 0, 0]
        m[2, :] = [0, 0, A * np.cos(gamma), 0]
        m[3, :] = [0, 0, 0, ((1 - 2 * P) / (1 - P)) * A * np.cos(gamma)]

        # matriz de fase
        # P = R_out * M * R_in
        r_out = np.dot(r_out, m)

        p = np.dot(r_out, r_in)

        # parâmetros de Stokes
        # S_out = [I,Q,U,V]^T
        I = np.dot(p[0, :], s_in)
        Q = np.dot(p[1, :], s_in)
        U = np.dot(p[2, :], s_in)
        # V = np.dot(p[3, :], s_in)

        DOP[i] = (banda[i] ** par_wave[i]) * np.sqrt(Q ** 2 + U ** 2) / I

    return DOP


def func_hor_DOP(allvars, N):
    crds1, crds2, cLUA1, cLUA2, banda, par_wave = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4))
        r_out = np.zeros((4, 4))

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4))

        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha = rotation_angles(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha_IN = -alpha[0]
        alpha_OUT = np.pi - alpha[1]

        # matriz de rotação
        r_out[0, :] = [1, 0, 0, 0]
        r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
        r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
        r_out[3, :] = [0, 0, 0, 1]

        r_in[0, :] = [1, 0, 0, 0]
        r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
        r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
        r_in[3, :] = [0, 0, 0, 1]

        # matriz reduzida para Rayleigh scattering
        m[0, :] = [1, -np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 0, 0]
        m[1, :] = [-np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 1, 0, 0]
        m[2, :] = [0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2), 0]
        m[3, :] = [0, 0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2)]

        # matriz de fase
        # P = R_out * M * R_in
        r_out = np.dot(r_out, m)

        p = np.dot(r_out, r_in)

        # parâmetros de Stokes
        # S_out = [I,Q,U,V]^T
        I = np.dot(p[0, :], s_in)
        Q = np.dot(p[1, :], s_in)
        U = np.dot(p[2, :], s_in)
        # V = np.dot(p[3, :], s_in)

        DOP[i] = (banda[i] ** par_wave[i]) * (np.cos(cLUA1[i]) ** (1 / N)) * np.sqrt(Q ** 2 + U ** 2) / I

    return DOP


def func_hor(allvars, N):
    crds1, crds2, cLUA1, cLUA2 = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4))
        r_out = np.zeros((4, 4))

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4))

        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha = rotation_angles(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha_IN = -alpha[0]
        alpha_OUT = np.pi - alpha[1]

        # matriz de rotação
        r_out[0, :] = [1, 0, 0, 0]
        r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
        r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
        r_out[3, :] = [0, 0, 0, 1]

        r_in[0, :] = [1, 0, 0, 0]
        r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
        r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
        r_in[3, :] = [0, 0, 0, 1]

        # matriz reduzida para Rayleigh scattering
        m[0, :] = [1, -np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 0, 0]
        m[1, :] = [-np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 1, 0, 0]
        m[2, :] = [0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2), 0]
        m[3, :] = [0, 0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2)]

        # matriz de fase
        # P = R_out * M * R_in
        r_out = np.dot(r_out, m)

        p = np.dot(r_out, r_in)

        # parâmetros de Stokes
        # S_out = [I,Q,U,V]^T
        I = np.dot(p[0, :], s_in)
        Q = np.dot(p[1, :], s_in)
        U = np.dot(p[2, :], s_in)
        # V = np.dot(p[3, :], s_in)

        DOP[i] = (np.cos(cLUA1[i]) ** (1 / N)) * np.sqrt(Q ** 2 + U ** 2) / I

    return DOP


def func_turb_DOP(allvars, k, d):
    crds1, crds2, cLUA1, cLUA2, par_int = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4))
        r_out = np.zeros((4, 4))

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4))

        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha = rotation_angles(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha_IN = -alpha[0]
        alpha_OUT = np.pi - alpha[1]

        # matriz de rotação
        r_out[0, :] = [1, 0, 0, 0]
        r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
        r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
        r_out[3, :] = [0, 0, 0, 1]

        r_in[0, :] = [1, 0, 0, 0]
        r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
        r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
        r_in[3, :] = [0, 0, 0, 1]

        # matriz reduzida para Rayleigh scattering
        m[0, :] = [1, -np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 0, 0]
        m[1, :] = [-np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 1, 0, 0]
        m[2, :] = [0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2), 0]
        m[3, :] = [0, 0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2)]

        # matriz de fase
        # P = R_out * M * R_in
        r_out = np.dot(r_out, m)

        p = np.dot(r_out, r_in)

        # parâmetros de Stokes
        # S_out = [I,Q,U,V]^T
        I = np.dot(p[0, :], s_in)
        Q = np.dot(p[1, :], s_in)
        U = np.dot(p[2, :], s_in)
        # V = np.dot(p[3, :], s_in)

        DOP[i] = par_int[i] * np.exp(-3 / k + d) * np.sqrt(Q ** 2 + U ** 2) / I

    return DOP


def func_DOP(allvars, N, k, d, c):
    crds1, crds2, cLUA1, cLUA2, cSOL1, cSOL2, alb, seeing, banda, par = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4))
        r_out = np.zeros((4, 4))

        r_in_sol = np.zeros((4, 4))
        r_out_sol = np.zeros((4, 4))

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4))
        m_sol = np.zeros((4, 4))

        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])
        gamma_sol = func_gamma(crds1[i], crds2[i], cSOL1[i], cSOL2[i])

        alpha = rotation_angles(crds1[i], crds2[i], cLUA1[i], cLUA2[i])
        alpha_sol = rotation_angles(crds1[i], crds2[i], cSOL1[i], cSOL2[i])

        alpha_IN = -alpha[0]
        alpha_OUT = np.pi - alpha[1]

        alpha_in_sol = -alpha_sol[0]
        alpha_out_sol = np.pi - alpha_sol[1]

        # matriz de rotação
        r_out[0, :] = [1, 0, 0, 0]
        r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
        r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
        r_out[3, :] = [0, 0, 0, 1]

        r_in[0, :] = [1, 0, 0, 0]
        r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
        r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
        r_in[3, :] = [0, 0, 0, 1]

        r_out_sol[0, :] = [1, 0, 0, 0]
        r_out_sol[1, :] = [0, np.cos(2 * alpha_out_sol), -np.sin(2 * alpha_out_sol), 0]
        r_out_sol[2, :] = [0, np.sin(2 * alpha_out_sol), np.cos(2 * alpha_out_sol), 0]
        r_out_sol[3, :] = [0, 0, 0, 1]

        r_in_sol[0, :] = [1, 0, 0, 0]
        r_in_sol[1, :] = [0, np.cos(2 * alpha_in_sol), -np.sin(2 * alpha_in_sol), 0]
        r_in_sol[2, :] = [0, np.sin(2 * alpha_in_sol), np.cos(2 * alpha_in_sol), 0]
        r_in_sol[3, :] = [0, 0, 0, 1]

        # matriz reduzida para Rayleigh scattering
        m[0, :] = [1, -np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 0, 0]
        m[1, :] = [-np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 1, 0, 0]
        m[2, :] = [0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2), 0]
        m[3, :] = [0, 0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2)]

        m_sol[0, :] = [1, -np.sin(gamma_sol) ** 2 / (1 + np.cos(gamma_sol) ** 2), 0, 0]
        m_sol[1, :] = [-np.sin(gamma_sol) ** 2 / (1 + np.cos(gamma_sol) ** 2), 1, 0, 0]
        m_sol[2, :] = [0, 0, 2 * np.cos(gamma_sol) / (1 + np.cos(gamma_sol) ** 2), 0]
        m_sol[3, :] = [0, 0, 0, 2 * np.cos(gamma_sol) / (1 + np.cos(gamma_sol) ** 2)]

        # matriz de fase
        # P = R_out * M * R_in
        r_out = np.dot(r_out, m)
        r_out_sol = np.dot(r_out_sol, m_sol)

        p = np.dot(r_out, r_in)
        p_sol = np.dot(r_out_sol, r_in_sol)

        # parâmetros de Stokes
        # S_out = [I,Q,U,V]^T
        I = np.dot(p[0, :], s_in)
        Q = np.dot(p[1, :], s_in)
        U = np.dot(p[2, :], s_in)
        # V = np.dot(p[3, :], s_in)

        I_sol = np.dot(p_sol[0, :], s_in)
        Q_sol = np.dot(p_sol[1, :], s_in)
        U_sol = np.dot(p_sol[2, :], s_in)
        # V_sol = np.dot(p_sol[3, :], s_in)

        alt = 90 - 180 / np.pi * cSOL1[i]

        if alt < -18:
            DOP[i] = cLUA1[i] * np.cos(cLUA1[i]) ** (1 / N) * np.exp(-seeing[i] / k + d) * banda[i] ** c * par[i] * alb[
                i] * np.sqrt(Q ** 2 + U ** 2) / I

        if alt >= -18:
            I = I + I_sol
            Q = Q * cLUA1[i] * np.cos(cLUA1[i]) ** (1 / N) * alb[i] + Q_sol * cSOL1[i] * np.cos(cSOL1[i]) ** (1 / N)
            U = U * cLUA1[i] * np.cos(cLUA1[i]) ** (1 / N) * alb[i] + U_sol * cSOL1[i] * np.cos(cSOL1[i]) ** (1 / N)

            DOP[i] = par[i] * banda[i] ** c * np.exp(-seeing[i] / k + d) * np.sqrt(Q ** 2 + U ** 2) / I

    return DOP


def func_seeing_DOP(allvars, k, d):
    crds1, crds2, cLUA1, cLUA2, seeing, banda, par_wave = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4))
        r_out = np.zeros((4, 4))

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4))

        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha = rotation_angles(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha_IN = -alpha[0]
        alpha_OUT = np.pi - alpha[1]

        # matriz de rotação
        r_out[0, :] = [1, 0, 0, 0]
        r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
        r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
        r_out[3, :] = [0, 0, 0, 1]

        r_in[0, :] = [1, 0, 0, 0]
        r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
        r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
        r_in[3, :] = [0, 0, 0, 1]

        # matriz reduzida para Rayleigh scattering
        m[0, :] = [1, -np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 0, 0]
        m[1, :] = [-np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 1, 0, 0]
        m[2, :] = [0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2), 0]
        m[3, :] = [0, 0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2)]

        # matriz de fase
        # P = R_out * M * R_in
        r_out = np.dot(r_out, m)

        p = np.dot(r_out, r_in)

        # parâmetros de Stokes
        # S_out = [I,Q,U,V]^T
        I = np.dot(p[0, :], s_in)
        Q = np.dot(p[1, :], s_in)
        U = np.dot(p[2, :], s_in)
        # V = np.dot(p[3, :], s_in)

        DOP[i] = (banda[i] ** par_wave[i]) * np.exp(-seeing[i] / k + d) * np.sqrt(Q ** 2 + U ** 2) / I

    return DOP


def func_sun_DOP(allvars, par):
    crds1, crds2, cLUA1, cLUA2, cSOL1, cSOL2, alb, par_wave = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4))
        r_out = np.zeros((4, 4))

        r_in_sol = np.zeros((4, 4))
        r_out_sol = np.zeros((4, 4))

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4))
        m_sol = np.zeros((4, 4))

        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])
        gamma_sol = func_gamma(crds1[i], crds2[i], cSOL1[i], cSOL2[i])

        alpha = rotation_angles(crds1[i], crds2[i], cLUA1[i], cLUA2[i])
        alpha_sol = rotation_angles(crds1[i], crds2[i], cSOL1[i], cSOL2[i])

        alpha_IN = -alpha[0]
        alpha_OUT = np.pi - alpha[1]

        alpha_in_sol = -alpha_sol[0]
        alpha_out_sol = np.pi - alpha_sol[1]

        # matriz de rotação
        r_out[0, :] = [1, 0, 0, 0]
        r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
        r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
        r_out[3, :] = [0, 0, 0, 1]

        r_in[0, :] = [1, 0, 0, 0]
        r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
        r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
        r_in[3, :] = [0, 0, 0, 1]

        r_out_sol[0, :] = [1, 0, 0, 0]
        r_out_sol[1, :] = [0, np.cos(2 * alpha_out_sol), -np.sin(2 * alpha_out_sol), 0]
        r_out_sol[2, :] = [0, np.sin(2 * alpha_out_sol), np.cos(2 * alpha_out_sol), 0]
        r_out_sol[3, :] = [0, 0, 0, 1]

        r_in_sol[0, :] = [1, 0, 0, 0]
        r_in_sol[1, :] = [0, np.cos(2 * alpha_in_sol), -np.sin(2 * alpha_in_sol), 0]
        r_in_sol[2, :] = [0, np.sin(2 * alpha_in_sol), np.cos(2 * alpha_in_sol), 0]
        r_in_sol[3, :] = [0, 0, 0, 1]

        # matriz reduzida para Rayleigh scattering
        m[0, :] = [1, -np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 0, 0]
        m[1, :] = [-np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 1, 0, 0]
        m[2, :] = [0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2), 0]
        m[3, :] = [0, 0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2)]

        m_sol[0, :] = [1, -np.sin(gamma_sol) ** 2 / (1 + np.cos(gamma_sol) ** 2), 0, 0]
        m_sol[1, :] = [-np.sin(gamma_sol) ** 2 / (1 + np.cos(gamma_sol) ** 2), 1, 0, 0]
        m_sol[2, :] = [0, 0, 2 * np.cos(gamma_sol) / (1 + np.cos(gamma_sol) ** 2), 0]
        m_sol[3, :] = [0, 0, 0, 2 * np.cos(gamma_sol) / (1 + np.cos(gamma_sol) ** 2)]

        # matriz de fase
        # P = R_out * M * R_in
        r_out = np.dot(r_out, m)
        r_out_sol = np.dot(r_out_sol, m_sol)

        p = np.dot(r_out, r_in)
        p_sol = np.dot(r_out_sol, r_in_sol)

        # parâmetros de Stokes
        # S_out = [I,Q,U,V]^T
        I = np.dot(p[0, :], s_in)
        Q = np.dot(p[1, :], s_in)
        U = np.dot(p[2, :], s_in)
        # V = np.dot(p[3, :], s_in)

        I_sol = np.dot(p_sol[0, :], s_in)
        Q_sol = np.dot(p_sol[1, :], s_in)
        U_sol = np.dot(p_sol[2, :], s_in)
        # V_sol = np.dot(p_sol[3, :], s_in)

        alt = 90 - 180 / np.pi * cSOL1[i]

        if alt < -18:
            DOP[i] = par_wave[i] * np.sqrt(Q ** 2 + U ** 2) / I

        if alt >= -18:
            I = I * par * alb[i] + I_sol * (1 - par * alb[i])
            Q = Q * par * alb[i] + Q_sol * (1 - par * alb[i])
            U = U * par * alb[i] + U_sol * (1 - par * alb[i])

            DOP[i] = par_wave[i] * np.sqrt(Q ** 2 + U ** 2) / I

    return DOP


def func_wav_DOP(allvars, c):
    crds1, crds2, cLUA1, cLUA2, banda = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4))
        r_out = np.zeros((4, 4))

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4))

        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha = rotation_angles(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha_IN = -alpha[0]
        alpha_OUT = np.pi - alpha[1]

        # matriz de rotação
        r_out[0, :] = [1, 0, 0, 0]
        r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
        r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
        r_out[3, :] = [0, 0, 0, 1]

        r_in[0, :] = [1, 0, 0, 0]
        r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
        r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
        r_in[3, :] = [0, 0, 0, 1]

        # matriz reduzida para Rayleigh scattering
        m[0, :] = [1, -np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 0, 0]
        m[1, :] = [-np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 1, 0, 0]
        m[2, :] = [0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2), 0]
        m[3, :] = [0, 0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2)]

        # matriz de fase
        # P = R_out * M * R_in
        r_out = np.dot(r_out, m)

        p = np.dot(r_out, r_in)

        # parâmetros de Stokes
        # S_out = [I,Q,U,V]^T
        I = np.dot(p[0, :], s_in)
        Q = np.dot(p[1, :], s_in)
        U = np.dot(p[2, :], s_in)
        # V = np.dot(p[3, :], s_in)

        DOP[i] = banda[i] ** c * np.sqrt(Q ** 2 + U ** 2) / I

    return DOP


def func_wav(allvars, c):
    crds1, crds2, cLUA1, cLUA2, banda, N = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4))
        r_out = np.zeros((4, 4))

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4))

        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha = rotation_angles(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha_IN = -alpha[0]
        alpha_OUT = np.pi - alpha[1]

        # matriz de rotação
        r_out[0, :] = [1, 0, 0, 0]
        r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
        r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
        r_out[3, :] = [0, 0, 0, 1]

        r_in[0, :] = [1, 0, 0, 0]
        r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
        r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
        r_in[3, :] = [0, 0, 0, 1]

        # matriz reduzida para Rayleigh scattering
        m[0, :] = [1, -np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 0, 0]
        m[1, :] = [-np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 1, 0, 0]
        m[2, :] = [0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2), 0]
        m[3, :] = [0, 0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2)]

        # matriz de fase
        # P = R_out * M * R_in
        r_out = np.dot(r_out, m)

        p = np.dot(r_out, r_in)

        # parâmetros de Stokes
        # S_out = [I,Q,U,V]^T
        I = np.dot(p[0, :], s_in)
        Q = np.dot(p[1, :], s_in)
        U = np.dot(p[2, :], s_in)
        # V = np.dot(p[3, :], s_in)

        DOP[i] = (np.cos(cLUA1[i]) ** (1 / N[i])) * banda[i] ** c * np.sqrt(Q ** 2 + U ** 2) / I

    return DOP


def func_simple_reg_DOP(theta_field, phi_field, theta_lua, phi_lua, par):
    # vector de Stokes para a luz natural
    s_in = np.array([1, 0, 0, 0], dtype=float)

    # matriz de rotação
    r_in = np.zeros((4, 4))
    r_out = np.zeros((4, 4))

    # matriz reduzida para o  Rayleigh scattering
    m = np.zeros((4, 4))

    gamma = func_gamma(theta_field, phi_field, theta_lua, phi_lua)

    # alpha = rotation_angles(theta_field, phi_field, theta_lua, phi_lua)

    alpha = rotation_angles(theta_field, phi_field, theta_lua, phi_lua)

    alpha_IN = -alpha[0]
    alpha_OUT = np.pi - alpha[1]

    # matriz de rotação
    r_out[0, :] = [1, 0, 0, 0]
    r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
    r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
    r_out[3, :] = [0, 0, 0, 1]

    r_in[0, :] = [1, 0, 0, 0]
    r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
    r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
    r_in[3, :] = [0, 0, 0, 1]

    # matriz reduzida para Rayleigh scattering
    m[0, :] = [1, -np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 0, 0]
    m[1, :] = [-np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 1, 0, 0]
    m[2, :] = [0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2), 0]
    m[3, :] = [0, 0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2)]

    # matriz de fase
    # P = R_out * M * R_in
    r_out = np.dot(r_out, m)

    p = np.dot(r_out, r_in)

    # parâmetros de Stokes
    # S_out = [I,Q,U,V]^T
    I = np.dot(p[0, :], s_in)
    Q = np.dot(p[1, :], s_in)
    U = np.dot(p[2, :], s_in)
    # V = np.dot(p[3, :], s_in)

    DOP = par * np.sqrt(Q ** 2 + U ** 2) / I

    return DOP


def func_reg_Q(allvars, par):
    crds1, crds2, cLUA1, cLUA2 = allvars
    Q = np.zeros(len(cLUA1), dtype=float)
    Q[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4))
        r_out = np.zeros((4, 4))

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4))

        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha = rotation_angles(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha_IN = - alpha[0]
        alpha_OUT = np.pi - alpha[1]

        # matriz de rotação
        r_out[0, :] = [1, 0, 0, 0]
        r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
        r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
        r_out[3, :] = [0, 0, 0, 1]

        r_in[0, :] = [1, 0, 0, 0]
        r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
        r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
        r_in[3, :] = [0, 0, 0, 1]

        # matriz reduzida para Rayleigh scattering
        m[0, :] = [1, -np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 0, 0]
        m[1, :] = [-np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 1, 0, 0]
        m[2, :] = [0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2), 0]
        m[3, :] = [0, 0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2)]

        # matriz de fase
        # P = R_out * M * R_in
        r_out = np.dot(r_out, m)

        p = np.dot(r_out, r_in)

        # parâmetros de Stokes
        # S_out = [I,Q,U,V]^T
        # I = np.dot(p[0, :], s_in)
        Q[i] = np.dot(p[1, :], s_in) * par
        # U = np.dot(p[2, :], s_in)
        # V = np.dot(p[3, :], s_in)

    return Q


def func_reg_U(allvars, par):
    crds1, crds2, cLUA1, cLUA2 = allvars
    U = np.zeros(len(cLUA1), dtype=float)
    U[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4))
        r_out = np.zeros((4, 4))

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4))

        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha = rotation_angles(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha_IN = - alpha[0]
        alpha_OUT = np.pi - alpha[1]

        # matriz de rotação
        r_out[0, :] = [1, 0, 0, 0]
        r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
        r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
        r_out[3, :] = [0, 0, 0, 1]

        r_in[0, :] = [1, 0, 0, 0]
        r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
        r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
        r_in[3, :] = [0, 0, 0, 1]

        # matriz reduzida para Rayleigh scattering
        m[0, :] = [1, -np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 0, 0]
        m[1, :] = [-np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 1, 0, 0]
        m[2, :] = [0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2), 0]
        m[3, :] = [0, 0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2)]

        # matriz de fase
        # P = R_out * M * R_in
        r_out = np.dot(r_out, m)

        p = np.dot(r_out, r_in)

        # parâmetros de Stokes
        # S_out = [I,Q,U,V]^T
        # I = np.dot(p[0, :], s_in)
        # Q = np.dot(p[1, :], s_in)
        U[i] = np.dot(p[2, :], s_in) * par
        # V = np.dot(p[3, :], s_in)

    return U


def func_reg_AOP(allvars, par):
    crds1, crds2, cLUA1, cLUA2 = allvars
    AOP = np.zeros(len(cLUA1), dtype=float)
    AOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4))
        r_out = np.zeros((4, 4))

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4))

        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha = rotation_angles(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha_IN = - alpha[0]
        alpha_OUT = np.pi - alpha[1]

        # matriz de rotação
        r_out[0, :] = [1, 0, 0, 0]
        r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
        r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
        r_out[3, :] = [0, 0, 0, 1]

        r_in[0, :] = [1, 0, 0, 0]
        r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
        r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
        r_in[3, :] = [0, 0, 0, 1]

        # matriz reduzida para Rayleigh scattering
        m[0, :] = [1, -np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 0, 0]
        m[1, :] = [-np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 1, 0, 0]
        m[2, :] = [0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2), 0]
        m[3, :] = [0, 0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2)]

        # matriz de fase
        # P = R_out * M * R_in
        r_out = np.dot(r_out, m)

        p = np.dot(r_out, r_in)

        # parâmetros de Stokes
        # S_out = [I,Q,U,V]^T
        # I = np.dot(p[0, :], s_in)
        Q = np.dot(p[1, :], s_in)
        U = np.dot(p[2, :], s_in)
        # V = np.dot(p[3, :], s_in)

        AOP[i] = par * 0.5 * np.arctan(U / Q) * 180 / np.pi

    return AOP


def func_wav_Q(allvars, c):
    crds1, crds2, cLUA1, cLUA2, banda, par_int = allvars
    Q = np.zeros(len(cLUA1), dtype=float)
    Q[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4))
        r_out = np.zeros((4, 4))

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4))

        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha = rotation_angles(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha_IN = - alpha[0]
        alpha_OUT = np.pi - alpha[1]

        # matriz de rotação
        r_out[0, :] = [1, 0, 0, 0]
        r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
        r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
        r_out[3, :] = [0, 0, 0, 1]

        r_in[0, :] = [1, 0, 0, 0]
        r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
        r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
        r_in[3, :] = [0, 0, 0, 1]

        # matriz reduzida para Rayleigh scattering
        m[0, :] = [1, -np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 0, 0]
        m[1, :] = [-np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 1, 0, 0]
        m[2, :] = [0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2), 0]
        m[3, :] = [0, 0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2)]

        # matriz de fase
        # P = R_out * M * R_in
        r_out = np.dot(r_out, m)

        p = np.dot(r_out, r_in)

        # parâmetros de Stokes
        # S_out = [I,Q,U,V]^T
        # I = np.dot(p[0, :], s_in)
        Q[i] = np.dot(p[1, :], s_in) * par_int[i] * banda[i] ** c
        # U = np.dot(p[2, :], s_in)
        # V = np.dot(p[3, :], s_in)

    return Q


def func_wav_U(allvars, c):
    crds1, crds2, cLUA1, cLUA2, banda, par_int = allvars
    U = np.zeros(len(cLUA1), dtype=float)
    U[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4))
        r_out = np.zeros((4, 4))

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4))

        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha = rotation_angles(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha_IN = - alpha[0]
        alpha_OUT = np.pi - alpha[1]

        # matriz de rotação
        r_out[0, :] = [1, 0, 0, 0]
        r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
        r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
        r_out[3, :] = [0, 0, 0, 1]

        r_in[0, :] = [1, 0, 0, 0]
        r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
        r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
        r_in[3, :] = [0, 0, 0, 1]

        # matriz reduzida para Rayleigh scattering
        m[0, :] = [1, -np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 0, 0]
        m[1, :] = [-np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 1, 0, 0]
        m[2, :] = [0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2), 0]
        m[3, :] = [0, 0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2)]

        # matriz de fase
        # P = R_out * M * R_in
        r_out = np.dot(r_out, m)

        p = np.dot(r_out, r_in)

        # parâmetros de Stokes
        # S_out = [I,Q,U,V]^T
        # I = np.dot(p[0, :], s_in)
        # Q = np.dot(p[1, :], s_in)
        U[i] = np.dot(p[2, :], s_in) * par_int[i] * banda[i] ** c
        # V = np.dot(p[3, :], s_in)

    return U


def func_wav_AOP(allvars, c):
    crds1, crds2, cLUA1, cLUA2, banda, par_int = allvars
    AOP = np.zeros(len(cLUA1), dtype=float)
    AOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4))
        r_out = np.zeros((4, 4))

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4))

        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha = rotation_angles(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        alpha_IN = - alpha[0]
        alpha_OUT = np.pi - alpha[1]

        # matriz de rotação
        r_out[0, :] = [1, 0, 0, 0]
        r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
        r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
        r_out[3, :] = [0, 0, 0, 1]

        r_in[0, :] = [1, 0, 0, 0]
        r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
        r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
        r_in[3, :] = [0, 0, 0, 1]

        # matriz reduzida para Rayleigh scattering
        m[0, :] = [1, -np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 0, 0]
        m[1, :] = [-np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 1, 0, 0]
        m[2, :] = [0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2), 0]
        m[3, :] = [0, 0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2)]

        # matriz de fase
        # P = R_out * M * R_in
        r_out = np.dot(r_out, m)

        p = np.dot(r_out, r_in)

        # parâmetros de Stokes
        # S_out = [I,Q,U,V]^T
        # I = np.dot(p[0, :], s_in)
        Q = np.dot(p[1, :], s_in)
        U = np.dot(p[2, :], s_in)
        # V = np.dot(p[3, :], s_in)

        AOP[i] = par_int[i] * 0.5 * np.arctan(U / Q) * banda[i] ** c

    return AOP


def map_observations(band, condition, Par1, Par2, P1='individual', P2='Regular'):
    url = 'https://drive.google.com/file/d/15FlGlJBC-c3Wh1e8bEv6fE8nycqUxebc/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
    df = pd.read_csv(path, sep=';')

    DATA = df[df['BAND'] == band]
    DATA = DATA[DATA['CONDITION'] == condition]

    field = DATA['FIELD'].to_numpy()
    RA = DATA['RA'].to_numpy()
    DEC = DATA['DEC'].to_numpy()
    # BEGIN = DATA['BEGIN OBS'].to_numpy()
    # END = DATA['END OBS'].to_numpy()
    MED = DATA['MED OBS'].to_numpy()
    # Ival = DATA['I'].to_numpy()
    Qval = DATA['Q'].to_numpy()
    errQval = DATA['error Q'].to_numpy()
    Uval = DATA['U'].to_numpy()
    errUval = DATA['error U '].to_numpy()
    seen = DATA['SEEING'].to_numpy()

    wave = 1

    if band == 'B':
        wave = 437
    if band == 'V':
        wave = 555
    if band == 'R':
        wave = 655
    if band == 'I':
        wave = 768

    a = 1
    b = 1
    c = 1
    d = 0
    e = 1

    par = 1

    k = 0

    for t in MED:

        C1field = []
        C2field = []

        DOP_field = []

        POL_OBS = []
        errPOL_OBS = []

        LUA = coordinates.truemoon(t)

        alt_lua = LUA[0].value
        az_lua = LUA[1].value
        phase = LUA[2].value

        t_lua = np.pi / 2 - alt_lua * np.pi / 180.0
        phi_lua = az_lua * np.pi / 180.0

        # SOL = coordinates.true_sun(t)

        # alt_sol = SOL[0].value
        # az_sol = SOL[1].value

        # t_sol = np.pi / 2 - alt_sol * np.pi / 180.0
        # phi_sol = az_sol * np.pi / 180.0

        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4))
        r_out = np.zeros((4, 4))

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4))

        A = 1

        if P1 == 'individual':
            if P2 == 'Regular':
                par = Par1[0]

                a = 1
                b = 1
                c = 1
                e = 1

            if P2 == 'Mix':
                N = Par1[0]
                k1 = Par1[1]
                k2 = Par1[2]

                b = np.exp(-seen[k] / k1 + k2)
                a = np.cos(t_lua) ** (1 / N)
                c = 1
                e = 1

            if P2 == 'Depolarization':
                P = Par1[0]

                a = 1
                b = 1
                c = (1 + P) / (1 - P)
                A = (3 / 2) * ((1 - P) / (1 + P / 2))
                e = ((1 - 2 * P) / (1 - P))

            if P2 == 'Wave':
                n = Par1[0]

                par = wave ** n

                a = 1
                b = 1
                c = 1

        if P1 == 'complex':
            if P2 == 'horizon fit':
                par = Par1[0]
                N = Par2[0]

                a = np.cos(t_lua) ** (1 / N)
                b = 1
                c = 1

            if P2 == 'depolarization fit':
                par = Par1[0]
                P = Par2[0]

                a = 1
                b = 1
                c = (1 + P) / (1 - P)
                A = (3 / 2) * ((1 - P) / (1 + P / 2))
                e = ((1 - 2 * P) / (1 - P))

            if P2 == 'fit seeing':
                par = Par1[0]
                k1 = Par2[0]
                k2 = Par2[1]

                a = 1
                b = np.exp(-seen[k] / k1 + k2)
                c = 1

            if P2 == 'fit wavelength':
                N = Par1[0]
                k1 = Par1[1]
                k2 = Par1[2]
                n = Par2[0]

                par = wave ** n
                b = np.exp(-seen[k] / k1 + k2)
                a = np.cos(t_lua) ** (1 / N)
                c = 1

        for n in range(0, len(RA)):
            # vector de Stokes para a luz natural
            s_in_field = np.array([1, 0, 0, 0], dtype=float)

            # matriz de rotação
            r_in_field = np.zeros((4, 4))
            r_out_field = np.zeros((4, 4))

            # matriz reduzida para o  Rayleigh scattering
            m_field = np.zeros((4, 4))

            altaz = coordinates.coord_radectoaltaz(float(RA[n]), float(DEC[n]), t)

            alt = altaz[0].value
            az = altaz[1].value

            tfield = np.pi / 2 - alt * np.pi / 180.0
            phifield = az * np.pi / 180.0

            C1field.append(float(tfield))
            C2field.append(float(phifield))

            gamma = func_gamma(tfield, phifield, t_lua, phi_lua)

            alpha_field = rotation_angles(tfield, phifield, t_lua, phi_lua)

            alpha_in_field = alpha_field[0]
            alpha_out_field = alpha_field[1]

            # matriz de rotação
            r_out_field[0, :] = [1, 0, 0, 0]
            r_out_field[1, :] = [0, np.cos(2 * alpha_out_field), -np.sin(2 * alpha_out_field), 0]
            r_out_field[2, :] = [0, np.sin(2 * alpha_out_field), np.cos(2 * alpha_out_field), 0]
            r_out_field[3, :] = [0, 0, 0, 1]

            r_in_field[0, :] = [1, 0, 0, 0]
            r_in_field[1, :] = [0, np.cos(2 * alpha_in_field), -np.sin(2 * alpha_in_field), 0]
            r_in_field[2, :] = [0, np.sin(2 * alpha_in_field), np.cos(2 * alpha_in_field), 0]
            r_in_field[3, :] = [0, 0, 0, 1]

            # matriz reduzida para Rayleigh scattering
            m_field[0, :] = [1, -np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 0, 0]
            m_field[1, :] = [-np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 1, 0, 0]
            m_field[2, :] = [0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2), 0]
            m_field[3, :] = [0, 0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2)]

            if P2 == 'depolarizationn fit' or P2 == 'Depolarization':

                m[0, :] = [0.5 * A * (c + np.cos(gamma) ** 2), -0.5 * A * (np.sin(gamma) ** 2), 0, 0]
                m[1, :] = [-0.5 * A * np.sin(gamma) ** 2, 0.5 * A * (1 + np.cos(gamma) ** 2), 0, 0]
                m[2, :] = [0, 0, A * np.cos(gamma), 0]
                m[3, :] = [0, 0, 0, e * A * np.cos(gamma)]

            # matriz de fase
            # P = R_out * M * R_in
            r_out_field = np.dot(r_out_field, m_field)

            p = np.dot(r_out_field, r_in_field)

            # parâmetros de Stokes
            # S_out = [I,Q,U,V]^T
            I = np.dot(p[0, :], s_in_field)
            Q = np.dot(p[1, :], s_in_field)
            U = np.dot(p[2, :], s_in_field)
            # V = np.dot(p[3, :], s_in)

            DOP_field.append(a * b * par * np.sqrt(Q ** 2 + U ** 2) / I)

            pol = np.sqrt(float(Qval[n]) ** 2 + float(Uval[n]) ** 2)
            POL_OBS.append(pol)
            errPOL_OBS.append(np.sqrt(
                float(Qval[n]) ** 2 * float(errQval[n]) ** 2 + float(Uval[n]) ** 2 * float(errUval[n]) ** 2) / pol)

        theta_sky = np.zeros((100, 400))
        phi_sky = np.zeros((100, 400))

        dop = np.zeros((100, 400))

        my_cmap_r = reverse_colourmap(mpl.cm.Spectral)

        i, j = 0, 0

        for Eo in np.linspace(0, np.pi / 2, 100, endpoint=True):
            for Azo in np.linspace(0, 2 * np.pi, 400, endpoint=True):
                to = np.pi / 2 - Eo
                phio = Azo

                theta_sky[i, j] = to * 180 / np.pi
                phi_sky[i, j] = phio

                gamma = func_gamma(to, phio, t_lua, phi_lua)

                alpha = rotation_angles(to, phio, t_lua, phi_lua)

                alpha_IN = alpha[0]
                alpha_OUT = alpha[1]

                # matriz de rotação
                r_out[0, :] = [1, 0, 0, 0]
                r_out[1, :] = [0, np.cos(2 * alpha_OUT), -np.sin(2 * alpha_OUT), 0]
                r_out[2, :] = [0, np.sin(2 * alpha_OUT), np.cos(2 * alpha_OUT), 0]
                r_out[3, :] = [0, 0, 0, 1]

                r_in[0, :] = [1, 0, 0, 0]
                r_in[1, :] = [0, np.cos(2 * alpha_IN), -np.sin(2 * alpha_IN), 0]
                r_in[2, :] = [0, np.sin(2 * alpha_IN), np.cos(2 * alpha_IN), 0]
                r_in[3, :] = [0, 0, 0, 1]

                # matriz reduzida para Rayleigh scattering
                m[0, :] = [1, -np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 0, 0]
                m[1, :] = [-np.sin(gamma) ** 2 / (1 + np.cos(gamma) ** 2), 1, 0, 0]
                m[2, :] = [0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2), 0]
                m[3, :] = [0, 0, 0, 2 * np.cos(gamma) / (1 + np.cos(gamma) ** 2)]

                if P2 == 'depolarizationn fit' or P2 == 'Depolarization':
                    m[0, :] = [0.5 * A * (c + np.cos(gamma) ** 2), -0.5 * A * (np.sin(gamma) ** 2), 0, 0]
                    m[1, :] = [-0.5 * A * np.sin(gamma) ** 2, 0.5 * A * (1 + np.cos(gamma) ** 2), 0, 0]
                    m[2, :] = [0, 0, A * np.cos(gamma), 0]
                    m[3, :] = [0, 0, 0, e * A * np.cos(gamma)]

                # matriz de fase
                # P = R_out * M * R_in
                r_out = np.dot(r_out, m)

                p = np.dot(r_out, r_in)

                # parâmetros de Stokes
                # S_out = [I,Q,U,V]^T
                I = np.dot(p[0, :], s_in)
                Q = np.dot(p[1, :], s_in)
                U = np.dot(p[2, :], s_in)
                # V = np.dot(p[3, :], s_in)

                dop[i, j] = a * b * par * np.sqrt(Q ** 2 + U ** 2) / I

                j += 1

            i += 1
            j = 0

        degree_sign = u'\N{DEGREE SIGN}'
        r_labels = [
            '90' + degree_sign,
            '',
            '60' + degree_sign,
            '',
            '30' + degree_sign,
            '',
            '0' + degree_sign + ' Alt.',
        ]

        az_label_offset = 0.0 * u.deg
        theta_labels = []
        for chunk in range(0, 7):
            label_angle = (az_label_offset * (1 / u.deg)) + (chunk * 45.0)
            while label_angle >= 360.0:
                label_angle -= 360.0
            if chunk == 0:
                theta_labels.append('N ' + '\n' + str(label_angle) + degree_sign + ' Az')
            elif chunk == 2:
                theta_labels.append('E' + '\n' + str(label_angle) + degree_sign)
            elif chunk == 4:
                theta_labels.append('S' + '\n' + str(label_angle) + degree_sign)
            elif chunk == 6:
                theta_labels.append('W' + '\n' + str(label_angle) + degree_sign)
            else:
                theta_labels.append(str(label_angle) + degree_sign)
        theta_labels.append('')

        fig = plt.figure(figsize=(7, 5))
        plt.clf()
        ax = fig.gca(projection='polar')
        ax.set_theta_zero_location('N')
        ax.set_rlim(1, 91)
        plt.pcolormesh(phi_sky, theta_sky, dop, vmax=1, vmin=0, cmap=my_cmap_r)
        if 0 <= t_lua <= np.pi / 2:
            ax.plot(phi_lua, t_lua * 180 / np.pi, 'o', color='blue')
            ax.annotate('moon phase: %.3f' % round(phase, 3), xy=(phi_lua, t_lua * 180 / np.pi), xycoords='data',
                        xytext=(0.1, 1), textcoords="axes fraction", size=6, bbox=dict(boxstyle="round", fc="0.8"),
                        ha='right', arrowprops=dict(arrowstyle="->", shrinkA=0, shrinkB=10,
                                                    connectionstyle="angle3,angleA=90,angleB=0"))
        for y in range(0, len(RA)):
            if 0 <= C1field[y] <= np.pi / 2:
                if t == MED[y]:
                    ax.plot(C2field[y], C1field[y] * 180 / np.pi, 'o', color='grey')
                    ax.annotate('observed field ' + field[y] + ': banda ' + str(band) + '\n DoP: ' + str(
                        round(POL_OBS[y], 3)) + ' $ \pm $ ' + str(round(errPOL_OBS[y], 3)) + '\n DoP sim: ' + str(
                        round(DOP_field[y], 3)), xy=(C2field[y], C1field[y] * 180 / np.pi), xycoords='data',
                                xytext=(0.1, 0.1), textcoords="axes fraction", size=6,
                                bbox=dict(boxstyle="round", fc="0.8"), ha='right',
                                arrowprops=dict(arrowstyle="->", shrinkA=0, shrinkB=10,
                                                connectionstyle="angle3,angleA=90,angleB=0"))
                    ax.annotate(field[y], xy=(C2field[y], C1field[y] * 180 / np.pi), xycoords='data')
                else:
                    ax.plot(C2field[y], C1field[y] * 180 / np.pi, 'x', color='black')
                    ax.annotate(field[y], xy=(C2field[y], C1field[y] * 180 / np.pi), xycoords='data')
        plt.title(str(t), loc='right', fontsize=8, color='black')
        ax.grid(True, which='major')
        ax.set_rgrids(range(1, 106, 15), r_labels, angle=-45)
        ax.set_thetagrids(range(0, 360, 45), theta_labels)
        plt.colorbar()
        nome = 'degree_of_polarization_moon_sim_stokes_' + str(band) + '_' + str(d) + '.png'
        plt.savefig(nome)
        ax.figure.canvas.draw()
        plt.show()

        del C1field
        del C2field
        del DOP_field

        d += 1


def FIT(band=None, condition=None, command='ALL'):
    # url = 'https://drive.google.com/file/d/15FlGlJBC-c3Wh1e8bEv6fE8nycqUxebc/view?usp=sharing'
    # path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
    global conjunto, cor_line, par_wave, cor, lab_wave, cor_unc, V_par_aop, V_par_q, V_par_u, B_par_u, B_par_q, B_par_aop, R_par_aop, R_par_q, R_par_u, I_par_aop, I_par_q, I_par_u, Rpar, n_par, barra, B_par_reg, B_par_hor, V_par_reg, V_par_hor, R_par_reg, R_par_hor, I_par_reg, I_par_hor, B_par_dep, V_par_dep, R_par_dep, I_par_dep
    df = pd.read_csv('data.csv', sep=';')

    df.isnull().sum()
    df.dropna(axis=1)

    COND = ['ok', 'clouds', 'sun']
    CONDT = ['ok', 'clouds', 'sun']

    DATA = df

    if condition is None:
        CONDT = DATA['CONDITION'].to_numpy()
        COND = ['ok', 'clouds', 'sun']

    if condition is not None:
        COND = condition
        if isinstance(condition, str):
            DATA = DATA[DATA['CONDITION'] == condition]
            CONDT = DATA['CONDITION'].to_numpy()
        else:
            DATA = DATA[DATA['CONDITION'].isin(condition)]
            CONDT = DATA['CONDITION'].to_numpy()

    BANDA = ['B', 'V', 'R', 'I']
    BAN = ['B', 'V', 'R', 'I']

    if band is None:
        BANDA = DATA['BAND'].to_numpy()
        conjunto = ['B', 'V', 'R', 'I']

    if band is not None and condition is not None:
        conjunto = band
        if isinstance(band, str):
            DATA = DATA[DATA['BAND'] == band]
            BANDA = DATA['BAND'].to_numpy()
        else:
            DATA = DATA[DATA['BAND'].isin(band)]
            BANDA = DATA['BAND'].to_numpy()

    LABEL = listToString(conjunto)
    LAB = listToString(COND)

    field = DATA['FIELD'].to_numpy()
    RA = DATA['RA'].to_numpy()
    DEC = DATA['DEC'].to_numpy()
    MED = DATA['MED OBS'].to_numpy()
    Ival = DATA['I'].to_numpy()
    Qval = DATA['Q'].to_numpy()
    errQval = DATA['error Q'].to_numpy()
    Uval = DATA['U'].to_numpy()
    errUval = DATA['error U '].to_numpy()
    seen = DATA['SEEING'].to_numpy()

    total = len(RA)
    fit_observations_resume = pd.DataFrame(
        {'FIELD': field, 'RA': RA, 'DEC': DEC, 'OB TIME MED': MED, 'I': Ival, 'Q': Qval, 'error Q': errQval, 'U': Uval,
         'error U': errUval, 'SEEING': seen})

    C1field = []
    C2field = []
    C1lua = []
    C2lua = []
    C1sol = []
    C2sol = []
    GAMMA = []
    GAMMA_SOL = []
    Q_OBS = []
    errQ_OBS = []
    U_OBS = []
    errU_OBS = []
    I_OBS = []
    POL_OBS = []
    errPOL_OBS = []
    ALBEDO = []
    AOP = []
    errAOP = []
    SEEING = []
    BAND = []
    WAV = []
    AOP_BAND = []
    Q_BAND = []
    U_BAND = []
    N_BAND = []

    label = []

    aop_par = 1
    q_par = 1
    u_par = 1

    if command == 'ALL' or command == 'individuals depolarization':
        B_par_dep, B_chisqr_dep, B_bic_dep, result_data_B_dep = field_functions.fit_base('B', condition, command='dep_stokes')
        V_par_dep, V_chisqr_dep, V_bic_dep, result_data_V_dep = field_functions.fit_base('V', condition, command='dep_stokes')
        R_par_dep, R_chisqr_dep, R_bic_dep, result_data_R_dep = field_functions.fit_base('R', condition, command='dep_stokes')
        I_par_dep, I_chisqr_dep, I_bic_dep, result_data_I_dep = field_functions.fit_base('I', condition, command='dep_stokes')

        column_names = ['FIELD', 'BAND', 'CONDITION', 'GAMMA', 'POL', 'ERROR POL', 'FIT IND', 'FIT IND UNC',
                        'FIT IND DIFF']

        result_data_dep = pd.DataFrame(columns=column_names)
        result_data_dep = pd.concat([result_data_dep, result_data_B_dep])
        result_data_dep = pd.concat([result_data_dep, result_data_V_dep])
        result_data_dep = pd.concat([result_data_dep, result_data_R_dep])
        result_data_dep = pd.concat([result_data_dep, result_data_I_dep])

        # field_functions.plot_all_bands(B_par, V_par, R_par, I_par, condition, command='regular_ray')
        field_functions.plot_all(result_data_dep, command='dep_stokes')

        par_ind_dep = [B_par_dep[0], V_par_dep[0], R_par_dep[0], I_par_dep[0]]
        chisqr_ind_dep = [B_chisqr_dep, V_chisqr_dep, R_chisqr_dep, I_chisqr_dep]
        bic_ind_dep = [B_bic_dep, V_bic_dep, R_bic_dep, I_bic_dep]
        index = ['B', 'V', 'R', 'I']

        df_ind_x = pd.DataFrame({'$\u03C1$': par_ind_dep, 'Chi-square': chisqr_ind_dep, 'BIC': bic_ind_dep}, index=index)

        # df_ind.plot(figsize=(10, 5))
        df_ind_x.plot.bar(rot=0, subplots=True)
        plt.savefig('indvidual_stokes_dop_dep_fit_separate_parameters.png')
        plt.pause(2)
        plt.close()

        if command == 'individuals depolarization':
            result_data_dep.to_csv("output.csv")
            exit(df_ind_x)

    if command == 'ALL' or command == 'individuals regular' or command == 'depolarization fit' or command == 'fit seeing' or command == 'horizon fit' or command == 'fit Sun' or command == 'fit all':
        B_par_reg, B_chisqr_reg, B_bic_reg, result_data_B_reg = field_functions.fit_base('B', condition, command='regular_stokes')
        V_par_reg, V_chisqr_reg, V_bic_reg, result_data_V_reg = field_functions.fit_base('V', condition, command='regular_stokes')
        R_par_reg, R_chisqr_reg, R_bic_reg, result_data_R_reg = field_functions.fit_base('R', condition, command='regular_stokes')
        I_par_reg, I_chisqr_reg, I_bic_reg, result_data_I_reg = field_functions.fit_base('I', condition, command='regular_stokes')

        column_names = ['FIELD', 'BAND', 'CONDITION', 'GAMMA', 'POL', 'ERROR POL', 'FIT IND', 'FIT IND UNC',
                        'FIT IND DIFF']

        result_data_reg = pd.DataFrame(columns=column_names)
        result_data_reg = pd.concat([result_data_reg, result_data_B_reg])
        result_data_reg = pd.concat([result_data_reg, result_data_V_reg])
        result_data_reg = pd.concat([result_data_reg, result_data_R_reg])
        result_data_reg = pd.concat([result_data_reg, result_data_I_reg])

        # field_functions.plot_all_bands(B_par, V_par, R_par, I_par, condition, command='regular_ray')
        field_functions.plot_all(result_data_reg, command='regular_stokes')

        par_ind_reg = [B_par_reg[0], V_par_reg[0], R_par_reg[0], I_par_reg[0]]
        chisqr_ind_reg = [B_chisqr_reg, V_chisqr_reg, R_chisqr_reg, I_chisqr_reg]
        bic_ind_reg = [B_bic_reg, V_bic_reg, R_bic_reg, I_bic_reg]
        index = ['B', 'V', 'R', 'I']
        df_ind_x_reg = pd.DataFrame({'A': par_ind_reg, 'Chi-square': chisqr_ind_reg, 'BIC': bic_ind_reg}, index=index)

        # df_ind.plot(figsize=(10, 5))
        df_ind_x_reg.plot.bar(rot=0, subplots=True)
        plt.savefig('indvidual_stokes_dop_reg_fit_separate_parameters.png')
        plt.pause(2)
        plt.close()

        if command == 'individuals regular':
            result_data_reg.to_csv("output.csv")
            exit(df_ind_x_reg)

    if command == 'ALL' or command == 'individuals horizon':
        B_par_hor, B_chisqr_hor, B_bic_hor, result_data_B_hor = field_functions.fit_base('B', condition, command='hor_stokes')
        V_par_hor, V_chisqr_hor, V_bic_hor, result_data_V_hor = field_functions.fit_base('V', condition, command='hor_stokes')
        R_par_hor, R_chisqr_hor, R_bic_hor, result_data_R_hor = field_functions.fit_base('R', condition, command='hor_stokes')
        I_par_hor, I_chisqr_hor, I_bic_hor, result_data_I_hor = field_functions.fit_base('I', condition, command='hor_stokes')

        par_ind_hor = [B_par_hor[0], V_par_hor[0], R_par_hor[0], I_par_hor[0]]
        chisqr_ind_hor = [B_chisqr_hor, V_chisqr_hor, R_chisqr_hor, I_chisqr_hor]
        bic_ind_hor = [B_bic_hor, V_bic_hor, R_bic_hor, I_bic_hor]
        index = ['B', 'V', 'R', 'I']
        df_ind_hor = pd.DataFrame({'N': par_ind_hor, 'Chi-square': chisqr_ind_hor, 'BIC': bic_ind_hor}, index=index)

        # df_ind.plot(figsize=(10, 5))
        df_ind_hor.plot.bar(rot=0, subplots=True)
        plt.savefig('indvidual_stokes_dop_hor_fit_separate_parameters.png')
        plt.pause(2)
        plt.close()

        column_names = ['FIELD', 'BAND', 'CONDITION', 'GAMMA', 'POL', 'ERROR POL', 'FIT IND', 'FIT IND UNC',
                        'FIT IND DIFF']

        result_data_hor = pd.DataFrame(columns=column_names)
        result_data_hor = pd.concat([result_data_hor, result_data_B_hor])
        result_data_hor = pd.concat([result_data_hor, result_data_V_hor])
        result_data_hor = pd.concat([result_data_hor, result_data_R_hor])
        result_data_hor = pd.concat([result_data_hor, result_data_I_hor])

        # field_functions.plot_all_bands(B_par, V_par, R_par, I_par, condition, command='regular_ray')
        field_functions.plot_all(result_data_hor, command='hor_stokes')

        if command == 'individuals horizon':
            result_data_hor.to_csv("output.csv")
            exit(df_ind_hor)

    if command != 'ALL' and isinstance(band, str) == False:
        barra = tqdm(total=100, desc='Processing ' + command)
    if isinstance(band, str) and command != 'ALL':
        barra = tqdm(total=127, desc='Processing ' + command)
    if command == 'ALL':
        barra = tqdm(total=500, desc='Processing')

    if search('aop', command):
        B_par_aop, B_chisqr_aop, result_data_B_aop = field_functions.fit_base('B', condition, command='regular_aop')
        V_par_aop, V_chisqr_aop, result_data_V_aop = field_functions.fit_base('V', condition, command='regular_aop')
        R_par_aop, R_chisqr_aop, result_data_R_aop = field_functions.fit_base('R', condition, command='regular_aop')
        I_par_aop, I_chisqr_aop, result_data_I_aop = field_functions.fit_base('I', condition, command='regular_aop')

        par_ind_aop = [B_par_aop[0], V_par_aop[0], R_par_aop[0], I_par_aop[0]]
        # beta1_ind_aop = [B_par_aop[1], V_par_aop[1], R_par_aop[1], I_par_aop[1]]
        # beta2_ind_aop = [B_par_aop[2], V_par_aop[2], R_par_aop[2], I_par_aop[2]]
        chisqr_ind_aop = [B_chisqr_aop, V_chisqr_aop, R_chisqr_aop, I_chisqr_aop]
        index = ['B', 'V', 'R', 'I']
        df_ind_aop = pd.DataFrame({'$P_{norm}$': par_ind_aop, 'Chi-square': chisqr_ind_aop}, index=index)

        # df_ind.plot(figsize=(10, 5))
        df_ind_aop.plot.bar(rot=0)
        plt.savefig('indvidual_stokes_aop_fit_together_parameters.png')
        plt.pause(2)
        plt.close()

        # df_ind.plot(figsize=(10, 5))
        df_ind_aop.plot.bar(rot=0, subplots=True)
        plt.savefig('indvidual_stokes_aop_fit_separate_parameters.png')
        plt.pause(2)
        plt.close()

        for n in range(0, total):
            if band == 'B':
                aop_par = B_par_aop[0]

            if band == 'V':
                aop_par = V_par_aop[0]

            if band == 'R':
                aop_par = R_par_aop[0]

            if band == 'I':
                aop_par = I_par_aop[0]

            if band is None or not isinstance(band, str):
                if BANDA[n] == 'B':
                    aop_par = B_par_aop[0]

                if BANDA[n] == 'V':
                    aop_par = V_par_aop[0]

                if BANDA[n] == 'R':
                    aop_par = R_par_aop[0]

                if BANDA[n] == 'I':
                    aop_par = I_par_aop[0]

            AOP_BAND.append(aop_par)

    if search('Q', command):
        B_par_q, B_chisqr_q, result_data_B_q = field_functions.fit_base('B', condition, command='regular_q')
        V_par_q, V_chisqr_q, result_data_V_q = field_functions.fit_base('V', condition, command='regular_q')
        R_par_q, R_chisqr_q, result_data_R_q = field_functions.fit_base('R', condition, command='regular_q')
        I_par_q, I_chisqr_q, result_data_I_q = field_functions.fit_base('I', condition, command='regular_q')

        par_ind_q = [B_par_q[0], V_par_q[0], R_par_q[0], I_par_q[0]]
        # beta1_ind_q = [B_par_q[1], V_par_q[1], R_par_q[1], I_par_q[1]]
        # beta2_ind_q = [B_par_q[2], V_par_q[2], R_par_q[2], I_par_q[2]]
        chisqr_ind_q = [B_chisqr_q, V_chisqr_q, R_chisqr_q, I_chisqr_q]
        index = ['B', 'V', 'R', 'I']
        df_ind_q = pd.DataFrame({'$P_{norm}$': par_ind_q, 'Chi-square': chisqr_ind_q}, index=index)

        # df_ind.plot(figsize=(10, 5))
        df_ind_q.plot.bar(rot=0)
        plt.savefig('indvidual_stokes_q_fit_together_parameters.png')
        plt.pause(2)
        plt.close()

        # df_ind.plot(figsize=(10, 5))
        df_ind_q.plot.bar(rot=0, subplots=True)
        plt.savefig('indvidual_stokes_q_fit_separate_parameters.png')
        plt.pause(2)
        plt.close()

        for n in range(0, total):
            if band == 'B':
                q_par = B_par_q[0]

            if band == 'V':
                q_par = V_par_q[0]

            if band == 'R':
                q_par = R_par_q[0]

            if band == 'I':
                q_par = I_par_q[0]

            if band is None or not isinstance(band, str):
                if BANDA[n] == 'B':
                    q_par = B_par_q[0]

                if BANDA[n] == 'V':
                    q_par = V_par_q[0]

                if BANDA[n] == 'R':
                    q_par = R_par_q[0]

                if BANDA[n] == 'I':
                    q_par = I_par_q[0]

            Q_BAND.append(q_par)

    if search('U', command):
        B_par_u, B_chisqr_u, result_data_B_u = field_functions.fit_base('B', condition, command='regular_u')
        V_par_u, V_chisqr_u, result_data_V_u = field_functions.fit_base('V', condition, command='regular_u')
        R_par_u, R_chisqr_u, result_data_R_u = field_functions.fit_base('R', condition, command='regular_u')
        I_par_u, I_chisqr_u, result_data_I_u = field_functions.fit_base('I', condition, command='regular_u')

        par_ind_u = [B_par_u[0], V_par_u[0], R_par_u[0], I_par_u[0]]
        # beta1_ind_u = [B_par_u[1], V_par_u[1], R_par_u[1], I_par_u[1]]
        # beta2_ind_u = [B_par_u[2], V_par_u[2], R_par_u[2], I_par_u[2]]
        chisqr_ind_u = [B_chisqr_u, V_chisqr_u, R_chisqr_u, I_chisqr_u]
        index = ['B', 'V', 'R', 'I']
        # df_ind_u = pd.DataFrame({'$P_{norm}$': par_ind_u, '$beta_{1}$': beta1_ind_u, '$beta_{2}$': beta2_ind_u, 'Chi-square': chisqr_ind_u}, index=index)
        df_ind_u = pd.DataFrame({'$P_{norm}$': par_ind_u, 'Chi-square': chisqr_ind_u}, index=index)

        # df_ind.plot(figsize=(10, 5))
        df_ind_u.plot.bar(rot=0)
        plt.savefig('indvidual_stokes_u_fit_together_parameters.png')
        plt.pause(2)
        plt.close()

        # df_ind.plot(figsize=(10, 5))
        df_ind_u.plot.bar(rot=0, subplots=True)
        plt.savefig('indvidual_stokes_u_fit_separate_parameters.png')
        plt.pause(2)
        plt.close()

        for n in range(0, total):
            if band == 'B':
                u_par = B_par_u[0]

            if band == 'V':
                u_par = V_par_u[0]

            if band == 'R':
                u_par = R_par_u[0]

            if band == 'I':
                u_par = I_par_u[0]

            if band is None or not isinstance(band, str):
                if BANDA[n] == 'B':
                    u_par = B_par_u[0]

                if BANDA[n] == 'V':
                    u_par = V_par_u[0]

                if BANDA[n] == 'R':
                    u_par = R_par_u[0]

                if BANDA[n] == 'I':
                    u_par = I_par_u[0]

            U_BAND.append(u_par)

    for n in range(0, total):
        wave = 1
        c_par = 1

        if band == 'B':
            wave = 437
            c_par = B_par_reg[0]
            n_par = 10

        if band == 'V':
            wave = 555
            c_par = V_par_reg[0]
            n_par = 10

        if band == 'R':
            wave = 655
            c_par = R_par_reg[0]
            n_par = 10

        if band == 'I':
            wave = 768
            c_par = I_par_reg[0]
            n_par = 10

        if band is None or not isinstance(band, str):
            if BANDA[n] == 'B':
                wave = 437
                c_par = B_par_reg[0]
                n_par = 10
                # beta1_par = B_par_aop[1]
                # beta2_par = B_par_aop[2]
            if BANDA[n] == 'V':
                wave = 555
                c_par = V_par_reg[0]
                n_par = 10
                # beta1_par = V_par_aop[1]
                # beta2_par = V_par_aop[2]
            if BANDA[n] == 'R':
                wave = 655
                c_par = R_par_reg[0]
                n_par = 10
                # beta1_par = R_par_aop[1]
                # beta2_par = R_par_aop[2]
            if BANDA[n] == 'I':
                wave = 768
                c_par = I_par_reg[0]
                n_par = 10
                # beta1_par = I_par_aop[1]
                # beta2_par = I_par_aop[2]

        observation = moon_functions.Moon(MED[n])
        observation.get_parameters()

        campo = field_functions.Field(BANDA[n], condition, field[n], float(RA[n]), float(DEC[n]))
        campo.get_observation(MED[n])

        observation.true_sun()

        alt_sol = observation.my_sun.alt
        az_sol = observation.my_sun.az

        t_sol = np.pi / 2 - alt_sol * np.pi / 180.0
        phi_sol = az_sol * np.pi / 180.0

        C1lua.append(float(observation.theta))
        C2lua.append(float(observation.phi))
        C1sol.append(float(t_sol))
        C2sol.append(float(phi_sol))
        C1field.append(float(campo.theta))
        C2field.append(float(campo.phi))
        GAMMA.append(campo.gamma * 180 / np.pi)
        GAMMA_SOL.append(campo.func_gamma(t_sol, phi_sol, units='degrees'))
        Q_OBS.append(float(Qval[n]))  # * float(Ival[n]))
        errQ_OBS.append(float(errQval[n]))  # * float(Ival[n]))
        U_OBS.append(float(Uval[n]))  # * float(Ival[n]))
        errU_OBS.append(float(errUval[n]))  # * float(Ival[n]))
        I_OBS.append(float(Ival[n]))
        pol = np.sqrt(float(Qval[n]) ** 2 + float(Uval[n]) ** 2)
        POL_OBS.append(pol)
        errPOL_OBS.append(np.sqrt(
            float(Qval[n]) ** 2 * float(errQval[n]) ** 2 + float(Uval[n]) ** 2 * float(
                errUval[n]) ** 2) / pol)
        alb = observation.plot_and_retrive_albedo(wave)
        ALBEDO.append(alb)
        AOP.append(0.5 * np.arctan(float(Uval[n]) / float(Qval[n])) * 180 / np.pi)
        errAOP.append(0.5 * np.sqrt((Qval[n] * errU_OBS[n]) ** 2 + (Uval[n] * errQ_OBS[n]) ** 2) / (
                (1 + (Uval[n] / Qval[n]) ** 2) * Qval[n] ** 2) * 180 / np.pi)
        SEEING.append(seen[n])
        BAND.append(wave)
        WAV.append(c_par)
        N_BAND.append(n_par)
        barra.update(int(50 / total))
        # Beta1_BAND.append(beta1_par)
        # Beta2_BAND.append(beta2_par)

    # C1sol, C2sol = np.asarray(C1sol, dtype=np.float32), np.asarray(C2sol, dtype=np.float32)
    # Q_OBS, errQ_OBS = np.asarray(Q_OBS, dtype=np.float32), np.asarray(errQ_OBS, dtype=np.float32)
    # U_OBS, errU_OBS = np.asarray(U_OBS, dtype=np.float32), np.asarray(errU_OBS, dtype=np.float32)

    C1field, C2field = np.asarray(C1field, dtype=np.float32), np.asarray(C2field, dtype=np.float32)
    C1lua, C2lua = np.asarray(C1lua, dtype=np.float32), np.asarray(C2lua, dtype=np.float32)
    C1sol, C2sol = np.asarray(C1sol, dtype=np.float32), np.asarray(C2sol, dtype=np.float32)
    POL_OBS, errPOL_OBS = np.asarray(POL_OBS, dtype=np.float32), np.asarray(errPOL_OBS, dtype=np.float32)
    GAMMA, AOP = np.asarray(GAMMA, dtype=np.float32), np.asarray(AOP, dtype=np.float32)
    ALBEDO, SEEING = np.asarray(ALBEDO, dtype=np.float32), np.asarray(SEEING, dtype=np.float32)
    GAMMA_SOL, BAND = np.asarray(GAMMA_SOL, dtype=np.float32), np.asarray(BAND, dtype=np.float32)
    WAV, AOP_BAND = np.asarray(WAV, dtype=np.float32), np.asarray(AOP_BAND, dtype=np.float32)
    N_BAND = np.asarray(N_BAND, dtype=np.float32)

    fit_observations_resume.insert(10, 'THETA MOON', C1lua)
    fit_observations_resume.insert(11, 'PHI MOON', C2lua)
    fit_observations_resume.insert(12, 'ALBEDO', ALBEDO)
    fit_observations_resume.insert(13, 'WAVELENGTH', BAND)
    fit_observations_resume.insert(14, 'PAR', WAV)
    fit_observations_resume.insert(15, 'THETA FIELD', C1field)
    fit_observations_resume.insert(16, 'PHI FIELD', C2field)
    fit_observations_resume.insert(17, 'GAMMA', GAMMA)
    fit_observations_resume.insert(18, 'AOP', AOP)
    fit_observations_resume.insert(19, 'AOP error', errAOP)
    fit_observations_resume.insert(20, 'POL OBS', POL_OBS)
    fit_observations_resume.insert(21, 'POL OBS error', errPOL_OBS)

    LABEL = listToString(conjunto)

    k1_mix = []
    k2_mix = []
    N_mix = []
    c_mix = []
    chi_mix = []
    index_names = []

    meto = 'leastsq'

    coluna = 22

    #        SCATTERING

    if command == 'ALL' or command == 'horizon fit':
        model = lmfit.Model(func_hor_DOP)
        model.set_param_hint('N', min=0, max=20)
        p = model.make_params(N=np.random.rand())
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee')
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, BAND, WAV],
                           weights=errPOL_OBS,
                           method=meto)

        result_emcee_hor = [result.params['N'].value]  # , result.params['N'].value]
        result_emcee_hor = np.asarray(result_emcee_hor)
        Rpar = result_emcee_hor

        k1_mix.append(np.nan)
        k2_mix.append(np.nan)
        N_mix.append(result.params['N'].value)
        c_mix.append(np.nan)
        chi_mix.append(result.chisqr)
        index_names.append('horizon stokes')

        txname = 'REPORT_' + LABEL + '_' + condition + '_horizon_stokes_' + meto + '.txt'
        model_name = 'MODEL_' + LABEL + '_' + condition + '_horizon_stokes_' + meto + '.sav'

        lmfit.model.save_modelresult(result, model_name)
        TXT = open(txname, "w+")

        model_fit_report = result.fit_report()
        TXT.write('***  Second Fit: now adding the horizon correction parameter  *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        barra.update(10)

        y1 = func_hor_DOP([C1field, C2field, C1lua, C2lua, BAND, WAV], *result_emcee_hor)
        fit_observations_resume.insert(coluna, 'HOR POL', y1)

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
        fit_observations_resume.insert(coluna, 'HOR UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])
        fit_observations_resume.insert(coluna, 'HOR DIFF', diff)
        coluna += 1

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['HOR POL'].to_numpy()
            c1 = b_points['HOR UNC'].to_numpy()
            d1 = b_points['HOR DIFF'].to_numpy()
            e1 = b_points['POL OBS'].to_numpy()
            f1 = b_points['POL OBS error'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], np.asarray(d1)[w], \
                                     np.asarray(e1)[
                                         w], np.asarray(f1)[w]

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['HOR POL'].to_numpy()
            c2 = v_points['HOR UNC'].to_numpy()
            d2 = v_points['HOR DIFF'].to_numpy()
            e2 = v_points['POL OBS'].to_numpy()
            f2 = v_points['POL OBS error'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], np.asarray(d2)[w], \
                                     np.asarray(e2)[
                                         w], np.asarray(f2)[w]

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['HOR POL'].to_numpy()
            c3 = r_points['HOR UNC'].to_numpy()
            d3 = r_points['HOR DIFF'].to_numpy()
            e3 = r_points['POL OBS'].to_numpy()
            f3 = r_points['POL OBS error'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], np.asarray(d3)[w], \
                                     np.asarray(e3)[
                                         w], np.asarray(f3)[w]

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['HOR POL'].to_numpy()
            c4 = i_points['HOR UNC'].to_numpy()
            d4 = i_points['HOR DIFF'].to_numpy()
            e4 = i_points['POL OBS'].to_numpy()
            f4 = i_points['POL OBS error'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], np.asarray(d4)[w], \
                                     np.asarray(e4)[
                                         w], np.asarray(f4)[w]
            barra.update(10)

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')

            barra.update(10)

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            h1 = np.add(b2, c2)
            h2 = np.subtract(b2, c2)
            plt.fill_between(a2, h2, h1, where=(h2 < h1), interpolate=True, color='beige')

            j1 = np.add(b3, c3)
            j2 = np.subtract(b3, c3)
            plt.fill_between(a3, j2, j1, where=(j2 < j1), interpolate=True, color='mistyrose')

            k1 = np.add(b4, c4)
            k2 = np.subtract(b4, c4)
            plt.fill_between(a4, k2, k1, where=(k2 < k1), interpolate=True, color='antiquewhite')

            plt.ylim(0, 0.8)
            if isinstance(result.params['N'].stderr, float):
                N_par = round(result.params['N'].stderr, 3)
            else:
                N_par = result.params['N'].stderr
            plt.ylabel('Polarization')
            label_text = 'fit parameters:   $N$ = ' + str(
                round(result.params['N'].value, 3)) + '$\pm$' + str(N_par) + '\n' + 'chi-square: ' + str(
                round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
                round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            leg_text = 'fit adding \nthe horizon correction parameter...\n \n$c_{B}$= ' + str(
                round(B_par_reg[0], 3)) + '\n$c_{V}$= ' + str(round(V_par_reg[0], 3)) + '\n$c_{R}$= ' + str(
                round(R_par_reg[0], 3)) + '\n$c_{I}$= ' + str(round(I_par_reg[0], 3))
            plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            barra.update(10)

            fig_x.add_axes((.1, .1, .6, .2))

            plt.errorbar(a1, d1, yerr=f1, ms=2.0, fmt='o', color='blue', label='diff B band')
            plt.plot(a1, c1, '-', color='cornflowerblue', markersize=2, label='uncertanties fit')
            plt.errorbar(a2, d2, yerr=f2, ms=2.0, fmt='o', color='green', label='diff V band')
            plt.plot(a2, c2, '-', color='mediumseagreen', markersize=2, label='uncertanties fit')
            plt.errorbar(a3, d3, yerr=f3, ms=2.0, fmt='o', color='red', label='diff B band')
            plt.plot(a3, c3, '-', color='indianred', markersize=2, label='uncertanties fit')
            plt.errorbar(a4, d4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='diff V band')
            plt.plot(a4, c4, '-', color='orange', markersize=2, label='uncertanties fit')

            plt.xlabel('Scattering Angle (degrees)')
            plt.ylabel('Residual data')
            plt.grid(True)
            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_horizon_stokes_' + meto + '.png', bbox_inches='tight')
            barra.update(15)
            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['HOR POL'].to_numpy()
                c = points['HOR UNC'].to_numpy()
                d = points['HOR DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], \
                                   np.asarray(f)[w]

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                # plt.ylim(0, 0.8)
                if isinstance(result.params['N'].stderr, float):
                    N_par = round(result.params['N'].stderr, 3)
                else:
                    N_par = result.params['N'].stderr
                plt.ylabel('Polarization')
                label_text = 'fit parameters:   $N$ = ' + str(
                    round(result.params['N'].value, 3)) + '$\pm$' + str(N_par) + '\n' + 'chi-square: ' + str(
                    round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
                    round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                leg_text = 'fit adding \nthe horizon correction parameter...\n \n$c_{B}$= ' + str(
                    round(B_par_reg[0], 3)) + '\n$c_{V}$= ' + str(round(V_par_reg[0], 3)) + '\n$c_{R}$= ' + str(
                    round(R_par_reg[0], 3)) + '\n$c_{I}$= ' + str(round(I_par_reg[0], 3))
                plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['HOR POL'].to_numpy()
                c = points['HOR UNC'].to_numpy()
                d = points['HOR DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], \
                                   np.asarray(f)[w]

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_horizon_stokes_' + meto + '.png', bbox_inches='tight')
            plt.close()
        TXT.close()

    # -------------------------------------------------------------------------------------------

    if command == 'ALL' or command == 'fit Sun':
        model = lmfit.Model(func_sun_DOP)
        model.set_param_hint('par')
        p = model.make_params(par=np.random.rand())  # , N=10)
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, C1sol, C2sol, ALBEDO],
        #                    method='emcee', weights=errPOL_OBS, nfev=500000)
        result = model.fit(data=POL_OBS, params=p,
                           allvars=[C1field, C2field, C1lua, C2lua, C1sol, C2sol, ALBEDO, WAV],
                           weights=errPOL_OBS, method=meto)

        result_emcee_sun = [result.params['par'].value]
        result_emcee_sun = np.asarray(result_emcee_sun)
        Rpar = result_emcee_sun

        txname = 'REPORT_' + LABEL + '_' + LAB + '_sun_stokes_' + meto + '.txt'
        TXT = open(txname, "w+")

        model_fit_report = result.fit_report()
        TXT.write('***  Third Fit: now considering the sun influence *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        barra.update(10)

        model_name = 'MODEL_' + LABEL + '_' + LAB + '_sun_stokes_' + meto + '.sav'
        lmfit.model.save_modelresult(result, model_name)

        y1 = func_sun_DOP([C1field, C2field, C1lua, C2lua, C1sol, C2sol, ALBEDO, WAV], *result_emcee_sun)
        fit_observations_resume.insert(coluna, 'SUN POL', y1)
        coluna += 1

        rsd = result.eval_uncertainty()
        fit_observations_resume.insert(coluna, 'SUN UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])
        fit_observations_resume.insert(coluna, 'SUN DIFF', diff)
        coluna += 1

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['SUN POL'].to_numpy()
            c1 = b_points['SUN UNC'].to_numpy()
            d1 = b_points['SUN DIFF'].to_numpy()
            e1 = b_points['POL OBS'].to_numpy()
            f1 = b_points['POL OBS error'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], np.asarray(d1)[w], \
                                     np.asarray(e1)[
                                         w], np.asarray(f1)[w]

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['SUN POL'].to_numpy()
            c2 = v_points['SUN UNC'].to_numpy()
            d2 = v_points['SUN DIFF'].to_numpy()
            e2 = v_points['POL OBS'].to_numpy()
            f2 = v_points['POL OBS error'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], np.asarray(d2)[w], \
                                     np.asarray(e2)[
                                         w], np.asarray(f2)[w]

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['SUN POL'].to_numpy()
            c3 = r_points['SUN UNC'].to_numpy()
            d3 = r_points['SUN DIFF'].to_numpy()
            e3 = r_points['POL OBS'].to_numpy()
            f3 = r_points['POL OBS error'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], np.asarray(d3)[w], \
                                     np.asarray(e3)[
                                         w], np.asarray(f3)[w]

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['SUN POL'].to_numpy()
            c4 = i_points['SUN UNC'].to_numpy()
            d4 = i_points['SUN DIFF'].to_numpy()
            e4 = i_points['POL OBS'].to_numpy()
            f4 = i_points['POL OBS error'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], np.asarray(d4)[w], \
                                     np.asarray(e4)[
                                         w], np.asarray(f4)[w]

            barra.update(10)

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            h1 = np.add(b2, c2)
            h2 = np.subtract(b2, c2)
            plt.fill_between(a2, h2, h1, where=(h2 < h1), interpolate=True, color='beige')

            j1 = np.add(b3, c3)
            j2 = np.subtract(b3, c3)
            plt.fill_between(a3, j2, j1, where=(j2 < j1), interpolate=True, color='mistyrose')

            k1 = np.add(b4, c4)
            k2 = np.subtract(b4, c4)
            plt.fill_between(a4, k2, k1, where=(k2 < k1), interpolate=True, color='antiquewhite')

            # plt.ylim(0, 0.8)
            plt.ylabel('Polarization')
            if isinstance(result.params['par'].stderr, float):
                par_par = round(result.params['par'].stderr, 3)
            else:
                par_par = result.params['par'].stderr
            label_text = 'fit parameters:  $f$ = ' + str(round(result.params['par'].value, 3)) + '$\pm$' + str(
                par_par) + '\n' + 'chi-square: ' + str(
                round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
                round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            leg_text = 'Individuals Amplitudes\n \n$A_{B}$= ' + str(
                round(B_par_reg[0], 3)) + '\n$A_{V}$= ' + str(round(V_par_reg[0], 3)) + '\n$A_{R}$= ' + str(
                round(R_par_reg[0], 3)) + '\n$A_{I}$= ' + str(round(I_par_reg[0], 3))
            plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            barra.update(10)

            fig_x.add_axes((.1, .1, .6, .2))

            plt.errorbar(a1, d1, yerr=f1, ms=2.0, fmt='o', color='blue', label='diff B band')
            plt.plot(a1, c1, '-', color='cornflowerblue', markersize=2, label='uncertanties fit')
            plt.errorbar(a2, d2, yerr=f2, ms=2.0, fmt='o', color='green', label='diff V band')
            plt.plot(a2, c2, '-', color='mediumseagreen', markersize=2, label='uncertanties fit')
            plt.errorbar(a3, d3, yerr=f3, ms=2.0, fmt='o', color='red', label='diff B band')
            plt.plot(a3, c3, '-', color='indianred', markersize=2, label='uncertanties fit')
            plt.errorbar(a4, d4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='diff V band')
            plt.plot(a4, c4, '-', color='orange', markersize=2, label='uncertanties fit')

            plt.xlabel('Scattering Angle (degrees)')
            plt.ylabel('Residual data')
            plt.grid(True)
            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_sun_stokes_' + meto + '.png', bbox_inches='tight')
            barra.update(15)
            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['SUN POL'].to_numpy()
                c = points['SUN UNC'].to_numpy()
                d = points['SUN DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], np.asarray(f)[w]

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                # plt.ylim(0, 0.8)
                plt.ylabel('Polarization')
                if isinstance(result.params['par'].stderr, float):
                    par_par = round(result.params['par'].stderr, 3)
                else:
                    par_par = result.params['par'].stderr
                label_text = 'fit parameters: ' + ' $c_{banda}$ = ' + str(
                    round(result.params['par'].value, 3)) + '$\pm$' + str(
                    par_par) + '\n' + 'chi-square: ' + str(
                    round(result.chisqr, 7)) + '\n' + 'reduced chi-square: ' + str(round(result.redchi, 7))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                leg_text = 'fit adding \nthe sun influence...\n \n$c_{B}$= ' + str(
                    round(B_par_reg[0], 3)) + '\n$c_{V}$= ' + str(round(V_par_reg[0], 3)) + '\n$c_{R}$= ' + str(
                    round(R_par_reg[0], 3)) + '\n$c_{I}$= ' + str(round(I_par_reg[0], 3))
                plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['SUN POL'].to_numpy()
                c = points['SUN UNC'].to_numpy()
                d = points['SUN DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], np.asarray(f)[w]

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + LAB+ '_sun_stokes_' + meto + '.png', bbox_inches='tight')
            plt.close()

    # --------------------------------------------------------------------------------------------

    if command == 'ALL' or command == 'fit seeing':
        model = lmfit.Model(func_seeing_DOP)
        model.set_param_hint('k', min=3, max=20)
        model.set_param_hint('d', min=0.1, max=1)
        p = model.make_params(k=np.random.rand(), d=np.random.rand())  # , N=10)
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, SEEING], method='emcee',
        #                    weights=errPOL_OBS, nfev=500000)
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, SEEING, BAND, WAV],
                           weights=errPOL_OBS,
                           method=meto)

        result_emcee_seeing = [result.params['k'].value, result.params['d'].value]
        result_emcee_seeing = np.asarray(result_emcee_seeing)
        Rpar = result_emcee_seeing

        k1_mix.append(result.params['k'].value)
        k2_mix.append(result.params['d'].value)
        N_mix.append(np.nan)
        c_mix.append(np.nan)
        chi_mix.append(result.chisqr)
        index_names.append('seeing stokes')

        txname = 'REPORT_' + LABEL + '_' + condition + '_seeing_stokes_' + meto + '.txt'
        TXT = open(txname, "w+")

        model_fit_report = result.fit_report()
        TXT.write('***  Fourth Fit: now considering the seeing  astronomical parameter  *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        barra.update(10)

        model_name = 'MODEL_' + LABEL + '_' + condition + '_seeing_stokes_' + meto + '.sav'
        lmfit.model.save_modelresult(result, model_name)

        y1 = func_seeing_DOP([C1field, C2field, C1lua, C2lua, SEEING, BAND, WAV], *result_emcee_seeing)
        fit_observations_resume.insert(coluna, 'SEEING POL', y1)
        coluna += 1

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
        fit_observations_resume.insert(coluna, 'SEEING UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])
        fit_observations_resume.insert(coluna, 'SEEING DIFF', diff)
        coluna += 1

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['SEEING POL'].to_numpy()
            c1 = b_points['SEEING UNC'].to_numpy()
            d1 = b_points['SEEING DIFF'].to_numpy()
            e1 = b_points['POL OBS'].to_numpy()
            f1 = b_points['POL OBS error'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], np.asarray(d1)[w], \
                                     np.asarray(e1)[
                                         w], np.asarray(f1)[w]

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['SEEING POL'].to_numpy()
            c2 = v_points['SEEING UNC'].to_numpy()
            d2 = v_points['SEEING DIFF'].to_numpy()
            e2 = v_points['POL OBS'].to_numpy()
            f2 = v_points['POL OBS error'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], np.asarray(d2)[w], \
                                     np.asarray(e2)[
                                         w], np.asarray(f2)[w]

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['SEEING POL'].to_numpy()
            c3 = r_points['SEEING UNC'].to_numpy()
            d3 = r_points['SEEING DIFF'].to_numpy()
            e3 = r_points['POL OBS'].to_numpy()
            f3 = r_points['POL OBS error'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], np.asarray(d3)[w], \
                                     np.asarray(e3)[
                                         w], np.asarray(f3)[w]

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['SEEING POL'].to_numpy()
            c4 = i_points['SEEING UNC'].to_numpy()
            d4 = i_points['SEEING DIFF'].to_numpy()
            e4 = i_points['POL OBS'].to_numpy()
            f4 = i_points['POL OBS error'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], np.asarray(d4)[w], \
                                     np.asarray(e4)[
                                         w], np.asarray(f4)[w]

            barra.update(10)

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            h1 = np.add(b2, c2)
            h2 = np.subtract(b2, c2)
            plt.fill_between(a2, h2, h1, where=(h2 < h1), interpolate=True, color='beige')

            j1 = np.add(b3, c3)
            j2 = np.subtract(b3, c3)
            plt.fill_between(a3, j2, j1, where=(j2 < j1), interpolate=True, color='mistyrose')

            k1 = np.add(b4, c4)
            k2 = np.subtract(b4, c4)
            plt.fill_between(a4, k2, k1, where=(k2 < k1), interpolate=True, color='antiquewhite')

            # plt.ylim(0, 0.8)
            plt.ylabel('Polarization')
            label_text = 'fit parameters:  $k_{1}$ = ' + str(round(result.params['k'].value, 3)) + '$\pm$' + str(
                round(result.params['k'].stderr, 3)) + ',   $k_{2}$ = ' + str(
                round(result.params['d'].value, 3)) + '$\pm$' + str(
                round(result.params['d'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
                round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            leg_text = 'fit adding \nthe seeing astronomical parameter...\n \n$c_{B}$= ' + str(
                round(B_par_reg[0], 3)) + '\n$c_{V}$= ' + str(round(V_par_reg[0], 3)) + '\n$c_{R}$= ' + str(
                round(R_par_reg[0], 3)) + '\n$c_{I}$= ' + str(round(I_par_reg[0], 3))
            plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            barra.update(10)

            fig_x.add_axes((.1, .1, .6, .2))

            plt.errorbar(a1, d1, yerr=f1, ms=2.0, fmt='o', color='blue', label='diff B band')
            plt.plot(a1, c1, '-', color='cornflowerblue', markersize=2, label='uncertanties fit')
            plt.errorbar(a2, d2, yerr=f2, ms=2.0, fmt='o', color='green', label='diff V band')
            plt.plot(a2, c2, '-', color='mediumseagreen', markersize=2, label='uncertanties fit')
            plt.errorbar(a3, d3, yerr=f3, ms=2.0, fmt='o', color='red', label='diff B band')
            plt.plot(a3, c3, '-', color='indianred', markersize=2, label='uncertanties fit')
            plt.errorbar(a4, d4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='diff V band')
            plt.plot(a4, c4, '-', color='orange', markersize=2, label='uncertanties fit')

            plt.xlabel('Scattering Angle (degrees)')
            plt.ylabel('Residual data')
            plt.grid(True)
            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_seeing_stokes_' + meto + '.png', bbox_inches='tight')
            barra.update(15)
            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['SEEING POL'].to_numpy()
                c = points['SEEING UNC'].to_numpy()
                d = points['SEEING DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], \
                                   np.asarray(f)[w]

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                # plt.ylim(0, 0.8)
                plt.ylabel('Polarization')
                label_text = 'fit parameters:  $k_{1}$ = ' + str(round(result.params['k'].value, 3)) + '$\pm$' + str(
                    round(result.params['k'].stderr, 3)) + ',   $k_{2}$ = ' + str(
                    round(result.params['d'].value, 3)) + '$\pm$' + str(
                    round(result.params['d'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                    round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
                    round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                leg_text = 'fit adding \nthe seeing astronomical parameter...\n \n$c_{B}$= ' + str(
                    round(B_par_reg[0], 3)) + '\n$c_{V}$= ' + str(round(V_par_reg[0], 3)) + '\n$c_{R}$= ' + str(
                    round(R_par_reg[0], 3)) + '\n$c_{I}$= ' + str(round(I_par_reg[0], 3))
                plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['SEEING POL'].to_numpy()
                c = points['SEEING UNC'].to_numpy()
                d = points['SEEING DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], \
                                   np.asarray(f)[w]

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_seeing_stokes_' + meto + '.png', bbox_inches='tight')
            plt.close()
        TXT.close()

    # ---------------------------------------------------------------------------------------------------

    if command == 'ALL' or command == 'fit wavelength':
        model = lmfit.Model(func_wav)
        model.set_param_hint('c')
        p = model.make_params(c=np.random.rand())  # , N=10)
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee',
        #                    weights=errPOL_OBS, nfev=500000)
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, BAND, N_BAND],
                           weights=errPOL_OBS, method='leastsq')

        result_emcee_wav = [result.params['c'].value]
        result_emcee_wav = np.asarray(result_emcee_wav)
        Rpar = result_emcee_wav

        k1_mix.append(np.nan)
        k2_mix.append(np.nan)
        N_mix.append(np.nan)
        c_mix.append(result.params['c'].value)
        chi_mix.append(result.chisqr)
        index_names.append('wave stokes')

        txname = 'REPORT_' + LABEL + '_' + condition + '_wave_stokes_' + meto + '.txt'
        TXT = open(txname, "w+")

        model_fit_report = result.fit_report()
        TXT.write('***  Fifth Fit: now considering the wavelength of light *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        model_name = 'MODEL_' + LABEL + '_' + condition + '_wave_stokes_' + meto + '.sav'
        lmfit.model.save_modelresult(result, model_name)

        y1 = func_wav([C1field, C2field, C1lua, C2lua, BAND, N_BAND], *result_emcee_wav)
        fit_observations_resume.insert(coluna, 'WAVE POL', y1)
        coluna += 1

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
        fit_observations_resume.insert(coluna, 'WAVE UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])
        fit_observations_resume.insert(coluna, 'WAVE DIFF', diff)

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['WAVE POL'].to_numpy()
            c1 = b_points['WAVE UNC'].to_numpy()
            d1 = b_points['WAVE DIFF'].to_numpy()
            e1 = b_points['POL OBS'].to_numpy()
            f1 = b_points['POL OBS error'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], np.asarray(d1)[w], \
                                     np.asarray(e1)[
                                         w], np.asarray(f1)[w]

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['WAVE POL'].to_numpy()
            c2 = v_points['WAVE UNC'].to_numpy()
            d2 = v_points['WAVE DIFF'].to_numpy()
            e2 = v_points['POL OBS'].to_numpy()
            f2 = v_points['POL OBS error'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], np.asarray(d2)[w], \
                                     np.asarray(e2)[
                                         w], np.asarray(f2)[w]

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['WAVE POL'].to_numpy()
            c3 = r_points['WAVE UNC'].to_numpy()
            d3 = r_points['WAVE DIFF'].to_numpy()
            e3 = r_points['POL OBS'].to_numpy()
            f3 = r_points['POL OBS error'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], np.asarray(d3)[w], \
                                     np.asarray(e3)[
                                         w], np.asarray(f3)[w]

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['WAVE POL'].to_numpy()
            c4 = i_points['WAVE UNC'].to_numpy()
            d4 = i_points['WAVE DIFF'].to_numpy()
            e4 = i_points['POL OBS'].to_numpy()
            f4 = i_points['POL OBS error'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], np.asarray(d4)[w], \
                                     np.asarray(e4)[
                                         w], np.asarray(f4)[w]

            barra.update(10)

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            h1 = np.add(b2, c2)
            h2 = np.subtract(b2, c2)
            plt.fill_between(a2, h2, h1, where=(h2 < h1), interpolate=True, color='beige')

            j1 = np.add(b3, c3)
            j2 = np.subtract(b3, c3)
            plt.fill_between(a3, j2, j1, where=(j2 < j1), interpolate=True, color='mistyrose')

            k1 = np.add(b4, c4)
            k2 = np.subtract(b4, c4)
            plt.fill_between(a4, k2, k1, where=(k2 < k1), interpolate=True, color='antiquewhite')

            # plt.ylim(0, 0.8)
            if isinstance(result.params['c'].stderr, float):
                c_par = round(result.params['c'].stderr, 3)
            else:
                c_par = result.params['c'].stderr
            plt.ylabel('Polarization')
            label_text = 'fit parameters: ' + '$c$ = ' + str(
                round(result.params['c'].value, 3)) + '$\pm$' + str(c_par) + '\n' + 'chi-square: ' + str(
                round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
                round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            barra.update(10)

            fig_x.add_axes((.1, .1, .6, .2))

            plt.errorbar(a1, d1, yerr=f1, ms=2.0, fmt='o', color='blue', label='diff B band')
            plt.plot(a1, c1, '-', color='cornflowerblue', markersize=2, label='uncertanties fit')
            plt.errorbar(a2, d2, yerr=f2, ms=2.0, fmt='o', color='green', label='diff V band')
            plt.plot(a2, c2, '-', color='mediumseagreen', markersize=2, label='uncertanties fit')
            plt.errorbar(a3, d3, yerr=f3, ms=2.0, fmt='o', color='red', label='diff B band')
            plt.plot(a3, c3, '-', color='indianred', markersize=2, label='uncertanties fit')
            plt.errorbar(a4, d4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='diff V band')
            plt.plot(a4, c4, '-', color='orange', markersize=2, label='uncertanties fit')

            plt.xlabel('Scattering Angle (degrees)')
            plt.ylabel('Residual data')
            plt.grid(True)
            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_wave_stokes_' + meto + '.png', bbox_inches='tight')
            barra.update(15)
            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['WAVE POL'].to_numpy()
                c = points['WAVE UNC'].to_numpy()
                d = points['WAVE DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], \
                                   np.asarray(f)[w]

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                # plt.ylim(0, 0.8)
                plt.ylabel('Polarization')
                if isinstance(result.params['c'].stderr, float):
                    c_par = round(result.params['c'].stderr, 3)
                else:
                    c_par = result.params['c'].stderr
                plt.ylabel('Polarization')
                label_text = 'fit parameters: ' + '$c$ = ' + str(
                    round(result.params['c'].value, 3)) + '$\pm$' + str(c_par) + '\n' + 'chi-square: ' + str(
                    round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
                    round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['WAVE POL'].to_numpy()
                c = points['WAVE UNC'].to_numpy()
                d = points['WAVE DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], \
                                   np.asarray(f)[w]

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_wave_stokes_' + meto + '.png', bbox_inches='tight')
            plt.close()
        TXT.close()

    # ---------------------------------------------------------------------------------------------------

    if command == 'ALL' or command == 'fit depolarization':
        model = lmfit.Model(func_depo_DOP)
        model.set_param_hint('P')
        p = model.make_params(P=np.random.rand())  # , N=10)
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee',
        #                    weights=errPOL_OBS, nfev=500000)
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, BAND, WAV],
                           weights=errPOL_OBS, method=meto)

        result_emcee_dep = [result.params['P'].value]
        result_emcee_dep = np.asarray(result_emcee_dep)
        Rpar = result_emcee_dep

        txname = 'REPORT_' + LABEL + '_' + condition + '_dep_stokes_' + meto + '.txt'
        TXT = open(txname, "w+")

        model_fit_report = result.fit_report()
        TXT.write('***  now considering the depolarization factor *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        barra.update(10)

        model_name = 'MODEL_' + LABEL + '_' + condition + '_dep_stokes_' + meto + '.sav'
        lmfit.model.save_modelresult(result, model_name)

        y1 = func_depo_DOP([C1field, C2field, C1lua, C2lua, BAND, WAV], *result_emcee_dep)
        fit_observations_resume.insert(coluna, 'DEP POL', y1)
        coluna += 1

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
        fit_observations_resume.insert(coluna, 'DEP UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])
        fit_observations_resume.insert(coluna, 'DEP DIFF', diff)

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['DEP POL'].to_numpy()
            c1 = b_points['DEP UNC'].to_numpy()
            d1 = b_points['DEP DIFF'].to_numpy()
            e1 = b_points['POL OBS'].to_numpy()
            f1 = b_points['POL OBS error'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], np.asarray(d1)[w], \
                                     np.asarray(e1)[
                                         w], np.asarray(f1)[w]

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['DEP POL'].to_numpy()
            c2 = v_points['DEP UNC'].to_numpy()
            d2 = v_points['DEP DIFF'].to_numpy()
            e2 = v_points['POL OBS'].to_numpy()
            f2 = v_points['POL OBS error'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], np.asarray(d2)[w], \
                                     np.asarray(e2)[
                                         w], np.asarray(f2)[w]

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['DEP POL'].to_numpy()
            c3 = r_points['DEP UNC'].to_numpy()
            d3 = r_points['DEP DIFF'].to_numpy()
            e3 = r_points['POL OBS'].to_numpy()
            f3 = r_points['POL OBS error'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], np.asarray(d3)[w], \
                                     np.asarray(e3)[
                                         w], np.asarray(f3)[w]

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['DEP POL'].to_numpy()
            c4 = i_points['DEP UNC'].to_numpy()
            d4 = i_points['DEP DIFF'].to_numpy()
            e4 = i_points['POL OBS'].to_numpy()
            f4 = i_points['POL OBS error'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], np.asarray(d4)[w], \
                                     np.asarray(e4)[
                                         w], np.asarray(f4)[w]

            barra.update(10)

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            h1 = np.add(b2, c2)
            h2 = np.subtract(b2, c2)
            plt.fill_between(a2, h2, h1, where=(h2 < h1), interpolate=True, color='beige')

            j1 = np.add(b3, c3)
            j2 = np.subtract(b3, c3)
            plt.fill_between(a3, j2, j1, where=(j2 < j1), interpolate=True, color='mistyrose')

            k1 = np.add(b4, c4)
            k2 = np.subtract(b4, c4)
            plt.fill_between(a4, k2, k1, where=(k2 < k1), interpolate=True, color='antiquewhite')

            # plt.ylim(0, 0.8)
            if isinstance(result.params['P'].stderr, float):
                P_par = round(result.params['P'].stderr, 3)
            else:
                P_par = result.params['P'].stderr
            plt.ylabel('Polarization')
            label_text = 'fit parameters: ' + '$P$ = ' + str(
                round(result.params['P'].value, 3)) + '$\pm$' + str(P_par) + '\n' + 'chi-square: ' + str(
                round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
                round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center',
                         bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            barra.update(10)

            fig_x.add_axes((.1, .1, .6, .2))

            plt.errorbar(a1, d1, yerr=f1, ms=2.0, fmt='o', color='blue', label='diff B band')
            plt.plot(a1, c1, '-', color='cornflowerblue', markersize=2, label='uncertanties fit')
            plt.errorbar(a2, d2, yerr=f2, ms=2.0, fmt='o', color='green', label='diff V band')
            plt.plot(a2, c2, '-', color='mediumseagreen', markersize=2, label='uncertanties fit')
            plt.errorbar(a3, d3, yerr=f3, ms=2.0, fmt='o', color='red', label='diff B band')
            plt.plot(a3, c3, '-', color='indianred', markersize=2, label='uncertanties fit')
            plt.errorbar(a4, d4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='diff V band')
            plt.plot(a4, c4, '-', color='orange', markersize=2, label='uncertanties fit')

            plt.xlabel('Scattering Angle (degrees)')
            plt.ylabel('Residual data')
            plt.grid(True)
            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_dep_stokes_' + meto + '.png', bbox_inches='tight')
            barra.update(15)
            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['DEP POL'].to_numpy()
                c = points['DEP UNC'].to_numpy()
                d = points['DEP DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], \
                                   np.asarray(f)[w]

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                # plt.ylim(0, 0.8)
                plt.ylabel('Polarization')
                if isinstance(result.params['P'].stderr, float):
                    P_par = round(result.params['P'].stderr, 3)
                else:
                    P_par = result.params['P'].stderr
                plt.ylabel('Polarization')
                label_text = 'fit parameters: ' + '$P$ = ' + str(
                    round(result.params['P'].value, 3)) + '$\pm$' + str(P_par) + '\n' + 'chi-square: ' + str(
                    round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
                    round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                    round(result.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['DEP POL'].to_numpy()
                c = points['DEP UNC'].to_numpy()
                d = points['DEP DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], \
                                   np.asarray(f)[w]

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_dep_stokes_' + meto + '.png', bbox_inches='tight')
            plt.close()
        TXT.close()

    # -------------------------------------------------------------------------------------------

    if command == 'ALL' or command == 'fit all':
        model = lmfit.Model(func_DOP)
        model.set_param_hint('k', min=2, max=20)
        model.set_param_hint('d', min=0.1, max=1)
        model.set_param_hint('N', min=0, max=20)
        model.set_param_hint('c', min=-4, max=0)
        p = model.make_params(N=np.random.rand(), k=np.random.rand(), c=np.random.rand(), d=np.random.rand())  # , N=10)
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee')
        result = model.fit(data=POL_OBS, params=p,
                           allvars=[C1field, C2field, C1lua, C2lua, C1sol, C2sol, ALBEDO, SEEING, BAND, WAV],
                           weights=errPOL_OBS, method=meto)

        result_emcee = [result.params['N'].value, result.params['k'].value, result.params['d'].value,
                        result.params['c'].value]
        result_emcee = np.asarray(result_emcee)
        Rpar = result_emcee

        k1_mix.append(result.params['k'].value)
        k2_mix.append(result.params['d'].value)
        N_mix.append(result.params['N'].value)
        c_mix.append(result.params['c'].value)
        chi_mix.append(result.chisqr)
        index_names.append('all stokes')

        txname = 'REPORT_' + LABEL + '_' + condition + '_all_stokes_' + meto + '.txt'
        TXT = open(txname, "w+")

        model_fit_report = result.fit_report()
        TXT.write('*** Sixth Fit: considering all previous corrections *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        barra.update(10)

        model_name = 'MODEL_' + LABEL + '_' + condition + '_all_ray_' + meto + '.sav'
        lmfit.model.save_modelresult(result, model_name)

        y1 = func_DOP([C1field, C2field, C1lua, C2lua, C1sol, C2sol, ALBEDO, SEEING, BAND, WAV], *result_emcee)
        fit_observations_resume.insert(coluna, 'ALL POL', y1)
        coluna += 1

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
        fit_observations_resume.insert(coluna, 'ALL UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])
        fit_observations_resume.insert(coluna, 'ALL DIFF', diff)
        coluna += 1

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['ALL POL'].to_numpy()
            c1 = b_points['ALL UNC'].to_numpy()
            d1 = b_points['ALL DIFF'].to_numpy()
            e1 = b_points['POL OBS'].to_numpy()
            f1 = b_points['POL OBS error'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], np.asarray(d1)[w], \
                                     np.asarray(e1)[
                                         w], np.asarray(f1)[w]

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['ALL POL'].to_numpy()
            c2 = v_points['ALL UNC'].to_numpy()
            d2 = v_points['ALL DIFF'].to_numpy()
            e2 = v_points['POL OBS'].to_numpy()
            f2 = v_points['POL OBS error'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], np.asarray(d2)[w], \
                                     np.asarray(e2)[
                                         w], np.asarray(f2)[w]

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['ALL POL'].to_numpy()
            c3 = r_points['ALL UNC'].to_numpy()
            d3 = r_points['ALL DIFF'].to_numpy()
            e3 = r_points['POL OBS'].to_numpy()
            f3 = r_points['POL OBS error'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], np.asarray(d3)[w], \
                                     np.asarray(e3)[
                                         w], np.asarray(f3)[w]

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['ALL POL'].to_numpy()
            c4 = i_points['ALL UNC'].to_numpy()
            d4 = i_points['ALL DIFF'].to_numpy()
            e4 = i_points['POL OBS'].to_numpy()
            f4 = i_points['POL OBS error'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], np.asarray(d4)[w], \
                                     np.asarray(e4)[
                                         w], np.asarray(f4)[w]

            barra.update(10)

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            h1 = np.add(b2, c2)
            h2 = np.subtract(b2, c2)
            plt.fill_between(a2, h2, h1, where=(h2 < h1), interpolate=True, color='beige')

            j1 = np.add(b3, c3)
            j2 = np.subtract(b3, c3)
            plt.fill_between(a3, j2, j1, where=(j2 < j1), interpolate=True, color='mistyrose')

            k1 = np.add(b4, c4)
            k2 = np.subtract(b4, c4)
            plt.fill_between(a4, k2, k1, where=(k2 < k1), interpolate=True, color='antiquewhite')

            if isinstance(result.params['k'].stderr, float):
                k_par = round(result.params['k'].stderr, 3)
            else:
                k_par = result.params['k'].stderr
            if isinstance(result.params['d'].stderr, float):
                d_par = round(result.params['d'].stderr, 3)
            else:
                d_par = result.params['d'].stderr
            plt.ylim(0, 0.8)
            plt.ylabel('Polarization')
            label_text = 'fit parameters:  $k_{1}$ = ' + str(round(result.params['k'].value, 3)) + '$\pm$' + str(
                k_par) + ',   $k_{2}$ = ' + str(round(result.params['d'].value, 3)) + '$\pm$' + str(
                d_par) + ',   $N$ = ' + str(round(result.params['N'].value, 3)) + '$\pm$' + str(
                round(result.params['N'].stderr, 3)) + ',   $c$ = ' + str(
                round(result.params['c'].value, 3)) + '$\pm$' + str(
                round(result.params['c'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
                round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            leg_text = 'fit with all corrections...\n \n$P_{B norm}$= ' + str(
                round(B_par_reg[0], 3)) + '\n$P_{V norm}$= ' + str(round(V_par_reg[0], 3)) + '\n$P_{R norm}$= ' + str(
                round(R_par_reg[0], 3)) + '\n$P_{I norm}$= ' + str(round(I_par_reg[0], 3))
            plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            barra.update(10)

            fig_x.add_axes((.1, .1, .6, .2))

            plt.errorbar(a1, d1, yerr=f1, ms=2.0, fmt='o', color='blue', label='diff B band')
            plt.plot(a1, c1, '-', color='cornflowerblue', markersize=2, label='uncertanties fit')
            plt.errorbar(a2, d2, yerr=f2, ms=2.0, fmt='o', color='green', label='diff V band')
            plt.plot(a2, c2, '-', color='mediumseagreen', markersize=2, label='uncertanties fit')
            plt.errorbar(a3, d3, yerr=f3, ms=2.0, fmt='o', color='red', label='diff B band')
            plt.plot(a3, c3, '-', color='indianred', markersize=2, label='uncertanties fit')
            plt.errorbar(a4, d4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='diff V band')
            plt.plot(a4, c4, '-', color='orange', markersize=2, label='uncertanties fit')

            plt.xlabel('Scattering Angle (degrees)')
            plt.ylabel('Residual data')
            plt.grid(True)
            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_all_stokes_' + meto + '.png', bbox_inches='tight')
            barra.update(15)
            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['ALL POL'].to_numpy()
                c = points['ALL UNC'].to_numpy()
                d = points['ALL DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], \
                                   np.asarray(f)[w]

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                # plt.ylim(0, 0.8)
                plt.ylabel('Polarization')
                if isinstance(result.params['c'].stderr, float):
                    c_par = round(result.params['c'].stderr, 3)
                else:
                    c_par = result.params['c'].stderr
                if isinstance(result.params['k'].stderr, float):
                    k_par = round(result.params['k'].stderr, 3)
                else:
                    k_par = result.params['k'].stderr
                label_text = 'fit parameters:  $k_{1}$ = ' + str(round(result.params['k'].value, 3)) + '$\pm$' + str(
                    k_par) + ',   $k_{2}$ = ' + str(round(result.params['d'].value, 3)) + '$\pm$' + str(
                    round(result.params['d'].stderr, 3)) + ',   $N$ = ' + str(
                    round(result.params['N'].value, 3)) + '$\pm$' + str(
                    round(result.params['N'].stderr, 3)) + ',   $c$ = ' + str(
                    round(result.params['c'].value, 3)) + '$\pm$' + str(
                    c_par) + '\n' + 'chi-square: ' + str(
                    round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
                    round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                leg_text = 'fit with all corrections...\n \n$P_{B norm}$= ' + str(
                    round(B_par_reg[0], 3)) + '\n$P_{V norm}$= ' + str(round(V_par_reg[0], 3)) + '\n$P_{R norm}$= ' + str(
                    round(R_par_reg[0], 3)) + '\n$P_{I norm}$= ' + str(round(I_par_reg[0], 3))
                plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['ALL POL'].to_numpy()
                c = points['ALL UNC'].to_numpy()
                d = points['ALL DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], \
                                   np.asarray(f)[w]

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_all_stokes_' + meto + '.png', bbox_inches='tight')
            plt.close()
        TXT.close()

    fit_observations_resume.to_csv('TABLE_results_POLARIZATION_Rayleigh_scattering.csv')
    # result_data.to_excel('TABLE.xlsx', sheet_name='results_POLARIZATION_Rayleigh_scattering')

    # ################################## PLOT FITS ###################################################################

    if command == 'ALL' or command == 'fit aop regular':
        model = lmfit.Model(func_reg_AOP)
        # print(model.independent_vars)
        model.set_param_hint('par', min=0.0, max=4.0)
        p = model.make_params(par=np.random.rand())  # , beta1=np.random.rand(), beta2=np.random.rand())
        model.eval(params=p, allvars=[C1field, C2field, C1lua, C2lua])
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee',
        #                    weights=errPOL_OBS, fit_kws={'steps': 500000})
        result = model.fit(data=AOP, params=p, allvars=[C1field, C2field, C1lua, C2lua], weights=errAOP, method=meto)

        # para obter o número de iterações: ite = result.nfev
        # para obter o erro residual do fit: res = result.residual

        txname = 'REPORT_' + LABEL + '_' + condition + '_regular_AOP_stokes_' + meto + '.txt'
        model_name = 'MODEL_' + LABEL + '_' + condition + '_regular_AOP_stokes_' + meto + '.sav'

        model_fit_report = result.fit_report()

        TXT = open(txname, "w+")
        lmfit.model.save_modelresult(result, model_name)

        TXT.write('***  First Fit: fit regular for AOP with only normalization parameter free *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        barra.update(10)

        result_aop_reg = [result.params['par'].value]  # , result.params['beta1'].value, result.params['beta2'].value]
        result_aop_reg = np.asarray(result_aop_reg)
        # TXT.write('\n \n')

        y = func_reg_AOP([C1field, C2field, C1lua, C2lua], *result_aop_reg)
        fit_observations_resume.insert(coluna, 'REG AOP', y)
        coluna += 1

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
        fit_observations_resume.insert(coluna, 'REG AOP UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(AOP)):
            diff.append(AOP[i] - y[i])
        fit_observations_resume.insert(coluna, 'REG AOP DIFF', diff)
        coluna += 1

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['REG AOP'].to_numpy()
            c1 = b_points['REG AOP UNC'].to_numpy()
            d1 = b_points['REG AOP DIFF'].to_numpy()
            e1 = b_points['AOP'].to_numpy()
            f1 = b_points['AOP error'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], np.asarray(d1)[w], \
                                     np.asarray(e1)[
                                         w], np.asarray(f1)[w]

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['REG AOP'].to_numpy()
            c2 = v_points['REG AOP UNC'].to_numpy()
            d2 = v_points['REG AOP DIFF'].to_numpy()
            e2 = v_points['AOP'].to_numpy()
            f2 = v_points['AOP error'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], np.asarray(d2)[w], \
                                     np.asarray(e2)[
                                         w], np.asarray(f2)[w]

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['REG AOP'].to_numpy()
            c3 = r_points['REG AOP UNC'].to_numpy()
            d3 = r_points['REG AOP DIFF'].to_numpy()
            e3 = r_points['AOP'].to_numpy()
            f3 = r_points['AOP error'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], np.asarray(d3)[w], \
                                     np.asarray(e3)[
                                         w], np.asarray(f3)[w]

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['REG AOP'].to_numpy()
            c4 = i_points['REG AOP UNC'].to_numpy()
            d4 = i_points['REG AOP DIFF'].to_numpy()
            e4 = i_points['AOP'].to_numpy()
            f4 = i_points['AOP error'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], np.asarray(d4)[w], \
                                     np.asarray(e4)[
                                         w], np.asarray(f4)[w]

            barra.update(10)

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            h1 = np.add(b2, c2)
            h2 = np.subtract(b2, c2)
            plt.fill_between(a2, h2, h1, where=(h2 < h1), interpolate=True, color='beige')

            j1 = np.add(b3, c3)
            j2 = np.subtract(b3, c3)
            plt.fill_between(a3, j2, j1, where=(j2 < j1), interpolate=True, color='mistyrose')

            k1 = np.add(b4, c4)
            k2 = np.subtract(b4, c4)
            plt.fill_between(a4, k2, k1, where=(k2 < k1), interpolate=True, color='antiquewhite')

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')

            plt.ylim(-50, 50)
            if isinstance(result.params['par'].stderr, float):
                par_par = round(result.params['par'].stderr, 3)
            else:
                par_par = result.params['par'].stderr
            plt.ylabel('Angle of Polarization')
            label_text = 'fit parameters:    $P_{norm}$ = ' + str(round(result.params['par'].value, 3)) + '$\pm$' + str(
                par_par) + '\n' + 'chi-square: ' + str(
                round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
                round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            barra.update(10)

            fig_x.add_axes((.1, .1, .6, .2))

            plt.errorbar(a1, d1, yerr=f1, ms=2.0, fmt='o', color='blue', label='diff B band')
            plt.plot(a1, c1, '-', color='cornflowerblue', markersize=2, label='uncertanties fit')
            plt.errorbar(a2, d2, yerr=f2, ms=2.0, fmt='o', color='green', label='diff V band')
            plt.plot(a2, c2, '-', color='mediumseagreen', markersize=2, label='uncertanties fit')
            plt.errorbar(a3, d3, yerr=f3, ms=2.0, fmt='o', color='red', label='diff B band')
            plt.plot(a3, c3, '-', color='indianred', markersize=2, label='uncertanties fit')
            plt.errorbar(a4, d4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='diff V band')
            plt.plot(a4, c4, '-', color='orange', markersize=2, label='uncertanties fit')

            plt.xlabel('Scattering Angle (degrees)')
            plt.ylabel('Residual data')
            plt.grid(True)
            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_regular_AOP_stokes_' + meto + '.png')
            barra.update(15)
            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['REG AOP'].to_numpy()
                c = points['REG AOP UNC'].to_numpy()
                d = points['REG AOP DIFF'].to_numpy()
                e = points['AOP'].to_numpy()
                f = points['AOP error'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], np.asarray(f)[w]

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylim(0, 0.8)
                if isinstance(result.params['par'].stderr, float):
                    par_par = round(result.params['par'].stderr, 3)
                else:
                    par_par = result.params['par'].stderr
                plt.ylabel('Angle of Polarization')
                label_text = 'fit parameters:    $P_{norm}$ = ' + str(
                    round(result.params['par'].value, 3)) + '$\pm$' + str(
                    par_par) + '\n' + 'chi-square: ' + str(
                    round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
                    round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['REG AOP'].to_numpy()
                c = points['REG AOP UNC'].to_numpy()
                d = points['REG AOP DIFF'].to_numpy()
                e = points['AOP'].to_numpy()
                f = points['AOP error'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], np.asarray(f)[w]

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_regular_AOP_stokes_' + meto + '.png',
                        bbox_inches='tight')
            plt.close()

        TXT.close()

    # ----------------------------------------------------------------------------------------------

    if command == 'ALL' or command == 'fit aop wave':
        model = lmfit.Model(func_wav_AOP)
        # print(model.independent_vars)
        model.set_param_hint('c', min=-4, max=0)
        p = model.make_params(c=np.random.rand())
        model.eval(params=p, allvars=[C1field, C2field, C1lua, C2lua, BAND, AOP_BAND])
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee',
        #                    weights=errPOL_OBS, fit_kws={'steps': 500000})
        result = model.fit(data=AOP, params=p, allvars=[C1field, C2field, C1lua, C2lua, BAND, AOP_BAND], weights=errAOP,
                           method='leastsq')

        model_fit_report = result.fit_report()
        model_name = 'MODEL_' + LABEL + '_' + condition + '_wave_AOP_stokes_' + meto + '.sav'
        txname = 'REPORT_' + LABEL + '_' + condition + '_wave_AOP_stokes_' + meto + '.txt'
        TXT = open(txname, "w+")
        lmfit.model.save_modelresult(result, model_name)

        TXT.write('***  Second Fit: fit considering the wavelength of light *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        barra.update(10)

        result_aop_wav = [result.params['c'].value]
        result_aop_wav = np.asarray(result_aop_wav)

        y = func_wav_AOP([C1field, C2field, C1lua, C2lua, BAND, AOP_BAND], *result_aop_wav)
        fit_observations_resume.insert(21, 'WAV AOP', y)

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
        fit_observations_resume.insert(22, 'WAV AOP UNC', rsd)

        diff = []
        for i in range(0, len(AOP)):
            diff.append(AOP[i] - y[i])
        fit_observations_resume.insert(23, 'WAV AOP DIFF', diff)

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['WAV AOP'].to_numpy()
            c1 = b_points['WAV AOP UNC'].to_numpy()
            d1 = b_points['WAV AOP DIFF'].to_numpy()
            e1 = b_points['AOP'].to_numpy()
            f1 = b_points['AOP error'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], np.asarray(d1)[w], \
                                     np.asarray(e1)[
                                         w], np.asarray(f1)[w]

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['WAV AOP'].to_numpy()
            c2 = v_points['WAV AOP UNC'].to_numpy()
            d2 = v_points['WAV AOP DIFF'].to_numpy()
            e2 = v_points['AOP'].to_numpy()
            f2 = v_points['AOP error'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], np.asarray(d2)[w], \
                                     np.asarray(e2)[
                                         w], np.asarray(f2)[w]

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['WAV AOP'].to_numpy()
            c3 = r_points['WAV AOP UNC'].to_numpy()
            d3 = r_points['WAV AOP DIFF'].to_numpy()
            e3 = r_points['AOP'].to_numpy()
            f3 = r_points['AOP error'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], np.asarray(d3)[w], \
                                     np.asarray(e3)[
                                         w], np.asarray(f3)[w]

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['WAV AOP'].to_numpy()
            c4 = i_points['WAV AOP UNC'].to_numpy()
            d4 = i_points['WAV AOP DIFF'].to_numpy()
            e4 = i_points['AOP'].to_numpy()
            f4 = i_points['AOP error'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], np.asarray(d4)[w], \
                                     np.asarray(e4)[
                                         w], np.asarray(f4)[w]

            barra.update(10)

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            h1 = np.add(b2, c2)
            h2 = np.subtract(b2, c2)
            plt.fill_between(a2, h2, h1, where=(h2 < h1), interpolate=True, color='beige')

            j1 = np.add(b3, c3)
            j2 = np.subtract(b3, c3)
            plt.fill_between(a3, j2, j1, where=(j2 < j1), interpolate=True, color='mistyrose')

            k1 = np.add(b4, c4)
            k2 = np.subtract(b4, c4)
            plt.fill_between(a4, k2, k1, where=(k2 < k1), interpolate=True, color='antiquewhite')

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')

            plt.ylim(-50, 50)
            if isinstance(result.params['c'].stderr, float):
                c_par = round(result.params['c'].stderr, 3)
            else:
                c_par = result.params['c'].stderr
            plt.ylabel('Angle of Polarization')
            label_text = 'fit parameters:   $c$ = ' + str(round(result.params['c'].value, 3)) + '$\pm$' + str(
                c_par) + '\n' + 'chi-square: ' + str(
                round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
                round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            leg_text = 'fit with all corrections...\n \n$P_{B norm}$= ' + str(
                round(B_par_aop[0], 3)) + '\n$P_{V norm}$= ' + str(round(V_par_aop[0], 3)) + '\n$P_{R norm}$= ' + str(
                round(R_par_aop[0], 3)) + '\n$P_{I norm}$= ' + str(round(I_par_aop[0], 3))
            plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            barra.update(10)

            fig_x.add_axes((.1, .1, .6, .2))

            plt.errorbar(a1, d1, yerr=f1, ms=2.0, fmt='o', color='blue', label='diff B band')
            plt.plot(a1, c1, '-', color='cornflowerblue', markersize=2, label='uncertanties fit')
            plt.errorbar(a2, d2, yerr=f2, ms=2.0, fmt='o', color='green', label='diff V band')
            plt.plot(a2, c2, '-', color='mediumseagreen', markersize=2, label='uncertanties fit')
            plt.errorbar(a3, d3, yerr=f3, ms=2.0, fmt='o', color='red', label='diff B band')
            plt.plot(a3, c3, '-', color='indianred', markersize=2, label='uncertanties fit')
            plt.errorbar(a4, d4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='diff V band')
            plt.plot(a4, c4, '-', color='orange', markersize=2, label='uncertanties fit')

            plt.xlabel('Scattering Angle (degrees)')
            plt.ylabel('Residual data')
            plt.grid(True)
            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_wave_AOP_stokes_' + meto + '.png')
            barra.update(15)
            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['WAV AOP'].to_numpy()
                c = points['WAV AOP UNC'].to_numpy()
                d = points['WAV AOP DIFF'].to_numpy()
                e = points['AOP'].to_numpy()
                f = points['AOP error'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], np.asarray(f)[w]
                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylim(-50, 50)
                if isinstance(result.params['c'].stderr, float):
                    c_par = round(result.params['c'].stderr, 3)
                else:
                    c_par = result.params['c'].stderr
                plt.ylabel('Angle of Polarization')
                label_text = 'fit parameters:   $c$ = ' + str(round(result.params['c'].value, 3)) + '$\pm$' + str(
                    c_par) + '\n' + 'chi-square: ' + str(
                    round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
                    round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                leg_text = 'fit with all corrections...\n \n$P_{B norm}$= ' + str(
                    round(B_par_aop[0], 3)) + '\n$P_{V norm}$= ' + str(
                    round(V_par_aop[0], 3)) + '\n$P_{R norm}$= ' + str(
                    round(R_par_aop[0], 3)) + '\n$P_{I norm}$= ' + str(round(I_par_aop[0], 3))
                plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['WAV AOP'].to_numpy()
                c = points['WAV AOP UNC'].to_numpy()
                d = points['WAV AOP DIFF'].to_numpy()
                e = points['AOP'].to_numpy()
                f = points['AOP error'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], np.asarray(f)[w]

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_wave_AOP_stokes_' + meto + '.png', bbox_inches='tight')
            plt.close()

        TXT.close()

    # ----------------------------------------------------------------------------------------------

    if command == 'ALL' or command == 'fit Q regular':
        model = lmfit.Model(func_reg_Q)
        # print(model.independent_vars)
        model.set_param_hint('par', min=0.0, max=4.0)
        p = model.make_params(par=np.random.rand())  # , beta1=np.random.rand(), beta2=np.random.rand())
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee',
        #                    weights=errPOL_OBS, fit_kws={'steps': 500000})
        result = model.fit(data=Q_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], weights=errQ_OBS,
                           method=meto)

        # para obter o número de iterações: ite = result.nfev
        # para obter o erro residual do fit: res = result.residual

        model_fit_report = result.fit_report()
        model_name = 'MODEL_' + LABEL + '_' + condition + '_regular_Q_stokes_' + meto + '.sav'
        txname = 'REPORT_' + LABEL + '_' + condition + '_regular_Q_stokes_' + meto + '.txt'
        TXT = open(txname, "w+")
        lmfit.model.save_modelresult(result, model_name)

        TXT.write('***  First Fit: fit regular for Q with only normalization parameter free *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        barra.update(10)

        result_q_reg = [result.params['par'].value]  # , result.params['beta1'].value, result.params['beta2'].value]
        result_q_reg = np.asarray(result_q_reg)
        # TXT.write('\n \n')

        y = func_reg_AOP([C1field, C2field, C1lua, C2lua], *result_q_reg)
        fit_observations_resume.insert(coluna, 'REG Q', y)
        coluna += 1

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
        fit_observations_resume.insert(coluna, 'REG Q UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(Q_OBS)):
            diff.append(Q_OBS[i] - y[i])
        fit_observations_resume.insert(coluna, 'REG Q DIFF', diff)
        coluna += 1

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['REG Q'].to_numpy()
            c1 = b_points['REG Q UNC'].to_numpy()
            d1 = b_points['REG Q DIFF'].to_numpy()
            e1 = b_points['Q'].to_numpy()
            f1 = b_points['error Q'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], np.asarray(d1)[w], \
                                     np.asarray(e1)[w], np.asarray(f1)[w]

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['REG Q'].to_numpy()
            c2 = v_points['REG Q UNC'].to_numpy()
            d2 = v_points['REG Q DIFF'].to_numpy()
            e2 = v_points['Q'].to_numpy()
            f2 = v_points['error Q'].to_numpy()
            w = np.argsort(a1)
            a2, b2, c2, d2, e2, f2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], np.asarray(d2)[w], \
                                     np.asarray(e2)[
                                         w], np.asarray(f2)[w]

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['REG Q'].to_numpy()
            c3 = r_points['REG Q UNC'].to_numpy()
            d3 = r_points['REG Q DIFF'].to_numpy()
            e3 = r_points['Q'].to_numpy()
            f3 = r_points['error Q'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], np.asarray(d3)[w], \
                                     np.asarray(e3)[
                                         w], np.asarray(f3)[w]

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['REG Q'].to_numpy()
            c4 = i_points['REG Q UNC'].to_numpy()
            d4 = i_points['REG Q DIFF'].to_numpy()
            e4 = i_points['Q'].to_numpy()
            f4 = i_points['error Q'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], np.asarray(d4)[w], \
                                     np.asarray(e4)[
                                         w], np.asarray(f4)[w]

            barra.update(10)

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            h1 = np.add(b2, c2)
            h2 = np.subtract(b2, c2)
            plt.fill_between(a2, h2, h1, where=(h2 < h1), interpolate=True, color='beige')

            j1 = np.add(b3, c3)
            j2 = np.subtract(b3, c3)
            plt.fill_between(a3, j2, j1, where=(j2 < j1), interpolate=True, color='mistyrose')

            k1 = np.add(b4, c4)
            k2 = np.subtract(b4, c4)
            plt.fill_between(a4, k2, k1, where=(k2 < k1), interpolate=True, color='antiquewhite')

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')

            plt.ylim(-0.4, 0.4)
            if isinstance(result.params['par'].stderr, float):
                par_par = round(result.params['par'].stderr, 3)
            else:
                par_par = result.params['par'].stderr
            plt.ylabel('Q parameter')
            label_text = 'fit parameters:    $P_{norm}$ = ' + str(round(result.params['par'].value, 3)) + '$\pm$' + str(
                par_par) + '\n' + 'chi-square: ' + str(round(result.chisqr, 7)) + ',  reduced chi-square: ' + str(
                round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            barra.update(10)

            fig_x.add_axes((.1, .1, .6, .2))

            plt.errorbar(a1, d1, yerr=f1, ms=2.0, fmt='o', color='blue', label='diff B band')
            plt.plot(a1, c1, '-', color='cornflowerblue', markersize=2, label='uncertanties fit')
            plt.errorbar(a2, d2, yerr=f2, ms=2.0, fmt='o', color='green', label='diff V band')
            plt.plot(a2, c2, '-', color='mediumseagreen', markersize=2, label='uncertanties fit')
            plt.errorbar(a3, d3, yerr=f3, ms=2.0, fmt='o', color='red', label='diff B band')
            plt.plot(a3, c3, '-', color='indianred', markersize=2, label='uncertanties fit')
            plt.errorbar(a4, d4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='diff V band')
            plt.plot(a4, c4, '-', color='orange', markersize=2, label='uncertanties fit')

            plt.xlabel('Scattering Angle (degrees)')
            plt.ylabel('Residual data')
            plt.grid(True)
            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_regular_Q_stokes_' + meto + '.png')
            barra.update(15)
            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['REG Q'].to_numpy()
                c = points['REG Q UNC'].to_numpy()
                d = points['REG Q DIFF'].to_numpy()
                e = points['Q'].to_numpy()
                f = points['error Q'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], np.asarray(f)[w]

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylim(0, 0.8)
                if isinstance(result.params['par'].stderr, float):
                    par_par = round(result.params['par'].stderr, 3)
                else:
                    par_par = result.params['par'].stderr
                plt.ylabel('Q parameter')
                label_text = 'fit parameters:    $P_{norm}$ = ' + str(
                    round(result.params['par'].value, 3)) + '$\pm$' + str(
                    par_par) + '\n' + 'chi-square: ' + str(round(result.chisqr, 7)) + ',  reduced chi-square: ' + str(
                    round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['REG Q'].to_numpy()
                c = points['REG Q UNC'].to_numpy()
                d = points['REG Q DIFF'].to_numpy()
                e = points['Q'].to_numpy()
                f = points['error Q'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], np.asarray(f)[w]

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_regular_Q_stokes_' + meto + '.png', bbox_inches='tight')
            plt.close()

        TXT.close()

    # ----------------------------------------------------------------------------------------------

    if command == 'ALL' or command == 'fit Q wave':
        model = lmfit.Model(func_wav_Q)
        # print(model.independent_vars)
        model.set_param_hint('c', min=-4, max=0)
        p = model.make_params(c=np.random.rand())
        model.eval(params=p, allvars=[C1field, C2field, C1lua, C2lua, BAND, Q_BAND])
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee',
        #                    weights=errPOL_OBS, fit_kws={'steps': 500000})
        result = model.fit(data=Q_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, BAND, Q_BAND], method=meto)

        model_fit_report = result.fit_report()
        model_name = 'MODEL_' + LABEL + '_' + condition + '_wave_Q_stokes_' + meto + '.sav'
        lmfit.model.save_modelresult(result, model_name)

        txname = 'REPORT_' + LABEL + '_' + condition + '_wave_Q_stokes_' + meto + '.txt'
        TXT = open(txname, "w+")

        TXT.write('***  Second Fit: fit considering the wavelength of light *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        barra.update(10)

        result_q_wav = [result.params['c'].value]
        result_q_wav = np.asarray(result_q_wav)

        y = func_wav_AOP([C1field, C2field, C1lua, C2lua, BAND, Q_BAND], *result_q_wav)
        fit_observations_resume.insert(coluna, 'WAV Q', y)
        coluna += 1

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
        fit_observations_resume.insert(coluna, 'WAV Q UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(Q_OBS)):
            diff.append(Q_OBS[i] - y[i])
        fit_observations_resume.insert(coluna, 'WAV Q DIFF', diff)
        coluna += 1

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['WAV Q'].to_numpy()
            c1 = b_points['WAV Q UNC'].to_numpy()
            d1 = b_points['WAV Q DIFF'].to_numpy()
            e1 = b_points['Q'].to_numpy()
            f1 = b_points['error Q'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], np.asarray(d1)[w], \
                                     np.asarray(e1)[
                                         w], np.asarray(f1)[w]

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['WAV Q'].to_numpy()
            c2 = v_points['WAV Q UNC'].to_numpy()
            d2 = v_points['WAV Q DIFF'].to_numpy()
            e2 = v_points['Q'].to_numpy()
            f2 = v_points['error Q'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], np.asarray(d2)[w], \
                                     np.asarray(e2)[
                                         w], np.asarray(f2)[w]

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['WAV Q'].to_numpy()
            c3 = r_points['WAV Q UNC'].to_numpy()
            d3 = r_points['WAV Q DIFF'].to_numpy()
            e3 = r_points['Q'].to_numpy()
            f3 = r_points['error Q'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], np.asarray(d3)[w], \
                                     np.asarray(e3)[
                                         w], np.asarray(f3)[w]

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['WAV Q'].to_numpy()
            c4 = i_points['WAV Q UNC'].to_numpy()
            d4 = i_points['WAV Q DIFF'].to_numpy()
            e4 = i_points['Q'].to_numpy()
            f4 = i_points['error Q'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], np.asarray(d4)[w], \
                                     np.asarray(e4)[
                                         w], np.asarray(f4)[w]

            barra.update(10)

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            h1 = np.add(b2, c2)
            h2 = np.subtract(b2, c2)
            plt.fill_between(a2, h2, h1, where=(h2 < h1), interpolate=True, color='beige')

            j1 = np.add(b3, c3)
            j2 = np.subtract(b3, c3)
            plt.fill_between(a3, j2, j1, where=(j2 < j1), interpolate=True, color='mistyrose')

            k1 = np.add(b4, c4)
            k2 = np.subtract(b4, c4)
            plt.fill_between(a4, k2, k1, where=(k2 < k1), interpolate=True, color='antiquewhite')

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')

            plt.ylim(-0.4, 0.4)
            if isinstance(result.params['c'].stderr, float):
                c_par = round(result.params['c'].stderr, 3)
            else:
                c_par = result.params['c'].stderr
            plt.ylabel('Q parameter')
            label_text = 'fit parameters:   $c$ = ' + str(round(result.params['c'].value, 3)) + '$\pm$' + str(
                c_par) + '\n' + 'chi-square: ' + str(
                round(result.chisqr, 7)) + ',  reduced chi-square: ' + str(
                round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            leg_text = 'fit with all corrections...\n \n$c_{B}$= ' + str(
                round(B_par_u[0], 3)) + '\n$c_{V}$= ' + str(round(V_par_u[0], 3)) + '\n$c_{R}$= ' + str(
                round(R_par_u[0], 3)) + '\n$c_{I}$= ' + str(round(I_par_u[0], 3))
            plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            barra.update(10)

            fig_x.add_axes((.1, .1, .6, .2))

            plt.errorbar(a1, d1, yerr=f1, ms=2.0, fmt='o', color='blue', label='diff B band')
            plt.plot(a1, c1, '-', color='cornflowerblue', markersize=2, label='uncertanties fit')
            plt.errorbar(a2, d2, yerr=f2, ms=2.0, fmt='o', color='green', label='diff V band')
            plt.plot(a2, c2, '-', color='mediumseagreen', markersize=2, label='uncertanties fit')
            plt.errorbar(a3, d3, yerr=f3, ms=2.0, fmt='o', color='red', label='diff B band')
            plt.plot(a3, c3, '-', color='indianred', markersize=2, label='uncertanties fit')
            plt.errorbar(a4, d4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='diff V band')
            plt.plot(a4, c4, '-', color='orange', markersize=2, label='uncertanties fit')

            barra.update(10)

            plt.xlabel('Scattering Angle (degrees)')
            plt.ylabel('Residual data')
            plt.grid(True)
            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_wave_Q_stokes_' + meto + '.png')
            barra.update(15)
            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['REG Q'].to_numpy()
                c = points['REG Q UNC'].to_numpy()
                d = points['REG Q DIFF'].to_numpy()
                e = points['Q'].to_numpy()
                f = points['error Q'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], np.asarray(f)[w]

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylim(0, 0.8)
                if isinstance(result.params['c'].stderr, float):
                    c_par = round(result.params['c'].stderr, 3)
                else:
                    c_par = result.params['c'].stderr
                plt.ylabel('Q parameter')
                label_text = 'fit parameters:   $c$ = ' + str(round(result.params['c'].value, 3)) + '$\pm$' + str(
                    c_par) + '\n' + 'chi-square: ' + str(
                    round(result.chisqr, 7)) + ',  reduced chi-square: ' + str(
                    round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                leg_text = 'fit with all corrections...\n \n$c_{B}$= ' + str(
                    round(B_par_u[0], 3)) + '\n$c_{V}$= ' + str(round(V_par_u[0], 3)) + '\n$c_{R}$= ' + str(
                    round(R_par_u[0], 3)) + '\n$c_{I}$= ' + str(round(I_par_u[0], 3))
                plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['REG Q'].to_numpy()
                c = points['REG Q UNC'].to_numpy()
                d = points['REG Q DIFF'].to_numpy()
                e = points['Q'].to_numpy()
                f = points['error Q'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], np.asarray(f)[w]
                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_wave_Q_stokes_' + meto + '.png', bbox_inches='tight')
            plt.close()

        TXT.close()

    # ----------------------------------------------------------------------------------------------

    if command == 'ALL' or command == 'fit U regular':
        model = lmfit.Model(func_reg_U)
        # print(model.independent_vars)
        model.set_param_hint('par', min=0.0, max=4.0)
        p = model.make_params(par=np.random.rand())  # , beta1=np.random.rand(), beta2=np.random.rand())
        model.eval(params=p, allvars=[C1field, C2field, C1lua, C2lua])
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee',
        #                    weights=errPOL_OBS, fit_kws={'steps': 500000})
        result = model.fit(data=U_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method=meto)

        # para obter o número de iterações: ite = result.nfev
        # para obter o erro residual do fit: res = result.residual

        model_fit_report = result.fit_report()
        model_name = 'MODEL_' + LABEL + '_' + condition + '_regular_U_stokes_' + meto + '.sav'
        txname = 'REPORT_' + LABEL + '_' + condition + '_regular_U_stokes_' + meto + '.txt'
        TXT = open(txname, "w+")
        lmfit.model.save_modelresult(result, model_name)

        TXT.write('***  First Fit: fit regular for U with only normalization parameter free *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        barra.update(10)

        result_u_reg = [result.params['par'].value]  # , result.params['beta1'].value, result.params['beta2'].value]
        result_u_reg = np.asarray(result_u_reg)
        # TXT.write('\n \n')

        y = func_reg_U([C1field, C2field, C1lua, C2lua], *result_u_reg)
        fit_observations_resume.insert(coluna, 'REG U', y)
        coluna += 1

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
        fit_observations_resume.insert(coluna, 'REG U UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(U_OBS)):
            diff.append(U_OBS[i] - y[i])
        fit_observations_resume.insert(coluna, 'REG U DIFF', diff)
        coluna += 1

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['REG U'].to_numpy()
            c1 = b_points['REG U UNC'].to_numpy()
            d1 = b_points['REG U DIFF'].to_numpy()
            e1 = b_points['U'].to_numpy()
            f1 = b_points['error U'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], np.asarray(d1)[w], \
                                     np.asarray(e1)[
                                         w], np.asarray(f1)[w]

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['REG U'].to_numpy()
            c2 = v_points['REG U UNC'].to_numpy()
            d2 = v_points['REG U DIFF'].to_numpy()
            e2 = v_points['U'].to_numpy()
            f2 = v_points['error U'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], np.asarray(d2)[w], \
                                     np.asarray(e2)[
                                         w], np.asarray(f2)[w]

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['REG U'].to_numpy()
            c3 = r_points['REG U UNC'].to_numpy()
            d3 = r_points['REG U DIFF'].to_numpy()
            e3 = r_points['U'].to_numpy()
            f3 = r_points['error U'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], np.asarray(d3)[w], \
                                     np.asarray(e3)[
                                         w], np.asarray(f3)[w]

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['REG U'].to_numpy()
            c4 = i_points['REG U UNC'].to_numpy()
            d4 = i_points['REG U DIFF'].to_numpy()
            e4 = i_points['U'].to_numpy()
            f4 = i_points['error U'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], np.asarray(d4)[w], \
                                     np.asarray(e4)[
                                         w], np.asarray(f4)[w]

            barra.update(10)

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            h1 = np.add(b2, c2)
            h2 = np.subtract(b2, c2)
            plt.fill_between(a2, h2, h1, where=(h2 < h1), interpolate=True, color='beige')

            j1 = np.add(b3, c3)
            j2 = np.subtract(b3, c3)
            plt.fill_between(a3, j2, j1, where=(j2 < j1), interpolate=True, color='mistyrose')

            k1 = np.add(b4, c4)
            k2 = np.subtract(b4, c4)
            plt.fill_between(a4, k2, k1, where=(k2 < k1), interpolate=True, color='antiquewhite')

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')

            plt.ylim(-0.6, 0.6)
            plt.ylabel('U parameter')
            if isinstance(result.params['par'].stderr, float):
                par_par = round(result.params['par'].stderr, 3)
            else:
                par_par = result.params['par'].stderr
            label_text = 'fit parameters:   $P_{norm}$ = ' + str(round(result.params['par'].value, 3)) + '$\pm$' + str(
                par_par) + '\n' + 'chi-square: ' + str(
                round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
                round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            barra.update(10)

            fig_x.add_axes((.1, .1, .6, .2))

            plt.errorbar(a1, d1, yerr=f1, ms=2.0, fmt='o', color='blue', label='diff B band')
            plt.plot(a1, c1, '-', color='cornflowerblue', markersize=2, label='uncertanties fit')
            plt.errorbar(a2, d2, yerr=f2, ms=2.0, fmt='o', color='green', label='diff V band')
            plt.plot(a2, c2, '-', color='mediumseagreen', markersize=2, label='uncertanties fit')
            plt.errorbar(a3, d3, yerr=f3, ms=2.0, fmt='o', color='red', label='diff B band')
            plt.plot(a3, c3, '-', color='indianred', markersize=2, label='uncertanties fit')
            plt.errorbar(a4, d4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='diff V band')
            plt.plot(a4, c4, '-', color='orange', markersize=2, label='uncertanties fit')

            plt.xlabel('Scattering Angle (degrees)')
            plt.ylabel('Residual data')
            plt.grid(True)
            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_regular_U_stokes_' + meto + '.png')
            barra.update(10)
            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['REG U'].to_numpy()
                c = points['REG U UNC'].to_numpy()
                d = points['REG U DIFF'].to_numpy()
                e = points['U'].to_numpy()
                f = points['error U'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], np.asarray(f)[w]
                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylim(0, 0.8)
                plt.ylabel('U parameter')
                if isinstance(result.params['par'].stderr, float):
                    par_par = round(result.params['par'].stderr, 3)
                else:
                    par_par = result.params['par'].stderr
                label_text = 'fit parameters:   $P_{norm}$ = ' + str(
                    round(result.params['par'].value, 3)) + '$\pm$' + str(
                    par_par) + '\n' + 'chi-square: ' + str(
                    round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
                    round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['REG U'].to_numpy()
                c = points['REG U UNC'].to_numpy()
                d = points['REG U DIFF'].to_numpy()
                e = points['U'].to_numpy()
                f = points['error U'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], np.asarray(f)[w]

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_regular_U_stokes_' + meto + '.png', bbox_inches='tight')
            plt.close()
        TXT.close()

    # ----------------------------------------------------------------------------------------------

    if command == 'ALL' or command == 'fit U wav':
        model = lmfit.Model(func_wav_U)
        # print(model.independent_vars)
        model.set_param_hint('c', min=-4, max=0)
        p = model.make_params(c=np.random.rand())
        model.eval(params=p, allvars=[C1field, C2field, C1lua, C2lua, BAND, U_BAND])
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee',
        #                    weights=errPOL_OBS, fit_kws={'steps': 500000})
        result = model.fit(data=U_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, BAND, U_BAND], method=meto)

        model_fit_report = result.fit_report()
        model_name = 'MODEL_' + LABEL + '_' + condition + '_wav_U_stokes_' + meto + '.sav'
        lmfit.model.save_modelresult(result, model_name)

        txname = 'REPORT_' + LABEL + '_' + condition + '_wav_U_stokes_' + meto + '.txt'
        TXT = open(txname, "w+")

        TXT.write('***  Second Fit: fit considering the wavelength of light *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        barra.update(10)

        result_u_wav = [result.params['c'].value]
        result_u_wav = np.asarray(result_u_wav)

        y = func_wav_AOP([C1field, C2field, C1lua, C2lua, BAND, U_BAND], *result_u_wav)
        fit_observations_resume.insert(coluna, 'WAV U', y)
        coluna += 1

        rsd = result.eval_uncertainty()
        fit_observations_resume.insert(coluna, 'WAV U UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(U_OBS)):
            diff.append(U_OBS[i] - y[i])
        fit_observations_resume.insert(coluna, 'WAV U DIFF', diff)
        coluna += 1

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['WAV U'].to_numpy()
            c1 = b_points['WAV U UNC'].to_numpy()
            d1 = b_points['WAV U DIFF'].to_numpy()
            e1 = b_points['U'].to_numpy()
            f1 = b_points['error U'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], np.asarray(d1)[w], \
                                     np.asarray(e1)[
                                         w], np.asarray(f1)[w]

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['WAV U'].to_numpy()
            c2 = v_points['WAV U UNC'].to_numpy()
            d2 = v_points['WAV U DIFF'].to_numpy()
            e2 = v_points['U'].to_numpy()
            f2 = v_points['error U'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], np.asarray(d2)[w], \
                                     np.asarray(e2)[
                                         w], np.asarray(f2)[w]

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['WAV U'].to_numpy()
            c3 = r_points['WAV U UNC'].to_numpy()
            d3 = r_points['WAV U DIFF'].to_numpy()
            e3 = r_points['U'].to_numpy()
            f3 = r_points['error U'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], np.asarray(d3)[w], \
                                     np.asarray(e3)[
                                         w], np.asarray(f3)[w]

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['WAV U'].to_numpy()
            c4 = i_points['WAV U UNC'].to_numpy()
            d4 = i_points['WAV U DIFF'].to_numpy()
            e4 = i_points['U'].to_numpy()
            f4 = i_points['error U'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], np.asarray(d4)[w], \
                                     np.asarray(e4)[
                                         w], np.asarray(f4)[w]

            barra.update(10)

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            h1 = np.add(b2, c2)
            h2 = np.subtract(b2, c2)
            plt.fill_between(a2, h2, h1, where=(h2 < h1), interpolate=True, color='beige')

            j1 = np.add(b3, c3)
            j2 = np.subtract(b3, c3)
            plt.fill_between(a3, j2, j1, where=(j2 < j1), interpolate=True, color='mistyrose')

            k1 = np.add(b4, c4)
            k2 = np.subtract(b4, c4)
            plt.fill_between(a4, k2, k1, where=(k2 < k1), interpolate=True, color='antiquewhite')

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')

            plt.ylim(-0.6, 0.6)
            if isinstance(result.params['c'].stderr, float):
                c_par = round(result.params['c'].stderr, 3)
            else:
                c_par = result.params['c'].stderr
            plt.ylabel('U parameter')
            label_text = 'fit parameters:   $c$ = ' + str(round(result.params['c'].value, 3)) + '$\pm$' + str(
                c_par) + '\n' + 'chi-square: ' + str(
                round(result.chisqr, 7)) + ',  reduced chi-square: ' + str(
                round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            leg_text = 'fit with all corrections...\n \n$c_{B}$= ' + str(
                round(B_par_u[0], 3)) + '\n$c_{V}$= ' + str(round(V_par_u[0], 3)) + '\n$c_{R}$= ' + str(
                round(R_par_u[0], 3)) + '\n$c_{I}$= ' + str(round(I_par_u[0], 3))
            plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            barra.update(10)

            fig_x.add_axes((.1, .1, .6, .2))

            plt.errorbar(a1, d1, yerr=f1, ms=2.0, fmt='o', color='blue', label='diff B band')
            plt.plot(a1, c1, '-', color='cornflowerblue', markersize=2, label='uncertanties fit')
            plt.errorbar(a2, d2, yerr=f2, ms=2.0, fmt='o', color='green', label='diff V band')
            plt.plot(a2, c2, '-', color='mediumseagreen', markersize=2, label='uncertanties fit')
            plt.errorbar(a3, d3, yerr=f3, ms=2.0, fmt='o', color='red', label='diff B band')
            plt.plot(a3, c3, '-', color='indianred', markersize=2, label='uncertanties fit')
            plt.errorbar(a4, d4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='diff V band')
            plt.plot(a4, c4, '-', color='orange', markersize=2, label='uncertanties fit')

            plt.xlabel('Scattering Angle (degrees)')
            plt.ylabel('Residual data')
            plt.grid(True)
            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_wav_U_stokes_' + meto + '.png')
            barra.update(15)
            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['WAV U'].to_numpy()
                c = points['WAV U UNC'].to_numpy()
                d = points['WAV U DIFF'].to_numpy()
                e = points['U'].to_numpy()
                f = points['error U'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], np.asarray(f)[w]

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylim(0, 0.8)
                if isinstance(result.params['c'].stderr, float):
                    c_par = round(result.params['c'].stderr, 3)
                else:
                    c_par = result.params['c'].stderr
                plt.ylabel('U parameter')
                label_text = 'fit parameters:   $c$ = ' + str(round(result.params['c'].value, 3)) + '$\pm$' + str(
                    c_par) + '\n' + 'chi-square: ' + str(
                    round(result.chisqr, 7)) + ',  reduced chi-square: ' + str(
                    round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                leg_text = 'fit with all corrections...\n \n$c_{B}$= ' + str(
                    round(B_par_u[0], 3)) + '\n$c_{V}$= ' + str(round(V_par_u[0], 3)) + '\n$c_{R}$= ' + str(
                    round(R_par_u[0], 3)) + '\n$c_{I}$= ' + str(round(I_par_u[0], 3))
                plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
                barra.update(int(20 / len(band)))
                if item == 'B':
                    par_wave = 437
                    cor_line = 'cornflowerblue'
                    cor = 'blue'
                    cor_unc = 'lavender'
                    lab_wave = 'B'
                if item == 'V':
                    par_wave = 555
                    cor_line = 'mediumseagreen'
                    cor = 'green'
                    cor_unc = 'beige'
                    lab_wave = 'V'
                if item == 'R':
                    par_wave = 655
                    cor_line = 'indianred'
                    cor = 'red'
                    cor_unc = 'mistyrose'
                    lab_wave = 'R'
                if item == 'I':
                    par_wave = 768
                    cor_line = 'orange'
                    cor = 'darkorange'
                    cor_unc = 'antiquewhite'
                    lab_wave = 'I'

                points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == par_wave]
                a = points['GAMMA'].to_numpy()
                b = points['WAV U'].to_numpy()
                c = points['WAV U UNC'].to_numpy()
                d = points['WAV U DIFF'].to_numpy()
                e = points['U'].to_numpy()
                f = points['error U'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], np.asarray(d)[w], \
                                   np.asarray(e)[w], np.asarray(f)[w]
                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + condition + '_wav_U_stokes_' + meto + '.png', bbox_inches='tight')
            plt.close()

        TXT.close()


    barra.close()

    if search('aop', command):
        print('Processo concluído')

    else:
        print('Pretende guardar mapas das observações?\n')
        answer = str(input('Please answer Yes or No.\n'))

        if answer == 'Yes':
            print('Do you wish plots with only individuals correction?\n')
            answer_2 = str(input('Please answer Yes or No.\n'))
            if answer_2 == 'Yes':
                plot = 'individual'
                answer_3 = str(input('Which one? Regular, Mix, Depolarization or Wave?\n'))

                if answer_3 == 'Regular':
                    print('*** Plotting B band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('B', condition, B_par_reg, Rpar, P1=plot, P2=answer_3)

                    print('*** Plotting V band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('V', condition, V_par_reg, Rpar, P1=plot, P2=answer_3)

                    print('*** Plotting R band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('R', condition, R_par_reg, Rpar, P1=plot, P2=answer_3)

                    print('*** Plotting I band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('I', condition, I_par_reg, Rpar, P1=plot, P2=answer_3)

                if answer_3 == 'Depolarization':
                    print('*** Plotting B band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('B', condition, B_par_dep, Rpar, P1=plot, P2=answer_3)

                    print('*** Plotting V band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('V', condition, V_par_dep, Rpar, P1=plot, P2=answer_3)

                    print('*** Plotting R band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('R', condition, R_par_dep, Rpar, P1=plot, P2=answer_3)

                    print('*** Plotting I band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('I', condition, I_par_dep, Rpar, P1=plot, P2=answer_3)

                else:
                    print('Wrong input!!')

            if answer_2 == 'No':
                plot = 'complex'
                if command != 'fit wavelength':
                    print('*** Plotting B band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('B', condition, B_par_reg, Rpar, P1=plot, P2=command)

                    print('*** Plotting V band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('V', condition, V_par_reg, Rpar, P1=plot, P2=command)

                    print('*** Plotting R band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('R', condition, R_par_reg, Rpar, P1=plot, P2=command)

                    print('*** Plotting I band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('I', condition, I_par_reg, Rpar, P1=plot, P2=command)

            else:
                print('Wrong input!!')

            print('Processo concluído')

        if answer == 'No':
            print('Processo concluído')

    return Rpar, fit_observations_resume



