import numpy as np
import sky


def func_simple_reg_DOP(gamma, A):
    DOP = A * ((np.sin(gamma * np.pi / 180)) ** 2) / (1 + np.cos(gamma * np.pi / 180) ** 2)

    return DOP


def func_reg_DOP(allvars, A):
    crds1, crds2, cLUA1, cLUA2 = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    for i in range(len(crds1)):
        gamma = sky.func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        DOP[i] = A * ((np.sin(gamma)) ** 2) / (1 + np.cos(gamma) ** 2)

    return DOP


def func_simple_dep_DOP(gamma, P):
    DOP = ((np.sin(gamma * np.pi / 180)) ** 2) / ((1 + P) / (1 - P) + np.cos(gamma * np.pi / 180) ** 2)

    return DOP


def func_dep_DOP(allvars, P):
    crds1, crds2, cLUA1, cLUA2 = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        gamma = sky.func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        DOP[i] = ((np.sin(gamma)) ** 2) / ((1 + P) / (1 - P) + np.cos(gamma) ** 2)

    return DOP


# ----------------------------------------------------------------------------------------------

def func_simple_hor_DOP(theta_lua, gamma, par_wave, N):
    DOP = np.cos(theta_lua) ** (1 / N) * ((np.sin(gamma * np.pi / 180)) ** 2) / (
            1 + np.cos(gamma * np.pi / 180) ** 2) * par_wave

    return DOP


def func_hor_DOP(allvars, N):
    crds1, crds2, cLUA1, cLUA2, par_wave = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        gamma = sky.func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        DOP[i] = np.cos(cLUA1[i]) ** (1 / N) * ((np.sin(gamma * np.pi / 180)) ** 2) / (
                1 + np.cos(gamma * np.pi / 180) ** 2) * par_wave[i]

    return DOP


def func_seeing_DOP(allvars, k, d):
    crds1, crds2, cLUA1, cLUA2, seeing, par_wave = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        gamma = sky.func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        DOP[i] = np.exp(-seeing[i] / k + d) * ((np.sin(gamma)) ** 2) / (1 + np.cos(gamma) ** 2) * par_wave[i]

    return DOP


def func_simple_seeing_DOP(gamma, seeing, par_wave, k, d):
    DOP = np.exp(-seeing / k + d) * ((np.sin(gamma * np.pi / 180)) ** 2) / (
            1 + np.cos(gamma * np.pi / 180) ** 2) * par_wave

    return DOP


# ------------------------------------------------------------------------------------------------

def func_simple_atm_DOP(gamma, theta_lua, seeing, N, k, d):
    DOP = np.cos(theta_lua) ** (1 / N) * np.exp(-seeing / k + d) * ((np.sin(gamma * np.pi / 180)) ** 2) / (
            1 + np.cos(gamma * np.pi / 180) ** 2)

    return DOP


def func_atm_DOP(allvars, N, k, d):
    crds1, crds2, cLUA1, cLUA2, seeing = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        gamma = sky.func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        DOP[i] = np.cos(cLUA1) ** (1 / N) * np.exp(-seeing[i] / k + d) * ((np.sin(gamma)) ** 2) / (
                1 + np.cos(gamma) ** 2)

    return DOP


def func_simple_atm_amp_DOP(gamma, theta_lua, seeing, par_wave, N, k, d):
    DOP = par_wave * np.cos(theta_lua) ** (1 / N) * np.exp(-seeing / k + d) * ((np.sin(gamma * np.pi / 180)) ** 2) / (
            1 + np.cos(gamma * np.pi / 180) ** 2)

    return DOP


def func_atm_amp_DOP(allvars, N, k, d):
    crds1, crds2, cLUA1, cLUA2, seeing, par_wave = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        gamma = sky.func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        DOP[i] = par_wave[i] * np.cos(cLUA1) ** (1 / N) * np.exp(-seeing[i] / k + d) * ((np.sin(gamma)) ** 2) / (
                1 + np.cos(gamma) ** 2)

    return DOP


# -------------------------------------------------------------------------------------------------

def func_wav_DOP(allvars, c):
    crds1, crds2, cLUA1, cLUA2, wave, seeing, n_par, k_par, d_par = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        gamma = sky.func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        DOP[i] = np.exp(-seeing[i] / k_par[i] + d_par[i]) * np.cos(cLUA1[i]) ** (1 / n_par[i]) * (
                    (np.sin(gamma)) ** 2) / (1 + np.cos(gamma) ** 2) * wave[i] ** c

    return DOP


def func_simple_wav_DOP(theta_lua, gamma, wavel, seeing, n_par, k_par, d_par, c):
    DOP = np.exp(-seeing / k_par + d_par) * (np.cos(theta_lua) ** (1 / n_par)) * (
                (np.sin(gamma * np.pi / 180)) ** 2) / (
                  1 + np.cos(gamma * np.pi / 180) ** 2) * wavel ** c

    return DOP


# -------------------------------------------------------------------------------------------------

def func_mix_DOP(allvars, P, N, k, d):
    crds1, crds2, cLUA1, cLUA2, seeing = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        gamma = sky.func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        DOP[i] = np.cos(cLUA1) ** (1 / N) * np.exp(-seeing[i] / k + d) * ((np.sin(gamma)) ** 2) / ((1 + P) / (1 - P) + np.cos(gamma) ** 2)

    return DOP
