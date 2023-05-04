import numpy as np
import pandas as pd
import coordinates
import lmfit
import matplotlib.pyplot as plt
import astropy.units as u
import albedo
import field_functions
import moon_functions
from tqdm import tqdm
from re import search
import matplotlib as mpl
import mplcursors


def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

        # return string
    return str1


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


def func_gamma(theta_obs, phi_obs, theta_lua, phi_lua):
    gamma = np.arccos(
        np.sin(theta_lua) * np.sin(theta_obs) * np.cos(phi_obs - phi_lua) + np.cos(theta_lua) * np.cos(theta_obs))

    return gamma


def _Lentz_Dn(z, N):
    """
    Compute the logarithmic derivative of the Ricatti-Bessel function.
    Args:
        z: function argument
        N: order of Ricatti-Bessel function
    Returns:
        This returns the Ricatti-Bessel function of order N with argument z
        using the continued fraction technique of Lentz, Appl. Opt., 15,
        668-671, (1976).
    """
    zinv = 2.0 / z
    alpha = (N + 0.5) * zinv
    aj = -(N + 1.5) * zinv
    alpha_j1 = aj + 1 / alpha
    alpha_j2 = aj
    ratio = alpha_j1 / alpha_j2
    runratio = alpha * ratio

    while np.abs(np.abs(ratio) - 1.0) > 1e-12:
        aj = zinv - aj
        alpha_j1 = 1.0 / alpha_j1 + aj
        alpha_j2 = 1.0 / alpha_j2 + aj
        ratio = alpha_j1 / alpha_j2
        zinv *= -1
        runratio = ratio * runratio

    return -N / z + runratio


def _D_downwards(z, N, D):
    """
    Compute the logarithmic derivative by downwards recurrence.
    Args:
        z: function argument
        N: order of Ricatti-Bessel function
        D: gets filled with the Ricatti-Bessel function values for orders
           from 0 to N for an argument z using the downwards recurrence relations.
    """
    last_D = _Lentz_Dn(z, N)
    for n in range(N, 0, -1):
        last_D = n / z - 1.0 / (last_D + n / z)
        D[n - 1] = last_D


def _D_upwards(z, N, D):
    """
    Compute the logarithmic derivative by upwards recurrence.
    Args:
        z: function argument
        N: order of Ricatti-Bessel function
        D: gets filled with the Ricatti-Bessel function values for orders
           from 0 to N for an argument z using the upwards recurrence relations.
    """
    exp = np.exp(-2j * z)
    D[1] = -1 / z + (1 - exp) / ((1 - exp) / z - 1j * (1 + exp))
    for n in range(2, N):
        D[n] = 1 / (n / z - D[n - 1]) - n / z


def _D_calc(m, x, N):
    """
    Compute the logarithmic derivative using best method.
    Args:
        m: the complex index of refraction of the sphere
        x: the size parameter of the sphere
        N: order of Ricatti-Bessel function
    Returns:
        The values of the Ricatti-Bessel function for orders from 0 to N.
    """
    n = m.real
    kappa = np.abs(m.imag)
    D = np.zeros(N, dtype=np.complex128)

    if n < 1 or n > 10 or kappa > 10 or x * kappa >= 3.9 - 10.8 * n + 13.78 * n ** 2:
        _D_downwards(m * x, N, D)
    else:
        _D_upwards(m * x, N, D)
    return D


def func_reg_DOP(allvars, A, m_part, x):
    crds1, crds2, cLUA1, cLUA2 = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    for i in range(len(crds1)):
        # vector de Stokes para a luz natural
        s_in = np.array([1, 0, 0, 0], dtype=float)

        # matriz de rotação
        r_in = np.zeros((4, 4), dtype=float)
        r_out = np.zeros((4, 4), dtype=float)

        # matriz reduzida para o  Rayleigh scattering
        m = np.zeros((4, 4), dtype=float)
        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        nangles = np.cos(gamma)

        nstop = int(x + 4.05 * x ** 0.33333 + 2.0) + 1

        a = np.zeros(nstop - 1, dtype=np.complex128)
        b = np.zeros(nstop - 1, dtype=np.complex128)

        psi_nm1 = np.sin(x)  # nm1 = n-1 = 0
        psi_n = psi_nm1 / x - np.cos(x)  # n = 1

        xi_nm1 = complex(psi_nm1, np.cos(x))
        xi_n = complex(psi_n, np.cos(x) / x + np.sin(x))

        if m_part.real > 0.0:
            D = _D_calc(m_part, x, nstop + 1)

            for n in range(1, nstop):
                temp = D[n] / m_part + n / x
                a[n - 1] = (temp * psi_n - psi_nm1) / (temp * xi_n - xi_nm1)
                temp = D[n] * m_part + n / x
                b[n - 1] = (temp * psi_n - psi_nm1) / (temp * xi_n - xi_nm1)
                xi = (2 * n + 1) * xi_n / x - xi_nm1
                xi_nm1 = xi_n
                xi_n = xi
                psi_nm1 = psi_n
                psi_n = xi_n.real

        else:
            for n in range(1, nstop):
                a[n - 1] = (n * psi_n / x - psi_nm1) / (n * xi_n / x - xi_nm1)
                b[n - 1] = psi_n / xi_n
                xi = (2 * n + 1) * xi_n / x - xi_nm1
                xi_nm1 = xi_n
                xi_n = xi
                psi_nm1 = psi_n
                psi_n = xi_n.real

        # amplitudes
        S1 = complex()
        S2 = complex()

        nstop = len(a)

        pi_nm2 = 0
        pi_nm1 = 1
        for n in range(1, nstop):
            tau_nm1 = n * nangles * pi_nm1 - (n + 1) * pi_nm2
            S1 += (2 * n + 1) * (pi_nm1 * a[n - 1] + tau_nm1 * b[n - 1]) / (n + 1) / n
            S2 += (2 * n + 1) * (tau_nm1 * a[n - 1] + pi_nm1 * b[n - 1]) / (n + 1) / n

            temp = pi_nm1
            pi_nm1 = ((2 * n + 1) * nangles * pi_nm1 - (n + 1) * pi_nm2) / n
            pi_nm2 = temp

        n = np.arange(1, len(a) + 1)
        cn = 2.0 * n + 1.0
        qext = 2 * np.sum(cn * (a.real + b.real)) / x ** 2

        normalization = np.sqrt(np.pi * x ** 2 * qext)

        S1 /= normalization
        S2 /= normalization

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

        i1 = np.abs(S1) ** 2
        i2 = np.abs(S2) ** 2
        i3 = S2 * np.conj(S1)
        i4 = S1 * np.conj(S2)

        P11 = i1.astype(np.float64) + i2.astype(np.float64)
        P12 = i2.astype(np.float64) - i1.astype(np.float64)
        P33 = i3 + i4
        P33 = float(P33.real)
        P34 = -(i4 - i3)
        P34 = float(P34.imag)

        # matriz reduzida para Rayleigh scattering
        m[0, :] = [P11, P12, 0, 0]
        m[1, :] = [P12, P11, 0, 0]
        m[2, :] = [0, 0, P33, -P34]
        m[3, :] = [0, 0, P34, P33]

        r_out = np.dot(r_out, m)

        p = np.dot(r_out, r_in)

        # parâmetros de Stokes
        # S_out = [I,Q,U,V]^T
        I = np.dot(p[0, :], s_in)
        Q = np.dot(p[1, :], s_in)
        U = np.dot(p[2, :], s_in)
        # V = np.dot(p[3, :], s_in)

        DOP[i] = A * np.sqrt(Q ** 2 + U ** 2) / I


def FIT(band=None, condition=None, command='ALL'):
    df = pd.read_csv('data_output.csv', sep=',')

    df.isnull().sum()
    df.dropna(axis=1)

    DATA = df

    COND = ['ok', 'clouds', 'sun']
    CONDT = ['ok', 'clouds', 'sun']

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
        BANDA = DATA['BANDA'].to_numpy()
        conjunto = ['B', 'V', 'R', 'I']

    if band is not None:
        conjunto = band
        if isinstance(band, str):
            DATA = DATA[DATA['BANDA'] == band]
            BANDA = DATA['BANDA'].to_numpy()
        else:
            DATA = DATA[DATA['BANDA'].isin(band)]
            BANDA = DATA['BANDA'].to_numpy()

    field = DATA['FIELD'].to_numpy()
    RA = DATA['RA'].to_numpy()
    DEC = DATA['DEC'].to_numpy()
    MED = DATA['OBS TIME MED'].to_numpy()
    Ival = DATA['I'].to_numpy()
    Qval = DATA['Q'].to_numpy()
    errQval = DATA['error Q'].to_numpy()
    Uval = DATA['U'].to_numpy()
    errUval = DATA['U error'].to_numpy()
    seen = DATA['SEEING'].to_numpy()
    theta_moon = DATA['THETA MOON'].to_numpy()
    phi_moon = DATA['PHI MOON'].to_numpy()
    albedo = DATA['ALBEDO'].to_numpy()
    wave = DATA['WAVELENGTH'].to_numpy()
    theta_field = DATA['THETA FIELD'].to_numpy()
    phi_field = DATA['PHI FIELD'].to_numpy()
    gamma = DATA['GAMMA'].to_numpy()
    AOPval = DATA['AOP'].to_numpy()
    errAOPval = DATA['AOP error'].to_numpy()
    POLval = DATA['POL OBS'].to_numpy()
    errPOLval = DATA['POL OBS error'].to_numpy()
    theta_sun = DATA['THETA SUN'].to_numpy()
    phi_sun = DATA['PHI SUN'].to_numpy()

    C1field, C2field = np.asarray(theta_field, dtype=np.float32), np.asarray(phi_field, dtype=np.float32)
    C1lua, C2lua = np.asarray(theta_moon, dtype=np.float32), np.asarray(phi_moon, dtype=np.float32)
    C1sol, C2sol = np.asarray(theta_sun, dtype=np.float32), np.asarray(phi_sun, dtype=np.float32)
    POL_OBS, errPOL_OBS = np.asarray(POLval, dtype=np.float32), np.asarray(errPOLval, dtype=np.float32)
    GAMMA, AOP = np.asarray(gamma, dtype=np.float32), np.asarray(AOPval, dtype=np.float32)
    ALBEDO, SEEING = np.asarray(albedo, dtype=np.float32), np.asarray(seen, dtype=np.float32)

    # LABEL = listToString(conjunto)
    # LAB = listToString(COND)
    meto = 'leastsq'

    if command == 'ALL' or command == 'individuals regular':
        B_par, B_chisqr, B_bic, result_data_B = field_functions.fit_base('B', condition,
                                                                         command='regular_mie')
        V_par, V_chisqr, V_bic, result_data_V = field_functions.fit_base('V', condition,
                                                                         command='regular_mie')
        R_par, R_chisqr, R_bic, result_data_R = field_functions.fit_base('R', condition,
                                                                         command='regular_mie')
        I_par, I_chisqr, I_bic, result_data_I = field_functions.fit_base('I', condition,
                                                                         command='regular_mie')

        A_ind_reg = [B_par[0], V_par[0], R_par[0], I_par[0]]
        x_ind_reg = [B_par[1], V_par[1], R_par[1], I_par[1]]
        m_part_ind_reg = [B_par[2], V_par[2], R_par[2], I_par[2]]
        chisqr_ind_reg = [B_chisqr, V_chisqr, R_chisqr, I_chisqr]
        bic_ind_reg = [B_bic, V_bic, R_bic, I_bic]
        index = ['B', 'V', 'R', 'I']

        df_ind_reg = pd.DataFrame(
            {'A': A_ind_reg, 'x': x_ind_reg, '$m_{part}$': m_part_ind_reg, 'Chi-square': chisqr_ind_reg, 'BIC': bic_ind_reg}, index=index)

        df_ind_reg.plot(rot=0, subplots=True, style=".--")
        plt.savefig('indvidual_multi_dop_reg_fit_separate_parameters.png')
        plt.pause(2)
        plt.close()

        column_names = ['FIELD', 'BAND', 'CONDITION', 'GAMMA', 'POL', 'ERROR POL', 'ALBEDO', 'FIT IND', 'FIT IND UNC',
                        'FIT IND DIFF']

        result_data_reg = pd.DataFrame(columns=column_names)
        result_data_reg = pd.concat([result_data_reg, result_data_B])
        result_data_reg = pd.concat([result_data_reg, result_data_V])
        result_data_reg = pd.concat([result_data_reg, result_data_R])
        result_data_reg = pd.concat([result_data_reg, result_data_I])

        field_functions.plot_all(result_data_reg, command='simple_regular_multi')

        if command == 'individuals regular':
            result_data_reg.to_csv("output.csv")
            df_ind_reg.to_csv("output_par.csv")
            exit(df_ind_reg)











