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


def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

        # return string
    return str1


# Fit for amplitude normalization


def func_reg_DOP(allvars, par):
    crds1, crds2, cLUA1, cLUA2 = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        DOP[i] = par * ((np.sin(gamma)) ** 2) / (1 + np.cos(gamma) ** 2)

    return DOP


def func_simple_reg_DOP(gamma, par):
    DOP = par * ((np.sin(gamma * np.pi / 180)) ** 2) / (1 + np.cos(gamma * np.pi / 180) ** 2)
    return DOP


# Fit the horizon correction

def func_hor_DOP(allvars, N):
    crds1, crds2, cLUA1, cLUA2, par_wave = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        DOP[i] = np.cos(cLUA1[i]) ** (1 / N) * ((np.sin(gamma)) ** 2) / (1 + np.cos(gamma) ** 2) * par_wave[i]

    return DOP


def func_simple_hor_DOP(theta_lua, gamma, par_wave, N):
    DOP = np.cos(theta_lua) ** (1 / N) * ((np.sin(gamma * np.pi / 180)) ** 2) / (
            1 + np.cos(gamma * np.pi / 180) ** 2) * par_wave

    return DOP


def func_simple_hor(theta_lua, gamma, N):
    DOP = np.cos(theta_lua) ** (1 / N) * ((np.sin(gamma * np.pi / 180)) ** 2) / (
            1 + np.cos(gamma * np.pi / 180) ** 2)

    return DOP


# Fit the depolarization correction

def func_dep_DOP(allvars, P):
    crds1, crds2, cLUA1, cLUA2 = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        DOP[i] = ((np.sin(gamma)) ** 2) / ((1 + P) / (1 - P) + np.cos(gamma) ** 2)

    return DOP


def func_depo_DOP(allvars, P):
    crds1, crds2, cLUA1, cLUA2, par_wave = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        DOP[i] = ((np.sin(gamma)) ** 2) / ((1 + P) / (1 - P) + np.cos(gamma) ** 2) * par_wave[i]

    return DOP


def func_simple_dep_DOP(gamma, P):
    DOP = ((np.sin(gamma * np.pi / 180)) ** 2) / ((1 + P) / (1 - P) + np.cos(gamma * np.pi / 180) ** 2)

    return DOP


def func_simple_depo_DOP(gamma, par_wave, P):
    DOP = ((np.sin(gamma * np.pi / 180)) ** 2) / ((1 + P) / (1 - P) + np.cos(gamma * np.pi / 180) ** 2) * par_wave

    return DOP


# Fit the turbulence correction

def func_turb_DOP(allvars, k, d):
    crds1, crds2, cLUA1, cLUA2, par_int = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        DOP[i] = par_int[i] * np.exp(-3 / k + d) * ((np.sin(gamma)) ** 2) / (1 + np.cos(gamma) ** 2)

    return DOP


# Fit the seeing correction

def func_seeing_DOP(allvars, k, d):
    crds1, crds2, cLUA1, cLUA2, seeing, par_wave = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        DOP[i] = np.exp(-seeing[i] / k + d) * ((np.sin(gamma)) ** 2) / (1 + np.cos(gamma) ** 2) * par_wave[i]

    return DOP


def func_simple_seeing_DOP(gamma, seeing, par_wave, k, d):
    DOP = np.exp(-seeing / k + d) * ((np.sin(gamma * np.pi / 180)) ** 2) / (
            1 + np.cos(gamma * np.pi / 180) ** 2) * par_wave

    return DOP


def func_simple_seeing(gamma, seeing, k, d):
    DOP = np.exp(-seeing / k + d) * ((np.sin(gamma * np.pi / 180)) ** 2) / (
            1 + np.cos(gamma * np.pi / 180) ** 2)

    return DOP


# Fit the Atmospheric Corrections

def func_simple_mix(gamma, theta_lua, seeing, N, k, d):
    DOP = np.cos(theta_lua) ** (1 / N) * np.exp(-seeing / k + d) * ((np.sin(gamma * np.pi / 180)) ** 2) / (
            1 + np.cos(gamma * np.pi / 180) ** 2)

    return DOP


def func_mix(allvars, N, k, d):
    crds1, crds2, cLUA1, cLUA2, seeing = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        DOP[i] = np.cos(cLUA1) ** (1 / N) * np.exp(-seeing[i] / k + d) * ((np.sin(gamma)) ** 2) / (1 + np.cos(gamma) ** 2)

    return DOP


def func_simple_mix_DOP(gamma, theta_lua, seeing, par_wave, N, k, d):
    DOP = par_wave * np.cos(theta_lua) ** (1 / N) * np.exp(-seeing / k + d) * ((np.sin(gamma * np.pi / 180)) ** 2) / (
            1 + np.cos(gamma * np.pi / 180) ** 2)

    return DOP


# Fit  considering the Sun influence

def func_sun_DOP(allvars, par):
    crds1, crds2, cLUA1, cLUA2, cSOL1, cSOL2, alb, par_wave = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):

        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])
        gamma_sol = func_gamma(crds1[i], crds2[i], cSOL1[i], cSOL2[i])

        dop = ((np.sin(gamma)) ** 2) / (1 + np.cos(gamma) ** 2) * par_wave[i]
        DOP_sol = ((np.sin(gamma_sol)) ** 2) / (1 + np.cos(gamma_sol) ** 2) * par_wave[i]

        alt = 90 - 180 / np.pi * cSOL1[i]

        if alt < -18:
            DOP[i] = dop
        if alt >= -18:
            DOP[i] = DOP_sol + dop * alb[i] * par  # * (1 - albedo)

    return DOP


def func_simple_sun_DOP(gamma, gamma_sol, theta_sol, alb, par_wave, par):
    DOP = ((np.sin(gamma * np.pi / 180)) ** 2) / (1 + np.cos(gamma * np.pi / 180) ** 2) * par_wave

    if (90 - 180 / np.pi * theta_sol >= -18).any():
        DOP = DOP * alb * par + ((np.sin(gamma_sol * np.pi / 180)) ** 2) / (
                1 + np.cos(gamma_sol * np.pi / 180) ** 2) * par_wave
    return DOP


# Fit the wavelength factor

def func_wav_DOP(allvars, c):
    crds1, crds2, cLUA1, cLUA2, banda = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        DOP[i] = ((np.sin(gamma)) ** 2) / (1 + np.cos(gamma) ** 2) * banda[i] ** c

    return DOP


def func_wav(allvars, c):
    crds1, crds2, cLUA1, cLUA2, banda, seeing, n_par, k_par, d_par = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        DOP[i] = np.exp(-seeing[i] / k_par[i] + d_par[i]) * np.cos(cLUA1[i]) ** (1 / n_par[i]) * ((np.sin(gamma)) ** 2) / (1 + np.cos(gamma) ** 2) * banda[i] ** c

    return DOP


def func_simple_wav_DOP(gamma, wavel, c):
    DOP = ((np.sin(gamma * np.pi / 180)) ** 2) / (1 + np.cos(gamma * np.pi / 180) ** 2) * wavel ** c

    return DOP


def func_simple_wav(theta_lua, gamma, wavel, seeing, n_par, k_par, d_par, c):
    DOP = np.exp(-seeing / k_par + d_par) * (np.cos(theta_lua) ** (1 / n_par))* ((np.sin(gamma * np.pi / 180)) ** 2) / (
            1 + np.cos(gamma * np.pi / 180) ** 2) * wavel ** c

    return DOP


# Fit all together

def func_DOP(allvars, N, k, d):
    crds1, crds2, cLUA1, cLUA2, seeing, wave_par = allvars
    DOP = np.zeros(len(cLUA1), dtype=float)
    DOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        gamma = func_gamma(crds1[i], crds2[i], cLUA1[i], cLUA2[i])

        DOP[i] = cLUA1[i] * np.cos(cLUA1[i]) ** (1 / N) * np.exp(-seeing[i] / k + d) * (wave_par[i]) * (
                (np.sin(gamma)) ** 2) / (
                         1 + np.cos(gamma) ** 2)

    return DOP


def func_simple_DOP(theta_lua, gamma, seeing, wave_par, N, k, d):
    DOP = wave_par * theta_lua * np.cos(theta_lua) ** (1 / N) * np.exp(-seeing / k + d) * (
            (np.sin(gamma * np.pi / 180)) ** 2) / (1 + np.cos(gamma * np.pi / 180) ** 2)

    return DOP


# AOP
# Fit the amplitude normalization

def func_reg_AOP(allvars, par):
    crds1, crds2, cLUA1, cLUA2 = allvars
    AOP = np.zeros(len(cLUA1), dtype=float)
    AOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        if np.sin(crds2[i] - cLUA2[i]) * np.sin(cLUA1[i]) == 0:
            AOP[i] = np.pi / 2
        else:
            AOP[i] = par * np.arctan((np.sin(crds1[i]) * np.cos(cLUA1[i]) - np.cos(crds1[i]) * np.cos(
                crds2[i] - cLUA2[i]) * np.sin(crds1[i])) / (
                                             np.sin(crds2[i] - cLUA2[i]) * np.sin(crds1[i])))

    return AOP


def func_simple_reg_AOP(phi_obs, theta_obs, phi_lua, theta_lua, par):
    if any(np.sin(phi_obs - phi_lua) * np.sin(theta_lua) == 0):
        AOP = np.pi / 2
    else:
        AOP = par * np.arctan((np.sin(theta_obs) * np.cos(theta_lua) - np.cos(theta_obs) * np.cos(
            phi_obs - phi_lua) * np.sin(theta_lua)) / (
                                      np.sin(phi_obs - phi_lua) * np.sin(theta_lua)))
    return AOP


# Fit the wavelength factor

def func_wav_AOP(allvars, n):
    crds1, crds2, cLUA1, cLUA2, banda, par = allvars
    AOP = np.zeros(len(cLUA1), dtype=float)
    AOP[:] = np.nan

    np.seterr(invalid='ignore')

    for i in range(len(crds1)):
        if np.sin(crds2[i] - cLUA2[i]) * np.sin(cLUA1[i]) == 0:
            AOP[i] = np.pi / 2
        else:
            AOP[i] = par[i] * (banda ** n) * np.arctan((np.sin(crds1[i]) * np.cos(cLUA1[i]) - np.cos(crds1[i]) * np.cos(
                crds2[i] - cLUA2[i]) * np.sin(crds1[i])) / (
                                             np.sin(crds2[i] - cLUA2[i]) * np.sin(crds1[i])))

    return AOP


def func_simple_wav_AOP(phi_obs, theta_obs, phi_lua, theta_lua, banda, par, n):
    if any(np.sin(phi_obs - phi_lua) * np.sin(theta_lua) == 0):
        AOP = np.pi / 2
    else:
        AOP = par * (banda ** n) * np.arctan((np.sin(theta_obs) * np.cos(theta_lua) - np.cos(theta_obs) * np.cos(
            phi_obs - phi_lua) * np.sin(theta_lua)) / (
                                      np.sin(phi_obs - phi_lua) * np.sin(theta_lua)))
    return AOP


# Map Functions

def map_observations(band, condition, Par1, Par2, P1='individual', P2='Regular'):
    ''' url = 'https://drive.google.com/file/d/15FlGlJBC-c3Wh1e8bEv6fE8nycqUxebc/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
    df = pd.read_csv(path, sep=';') '''

    df = pd.read_csv('data.csv', sep=';')

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

        if P1 == 'individual':
            if P2 == 'Regular':
                par = Par1[0]

                a = 1
                b = 1
                c = 1

            if P2 == 'Mix':
                N = Par1[0]
                k1 = Par1[1]
                k2 = Par1[2]

                b = np.exp(-seen[k] / k1 + k2)
                a = np.cos(t_lua) ** (1 / N)
                c = 1

            if P2 == 'Depolarization':
                P = Par1[0]

                a = 1
                b = 1
                c = (1 + P) / (1 - P)

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
            altaz = coordinates.coord_radectoaltaz(float(RA[n]), float(DEC[n]), t)

            alt = altaz[0].value
            az = altaz[1].value

            tfield = np.pi / 2 - alt * np.pi / 180.0
            phifield = az * np.pi / 180.0

            C1field.append(float(tfield))
            C2field.append(float(phifield))

            gamma = func_gamma(tfield, phifield, t_lua, phi_lua)

            DOP_field.append(a * b * par * ((np.sin(gamma)) ** 2) / (c + np.cos(gamma) ** 2))

            pol = np.sqrt(float(Qval[n]) ** 2 + float(Uval[n]) ** 2)
            POL_OBS.append(pol)
            errPOL_OBS.append(np.sqrt(
                float(Qval[n]) ** 2 * float(errQval[n]) ** 2 + float(Uval[n]) ** 2 * float(errUval[n]) ** 2) / pol)

        theta_sky = np.zeros((100, 400))
        phi_sky = np.zeros((100, 400))

        dop = np.zeros((100, 400))

        i, j = 0, 0

        my_cmap_r = reverse_colourmap(mpl.cm.Spectral)

        for Eo in np.linspace(0, np.pi / 2, 100, endpoint=True):
            for Azo in np.linspace(0, 2 * np.pi, 400, endpoint=True):
                to = np.pi / 2 - Eo
                phio = Azo

                theta_sky[i, j] = to * 180 / np.pi
                phi_sky[i, j] = phio

                gamma = func_gamma(to, phio, t_lua, phi_lua)

                dop[i, j] = a * b * par * ((np.sin(gamma)) ** 2) / (c + np.cos(gamma) ** 2)

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
        nome = 'MAP_rayleigh_' + P1 + '_' + P2 + '_' + str(band) + '_' + str(d) + '.png'
        plt.savefig(nome)
        ax.figure.canvas.draw()
        plt.pause(0.2)
        plt.close()

        del C1field
        del C2field
        del DOP_field

        d += 1
        k += 1


def FIT(band=None, condition=None, command='ALL', ryw='regular'):
    # url = 'https://drive.google.com/file/d/15FlGlJBC-c3Wh1e8bEv6fE8nycqUxebc/view?usp=sharing'
    # path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]

    global conjunto, cor_line, cor, lab_wave, cor_unc, par_wave, Rpar, barra, k_par, d_par, B_par, B_par_mix, V_par, V_par_mix, R_par, R_par_mix, I_par, I_par_mix, B_par_dep, V_par_dep, R_par_dep, I_par_dep
    df = pd.read_csv('data.csv', sep=';')

    df.isnull().sum()
    df.dropna(axis=1)

    COND = ['ok', 'clouds', 'sun']
    CONDT = ['ok', 'clouds', 'sun']

    DATA = df
    POINT_DATA = df

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

    field = DATA['FIELD'].to_numpy()
    RA = DATA['RA'].to_numpy()
    DEC = DATA['DEC'].to_numpy()
    # BEGIN = DATA['BEGIN OBS'].to_numpy()
    # END = DATA['END OBS'].to_numpy()
    MED = DATA['MED OBS'].to_numpy()
    Ival = DATA['I'].to_numpy()
    Qval = DATA['Q'].to_numpy()
    errQval = DATA['error Q'].to_numpy()
    Uval = DATA['U'].to_numpy()
    errUval = DATA['error U '].to_numpy()
    seen = DATA['SEEING'].to_numpy()

    fit_observations_resume = pd.DataFrame(
        {'FIELD': field, 'RA': RA, 'DEC': DEC, 'OB TIME MED': MED, 'I': Ival, 'Q': Qval, 'error Q': errQval, 'U': Uval,
         'U error': errUval, 'SEEING': seen})

    total = len(RA)

    C1field = []
    C2field = []
    C1lua = []
    C2lua = []
    C1sol = []
    C2sol = []
    GAMMA = []
    GAMMA_point = []
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
    SEEING = []
    BAND = []
    WAV = []
    errAOP = []
    N_BAND = []
    K_BAND = []
    D_BAND = []
    AOP_BAND = []

    if command == 'ALL' or command == 'individuals regular' or command == 'depolarization fit' or command == 'fit wavelength' or command == 'fit seeing' or command == 'horizon fit' or command == 'fit Sun' or command == 'fit all':
        B_par, B_chisqr, B_bic, result_data_B = field_functions.fit_base('B', condition, command='regular_ray_simple')
        V_par, V_chisqr, V_bic, result_data_V = field_functions.fit_base('V', condition, command='regular_ray_simple')
        R_par, R_chisqr, R_bic, result_data_R = field_functions.fit_base('R', condition, command='regular_ray_simple')
        I_par, I_chisqr, I_bic, result_data_I = field_functions.fit_base('I', condition, command='regular_ray_simple')

        column_names = ['FIELD', 'BAND', 'CONDITION', 'GAMMA', 'POL', 'ERROR POL', 'FIT IND', 'FIT IND UNC',
                        'FIT IND DIFF']

        result_data_reg = pd.DataFrame(columns=column_names)
        result_data_reg = pd.concat([result_data_reg, result_data_B])
        result_data_reg = pd.concat([result_data_reg, result_data_V])
        result_data_reg = pd.concat([result_data_reg, result_data_R])
        result_data_reg = pd.concat([result_data_reg, result_data_I])

        # field_functions.plot_all_bands(B_par, V_par, R_par, I_par, condition, command='regular_ray')
        field_functions.plot_all(result_data_reg, command='regular_ray')

        par_ind_reg = [B_par[0], V_par[0], R_par[0], I_par[0]]
        chisqr_ind_reg = [B_chisqr, V_chisqr, R_chisqr, I_chisqr]
        bic_ind_reg = [B_bic, V_bic, R_bic, I_bic]
        index = ['B', 'V', 'R', 'I']

        df_ind_reg = pd.DataFrame({'$A$': par_ind_reg, 'Chi-square': chisqr_ind_reg, 'BIC': bic_ind_reg}, index=index)

        df_ind_reg.plot(rot=0, subplots=True, style=".--")
        plt.savefig('indvidual_ray_reg_fit_separate_parameters.png')
        plt.pause(2)
        plt.close()

        if command == 'individuals regular':
            result_data_reg.to_csv("output.csv")
            df_ind_reg.to_csv("output_par.csv")
            exit(df_ind_reg)

    if command == 'ALL' or command == 'individuals mix' or command == 'fit wavelength' or command == 'fit seeing' or command == 'horizon fit':
        B_par_mix, B_chisqr_mix, B_bic_mix, result_data_B_mix = field_functions.fit_base('B', condition, command='mix_ray_simple')
        V_par_mix, V_chisqr_mix, V_bic_mix, result_data_V_mix = field_functions.fit_base('V', condition, command='mix_ray_simple')
        R_par_mix, R_chisqr_mix, R_bic_mix, result_data_R_mix = field_functions.fit_base('R', condition, command='mix_ray_simple')
        I_par_mix, I_chisqr_mix, I_bic_mix, result_data_I_mix = field_functions.fit_base('I', condition, command='mix_ray_simple')

        N_ind_mix = [B_par_mix[0], V_par_mix[0], R_par_mix[0], I_par_mix[0]]
        k_ind_mix = [B_par_mix[1], V_par_mix[1], R_par_mix[1], I_par_mix[1]]
        d_ind_mix = [B_par_mix[2], V_par_mix[2], R_par_mix[2], I_par_mix[2]]
        chisqr_ind_mix = [B_chisqr_mix, V_chisqr_mix, R_chisqr_mix, I_chisqr_mix]
        bic_ind_mix = [B_bic_mix, V_bic_mix, R_bic_mix, I_bic_mix]
        index = ['B', 'V', 'R', 'I']

        df_ind_mix = pd.DataFrame(
            {'N': N_ind_mix, '$k_{1}$': k_ind_mix, '$k_{2}$': d_ind_mix, 'Chi-square': chisqr_ind_mix, 'BIC': bic_ind_mix}, index=index)

        df_ind_mix.plot(rot=0, subplots=True, style=".--")
        plt.savefig('indvidual_ray_dop_mix_fit_separate_parameters.png')
        plt.pause(2)
        plt.close()

        column_names = ['FIELD', 'BAND', 'CONDITION', 'GAMMA', 'POL', 'ERROR POL', 'FIT IND', 'FIT IND UNC', 'FIT IND DIFF']

        result_data_mix = pd.DataFrame(columns=column_names)
        result_data_mix = pd.concat([result_data_mix, result_data_B_mix])
        result_data_mix = pd.concat([result_data_mix, result_data_V_mix])
        result_data_mix = pd.concat([result_data_mix, result_data_R_mix])
        result_data_mix = pd.concat([result_data_mix, result_data_I_mix])

        # field_functions.plot_all_bands(B_par, V_par, R_par, I_par, condition, command='regular_multi')
        field_functions.plot_all(result_data_mix, command='simple_mix_ray')

        if command == 'individuals mix':
            result_data_mix.to_csv("output.csv")
            exit(df_ind_mix)

    if command == 'ALL' or command == 'individuals depolarization':
        B_par_dep, B_chisqr_dep, B_bic_dep, result_data_B_dep = field_functions.fit_base('B', condition, command='dep_ray_simple')
        V_par_dep, V_chisqr_dep, V_bic_dep, result_data_V_dep = field_functions.fit_base('V', condition, command='dep_ray_simple')
        R_par_dep, R_chisqr_dep, R_bic_dep, result_data_R_dep = field_functions.fit_base('R', condition, command='dep_ray_simple')
        I_par_dep, I_chisqr_dep, I_bic_dep, result_data_I_dep = field_functions.fit_base('I', condition, command='dep_ray_simple')

        column_names = ['FIELD', 'BAND', 'CONDITION', 'GAMMA', 'POL', 'ERROR POL', 'FIT IND', 'FIT IND UNC',
                        'FIT IND DIFF']

        result_data_dep = pd.DataFrame(columns=column_names)
        result_data_dep = pd.concat([result_data_dep, result_data_B_dep])
        result_data_dep = pd.concat([result_data_dep, result_data_V_dep])
        result_data_dep = pd.concat([result_data_dep, result_data_R_dep])
        result_data_dep = pd.concat([result_data_dep, result_data_I_dep])

        # field_functions.plot_all_bands(B_par, V_par, R_par, I_par, condition, command='regular_ray')
        field_functions.plot_all(result_data_dep, command='dep_ray')

        par_ind_dep = [B_par_dep[0], V_par_dep[0], R_par_dep[0], I_par_dep[0]]
        chisqr_ind_dep = [B_chisqr_dep, V_chisqr_dep, R_chisqr_dep, I_chisqr_dep]
        bic_ind_dep = [B_bic_dep, V_bic_dep, R_bic_dep, I_bic_dep]
        index = ['B', 'V', 'R', 'I']

        df_ind_1 = pd.DataFrame({'$\u03C1$': par_ind_dep, 'Chi-square': chisqr_ind_dep, 'BIC': bic_ind_dep}, index=index)

        df_ind_1.plot(rot=0, subplots=True, style=".--")
        plt.savefig('indvidual_ray_dep_fit_separate_parameters.png')
        plt.pause(2)
        plt.close()

        if command == 'individuals depolarization':
            result_data_dep.to_csv("output.csv")
            exit(df_ind_1)

    if command == 'ALL' or command == 'individuals horizon':
        B_par_hor, B_chisqr_hor, B_bic_hor, result_data_B_hor = field_functions.fit_base('B', condition, command='hor_simple_ray')
        V_par_hor, V_chisqr_hor, V_bic_hor, result_data_V_hor = field_functions.fit_base('V', condition, command='hor_simple_ray')
        R_par_hor, R_chisqr_hor, R_bic_hor, result_data_R_hor = field_functions.fit_base('R', condition, command='hor_simple_ray')
        I_par_hor, I_chisqr_hor, I_bic_hor, result_data_I_hor = field_functions.fit_base('I', condition, command='hor_simple_ray')

        par_ind_hor = [B_par_hor[0], V_par_hor[0], R_par_hor[0], I_par_hor[0]]
        chisqr_ind_hor = [B_chisqr_hor, V_chisqr_hor, R_chisqr_hor, I_chisqr_hor]
        bic_ind_hor = [B_bic_hor, V_bic_hor, R_bic_hor, I_bic_hor]
        index = ['B', 'V', 'R', 'I']

        df_ind_hor = pd.DataFrame({'N': par_ind_hor, 'Chi-square': chisqr_ind_hor, 'BIC': bic_ind_hor}, index=index)

        df_ind_hor.plot(rot=0, subplots=True, style=".--")
        plt.savefig('indvidual_ray_dop_hor_fit_separate_parameters.png')
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
        field_functions.plot_all(result_data_hor, command='hor_ray')

        if command == 'individuals horizon':
            result_data_hor.to_csv("output.csv")
            exit(df_ind_hor)

    if command == 'ALL' or command == 'individuals seeing':
        B_par_seen, B_chisqr_seen, B_bic_seen, result_data_B_seen = field_functions.fit_base('B', condition, command='seeing_simple_ray')
        V_par_seen, V_chisqr_seen, V_bic_seen, result_data_V_seen = field_functions.fit_base('V', condition, command='seeing_simple_ray')
        R_par_seen, R_chisqr_seen, R_bic_seen, result_data_R_seen = field_functions.fit_base('R', condition, command='seeing_simple_ray')
        I_par_seen, I_chisqr_seen, I_bic_seen, result_data_I_seen = field_functions.fit_base('I', condition, command='seeing_simple_ray')

        k_ind_seen = [B_par_seen[0], V_par_seen[0], R_par_seen[0], I_par_seen[0]]
        d_ind_seen = [B_par_seen[1], V_par_seen[1], R_par_seen[1], I_par_seen[1]]
        chisqr_ind_seen = [B_chisqr_seen, V_chisqr_seen, R_chisqr_seen, I_chisqr_seen]
        bic_ind_seen = [B_bic_seen, V_bic_seen, R_bic_seen, I_bic_seen]
        index = ['B', 'V', 'R', 'I']

        df_ind_seen = pd.DataFrame({'$k_{1}$': k_ind_seen, '$k_{2}$': d_ind_seen, 'Chi-square': chisqr_ind_seen, 'BIC': bic_ind_seen}, index=index)

        df_ind_seen.plot(rot=0, subplots=True, style=".--")
        plt.savefig('indvidual_ray_dop_seen_fit_separate_parameters.png')
        plt.pause(2)
        plt.close()

        column_names = ['FIELD', 'BAND', 'CONDITION', 'GAMMA', 'POL', 'ERROR POL', 'FIT IND', 'FIT IND UNC',
                        'FIT IND DIFF']

        result_data_seen = pd.DataFrame(columns=column_names)
        result_data_seen = pd.concat([result_data_seen, result_data_B_seen])
        result_data_seen = pd.concat([result_data_seen, result_data_V_seen])
        result_data_seen = pd.concat([result_data_seen, result_data_R_seen])
        result_data_seen = pd.concat([result_data_seen, result_data_I_seen])

        # field_functions.plot_all_bands(B_par, V_par, R_par, I_par, condition, command='regular_ray')
        field_functions.plot_all(result_data_seen, command='seen_ray')

        if command == 'individuals seeing':
            result_data_seen.to_csv("output.csv")
            exit(df_ind_seen)

    if command == 'ALL' or command == 'individuals wave':
        B_par, B_chisqr, B_bic, result_data_B = field_functions.fit_base('B', condition, command='simple_wave_ray')
        V_par, V_chisqr, V_bic, result_data_V = field_functions.fit_base('V', condition, command='simple_wave_ray')
        R_par, R_chisqr, R_bic, result_data_R = field_functions.fit_base('R', condition, command='simple_wave_ray')
        I_par, I_chisqr, I_bic, result_data_I = field_functions.fit_base('I', condition, command='simple_wave_ray')

        column_names = ['FIELD', 'BAND', 'CONDITION', 'GAMMA', 'POL', 'ERROR POL', 'FIT IND', 'FIT IND UNC',
                        'FIT IND DIFF']

        result_data = pd.DataFrame(columns=column_names)
        result_data = pd.concat([result_data, result_data_B])
        result_data = pd.concat([result_data, result_data_V])
        result_data = pd.concat([result_data, result_data_R])
        result_data = pd.concat([result_data, result_data_I])

        # field_functions.plot_all_bands(B_par, V_par, R_par, I_par, condition, command='regular_ray')
        field_functions.plot_all(result_data, command='wave_ray')

        par_ind = [B_par[0], V_par[0], R_par[0], I_par[0]]
        chisqr_ind = [B_chisqr, V_chisqr, R_chisqr, I_chisqr]
        bic_ind = [B_bic, V_bic, R_bic, I_bic]
        index = ['B', 'V', 'R', 'I']

        df_ind = pd.DataFrame({'$c_{band}$': par_ind, 'Chi-square': chisqr_ind, 'BIC': bic_ind}, index=index)

        df_ind.plot(rot=0, subplots=True, style=".--")
        plt.savefig('indvidual_ray_fit_separate_parameters.png')
        plt.pause(2)
        plt.close()

        if command == 'individuals wave':
            result_data.to_csv("output.csv")
            exit(df_ind)

    if command != 'ALL' and isinstance(band, str) == False:
        barra = tqdm(total=100, desc='Processing ' + command)
    if isinstance(band, str) and command != 'ALL':
        barra = tqdm(total=127, desc='Processing ' + command)
    if command == 'ALL':
        barra = tqdm(total=500, desc='Processing')

    if search('aop', command):
        B_par_aop, B_chisqr_aop, result_data_B_aop = field_functions.fit_base('B', condition, command='regular_ray_simple_aop')
        V_par_aop, V_chisqr_aop, result_data_V_aop = field_functions.fit_base('V', condition, command='regular_ray_simple_aop')
        R_par_aop, R_chisqr_aop, result_data_R_aop = field_functions.fit_base('R', condition, command='regular_ray_simple_aop')
        I_par_aop, I_chisqr_aop, result_data_I_aop = field_functions.fit_base('I', condition, command='regular_ray_simple_aop')

        par_ind_aop = [B_par_aop[0], V_par_aop[0], R_par_aop[0], I_par_aop[0]]
        # beta1_ind_aop = [B_par_aop[1], V_par_aop[1], R_par_aop[1], I_par_aop[1]]
        # beta2_ind_aop = [B_par_aop[2], V_par_aop[2], R_par_aop[2], I_par_aop[2]]
        chisqr_ind_aop = [B_chisqr_aop, V_chisqr_aop, R_chisqr_aop, I_chisqr_aop]
        index = ['B', 'V', 'R', 'I']
        df_ind_aop = pd.DataFrame({'$P_{norm}$': par_ind_aop, 'Chi-square': chisqr_ind_aop}, index=index)

        # df_ind.plot(figsize=(10, 5))
        df_ind_aop.plot(rot=0)
        plt.savefig('indvidual_rayleigh_aop_fit_together_parameters.png')
        plt.pause(2)
        plt.close()

        # df_ind.plot(figsize=(10, 5))
        df_ind_aop.plot(rot=0, subplots=True)
        plt.savefig('indvidual_rayleigh_aop_fit_separate_parameters.png')
        plt.pause(2)
        plt.close()

        aop_par = 1

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

    for n in range(0, total):
        wave = 1
        c_par = 1
        n_par = 1

        if band == 'B':
            wave = 437
            c_par = B_par[0]
            n_par = B_par_mix[0]
            k_par = B_par_mix[1]
            d_par = B_par_mix[2]

        if band == 'V':
            wave = 555
            c_par = V_par[0]
            n_par = V_par_mix[0]
            k_par = V_par_mix[1]
            d_par = V_par_mix[2]

        if band == 'R':
            wave = 655
            c_par = R_par[0]
            n_par = R_par_mix[0]
            k_par = R_par_mix[1]
            d_par = R_par_mix[2]

        if band == 'I':
            wave = 768
            c_par = I_par[0]
            n_par = I_par_mix[0]
            k_par = I_par_mix[1]
            d_par = I_par_mix[2]

        if band is None or not isinstance(band, str):
            if BANDA[n] == 'B':
                wave = 437
                c_par = B_par[0]
                n_par = B_par_mix[0]
                k_par = B_par_mix[1]
                d_par = B_par_mix[2]
                # beta1_par = B_par_aop[1]
                # beta2_par = B_par_aop[2]
            if BANDA[n] == 'V':
                wave = 555
                c_par = V_par[0]
                n_par = V_par_mix[0]
                k_par = V_par_mix[1]
                d_par = V_par_mix[2]
                # beta1_par = V_par_aop[1]
                # beta2_par = V_par_aop[2]
            if BANDA[n] == 'R':
                wave = 655
                c_par = R_par[0]
                n_par = R_par_mix[0]
                k_par = R_par_mix[1]
                d_par = R_par_mix[2]
                # beta1_par = R_par_aop[1]
                # beta2_par = R_par_aop[2]
            if BANDA[n] == 'I':
                wave = 768
                c_par = I_par[0]
                n_par = I_par_mix[0]
                k_par = I_par_mix[1]
                d_par = I_par_mix[2]

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
            float(Qval[n]) ** 2 * float(errQval[n]) ** 2 + float(Uval[n]) ** 2 * float(errUval[n]) ** 2) / pol)
        alb = albedo.albedo_moon(wave, MED[n])
        a = float(alb[0]) / float(alb[1])
        ALBEDO.append(a)
        AOP.append(0.5 * np.arctan(float(Uval[n]) / float(Qval[n])) * 180 / np.pi)
        errAOP.append(0.5 * np.sqrt((Qval[n] * errU_OBS[n]) ** 2 + (Uval[n] * errQ_OBS[n]) ** 2) / (
                (1 + (Uval[n] / Qval[n]) ** 2) * Qval[n] ** 2) * 180 / np.pi)
        SEEING.append(seen[n])
        BAND.append(wave)
        WAV.append(c_par)
        N_BAND.append(n_par)
        K_BAND.append(k_par)
        D_BAND.append(d_par)

        barra.update(int(50 / total))

    C1field, C2field = np.asarray(C1field, dtype=np.float32), np.asarray(C2field, dtype=np.float32)
    C1lua, C2lua = np.asarray(C1lua, dtype=np.float32), np.asarray(C2lua, dtype=np.float32)
    C1sol, C2sol = np.asarray(C1sol, dtype=np.float32), np.asarray(C2sol, dtype=np.float32)
    POL_OBS, errPOL_OBS = np.asarray(POL_OBS, dtype=np.float32), np.asarray(errPOL_OBS, dtype=np.float32)
    GAMMA, AOP = np.asarray(GAMMA, dtype=np.float32), np.asarray(AOP, dtype=np.float32)
    GAMMA_SOL, BAND = np.asarray(GAMMA_SOL, dtype=np.float32), np.asarray(BAND, dtype=np.float32)
    ALBEDO, SEEING = np.asarray(ALBEDO, dtype=np.float32), np.asarray(SEEING, dtype=np.float32)
    N_BAND, errAOP = np.asarray(N_BAND, dtype=np.float32), np.asarray(errAOP, dtype=np.float32)

    LABEL = listToString(conjunto)
    LAB = listToString(COND)

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

    coluna = 22

    meto = 'leastsq'

    #         SCATTERING

    if command == 'ALL' or command == 'horizon fit':
        model = lmfit.Model(func_hor_DOP)
        model.set_param_hint('N', max=50)
        p = model.make_params(N=np.random.rand())
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee')
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, WAV],
                           weights=errPOL_OBS,
                           method=meto)

        result_emcee_hor = [result.params['N'].value]
        result_emcee_hor = np.asarray(result_emcee_hor)
        Rpar = result_emcee_hor

        txname = 'REPORT_' + LABEL + '_' + LAB + '_horizon_ray_' + meto + '.txt'
        model_name = 'MODEL_' + LABEL + '_' + LAB + '_horizon_ray_' + meto + '.sav'

        lmfit.model.save_modelresult(result, model_name)
        TXT = open(txname, "w+")

        model_fit_report = result.fit_report()
        TXT.write('***  Fit: horizon correction parameter  *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        y1 = func_hor_DOP([C1field, C2field, C1lua, C2lua, WAV], *result_emcee_hor)
        fit_observations_resume.insert(coluna, 'HOR POL', y1)
        coluna += 1

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
            g1 = b_points['THETA MOON'].to_numpy()
            h1 = b_points['PHI MOON'].to_numpy()
            i1 = b_points['THETA FIELD'].to_numpy()
            j1 = b_points['PHI FIELD'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1, g1, h1, i1, j1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], \
                                                     np.asarray(d1)[w], np.asarray(e1)[w], np.asarray(f1)[w], \
                                                     np.asarray(g1)[w], np.asarray(h1)[w], np.asarray(i1)[w], \
                                                     np.asarray(j1)[w]

            label1 = []
            for d in range(0, len(a1)):
                label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i1[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(round(j1[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g1[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h1[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                    round(e1[d], 3)) + ' $ \pm $ ' + str(round(f1[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                    round(b1[d], 3)) + ' $ \pm $ ' + str(round(c1[d], 3)))

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['HOR POL'].to_numpy()
            c2 = v_points['HOR UNC'].to_numpy()
            d2 = v_points['HOR DIFF'].to_numpy()
            e2 = v_points['POL OBS'].to_numpy()
            f2 = v_points['POL OBS error'].to_numpy()
            g2 = v_points['THETA MOON'].to_numpy()
            h2 = v_points['PHI MOON'].to_numpy()
            i2 = v_points['THETA FIELD'].to_numpy()
            j2 = v_points['PHI FIELD'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2, g2, h2, i2, j2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], \
                                                     np.asarray(d2)[w], np.asarray(e2)[w], np.asarray(f2)[w], \
                                                     np.asarray(g2)[w], np.asarray(h2)[w], np.asarray(i2)[w], \
                                                     np.asarray(j2)[w]

            label2 = []
            for d in range(0, len(a2)):
                label2.append('Field observed in band V: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i2[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j2[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g2[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h2[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e2[d], 3)) + ' $ \pm $ ' + str(
                    round(f2[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b2[d], 3)) + ' $ \pm $ ' + str(
                    round(c2[d], 3)))

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['HOR POL'].to_numpy()
            c3 = r_points['HOR UNC'].to_numpy()
            d3 = r_points['HOR DIFF'].to_numpy()
            e3 = r_points['POL OBS'].to_numpy()
            f3 = r_points['POL OBS error'].to_numpy()
            g3 = r_points['THETA MOON'].to_numpy()
            h3 = r_points['PHI MOON'].to_numpy()
            i3 = r_points['THETA FIELD'].to_numpy()
            j3 = r_points['PHI FIELD'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3, g3, h3, i3, j3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], \
                                                     np.asarray(d3)[w], np.asarray(e3)[w], np.asarray(f3)[w], \
                                                     np.asarray(g3)[w], np.asarray(h3)[w], np.asarray(i3)[w], \
                                                     np.asarray(j3)[w]

            label3 = []
            for d in range(0, len(a3)):
                label3.append('Field observed in band R: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i3[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j3[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g3[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h3[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e3[d], 3)) + ' $ \pm $ ' + str(
                    round(f3[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b3[d], 3)) + ' $ \pm $ ' + str(
                    round(c3[d], 3)))

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['HOR POL'].to_numpy()
            c4 = i_points['HOR UNC'].to_numpy()
            d4 = i_points['HOR DIFF'].to_numpy()
            e4 = i_points['POL OBS'].to_numpy()
            f4 = i_points['POL OBS error'].to_numpy()
            g4 = i_points['THETA MOON'].to_numpy()
            h4 = i_points['PHI MOON'].to_numpy()
            i4 = i_points['THETA FIELD'].to_numpy()
            j4 = i_points['PHI FIELD'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4, g4, h4, i4, j4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], \
                                                     np.asarray(d4)[w], np.asarray(e4)[w], np.asarray(f4)[w], \
                                                     np.asarray(g4)[w], np.asarray(h4)[w], np.asarray(i4)[w], \
                                                     np.asarray(j4)[w]

            label4 = []
            for d in range(0, len(a4)):
                label4.append('Field observed in band I: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i4[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j4[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g4[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h4[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e4[d], 3)) + ' $ \pm $ ' + str(
                    round(f4[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b4[d], 3)) + ' $ \pm $ ' + str(
                    round(c4[d], 3)))

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            band1 = plt.scatter(a1, e1, color='none')
            mplcursors.cursor(band1, hover=True).connect("add", lambda sel: sel.annotation.set_text(label1[sel.index]))

            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            band2 = plt.scatter(a2, e2, color='none')
            mplcursors.cursor(band2, hover=True).connect("add", lambda sel: sel.annotation.set_text(label2[sel.index]))

            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            band3 = plt.scatter(a3, e3, color='none')
            mplcursors.cursor(band3, hover=True).connect("add", lambda sel: sel.annotation.set_text(label3[sel.index]))

            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')
            band4 = plt.scatter(a4, e4, color='none')
            mplcursors.cursor(band4, hover=True).connect("add", lambda sel: sel.annotation.set_text(label4[sel.index]))

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            k1 = np.add(b2, c2)
            k2 = np.subtract(b2, c2)
            plt.fill_between(a2, k2, k1, where=(k2 < k1), interpolate=True, color='beige')

            l1 = np.add(b3, c3)
            l2 = np.subtract(b3, c3)
            plt.fill_between(a3, l2, l1, where=(l2 < l1), interpolate=True, color='mistyrose')

            m1 = np.add(b4, c4)
            m2 = np.subtract(b4, c4)
            plt.fill_between(a4, m2, m1, where=(m2 < m1), interpolate=True, color='antiquewhite')

            plt.ylim(0, 0.8)
            plt.ylabel('Polarization')
            if isinstance(result.params['N'].stderr, float):
                N_par = round(result.params['N'].stderr, 3)
            else:
                N_par = result.params['N'].stderr
            label_text = 'fit parameters: ' + ' $N$ = ' + str(
                round(result.params['N'].value, 3)) + '$\pm$' + str(
                N_par) + '\n' + 'chi-square: ' + str(
                round(result.chisqr, 10)) + ',   reduced chi-square: ' + str(
                round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                round(result.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            leg_text = 'Individual amplitudes:\n \n$A_{B}$= ' + str(
                round(B_par[0], 3)) + '\n$A_{V}$= ' + str(round(V_par[0], 3)) + '\n$A_{R}$= ' + str(
                round(R_par[0], 3)) + '\n$A_{I}$= ' + str(round(I_par[0], 3))
            plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

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

            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_horizon_ray_' + meto + '.png', bbox_inches='tight')

            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
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
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                band = plt.scatter(a, e, color='none')
                mplcursors.cursor(band, hover=True).connect("add",
                                                            lambda sel: sel.annotation.set_text(label2[sel.index]))

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylim(0, 0.8)

                label_text = 'fit parameters: ' + ' $N$ = ' + str(
                    round(result.params['N'].value, 3)) + '$\pm$' + str(
                    round(result.params['N'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                    round(result.chisqr, 10)) + ',   reduced chi-square: ' + str(
                    round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                    round(result.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                leg_text = 'Individual amplitudes:\n \n$c_{B}$= ' + str(
                    round(B_par[0], 3)) + '\n$c_{V}$= ' + str(round(V_par[0], 3)) + '\n$c_{R}$= ' + str(
                    round(R_par[0], 3)) + '\n$c_{I}$= ' + str(round(I_par[0], 3))
                plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
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
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for d in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[d], 3)) + ' $ \pm $ ' + str(round(f[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[d], 3)) + ' $ \pm $ ' + str(round(c[d], 3)))

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_horizon_ray_' + meto + '.png', bbox_inches='tight')
            plt.close()

        # ___________________________________________________________________________________________

        model = lmfit.Model(func_simple_hor_DOP, independent_vars=['theta_lua', 'gamma', 'par_wave'])
        model.set_param_hint('N', max=50)
        p = model.make_params(N=np.random.rand())
        result_simple = model.fit(data=POL_OBS, params=p, theta_lua=C1lua, gamma=GAMMA, par_wave=WAV,
                                  weights=errPOL_OBS, method=meto)

        result_simple_hor = [result_simple.params['N'].value]
        result_simple_hor = np.asarray(result_simple_hor)

        model_fit_report = result_simple.fit_report()
        TXT.write('***  Fit: horizon correction parameter in a simplier way *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        model_name = 'MODEL_' + LABEL + '_' + LAB + '_simple_horizon_ray_' + meto + '.sav'
        lmfit.model.save_modelresult(result_simple, model_name)

        y1 = func_simple_hor_DOP(theta_lua=C1lua, gamma=GAMMA, par_wave=WAV, N=result_simple_hor[0])
        fit_observations_resume.insert(coluna, 'SIM HOR POL', y1)
        coluna += 1

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
        fit_observations_resume.insert(coluna, 'SIM HOR UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])
        fit_observations_resume.insert(coluna, 'SIM HOR DIFF', diff)
        coluna += 1

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['SIM HOR POL'].to_numpy()
            c1 = b_points['SIM HOR UNC'].to_numpy()
            d1 = b_points['SIM HOR DIFF'].to_numpy()
            e1 = b_points['POL OBS'].to_numpy()
            f1 = b_points['POL OBS error'].to_numpy()
            g1 = b_points['THETA MOON'].to_numpy()
            h1 = b_points['PHI MOON'].to_numpy()
            i1 = b_points['THETA FIELD'].to_numpy()
            j1 = b_points['PHI FIELD'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1, g1, h1, i1, j1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], \
                                                     np.asarray(d1)[w], np.asarray(e1)[w], np.asarray(f1)[w], \
                                                     np.asarray(g1)[w], np.asarray(h1)[w], np.asarray(i1)[w], \
                                                     np.asarray(j1)[w]

            label1 = []
            for d in range(0, len(a1)):
                label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i1[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(round(j1[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g1[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h1[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                    round(e1[d], 3)) + ' $ \pm $ ' + str(round(f1[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                    round(b1[d], 3)) + ' $ \pm $ ' + str(round(c1[d], 3)))

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['SIM HOR POL'].to_numpy()
            c2 = v_points['SIM HOR UNC'].to_numpy()
            d2 = v_points['SIM HOR DIFF'].to_numpy()
            e2 = v_points['POL OBS'].to_numpy()
            f2 = v_points['POL OBS error'].to_numpy()
            g2 = v_points['THETA MOON'].to_numpy()
            h2 = v_points['PHI MOON'].to_numpy()
            i2 = v_points['THETA FIELD'].to_numpy()
            j2 = v_points['PHI FIELD'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2, g2, h2, i2, j2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], \
                                                     np.asarray(d2)[w], np.asarray(e2)[w], np.asarray(f2)[w], \
                                                     np.asarray(g2)[w], np.asarray(h2)[w], np.asarray(i2)[w], \
                                                     np.asarray(j2)[w]

            label2 = []
            for d in range(0, len(a2)):
                label2.append('Field observed in band V: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i2[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j2[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g2[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h2[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e2[d], 3)) + ' $ \pm $ ' + str(
                    round(f2[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b2[d], 3)) + ' $ \pm $ ' + str(
                    round(c2[d], 3)))

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['SIM HOR POL'].to_numpy()
            c3 = r_points['SIM HOR UNC'].to_numpy()
            d3 = r_points['SIM HOR DIFF'].to_numpy()
            e3 = r_points['POL OBS'].to_numpy()
            f3 = r_points['POL OBS error'].to_numpy()
            g3 = r_points['THETA MOON'].to_numpy()
            h3 = r_points['PHI MOON'].to_numpy()
            i3 = r_points['THETA FIELD'].to_numpy()
            j3 = r_points['PHI FIELD'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3, g3, h3, i3, j3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], \
                                                     np.asarray(d3)[w], np.asarray(e3)[w], np.asarray(f3)[w], \
                                                     np.asarray(g3)[w], np.asarray(h3)[w], np.asarray(i3)[w], \
                                                     np.asarray(j3)[w]

            label3 = []
            for d in range(0, len(a3)):
                label3.append('Field observed in band R: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i3[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j3[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g3[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h3[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e3[d], 3)) + ' $ \pm $ ' + str(
                    round(f3[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b3[d], 3)) + ' $ \pm $ ' + str(
                    round(c3[d], 3)))

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['SIM HOR POL'].to_numpy()
            c4 = i_points['SIM HOR UNC'].to_numpy()
            d4 = i_points['SIM HOR DIFF'].to_numpy()
            e4 = i_points['POL OBS'].to_numpy()
            f4 = i_points['POL OBS error'].to_numpy()
            g4 = i_points['THETA MOON'].to_numpy()
            h4 = i_points['PHI MOON'].to_numpy()
            i4 = i_points['THETA FIELD'].to_numpy()
            j4 = i_points['PHI FIELD'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4, g4, h4, i4, j4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], \
                                                     np.asarray(d4)[w], np.asarray(e4)[w], np.asarray(f4)[w], \
                                                     np.asarray(g4)[w], np.asarray(h4)[w], np.asarray(i4)[w], \
                                                     np.asarray(j4)[w]

            label4 = []
            for d in range(0, len(a4)):
                label4.append('Field observed in band I: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i4[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j4[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g4[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h4[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e4[d], 3)) + ' $ \pm $ ' + str(
                    round(f4[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b4[d], 3)) + ' $ \pm $ ' + str(
                    round(c4[d], 3)))

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            band1 = plt.scatter(a1, e1, color='none')
            mplcursors.cursor(band1, hover=True).connect("add", lambda sel: sel.annotation.set_text(label1[sel.index]))

            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            band2 = plt.scatter(a2, e2, color='none')
            mplcursors.cursor(band2, hover=True).connect("add", lambda sel: sel.annotation.set_text(label2[sel.index]))

            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            band3 = plt.scatter(a3, e3, color='none')
            mplcursors.cursor(band3, hover=True).connect("add", lambda sel: sel.annotation.set_text(label3[sel.index]))

            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')
            band4 = plt.scatter(a4, e4, color='none')
            mplcursors.cursor(band4, hover=True).connect("add", lambda sel: sel.annotation.set_text(label4[sel.index]))

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            k1 = np.add(b2, c2)
            k2 = np.subtract(b2, c2)
            plt.fill_between(a2, k2, k1, where=(k2 < k1), interpolate=True, color='beige')

            l1 = np.add(b3, c3)
            l2 = np.subtract(b3, c3)
            plt.fill_between(a3, l2, l1, where=(l2 < l1), interpolate=True, color='mistyrose')

            m1 = np.add(b4, c4)
            m2 = np.subtract(b4, c4)
            plt.fill_between(a4, m2, m1, where=(m2 < m1), interpolate=True, color='antiquewhite')

            plt.ylim(0, 0.8)
            plt.ylabel('Polarization')
            if isinstance(result.params['N'].stderr, float):
                N_par = round(result.params['N'].stderr, 3)
            else:
                N_par = result.params['N'].stderr
            label_text = 'fit parameters:  ' + ' $N$ = ' + str(
                round(result.params['N'].value, 3)) + '$\pm$' + str(
                N_par) + '\n' + 'chi-square: ' + str(
                round(result_simple.chisqr, 10)) + ',   reduced chi-square: ' + str(
                round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                round(result_simple.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            leg_text = 'Individual amplitudes:\n \n$A_{B}$= ' + str(
                round(B_par[0], 3)) + '\n$A_{V}$= ' + str(round(V_par[0], 3)) + '\n$A_{R}$= ' + str(
                round(R_par[0], 3)) + '\n$A_{I}$= ' + str(round(I_par[0], 3))
            plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

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
            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_simple_horizon_ray_' + meto + '.png',
                        bbox_inches='tight')

            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
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
                b = points['SIM HOR POL'].to_numpy()
                c = points['SIM HOR UNC'].to_numpy()
                d = points['SIM HOR DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                band = plt.scatter(a, e, color='none')
                mplcursors.cursor(band, hover=True).connect("add",
                                                            lambda sel: sel.annotation.set_text(label2[sel.index]))

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylim(0, 0.8)
                plt.ylabel('Polarization')
                label_text = 'fit parameters:  ' + ' $N$ = ' + str(
                    round(result.params['N'].value, 3)) + '$\pm$' + str(
                    round(result.params['N'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                    round(result_simple.chisqr, 10)) + ',   reduced chi-square: ' + str(
                    round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                    round(result_simple.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                leg_text = 'simple fit adding \nthe horizon correction parameter...\n \n$c_{B}$= ' + str(
                    round(B_par[0], 3)) + '\n$c_{V}$= ' + str(round(V_par[0], 3)) + '\n$c_{R}$= ' + str(
                    round(R_par[0], 3)) + '\n$c_{I}$= ' + str(round(I_par[0], 3))
                plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
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
                b = points['SIM HOR POL'].to_numpy()
                c = points['SIM HOR UNC'].to_numpy()
                d = points['SIM HOR DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))
                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_simple_horizon_ray_' + meto + '.png',
                        bbox_inches='tight')
            plt.close()
            TXT.close()

    # -------------------------------------------------------------------------------------------

    if command == 'ALL' or command == 'depolarization fit':

        model = lmfit.Model(func_depo_DOP)
        model.set_param_hint('P', min=0, max=1)
        p = model.make_params(P=np.random.rand())
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee')
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, WAV],
                           weights=errPOL_OBS,
                           method=meto)

        result_emcee_dep = [result.params['P'].value]
        result_emcee_dep = np.asarray(result_emcee_dep)
        Rpar = result_emcee_dep

        txname = 'REPORT_' + LABEL + '_' + LAB + '_dep_ray_' + meto + '.txt'
        model_name = 'MODEL_' + LABEL + '_' + LAB + '_dep_ray_' + meto + '.sav'

        lmfit.model.save_modelresult(result, model_name)
        TXT = open(txname, "w+")

        model_fit_report = result.fit_report()
        TXT.write('***  Fit: depolarization correction parameter  *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        y1 = func_depo_DOP([C1field, C2field, C1lua, C2lua, WAV], *result_emcee_dep)
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
        coluna += 1

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
            g1 = b_points['THETA MOON'].to_numpy()
            h1 = b_points['PHI MOON'].to_numpy()
            i1 = b_points['THETA FIELD'].to_numpy()
            j1 = b_points['PHI FIELD'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1, g1, h1, i1, j1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], \
                                                     np.asarray(d1)[w], np.asarray(e1)[w], np.asarray(f1)[w], \
                                                     np.asarray(g1)[w], np.asarray(h1)[w], np.asarray(i1)[w], \
                                                     np.asarray(j1)[w]

            label1 = []
            for d in range(0, len(a1)):
                label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i1[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(round(j1[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g1[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h1[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                    round(e1[d], 3)) + ' $ \pm $ ' + str(round(f1[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                    round(b1[d], 3)) + ' $ \pm $ ' + str(round(c1[d], 3)))

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['DEP POL'].to_numpy()
            c2 = v_points['DEP UNC'].to_numpy()
            d2 = v_points['DEP DIFF'].to_numpy()
            e2 = v_points['POL OBS'].to_numpy()
            f2 = v_points['POL OBS error'].to_numpy()
            g2 = v_points['THETA MOON'].to_numpy()
            h2 = v_points['PHI MOON'].to_numpy()
            i2 = v_points['THETA FIELD'].to_numpy()
            j2 = v_points['PHI FIELD'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2, g2, h2, i2, j2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], \
                                                     np.asarray(d2)[w], np.asarray(e2)[w], np.asarray(f2)[w], \
                                                     np.asarray(g2)[w], np.asarray(h2)[w], np.asarray(i2)[w], \
                                                     np.asarray(j2)[w]

            label2 = []
            for d in range(0, len(a2)):
                label2.append('Field observed in band V: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i2[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j2[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g2[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h2[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e2[d], 3)) + ' $ \pm $ ' + str(
                    round(f2[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b2[d], 3)) + ' $ \pm $ ' + str(
                    round(c2[d], 3)))

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['DEP POL'].to_numpy()
            c3 = r_points['DEP UNC'].to_numpy()
            d3 = r_points['DEP DIFF'].to_numpy()
            e3 = r_points['POL OBS'].to_numpy()
            f3 = r_points['POL OBS error'].to_numpy()
            g3 = r_points['THETA MOON'].to_numpy()
            h3 = r_points['PHI MOON'].to_numpy()
            i3 = r_points['THETA FIELD'].to_numpy()
            j3 = r_points['PHI FIELD'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3, g3, h3, i3, j3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], \
                                                     np.asarray(d3)[w], np.asarray(e3)[w], np.asarray(f3)[w], \
                                                     np.asarray(g3)[w], np.asarray(h3)[w], np.asarray(i3)[w], \
                                                     np.asarray(j3)[w]

            label3 = []
            for d in range(0, len(a3)):
                label3.append('Field observed in band R: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i3[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j3[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g3[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h3[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e3[d], 3)) + ' $ \pm $ ' + str(
                    round(f3[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b3[d], 3)) + ' $ \pm $ ' + str(
                    round(c3[d], 3)))

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['DEP POL'].to_numpy()
            c4 = i_points['DEP UNC'].to_numpy()
            d4 = i_points['DEP DIFF'].to_numpy()
            e4 = i_points['POL OBS'].to_numpy()
            f4 = i_points['POL OBS error'].to_numpy()
            g4 = i_points['THETA MOON'].to_numpy()
            h4 = i_points['PHI MOON'].to_numpy()
            i4 = i_points['THETA FIELD'].to_numpy()
            j4 = i_points['PHI FIELD'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4, g4, h4, i4, j4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], \
                                                     np.asarray(d4)[w], np.asarray(e4)[w], np.asarray(f4)[w], \
                                                     np.asarray(g4)[w], np.asarray(h4)[w], np.asarray(i4)[w], \
                                                     np.asarray(j4)[w]

            label4 = []
            for d in range(0, len(a4)):
                label4.append('Field observed in band I: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i4[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j4[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g4[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h4[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e4[d], 3)) + ' $ \pm $ ' + str(
                    round(f4[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b4[d], 3)) + ' $ \pm $ ' + str(
                    round(c4[d], 3)))

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            band1 = plt.scatter(a1, e1, color='none')
            mplcursors.cursor(band1, hover=True).connect("add", lambda sel: sel.annotation.set_text(label1[sel.index]))

            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            band2 = plt.scatter(a2, e2, color='none')
            mplcursors.cursor(band2, hover=True).connect("add", lambda sel: sel.annotation.set_text(label2[sel.index]))

            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            band3 = plt.scatter(a3, e3, color='none')
            mplcursors.cursor(band3, hover=True).connect("add", lambda sel: sel.annotation.set_text(label3[sel.index]))

            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')
            band4 = plt.scatter(a4, e4, color='none')
            mplcursors.cursor(band4, hover=True).connect("add", lambda sel: sel.annotation.set_text(label4[sel.index]))

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            k1 = np.add(b2, c2)
            k2 = np.subtract(b2, c2)
            plt.fill_between(a2, k2, k1, where=(k2 < k1), interpolate=True, color='beige')

            l1 = np.add(b3, c3)
            l2 = np.subtract(b3, c3)
            plt.fill_between(a3, l2, l1, where=(l2 < l1), interpolate=True, color='mistyrose')

            m1 = np.add(b4, c4)
            m2 = np.subtract(b4, c4)
            plt.fill_between(a4, m2, m1, where=(m2 < m1), interpolate=True, color='antiquewhite')

            plt.ylim(0, 0.8)
            plt.ylabel('Polarization')
            label_text = 'fit parameters: ' + ' $\u03C1$ = ' + str(
                round(result.params['P'].value, 3)) + '$\pm$' + str(
                round(result.params['P'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                round(result.chisqr, 10)) + ',   reduced chi-square: ' + str(
                round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                round(result.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            leg_text = 'fit adding \nthe horizon correction parameter...\n \n$c_{B}$= ' + str(
                round(B_par[0], 3)) + '\n$c_{V}$= ' + str(round(V_par[0], 3)) + '\n$c_{R}$= ' + str(
                round(R_par[0], 3)) + '\n$c_{I}$= ' + str(round(I_par[0], 3))
            plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

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
            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_dep_ray_' + meto + '.png', bbox_inches='tight')
            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
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
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                band = plt.scatter(a, e, color='none')
                mplcursors.cursor(band, hover=True).connect("add",
                                                            lambda sel: sel.annotation.set_text(label2[sel.index]))
                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylim(0, 0.8)

                label_text = 'fit parameters: ' + ' $\u03C1$ = ' + str(
                    round(result.params['P'].value, 3)) + '$\pm$' + str(
                    round(result.params['P'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                    round(result.chisqr, 10)) + ',   reduced chi-square: ' + str(
                    round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                    round(result.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                leg_text = 'fit adding \nthe horizon correction parameter...\n \n$c_{B}$= ' + str(
                    round(B_par[0], 3)) + '\n$c_{V}$= ' + str(round(V_par[0], 3)) + '\n$c_{R}$= ' + str(
                    round(R_par[0], 3)) + '\n$c_{I}$= ' + str(round(I_par[0], 3))
                plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
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
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_dep_ray_' + meto + '.png', bbox_inches='tight')
            plt.close()

        # ___________________________________________________________________________________________

        model = lmfit.Model(func_simple_depo_DOP, independent_vars=['gamma', 'par_wave'])
        model.set_param_hint('P', min=0, max=1)
        p = model.make_params(P=np.random.rand())
        result_simple = model.fit(data=POL_OBS, params=p, gamma=GAMMA, par_wave=WAV,
                                  weights=errPOL_OBS, method=meto)

        result_simple_dep = [result_simple.params['P'].value]
        result_simple_dep = np.asarray(result_simple_dep)

        model_fit_report = result_simple.fit_report()
        TXT.write('***  Fit: depolarization correction parameter in a simplier way *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        model_name = 'MODEL_' + LABEL + '_' + LAB + '_simple_dep_ray_' + meto + '.sav'
        lmfit.model.save_modelresult(result_simple, model_name)

        y1 = func_simple_depo_DOP(gamma=GAMMA, par_wave=WAV, P=result_simple_dep[0])
        fit_observations_resume.insert(coluna, 'SIM DEP POL', y1)
        coluna += 1

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
        fit_observations_resume.insert(coluna, 'SIM DEP UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])
        fit_observations_resume.insert(coluna, 'SIM DEP DIFF', diff)
        coluna += 1

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['SIM DEP POL'].to_numpy()
            c1 = b_points['SIM DEP UNC'].to_numpy()
            d1 = b_points['SIM DEP DIFF'].to_numpy()
            e1 = b_points['POL OBS'].to_numpy()
            f1 = b_points['POL OBS error'].to_numpy()
            f1 = b_points['POL OBS error'].to_numpy()
            g1 = b_points['THETA MOON'].to_numpy()
            h1 = b_points['PHI MOON'].to_numpy()
            i1 = b_points['THETA FIELD'].to_numpy()
            j1 = b_points['PHI FIELD'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1, g1, h1, i1, j1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], \
                                                     np.asarray(d1)[w], np.asarray(e1)[w], np.asarray(f1)[w], \
                                                     np.asarray(g1)[w], np.asarray(h1)[w], np.asarray(i1)[w], \
                                                     np.asarray(j1)[w]

            label1 = []
            for d in range(0, len(a1)):
                label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i1[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(round(j1[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g1[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h1[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                    round(e1[d], 3)) + ' $ \pm $ ' + str(round(f1[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                    round(b1[d], 3)) + ' $ \pm $ ' + str(round(c1[d], 3)))

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['SIM DEP POL'].to_numpy()
            c2 = v_points['SIM DEP UNC'].to_numpy()
            d2 = v_points['SIM DEP DIFF'].to_numpy()
            e2 = v_points['POL OBS'].to_numpy()
            f2 = v_points['POL OBS error'].to_numpy()
            g2 = v_points['THETA MOON'].to_numpy()
            h2 = v_points['PHI MOON'].to_numpy()
            i2 = v_points['THETA FIELD'].to_numpy()
            j2 = v_points['PHI FIELD'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2, g2, h2, i2, j2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], \
                                                     np.asarray(d2)[w], np.asarray(e2)[w], np.asarray(f2)[w], \
                                                     np.asarray(g2)[w], np.asarray(h2)[w], np.asarray(i2)[w], \
                                                     np.asarray(j2)[w]

            label2 = []
            for d in range(0, len(a2)):
                label2.append('Field observed in band V: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i2[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j2[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g2[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h2[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e2[d], 3)) + ' $ \pm $ ' + str(
                    round(f2[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b2[d], 3)) + ' $ \pm $ ' + str(
                    round(c2[d], 3)))

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['SIM DEP POL'].to_numpy()
            c3 = r_points['SIM DEP UNC'].to_numpy()
            d3 = r_points['SIM DEP DIFF'].to_numpy()
            e3 = r_points['POL OBS'].to_numpy()
            f3 = r_points['POL OBS error'].to_numpy()
            g3 = r_points['THETA MOON'].to_numpy()
            h3 = r_points['PHI MOON'].to_numpy()
            i3 = r_points['THETA FIELD'].to_numpy()
            j3 = r_points['PHI FIELD'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3, g3, h3, i3, j3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], \
                                                     np.asarray(d3)[w], np.asarray(e3)[w], np.asarray(f3)[w], \
                                                     np.asarray(g3)[w], np.asarray(h3)[w], np.asarray(i3)[w], \
                                                     np.asarray(j3)[w]

            label3 = []
            for d in range(0, len(a3)):
                label3.append('Field observed in band R: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i3[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j3[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g3[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h3[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e3[d], 3)) + ' $ \pm $ ' + str(
                    round(f3[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b3[d], 3)) + ' $ \pm $ ' + str(
                    round(c3[d], 3)))

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['SIM DEP POL'].to_numpy()
            c4 = i_points['SIM DEP UNC'].to_numpy()
            d4 = i_points['SIM DEP DIFF'].to_numpy()
            e4 = i_points['POL OBS'].to_numpy()
            f4 = i_points['POL OBS error'].to_numpy()
            g4 = i_points['THETA MOON'].to_numpy()
            h4 = i_points['PHI MOON'].to_numpy()
            i4 = i_points['THETA FIELD'].to_numpy()
            j4 = i_points['PHI FIELD'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4, g4, h4, i4, j4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], \
                                                     np.asarray(d4)[w], np.asarray(e4)[w], np.asarray(f4)[w], \
                                                     np.asarray(g4)[w], np.asarray(h4)[w], np.asarray(i4)[w], \
                                                     np.asarray(j4)[w]

            label4 = []
            for d in range(0, len(a4)):
                label4.append('Field observed in band I: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i4[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j4[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g4[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h4[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e4[d], 3)) + ' $ \pm $ ' + str(
                    round(f4[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b4[d], 3)) + ' $ \pm $ ' + str(
                    round(c4[d], 3)))

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            band1 = plt.scatter(a1, e1, color='none')
            mplcursors.cursor(band1, hover=True).connect("add", lambda sel: sel.annotation.set_text(label1[sel.index]))

            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            band2 = plt.scatter(a2, e2, color='none')
            mplcursors.cursor(band2, hover=True).connect("add", lambda sel: sel.annotation.set_text(label2[sel.index]))

            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            band3 = plt.scatter(a3, e3, color='none')
            mplcursors.cursor(band3, hover=True).connect("add", lambda sel: sel.annotation.set_text(label3[sel.index]))

            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')
            band4 = plt.scatter(a4, e4, color='none')
            mplcursors.cursor(band4, hover=True).connect("add", lambda sel: sel.annotation.set_text(label4[sel.index]))

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            k1 = np.add(b2, c2)
            k2 = np.subtract(b2, c2)
            plt.fill_between(a2, k2, k1, where=(k2 < k1), interpolate=True, color='beige')

            l1 = np.add(b3, c3)
            l2 = np.subtract(b3, c3)
            plt.fill_between(a3, l2, l1, where=(l2 < l1), interpolate=True, color='mistyrose')

            m1 = np.add(b4, c4)
            m2 = np.subtract(b4, c4)
            plt.fill_between(a4, m2, m1, where=(m2 < m1), interpolate=True, color='antiquewhite')

            plt.ylim(0, 0.8)
            plt.ylabel('Polarization')
            label_text = 'fit parameters:  ' + ' $\u03C1$ = ' + str(
                round(result.params['P'].value, 3)) + '$\pm$' + str(
                round(result.params['P'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                round(result_simple.chisqr, 10)) + ',   reduced chi-square: ' + str(
                round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                round(result_simple.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            leg_text = 'simple fit adding \nthe horizon correction parameter...\n \n$c_{B}$= ' + str(
                round(B_par[0], 3)) + '\n$c_{V}$= ' + str(round(V_par[0], 3)) + '\n$c_{R}$= ' + str(
                round(R_par[0], 3)) + '\n$c_{I}$= ' + str(round(I_par[0], 3))
            plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

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
            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_simple_dep_ray_' + meto + '.png',
                        bbox_inches='tight')

            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
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
                b = points['SIM DEP POL'].to_numpy()
                c = points['SIM DEP UNC'].to_numpy()
                d = points['SIM DEP DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                barra.update(10 / len(band))

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                band = plt.scatter(a, e, color='none')
                mplcursors.cursor(band, hover=True).connect("add",
                                                            lambda sel: sel.annotation.set_text(label2[sel.index]))

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylim(0, 0.8)
                plt.ylabel('Polarization')
                label_text = 'fit parameters:  ' + ' $\u03C1$ = ' + str(
                    round(result.params['P'].value, 3)) + '$\pm$' + str(
                    round(result.params['P'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                    round(result_simple.chisqr, 10)) + ',   reduced chi-square: ' + str(
                    round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                    round(result_simple.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                leg_text = 'simple fit adding \nthe horizon correction parameter...\n \n$c_{B}$= ' + str(
                    round(B_par[0], 3)) + '\n$c_{V}$= ' + str(round(V_par[0], 3)) + '\n$c_{R}$= ' + str(
                    round(R_par[0], 3)) + '\n$c_{I}$= ' + str(round(I_par[0], 3))
                plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
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
                b = points['SIM DEP POL'].to_numpy()
                c = points['SIM DEP UNC'].to_numpy()
                d = points['SIM DEP DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_simple_dep_ray_' + meto + '.png',
                        bbox_inches='tight')
            plt.close()
            TXT.close()

    # -------------------------------------------------------------------------------------------

    if command == 'ALL' or command == 'fit Sun':
        model = lmfit.Model(func_sun_DOP)
        model.set_param_hint('par', min=0, max=1)
        p = model.make_params(par=np.random.rand())  # , N=10)
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, C1sol, C2sol, ALBEDO],
        #                    weights=errPOL_OBS, nfev=500000)
        result = model.fit(data=POL_OBS, params=p,
                           allvars=[C1field, C2field, C1lua, C2lua, C1sol, C2sol, ALBEDO, WAV],
                           weights=errPOL_OBS, method=meto)

        result_emcee_sun = [result.params['par'].value]
        result_emcee_sun = np.asarray(result_emcee_sun)
        Rpar = result_emcee_sun

        txname = 'REPORT_' + LABEL + '_' + LAB + '_sun_ray_' + meto + '.txt'
        TXT = open(txname, "w+")

        model_fit_report = result.fit_report()
        TXT.write('***  Fit: now considering the sun influence *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        model_name = 'MODEL_' + LABEL + '_' + LAB + '_sun_ray_' + meto + '.sav'
        lmfit.model.save_modelresult(result, model_name)

        y1 = func_sun_DOP([C1field, C2field, C1lua, C2lua, C1sol, C2sol, ALBEDO, WAV], *result_emcee_sun)
        fit_observations_resume.insert(coluna, 'SUN POL', y1)
        coluna += 1

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
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
            g1 = b_points['THETA MOON'].to_numpy()
            h1 = b_points['PHI MOON'].to_numpy()
            i1 = b_points['THETA FIELD'].to_numpy()
            j1 = b_points['PHI FIELD'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1, g1, h1, i1, j1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], \
                                                     np.asarray(d1)[w], np.asarray(e1)[w], np.asarray(f1)[w], \
                                                     np.asarray(g1)[w], np.asarray(h1)[w], np.asarray(i1)[w], \
                                                     np.asarray(j1)[w]

            label1 = []
            for d in range(0, len(a1)):
                label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i1[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(round(j1[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g1[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h1[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                    round(e1[d], 3)) + ' $ \pm $ ' + str(round(f1[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                    round(b1[d], 3)) + ' $ \pm $ ' + str(round(c1[d], 3)))

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['SUN POL'].to_numpy()
            c2 = v_points['SUN UNC'].to_numpy()
            d2 = v_points['SUN DIFF'].to_numpy()
            e2 = v_points['POL OBS'].to_numpy()
            f2 = v_points['POL OBS error'].to_numpy()
            g2 = v_points['THETA MOON'].to_numpy()
            h2 = v_points['PHI MOON'].to_numpy()
            i2 = v_points['THETA FIELD'].to_numpy()
            j2 = v_points['PHI FIELD'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2, g2, h2, i2, j2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], \
                                                     np.asarray(d2)[w], np.asarray(e2)[w], np.asarray(f2)[w], \
                                                     np.asarray(g2)[w], np.asarray(h2)[w], np.asarray(i2)[w], \
                                                     np.asarray(j2)[w]

            label2 = []
            for d in range(0, len(a2)):
                label2.append('Field observed in band V: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i2[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j2[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g2[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h2[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e2[d], 3)) + ' $ \pm $ ' + str(
                    round(f2[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b2[d], 3)) + ' $ \pm $ ' + str(
                    round(c2[d], 3)))

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['SUN POL'].to_numpy()
            c3 = r_points['SUN UNC'].to_numpy()
            d3 = r_points['SUN DIFF'].to_numpy()
            e3 = r_points['POL OBS'].to_numpy()
            f3 = r_points['POL OBS error'].to_numpy()
            g3 = r_points['THETA MOON'].to_numpy()
            h3 = r_points['PHI MOON'].to_numpy()
            i3 = r_points['THETA FIELD'].to_numpy()
            j3 = r_points['PHI FIELD'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3, g3, h3, i3, j3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], \
                                                     np.asarray(d3)[w], np.asarray(e3)[w], np.asarray(f3)[w], \
                                                     np.asarray(g3)[w], np.asarray(h3)[w], np.asarray(i3)[w], \
                                                     np.asarray(j3)[w]

            label3 = []
            for d in range(0, len(a3)):
                label3.append('Field observed in band R: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i3[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j3[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g3[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h3[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e3[d], 3)) + ' $ \pm $ ' + str(
                    round(f3[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b3[d], 3)) + ' $ \pm $ ' + str(
                    round(c3[d], 3)))

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['SUN POL'].to_numpy()
            c4 = i_points['SUN UNC'].to_numpy()
            d4 = i_points['SUN DIFF'].to_numpy()
            e4 = i_points['POL OBS'].to_numpy()
            f4 = i_points['POL OBS error'].to_numpy()
            g4 = i_points['THETA MOON'].to_numpy()
            h4 = i_points['PHI MOON'].to_numpy()
            i4 = i_points['THETA FIELD'].to_numpy()
            j4 = i_points['PHI FIELD'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4, g4, h4, i4, j4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], \
                                                     np.asarray(d4)[w], np.asarray(e4)[w], np.asarray(f4)[w], \
                                                     np.asarray(g4)[w], np.asarray(h4)[w], np.asarray(i4)[w], \
                                                     np.asarray(j4)[w]

            label4 = []
            for d in range(0, len(a4)):
                label4.append('Field observed in band I: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i4[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j4[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g4[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h4[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e4[d], 3)) + ' $ \pm $ ' + str(
                    round(f4[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b4[d], 3)) + ' $ \pm $ ' + str(
                    round(c4[d], 3)))

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            band1 = plt.scatter(a1, e1, color='none')
            mplcursors.cursor(band1, hover=True).connect("add", lambda sel: sel.annotation.set_text(label1[sel.index]))

            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            band2 = plt.scatter(a2, e2, color='none')
            mplcursors.cursor(band2, hover=True).connect("add", lambda sel: sel.annotation.set_text(label2[sel.index]))

            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            band3 = plt.scatter(a3, e3, color='none')
            mplcursors.cursor(band3, hover=True).connect("add", lambda sel: sel.annotation.set_text(label3[sel.index]))

            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')
            band4 = plt.scatter(a4, e4, color='none')
            mplcursors.cursor(band4, hover=True).connect("add", lambda sel: sel.annotation.set_text(label4[sel.index]))

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            k1 = np.add(b2, c2)
            k2 = np.subtract(b2, c2)
            plt.fill_between(a2, k2, k1, where=(k2 < k1), interpolate=True, color='beige')

            l1 = np.add(b3, c3)
            l2 = np.subtract(b3, c3)
            plt.fill_between(a3, l2, l1, where=(l2 < l1), interpolate=True, color='mistyrose')

            m1 = np.add(b4, c4)
            m2 = np.subtract(b4, c4)
            plt.fill_between(a4, m2, m1, where=(m2 < m1), interpolate=True, color='antiquewhite')

            plt.ylim(0, 0.8)
            plt.ylabel('Polarization')
            if isinstance(result.params['par'].stderr, float):
                par_par = round(result.params['par'].stderr, 3)
            else:
                par_par = result.params['par'].stderr
            label_text = 'fit parameters: ' + ' $A$ = ' + str(
                round(result.params['par'].value, 3)) + '$\pm$' + str(par_par) + '\n' + 'chi-square: ' + str(
                round(result.chisqr, 10)) + ',   reduced chi-square: ' + str(
                round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                round(result.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction', horizontalalignment='left', verticalalignment='center',
                         bbox=dict(boxstyle="round", fc="w"))
            leg_text = 'fit adding \nthe sun influence...\n \n$c_{B}$= ' + str(
                round(B_par[0], 3)) + '\n$c_{V}$= ' + str(round(V_par[0], 3)) + '\n$c_{R}$= ' + str(
                round(R_par[0], 3)) + '\n$c_{I}$= ' + str(round(I_par[0], 3))
            plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                         textcoords='axes fraction', horizontalalignment='left', verticalalignment='center',
                         bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

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
            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_sun_ray_' + meto + '.png', bbox_inches='tight')

            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
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
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                band = plt.scatter(a, e, color='none')
                mplcursors.cursor(band, hover=True).connect("add",
                                                            lambda sel: sel.annotation.set_text(label2[sel.index]))

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylim(0, 0.8)
                plt.ylabel('Polarization')
                if isinstance(result.params['par'].stderr, float):
                    par_par = round(result.params['par'].stderr, 3)
                else:
                    par_par = result.params['par'].stderr
                label_text = 'fit parameters: ' + ' $A$ = ' + str(
                    round(result.params['par'].value, 3)) + '$\pm$' + str(par_par) + '\n' + 'chi-square: ' + str(
                    round(result.chisqr, 10)) + ',   reduced chi-square: ' + str(
                    round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                    round(result.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction', horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                leg_text = 'fit adding \nthe sun influence...\n \n$c_{B}$= ' + str(
                    round(B_par[0], 3)) + '\n$c_{V}$= ' + str(round(V_par[0], 3)) + '\n$c_{R}$= ' + str(
                    round(R_par[0], 3)) + '\n$c_{I}$= ' + str(round(I_par[0], 3))
                plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                             textcoords='axes fraction', horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
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
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_sun_ray_' + meto + '.png', bbox_inches='tight')
            plt.close()

            # ___________________________________________________________________________________________

        model = lmfit.Model(func_simple_sun_DOP,
                            independent_vars=['theta_sol', 'gamma', 'gamma_sol', 'alb', 'par_wave'])
        model.set_param_hint('par', min=0, max=1)
        p = model.make_params(par=np.random.rand())  # , N=10)
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, C1sol, C2sol, ALBEDO],
        #                    weights=errPOL_OBS, nfev=500000)
        result_simple = model.fit(data=POL_OBS, params=p, gamma=GAMMA, gamma_sol=GAMMA_SOL, theta_sol=C1sol, alb=ALBEDO,
                                  par_wave=WAV, weights=errPOL_OBS, method=meto)

        result_simple_sun = [result_simple.params['par'].value]
        result_simple_sun = np.asarray(result_simple_sun)

        model_fit_report = result_simple.fit_report()
        TXT.write('***  Fit: now considering the sun influence in a simplier way  *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        model_name = 'MODEL_' + LABEL + '_' + LAB + '_simple_sun_ray_' + meto + '.sav'

        lmfit.model.save_modelresult(result_simple, model_name)

        y1 = func_simple_sun_DOP(gamma=GAMMA, gamma_sol=GAMMA_SOL, theta_sol=C1sol, alb=ALBEDO,
                                 par_wave=WAV, par=result_simple_sun[0])
        fit_observations_resume.insert(coluna, 'SIM SUN POL', y1)
        coluna += 1

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
        fit_observations_resume.insert(coluna, 'SIM SUN UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])
        fit_observations_resume.insert(coluna, 'SIM SUN DIFF', diff)
        coluna += 1

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['SIM SUN POL'].to_numpy()
            c1 = b_points['SIM SUN UNC'].to_numpy()
            d1 = b_points['SIM SUN DIFF'].to_numpy()
            e1 = b_points['POL OBS'].to_numpy()
            f1 = b_points['POL OBS error'].to_numpy()
            g1 = b_points['THETA MOON'].to_numpy()
            h1 = b_points['PHI MOON'].to_numpy()
            i1 = b_points['THETA FIELD'].to_numpy()
            j1 = b_points['PHI FIELD'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1, g1, h1, i1, j1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], \
                                                     np.asarray(d1)[w], np.asarray(e1)[w], np.asarray(f1)[w], \
                                                     np.asarray(g1)[w], np.asarray(h1)[w], np.asarray(i1)[w], \
                                                     np.asarray(j1)[w]

            label1 = []
            for d in range(0, len(a1)):
                label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i1[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(round(j1[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g1[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h1[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                    round(e1[d], 3)) + ' $ \pm $ ' + str(round(f1[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                    round(b1[d], 3)) + ' $ \pm $ ' + str(round(c1[d], 3)))

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['SIM SUN POL'].to_numpy()
            c2 = v_points['SIM SUN UNC'].to_numpy()
            d2 = v_points['SIM SUN DIFF'].to_numpy()
            e2 = v_points['POL OBS'].to_numpy()
            f2 = v_points['POL OBS error'].to_numpy()
            g2 = v_points['THETA MOON'].to_numpy()
            h2 = v_points['PHI MOON'].to_numpy()
            i2 = v_points['THETA FIELD'].to_numpy()
            j2 = v_points['PHI FIELD'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2, g2, h2, i2, j2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], \
                                                     np.asarray(d2)[w], np.asarray(e2)[w], np.asarray(f2)[w], \
                                                     np.asarray(g2)[w], np.asarray(h2)[w], np.asarray(i2)[w], \
                                                     np.asarray(j2)[w]

            label2 = []
            for d in range(0, len(a2)):
                label2.append('Field observed in band V: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i2[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j2[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g2[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h2[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e2[d], 3)) + ' $ \pm $ ' + str(
                    round(f2[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b2[d], 3)) + ' $ \pm $ ' + str(
                    round(c2[d], 3)))

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['SIM SUN POL'].to_numpy()
            c3 = r_points['SIM SUN UNC'].to_numpy()
            d3 = r_points['SIM SUN DIFF'].to_numpy()
            e3 = r_points['POL OBS'].to_numpy()
            f3 = r_points['POL OBS error'].to_numpy()
            g3 = r_points['THETA MOON'].to_numpy()
            h3 = r_points['PHI MOON'].to_numpy()
            i3 = r_points['THETA FIELD'].to_numpy()
            j3 = r_points['PHI FIELD'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3, g3, h3, i3, j3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], \
                                                     np.asarray(d3)[w], np.asarray(e3)[w], np.asarray(f3)[w], \
                                                     np.asarray(g3)[w], np.asarray(h3)[w], np.asarray(i3)[w], \
                                                     np.asarray(j3)[w]

            label3 = []
            for d in range(0, len(a3)):
                label3.append('Field observed in band R: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i3[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j3[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g3[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h3[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e3[d], 3)) + ' $ \pm $ ' + str(
                    round(f3[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b3[d], 3)) + ' $ \pm $ ' + str(
                    round(c3[d], 3)))

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['SIM SUN POL'].to_numpy()
            c4 = i_points['SIM SUN UNC'].to_numpy()
            d4 = i_points['SIM SUN DIFF'].to_numpy()
            e4 = i_points['POL OBS'].to_numpy()
            f4 = i_points['POL OBS error'].to_numpy()
            g4 = i_points['THETA MOON'].to_numpy()
            h4 = i_points['PHI MOON'].to_numpy()
            i4 = i_points['THETA FIELD'].to_numpy()
            j4 = i_points['PHI FIELD'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4, g4, h4, i4, j4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], \
                                                     np.asarray(d4)[w], np.asarray(e4)[w], np.asarray(f4)[w], \
                                                     np.asarray(g4)[w], np.asarray(h4)[w], np.asarray(i4)[w], \
                                                     np.asarray(j4)[w]

            label4 = []
            for d in range(0, len(a4)):
                label4.append('Field observed in band I: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i4[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j4[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g4[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h4[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e4[d], 3)) + ' $ \pm $ ' + str(
                    round(f4[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b4[d], 3)) + ' $ \pm $ ' + str(
                    round(c4[d], 3)))

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            band1 = plt.scatter(a1, e1, color='none')
            mplcursors.cursor(band1, hover=True).connect("add", lambda sel: sel.annotation.set_text(label1[sel.index]))

            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            band2 = plt.scatter(a2, e2, color='none')
            mplcursors.cursor(band2, hover=True).connect("add", lambda sel: sel.annotation.set_text(label2[sel.index]))

            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            band3 = plt.scatter(a3, e3, color='none')
            mplcursors.cursor(band3, hover=True).connect("add", lambda sel: sel.annotation.set_text(label3[sel.index]))

            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')
            band4 = plt.scatter(a4, e4, color='none')
            mplcursors.cursor(band4, hover=True).connect("add", lambda sel: sel.annotation.set_text(label4[sel.index]))

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            k1 = np.add(b2, c2)
            k2 = np.subtract(b2, c2)
            plt.fill_between(a2, k2, k1, where=(k2 < k1), interpolate=True, color='beige')

            l1 = np.add(b3, c3)
            l2 = np.subtract(b3, c3)
            plt.fill_between(a3, l2, l1, where=(l2 < l1), interpolate=True, color='mistyrose')

            m1 = np.add(b4, c4)
            m2 = np.subtract(b4, c4)
            plt.fill_between(a4, m2, m1, where=(m2 < m1), interpolate=True, color='antiquewhite')

            plt.ylim(0, 0.8)
            plt.ylabel('Polarization')
            if isinstance(result.params['par'].stderr, float):
                par_par = round(result.params['par'].stderr, 3)
            else:
                par_par = result.params['par'].stderr
            label_text = 'fit parameters: ' + ' $A$ = ' + str(
                round(result_simple.params['par'].value, 3)) + '$\pm$' + str(par_par) + '\n' + 'chi-square: ' + str(
                round(result_simple.chisqr, 10)) + ',   reduced chi-square: ' + str(
                round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                round(result_simple.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction', horizontalalignment='left', verticalalignment='center',
                         bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

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
            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_simple_sun_ray_' + meto + '.png', bbox_inches='tight')

            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
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
                b = points['SIM SUN POL'].to_numpy()
                c = points['SIM SUN UNC'].to_numpy()
                d = points['SIM SUN DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                band = plt.scatter(a, e, color='none')
                mplcursors.cursor(band, hover=True).connect("add",
                                                            lambda sel: sel.annotation.set_text(label2[sel.index]))

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylim(0, 0.8)
                plt.ylabel('Polarization')
                if isinstance(result.params['par'].stderr, float):
                    par_par = round(result.params['par'].stderr, 3)
                else:
                    par_par = result.params['par'].stderr
                label_text = 'fit parameters: ' + ' $A$ = ' + str(
                    round(result_simple.params['par'].value, 3)) + '$\pm$' + str(par_par) + '\n' + 'chi-square: ' + str(
                    round(result_simple.chisqr, 10)) + ',   reduced chi-square: ' + str(
                    round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                    round(result_simple.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction', horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
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
                b = points['SIM SUN POL'].to_numpy()
                c = points['SIM SUN UNC'].to_numpy()
                d = points['SIM SUN DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_simple_sun_ray_' + meto + '.png', bbox_inches='tight')
            plt.close()
            TXT.close()

    # --------------------------------------------------------------------------------------------

    if command == 'ALL' or command == 'fit seeing':
        model = lmfit.Model(func_seeing_DOP)
        model.set_param_hint('k', min=3, max=20)
        model.set_param_hint('d', min=0.1, max=1)
        p = model.make_params(k=np.random.rand(), d=np.random.rand())
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, SEEING], method='emcee',
        #                    weights=errPOL_OBS, nfev=500000)
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, SEEING, WAV],
                           weights=errPOL_OBS, method=meto)

        result_emcee_seeing = [result.params['k'].value, result.params['d'].value]
        result_emcee_seeing = np.asarray(result_emcee_seeing)
        Rpar = result_emcee_seeing

        txname = 'REPORT_' + LABEL + '_' + LAB + '_seeing_ray_' + meto + '.txt'
        TXT = open(txname, "w+")

        model_fit_report = result.fit_report()
        TXT.write('***  Fit: the seeing  astronomical parameter  *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        model_name = 'MODEL_' + LABEL + '_' + LAB + '_seeing_multi_' + meto + '.sav'

        lmfit.model.save_modelresult(result, model_name)

        y1 = func_seeing_DOP([C1field, C2field, C1lua, C2lua, SEEING, WAV], *result_emcee_seeing)
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
            g1 = b_points['THETA MOON'].to_numpy()
            h1 = b_points['PHI MOON'].to_numpy()
            i1 = b_points['THETA FIELD'].to_numpy()
            j1 = b_points['PHI FIELD'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1, g1, h1, i1, j1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], \
                                                     np.asarray(d1)[w], np.asarray(e1)[w], np.asarray(f1)[w], \
                                                     np.asarray(g1)[w], np.asarray(h1)[w], np.asarray(i1)[w], \
                                                     np.asarray(j1)[w]

            label1 = []
            for d in range(0, len(a1)):
                label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i1[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(round(j1[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g1[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h1[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                    round(e1[d], 3)) + ' $ \pm $ ' + str(round(f1[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                    round(b1[d], 3)) + ' $ \pm $ ' + str(round(c1[d], 3)))

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['SEEING POL'].to_numpy()
            c2 = v_points['SEEING UNC'].to_numpy()
            d2 = v_points['SEEING DIFF'].to_numpy()
            e2 = v_points['POL OBS'].to_numpy()
            f2 = v_points['POL OBS error'].to_numpy()
            g2 = v_points['THETA MOON'].to_numpy()
            h2 = v_points['PHI MOON'].to_numpy()
            i2 = v_points['THETA FIELD'].to_numpy()
            j2 = v_points['PHI FIELD'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2, g2, h2, i2, j2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], \
                                                     np.asarray(d2)[w], np.asarray(e2)[w], np.asarray(f2)[w], \
                                                     np.asarray(g2)[w], np.asarray(h2)[w], np.asarray(i2)[w], \
                                                     np.asarray(j2)[w]

            label2 = []
            for d in range(0, len(a2)):
                label2.append('Field observed in band V: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i2[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j2[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g2[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h2[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e2[d], 3)) + ' $ \pm $ ' + str(
                    round(f2[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b2[d], 3)) + ' $ \pm $ ' + str(
                    round(c2[d], 3)))

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['SEEING POL'].to_numpy()
            c3 = r_points['SEEING UNC'].to_numpy()
            d3 = r_points['SEEING DIFF'].to_numpy()
            e3 = r_points['POL OBS'].to_numpy()
            f3 = r_points['POL OBS error'].to_numpy()
            g3 = r_points['THETA MOON'].to_numpy()
            h3 = r_points['PHI MOON'].to_numpy()
            i3 = r_points['THETA FIELD'].to_numpy()
            j3 = r_points['PHI FIELD'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3, g3, h3, i3, j3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], \
                                                     np.asarray(d3)[w], np.asarray(e3)[w], np.asarray(f3)[w], \
                                                     np.asarray(g3)[w], np.asarray(h3)[w], np.asarray(i3)[w], \
                                                     np.asarray(j3)[w]

            label3 = []
            for d in range(0, len(a3)):
                label3.append('Field observed in band R: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i3[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j3[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g3[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h3[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e3[d], 3)) + ' $ \pm $ ' + str(
                    round(f3[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b3[d], 3)) + ' $ \pm $ ' + str(
                    round(c3[d], 3)))

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['SEEING POL'].to_numpy()
            c4 = i_points['SEEING UNC'].to_numpy()
            d4 = i_points['SEEING DIFF'].to_numpy()
            e4 = i_points['POL OBS'].to_numpy()
            f4 = i_points['POL OBS error'].to_numpy()
            g4 = i_points['THETA MOON'].to_numpy()
            h4 = i_points['PHI MOON'].to_numpy()
            i4 = i_points['THETA FIELD'].to_numpy()
            j4 = i_points['PHI FIELD'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4, g4, h4, i4, j4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], \
                                                     np.asarray(d4)[w], np.asarray(e4)[w], np.asarray(f4)[w], \
                                                     np.asarray(g4)[w], np.asarray(h4)[w], np.asarray(i4)[w], \
                                                     np.asarray(j4)[w]

            label4 = []
            for d in range(0, len(a4)):
                label4.append('Field observed in band I: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i4[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j4[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g4[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h4[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e4[d], 3)) + ' $ \pm $ ' + str(
                    round(f4[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b4[d], 3)) + ' $ \pm $ ' + str(
                    round(c4[d], 3)))

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            band1 = plt.scatter(a1, e1, color='none')
            mplcursors.cursor(band1, hover=True).connect("add", lambda sel: sel.annotation.set_text(label1[sel.index]))

            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            band2 = plt.scatter(a2, e2, color='none')
            mplcursors.cursor(band2, hover=True).connect("add", lambda sel: sel.annotation.set_text(label2[sel.index]))

            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            band3 = plt.scatter(a3, e3, color='none')
            mplcursors.cursor(band3, hover=True).connect("add", lambda sel: sel.annotation.set_text(label3[sel.index]))

            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')
            band4 = plt.scatter(a4, e4, color='none')
            mplcursors.cursor(band4, hover=True).connect("add", lambda sel: sel.annotation.set_text(label4[sel.index]))

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            k1 = np.add(b2, c2)
            k2 = np.subtract(b2, c2)
            plt.fill_between(a2, k2, k1, where=(k2 < k1), interpolate=True, color='beige')

            l1 = np.add(b3, c3)
            l2 = np.subtract(b3, c3)
            plt.fill_between(a3, l2, l1, where=(l2 < l1), interpolate=True, color='mistyrose')

            m1 = np.add(b4, c4)
            m2 = np.subtract(b4, c4)
            plt.fill_between(a4, m2, m1, where=(m2 < m1), interpolate=True, color='antiquewhite')

            plt.ylim(0, 0.8)
            plt.ylabel('Polarization')
            label_text = 'fit parameters: ' + ' $k_{1}$ = ' + str(
                round(result.params['k'].value, 3)) + '$\pm$' + str(
                round(result.params['k'].stderr, 3)) + ',   $k_{2}$ = ' + str(
                round(result.params['d'].value, 3)) + '$\pm$' + str(
                round(result.params['d'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                round(result.chisqr, 10)) + ',   reduced chi-square: ' + str(
                round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                round(result.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            leg_text = 'Individuals Amplitudes: \n \n$A_{B}$= ' + str(
                round(B_par[0], 3)) + '\n$A_{V}$= ' + str(round(V_par[0], 3)) + '\n$A_{R}$= ' + str(
                round(R_par[0], 3)) + '\n$A_{I}$= ' + str(round(I_par[0], 3))
            plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

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
            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_seeing_ray_' + meto + '.png', bbox_inches='tight')

            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
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
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                band = plt.scatter(a, e, color='none')
                mplcursors.cursor(band, hover=True).connect("add",
                                                            lambda sel: sel.annotation.set_text(label2[sel.index]))

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylim(0, 0.8)
                plt.ylabel('Polarization')
                label_text = 'fit parameters: ' + ' $k_{1}$ = ' + str(
                    round(result.params['k'].value, 3)) + '$\pm$' + str(
                    round(result.params['k'].stderr, 3)) + ',   $k_{2}$ = ' + str(
                    round(result.params['d'].value, 3)) + '$\pm$' + str(
                    round(result.params['d'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                    round(result.chisqr, 10)) + ',   reduced chi-square: ' + str(
                    round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                    round(result.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                leg_text = 'fit adding \nthe seeing astronomical parameter...\n \n$c_{B}$= ' + str(
                    round(B_par[0], 3)) + '\n$c_{V}$= ' + str(round(V_par[0], 3)) + '\n$c_{R}$= ' + str(
                    round(R_par[0], 3)) + '\n$c_{I}$= ' + str(round(I_par[0], 3))
                plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
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
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_seeing_ray_' + meto + '.png', bbox_inches='tight')
            plt.close()

        # ___________________________________________________________________________________________

        model = lmfit.Model(func_simple_seeing_DOP, independent_vars=['gamma', 'seeing', 'par_wave'])
        model.set_param_hint('k', min=3, max=20)
        model.set_param_hint('d', min=0.1, max=1)
        p = model.make_params(k=np.random.rand(), d=np.random.rand())  # , N=10)
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, SEEING], method='emcee',
        #                    weights=errPOL_OBS, nfev=500000)
        result_simple = model.fit(data=POL_OBS, params=p, gamma=GAMMA, seeing=SEEING, par_wave=WAV,
                                  weights=errPOL_OBS, method=meto)

        result_simple_seeing = [result_simple.params['k'].value, result_simple.params['d'].value]
        result_simple_seeing = np.asarray(result_simple_seeing)

        model_fit_report = result_simple.fit_report()
        TXT.write('***  Fit: the seeing  astronomical parameter in a simplier way *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        model_name = 'MODEL_' + LABEL + '_' + LAB + '_simple_seeing_multi_' + meto + '.sav'

        lmfit.model.save_modelresult(result_simple, model_name)

        y1 = func_simple_seeing_DOP(gamma=GAMMA, seeing=SEEING, par_wave=WAV, k=result_simple_seeing[0],
                                    d=result_simple_seeing[1])
        fit_observations_resume.insert(coluna, 'SIM SEEING POL', y1)
        coluna += 1

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
        fit_observations_resume.insert(coluna, 'SIM SEEING UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])
        fit_observations_resume.insert(coluna, 'SIM SEEING DIFF', diff)
        coluna += 1

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['SIM SEEING POL'].to_numpy()
            c1 = b_points['SIM SEEING UNC'].to_numpy()
            d1 = b_points['SIM SEEING DIFF'].to_numpy()
            e1 = b_points['POL OBS'].to_numpy()
            f1 = b_points['POL OBS error'].to_numpy()
            g1 = b_points['THETA MOON'].to_numpy()
            h1 = b_points['PHI MOON'].to_numpy()
            i1 = b_points['THETA FIELD'].to_numpy()
            j1 = b_points['PHI FIELD'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1, g1, h1, i1, j1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], \
                                                     np.asarray(d1)[w], np.asarray(e1)[w], np.asarray(f1)[w], \
                                                     np.asarray(g1)[w], np.asarray(h1)[w], np.asarray(i1)[w], \
                                                     np.asarray(j1)[w]

            label1 = []
            for d in range(0, len(a1)):
                label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i1[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(round(j1[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g1[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h1[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                    round(e1[d], 3)) + ' $ \pm $ ' + str(round(f1[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                    round(b1[d], 3)) + ' $ \pm $ ' + str(round(c1[d], 3)))

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['SIM SEEING POL'].to_numpy()
            c2 = v_points['SIM SEEING UNC'].to_numpy()
            d2 = v_points['SIM SEEING DIFF'].to_numpy()
            e2 = v_points['POL OBS'].to_numpy()
            f2 = v_points['POL OBS error'].to_numpy()
            g2 = v_points['THETA MOON'].to_numpy()
            h2 = v_points['PHI MOON'].to_numpy()
            i2 = v_points['THETA FIELD'].to_numpy()
            j2 = v_points['PHI FIELD'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2, g2, h2, i2, j2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], \
                                                     np.asarray(d2)[w], np.asarray(e2)[w], np.asarray(f2)[w], \
                                                     np.asarray(g2)[w], np.asarray(h2)[w], np.asarray(i2)[w], \
                                                     np.asarray(j2)[w]

            label2 = []
            for d in range(0, len(a2)):
                label2.append('Field observed in band V: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i2[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j2[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g2[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h2[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e2[d], 3)) + ' $ \pm $ ' + str(
                    round(f2[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b2[d], 3)) + ' $ \pm $ ' + str(
                    round(c2[d], 3)))

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['SIM SEEING POL'].to_numpy()
            c3 = r_points['SIM SEEING UNC'].to_numpy()
            d3 = r_points['SIM SEEING DIFF'].to_numpy()
            e3 = r_points['POL OBS'].to_numpy()
            f3 = r_points['POL OBS error'].to_numpy()
            g3 = r_points['THETA MOON'].to_numpy()
            h3 = r_points['PHI MOON'].to_numpy()
            i3 = r_points['THETA FIELD'].to_numpy()
            j3 = r_points['PHI FIELD'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3, g3, h3, i3, j3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], \
                                                     np.asarray(d3)[w], np.asarray(e3)[w], np.asarray(f3)[w], \
                                                     np.asarray(g3)[w], np.asarray(h3)[w], np.asarray(i3)[w], \
                                                     np.asarray(j3)[w]

            label3 = []
            for d in range(0, len(a3)):
                label3.append('Field observed in band R: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i3[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j3[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g3[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h3[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e3[d], 3)) + ' $ \pm $ ' + str(
                    round(f3[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b3[d], 3)) + ' $ \pm $ ' + str(
                    round(c3[d], 3)))

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['SIM SEEING POL'].to_numpy()
            c4 = i_points['SIM SEEING UNC'].to_numpy()
            d4 = i_points['SIM SEEING DIFF'].to_numpy()
            e4 = i_points['POL OBS'].to_numpy()
            f4 = i_points['POL OBS error'].to_numpy()
            g4 = i_points['THETA MOON'].to_numpy()
            h4 = i_points['PHI MOON'].to_numpy()
            i4 = i_points['THETA FIELD'].to_numpy()
            j4 = i_points['PHI FIELD'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4, g4, h4, i4, j4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], \
                                                     np.asarray(d4)[w], np.asarray(e4)[w], np.asarray(f4)[w], \
                                                     np.asarray(g4)[w], np.asarray(h4)[w], np.asarray(i4)[w], \
                                                     np.asarray(j4)[w]

            label4 = []
            for d in range(0, len(a4)):
                label4.append('Field observed in band I: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i4[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j4[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g4[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h4[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e4[d], 3)) + ' $ \pm $ ' + str(
                    round(f4[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b4[d], 3)) + ' $ \pm $ ' + str(
                    round(c4[d], 3)))

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            band1 = plt.scatter(a1, e1, color='none')
            mplcursors.cursor(band1, hover=True).connect("add", lambda sel: sel.annotation.set_text(label1[sel.index]))

            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            band2 = plt.scatter(a2, e2, color='none')
            mplcursors.cursor(band2, hover=True).connect("add", lambda sel: sel.annotation.set_text(label2[sel.index]))

            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            band3 = plt.scatter(a3, e3, color='none')
            mplcursors.cursor(band3, hover=True).connect("add", lambda sel: sel.annotation.set_text(label3[sel.index]))

            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')
            band4 = plt.scatter(a4, e4, color='none')
            mplcursors.cursor(band4, hover=True).connect("add", lambda sel: sel.annotation.set_text(label4[sel.index]))

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            k1 = np.add(b2, c2)
            k2 = np.subtract(b2, c2)
            plt.fill_between(a2, k2, k1, where=(k2 < k1), interpolate=True, color='beige')

            l1 = np.add(b3, c3)
            l2 = np.subtract(b3, c3)
            plt.fill_between(a3, l2, l1, where=(l2 < l1), interpolate=True, color='mistyrose')

            m1 = np.add(b4, c4)
            m2 = np.subtract(b4, c4)
            plt.fill_between(a4, m2, m1, where=(m2 < m1), interpolate=True, color='antiquewhite')

            plt.ylim(0, 0.8)
            plt.ylabel('Polarization')
            label_text = 'fit parameters: ' + ' $k_{1}$ = ' + str(
                round(result_simple.params['k'].value, 3)) + '$\pm$' + str(
                round(result_simple.params['k'].stderr, 3)) + ',   $k_{2}$ = ' + str(
                round(result_simple.params['d'].value, 3)) + '$\pm$' + str(
                round(result_simple.params['d'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                round(result_simple.chisqr, 10)) + ',   reduced chi-square: ' + str(
                round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                round(result_simple.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            leg_text = 'Individuals Amplitudes: \n \n$A_{B}$= ' + str(
                round(B_par[0], 3)) + '\n$A_{V}$= ' + str(round(V_par[0], 3)) + '\n$A_{R}$= ' + str(
                round(R_par[0], 3)) + '\n$A_{I}$= ' + str(round(I_par[0], 3))
            plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

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
            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_simple_seeing_ray_' + meto + '.png', bbox_inches='tight')
            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
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
                b = points['SIM SEEING POL'].to_numpy()
                c = points['SIM SEEING UNC'].to_numpy()
                d = points['SIM SEEING DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                band = plt.scatter(a, e, color='none')
                mplcursors.cursor(band, hover=True).connect("add",
                                                            lambda sel: sel.annotation.set_text(label2[sel.index]))

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylim(0, 0.8)
                plt.ylabel('Polarization')
                label_text = 'fit parameters: ' + ' $k_{1}$ = ' + str(
                    round(result_simple.params['k'].value, 3)) + '$\pm$' + str(
                    round(result_simple.params['k'].stderr, 3)) + ',   $k_{2}$ = ' + str(
                    round(result_simple.params['d'].value, 3)) + '$\pm$' + str(
                    round(result_simple.params['d'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                    round(result_simple.chisqr, 10)) + ',   reduced chi-square: ' + str(
                    round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                    round(result_simple.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                leg_text = 'simple fit adding \nthe seeing astronomical parameter...\n \n$c_{B}$= ' + str(
                    round(B_par[0], 3)) + '\n$c_{V}$= ' + str(round(V_par[0], 3)) + '\n$c_{R}$= ' + str(
                    round(R_par[0], 3)) + '\n$c_{I}$= ' + str(round(I_par[0], 3))
                plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
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
                b = points['SIM SEEING POL'].to_numpy()
                c = points['SIM SEEING UNC'].to_numpy()
                d = points['SIM SEEING DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_simple_seeing_ray_' + meto + '.png', bbox_inches='tight')
            plt.close()
            TXT.close()

    # ---------------------------------------------------------------------------------------------------

    if command == 'ALL' or command == 'fit wavelength':
        model = lmfit.Model(func_wav)
        model.set_param_hint('c', min=-4)
        p = model.make_params(c=np.random.rand())  # , N=10)
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee',
        #                   weights=errPOL_OBS, nfev=500000)
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, BAND, SEEING, N_BAND, K_BAND, D_BAND],
                           weights=errPOL_OBS,
                           method=meto)
        # crds1, crds2, cLUA1, cLUA2, banda, seeing, n_par, k_par, d_par

        result_emcee_wav = [result.params['c'].value]
        result_emcee_wav = np.asarray(result_emcee_wav)
        Rpar = result_emcee_wav

        txname = 'REPORT_' + LABEL + '_' + LAB + '_wave_ray_' + meto + '.txt'
        TXT = open(txname, "w+")

        model_fit_report = result.fit_report()
        TXT.write('***  Fit: the wavelength of light *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        model_name = 'MODEL_' + LABEL + '_' + LAB + '_wave_ray_' + meto + '.sav'

        lmfit.model.save_modelresult(result, model_name)

        y1 = func_wav([C1field, C2field, C1lua, C2lua, BAND, SEEING, N_BAND, K_BAND, D_BAND], *result_emcee_wav)
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
        coluna += 1
        barra.update(10)

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
            g1 = b_points['THETA MOON'].to_numpy()
            h1 = b_points['PHI MOON'].to_numpy()
            i1 = b_points['THETA FIELD'].to_numpy()
            j1 = b_points['PHI FIELD'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1, g1, h1, i1, j1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], \
                                                     np.asarray(d1)[w], np.asarray(e1)[w], np.asarray(f1)[w], \
                                                     np.asarray(g1)[w], np.asarray(h1)[w], np.asarray(i1)[w], \
                                                     np.asarray(j1)[w]

            label1 = []
            for d in range(0, len(a1)):
                label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i1[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(round(j1[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g1[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h1[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                    round(e1[d], 3)) + ' $ \pm $ ' + str(round(f1[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                    round(b1[d], 3)) + ' $ \pm $ ' + str(round(c1[d], 3)))

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['WAVE POL'].to_numpy()
            c2 = v_points['WAVE UNC'].to_numpy()
            d2 = v_points['WAVE DIFF'].to_numpy()
            e2 = v_points['POL OBS'].to_numpy()
            f2 = v_points['POL OBS error'].to_numpy()
            g2 = v_points['THETA MOON'].to_numpy()
            h2 = v_points['PHI MOON'].to_numpy()
            i2 = v_points['THETA FIELD'].to_numpy()
            j2 = v_points['PHI FIELD'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2, g2, h2, i2, j2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], \
                                                     np.asarray(d2)[w], np.asarray(e2)[w], np.asarray(f2)[w], \
                                                     np.asarray(g2)[w], np.asarray(h2)[w], np.asarray(i2)[w], \
                                                     np.asarray(j2)[w]

            label2 = []
            for d in range(0, len(a2)):
                label2.append('Field observed in band V: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i2[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j2[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g2[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h2[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e2[d], 3)) + ' $ \pm $ ' + str(
                    round(f2[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b2[d], 3)) + ' $ \pm $ ' + str(
                    round(c2[d], 3)))

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['WAVE POL'].to_numpy()
            c3 = r_points['WAVE UNC'].to_numpy()
            d3 = r_points['WAVE DIFF'].to_numpy()
            e3 = r_points['POL OBS'].to_numpy()
            f3 = r_points['POL OBS error'].to_numpy()
            g3 = r_points['THETA MOON'].to_numpy()
            h3 = r_points['PHI MOON'].to_numpy()
            i3 = r_points['THETA FIELD'].to_numpy()
            j3 = r_points['PHI FIELD'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3, g3, h3, i3, j3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], \
                                                     np.asarray(d3)[w], np.asarray(e3)[w], np.asarray(f3)[w], \
                                                     np.asarray(g3)[w], np.asarray(h3)[w], np.asarray(i3)[w], \
                                                     np.asarray(j3)[w]

            label3 = []
            for d in range(0, len(a3)):
                label3.append('Field observed in band R: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i3[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j3[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g3[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h3[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e3[d], 3)) + ' $ \pm $ ' + str(
                    round(f3[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b3[d], 3)) + ' $ \pm $ ' + str(
                    round(c3[d], 3)))

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['WAVE POL'].to_numpy()
            c4 = i_points['WAVE UNC'].to_numpy()
            d4 = i_points['WAVE DIFF'].to_numpy()
            e4 = i_points['POL OBS'].to_numpy()
            f4 = i_points['POL OBS error'].to_numpy()
            g4 = i_points['THETA MOON'].to_numpy()
            h4 = i_points['PHI MOON'].to_numpy()
            i4 = i_points['THETA FIELD'].to_numpy()
            j4 = i_points['PHI FIELD'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4, g4, h4, i4, j4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], \
                                                     np.asarray(d4)[w], np.asarray(e4)[w], np.asarray(f4)[w], \
                                                     np.asarray(g4)[w], np.asarray(h4)[w], np.asarray(i4)[w], \
                                                     np.asarray(j4)[w]

            label4 = []
            for d in range(0, len(a4)):
                label4.append('Field observed in band I: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i4[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j4[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g4[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h4[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e4[d], 3)) + ' $ \pm $ ' + str(
                    round(f4[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b4[d], 3)) + ' $ \pm $ ' + str(
                    round(c4[d], 3)))

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            band1 = plt.scatter(a1, e1, color='none')
            mplcursors.cursor(band1, hover=True).connect("add", lambda sel: sel.annotation.set_text(label1[sel.index]))

            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            band2 = plt.scatter(a2, e2, color='none')
            mplcursors.cursor(band2, hover=True).connect("add", lambda sel: sel.annotation.set_text(label2[sel.index]))

            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            band3 = plt.scatter(a3, e3, color='none')
            mplcursors.cursor(band3, hover=True).connect("add", lambda sel: sel.annotation.set_text(label3[sel.index]))

            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')
            band4 = plt.scatter(a4, e4, color='none')
            mplcursors.cursor(band4, hover=True).connect("add", lambda sel: sel.annotation.set_text(label4[sel.index]))

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            k1 = np.add(b2, c2)
            k2 = np.subtract(b2, c2)
            plt.fill_between(a2, k2, k1, where=(k2 < k1), interpolate=True, color='beige')

            l1 = np.add(b3, c3)
            l2 = np.subtract(b3, c3)
            plt.fill_between(a3, l2, l1, where=(l2 < l1), interpolate=True, color='mistyrose')

            m1 = np.add(b4, c4)
            m2 = np.subtract(b4, c4)
            plt.fill_between(a4, m2, m1, where=(m2 < m1), interpolate=True, color='antiquewhite')

            if isinstance(result.params['c'].stderr, float):
                c_par = round(result.params['c'].stderr,8)
            else:
                c_par = result.params['c'].stderr
            plt.ylim(0, 0.8)
            plt.ylabel('Polarization')
            label_text = 'fit parameters: ' + ' $c$ = ' + str(round(result.params['c'].value,8)) + '$\pm$' + str(
                c_par) + '\n' + 'chi-square: ' + str(
                round(result.chisqr, 10)) + ',   reduced chi-square: ' + str(
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
            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_wave_ray_' + meto + '.png', bbox_inches='tight')

            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
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
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                band = plt.scatter(a, e, color='none')
                mplcursors.cursor(band, hover=True).connect("add",
                                                            lambda sel: sel.annotation.set_text(label2[sel.index]))

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylim(0, 0.8)
                plt.ylabel('Polarization')
                if isinstance(result.params['c'].stderr, float):
                    c_par = round(result.params['c'].stderr, 3)
                else:
                    c_par = result.params['c'].stderr
                plt.ylim(0, 0.8)
                plt.ylabel('Polarization')
                label_text = 'fit parameters: ' + ' $c$ = ' + str(round(result.params['c'].value, 3)) + '$\pm$' + str(
                    c_par) + '\n' + 'chi-square: ' + str(
                    round(result.chisqr, 10)) + ',   reduced chi-square: ' + str(
                    round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                barra.update(10)

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
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
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                barra.update(10)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_wave_ray_' + meto + '.png', bbox_inches='tight')
            plt.close()

        # ___________________________________________________________________________________________

        model = lmfit.Model(func_simple_wav, independent_vars=['theta_lua', 'gamma', 'wavel', 'seeing', 'n_par', 'k_par', 'd_par'])
        # theta_lua, gamma, wavel, seeing, n_par, k_par, d_par, c
        model.set_param_hint('c')
        p = model.make_params(c=np.random.rand())  # , N=10)
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee',
        #                    weights=errPOL_OBS, nfev=500000)
        result_simple = model.fit(data=POL_OBS, params=p, theta_lua=C1lua, gamma=GAMMA, wavel=BAND, seeing=SEEING, n_par=N_BAND, k_par=K_BAND, d_par=D_BAND,
                                  weights=errPOL_OBS, method=meto)

        result_simple_wav = [result_simple.params['c'].value]
        result_simple_wav = np.asarray(result_simple_wav)

        model_fit_report = result_simple.fit_report()
        TXT.write('*** Fit: wavelength in a simplier way *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        model_name = 'MODEL_' + LABEL + '_' + LAB + '_simple_wave_ray_' + meto + '.sav'

        lmfit.model.save_modelresult(result_simple, model_name)

        y1 = func_simple_wav(theta_lua=C1lua, gamma=GAMMA, wavel=BAND, seeing=SEEING, n_par=N_BAND, k_par=K_BAND, d_par=D_BAND, c=result_simple_wav[0])
        fit_observations_resume.insert(coluna, 'SIM WAVE POL', y1)
        coluna += 1

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
        fit_observations_resume.insert(coluna, 'SIM WAVE UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])
        fit_observations_resume.insert(coluna, 'SIM WAVE DIFF', diff)
        coluna += 1

        barra.update(10)

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['SIM WAVE POL'].to_numpy()
            c1 = b_points['SIM WAVE UNC'].to_numpy()
            d1 = b_points['SIM WAVE DIFF'].to_numpy()
            e1 = b_points['POL OBS'].to_numpy()
            f1 = b_points['POL OBS error'].to_numpy()
            g1 = b_points['THETA MOON'].to_numpy()
            h1 = b_points['PHI MOON'].to_numpy()
            i1 = b_points['THETA FIELD'].to_numpy()
            j1 = b_points['PHI FIELD'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1, g1, h1, i1, j1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], \
                                                     np.asarray(d1)[w], np.asarray(e1)[w], np.asarray(f1)[w], \
                                                     np.asarray(g1)[w], np.asarray(h1)[w], np.asarray(i1)[w], \
                                                     np.asarray(j1)[w]

            label1 = []
            for d in range(0, len(a1)):
                label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i1[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(round(j1[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g1[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h1[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                    round(e1[d], 3)) + ' $ \pm $ ' + str(round(f1[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                    round(b1[d], 3)) + ' $ \pm $ ' + str(round(c1[d], 3)))

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['SIM WAVE POL'].to_numpy()
            c2 = v_points['SIM WAVE UNC'].to_numpy()
            d2 = v_points['SIM WAVE DIFF'].to_numpy()
            e2 = v_points['POL OBS'].to_numpy()
            f2 = v_points['POL OBS error'].to_numpy()
            g2 = v_points['THETA MOON'].to_numpy()
            h2 = v_points['PHI MOON'].to_numpy()
            i2 = v_points['THETA FIELD'].to_numpy()
            j2 = v_points['PHI FIELD'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2, g2, h2, i2, j2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], \
                                                     np.asarray(d2)[w], np.asarray(e2)[w], np.asarray(f2)[w], \
                                                     np.asarray(g2)[w], np.asarray(h2)[w], np.asarray(i2)[w], \
                                                     np.asarray(j2)[w]

            label2 = []
            for d in range(0, len(a2)):
                label2.append('Field observed in band V: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i2[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j2[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g2[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h2[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e2[d], 3)) + ' $ \pm $ ' + str(
                    round(f2[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b2[d], 3)) + ' $ \pm $ ' + str(
                    round(c2[d], 3)))

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['SIM WAVE POL'].to_numpy()
            c3 = r_points['SIM WAVE UNC'].to_numpy()
            d3 = r_points['SIM WAVE DIFF'].to_numpy()
            e3 = r_points['POL OBS'].to_numpy()
            f3 = r_points['POL OBS error'].to_numpy()
            g3 = r_points['THETA MOON'].to_numpy()
            h3 = r_points['PHI MOON'].to_numpy()
            i3 = r_points['THETA FIELD'].to_numpy()
            j3 = r_points['PHI FIELD'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3, g3, h3, i3, j3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], \
                                                     np.asarray(d3)[w], np.asarray(e3)[w], np.asarray(f3)[w], \
                                                     np.asarray(g3)[w], np.asarray(h3)[w], np.asarray(i3)[w], \
                                                     np.asarray(j3)[w]

            label3 = []
            for d in range(0, len(a3)):
                label3.append('Field observed in band R: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i3[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j3[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g3[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h3[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e3[d], 3)) + ' $ \pm $ ' + str(
                    round(f3[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b3[d], 3)) + ' $ \pm $ ' + str(
                    round(c3[d], 3)))

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['SIM WAVE POL'].to_numpy()
            c4 = i_points['SIM WAVE UNC'].to_numpy()
            d4 = i_points['SIM WAVE DIFF'].to_numpy()
            e4 = i_points['POL OBS'].to_numpy()
            f4 = i_points['POL OBS error'].to_numpy()
            g4 = i_points['THETA MOON'].to_numpy()
            h4 = i_points['PHI MOON'].to_numpy()
            i4 = i_points['THETA FIELD'].to_numpy()
            j4 = i_points['PHI FIELD'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4, g4, h4, i4, j4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], \
                                                     np.asarray(d4)[w], np.asarray(e4)[w], np.asarray(f4)[w], \
                                                     np.asarray(g4)[w], np.asarray(h4)[w], np.asarray(i4)[w], \
                                                     np.asarray(j4)[w]

            label4 = []
            for d in range(0, len(a4)):
                label4.append('Field observed in band I: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i4[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j4[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g4[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h4[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e4[d], 3)) + ' $ \pm $ ' + str(
                    round(f4[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b4[d], 3)) + ' $ \pm $ ' + str(
                    round(c4[d], 3)))

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            band1 = plt.scatter(a1, e1, color='none')
            mplcursors.cursor(band1, hover=True).connect("add", lambda sel: sel.annotation.set_text(label1[sel.index]))

            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            band2 = plt.scatter(a2, e2, color='none')
            mplcursors.cursor(band2, hover=True).connect("add", lambda sel: sel.annotation.set_text(label2[sel.index]))

            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            band3 = plt.scatter(a3, e3, color='none')
            mplcursors.cursor(band3, hover=True).connect("add", lambda sel: sel.annotation.set_text(label3[sel.index]))

            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')
            band4 = plt.scatter(a4, e4, color='none')
            mplcursors.cursor(band4, hover=True).connect("add", lambda sel: sel.annotation.set_text(label4[sel.index]))

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            k1 = np.add(b2, c2)
            k2 = np.subtract(b2, c2)
            plt.fill_between(a2, k2, k1, where=(k2 < k1), interpolate=True, color='beige')

            l1 = np.add(b3, c3)
            l2 = np.subtract(b3, c3)
            plt.fill_between(a3, l2, l1, where=(l2 < l1), interpolate=True, color='mistyrose')

            m1 = np.add(b4, c4)
            m2 = np.subtract(b4, c4)
            plt.fill_between(a4, m2, m1, where=(m2 < m1), interpolate=True, color='antiquewhite')

            plt.ylim(0, 0.8)
            plt.ylabel('Polarization')
            if isinstance(result_simple.params['c'].stderr, float):
                c_par = round(result_simple.params['c'].stderr,8)
            else:
                c_par = result_simple.params['c'].stderr
            label_text = 'fit parameters: ' + '  $c$ = ' + str(
                round(result_simple.params['c'].value,8)) + '$\pm$' + str(
                c_par) + '\n' + 'chi-square: ' + str(
                round(result_simple.chisqr, 10)) + ',   reduced chi-square: ' + str(
                round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                round(result_simple.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

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
            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_simple_wave_ray_' + meto + '.png', bbox_inches='tight')

            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
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
                b = points['SIM WAVE POL'].to_numpy()
                c = points['SIM WAVE UNC'].to_numpy()
                d = points['SIM WAVE DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                barra.update(10 / len(band))

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                band = plt.scatter(a, e, color='none')
                mplcursors.cursor(band, hover=True).connect("add",
                                                            lambda sel: sel.annotation.set_text(label2[sel.index]))

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylim(0, 0.8)
                plt.ylabel('Polarization')
                if isinstance(result_simple.params['c'].stderr, float):
                    c_par = round(result_simple.params['c'].stderr, 3)
                else:
                    c_par = result_simple.params['c'].stderr
                label_text = 'fit parameters: ' + '  $c$ = ' + str(
                    round(result_simple.params['c'].value, 3)) + '$\pm$' + str(
                    c_par) + '\n' + 'chi-square: ' + str(
                    round(result_simple.chisqr, 10)) + ',   reduced chi-square: ' + str(
                    round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                    round(result_simple.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

                barra.update(10)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
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
                b = points['SIM WAVE POL'].to_numpy()
                c = points['SIM WAVE UNC'].to_numpy()
                d = points['SIM WAVE DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

                barra.update(10)

            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_simple_wave_ray_' + meto + '.png', bbox_inches='tight')
            plt.close()
            TXT.close()

    # -------------------------------------------------------------------------------------------

    if command == 'ALL' or command == 'fit all':
        model = lmfit.Model(func_DOP)
        model.set_param_hint('k', min=2, max=20)
        model.set_param_hint('d', min=0.1, max=1)
        model.set_param_hint('N', min=0, max=20)
        p = model.make_params(N=np.random.rand(), k=np.random.rand(), d=np.random.rand())  # , N=10)
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, SEEING, BAND, WAV],
                           weights=errPOL_OBS, method='leastsq')

        result_emcee = [result.params['N'].value, result.params['k'].value, result.params['d'].value]
        result_emcee = np.asarray(result_emcee)
        Rpar = result_emcee

        txname = 'REPORT_' + LABEL + '_' + LAB+ '_all_ray_' + meto + '.txt'
        TXT = open(txname, "w+")

        model_fit_report = result.fit_report()
        TXT.write('*** Fit: all previous corrections *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        model_name = 'MODEL_' + LABEL + '_' + LAB + '_all_ray_' + meto + '.sav'

        lmfit.model.save_modelresult(result, model_name)

        y1 = func_DOP([C1field, C2field, C1lua, C2lua, SEEING, BAND, WAV], *result_emcee)
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
            g1 = b_points['THETA MOON'].to_numpy()
            h1 = b_points['PHI MOON'].to_numpy()
            i1 = b_points['THETA FIELD'].to_numpy()
            j1 = b_points['PHI FIELD'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1, g1, h1, i1, j1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], \
                                                     np.asarray(d1)[w], np.asarray(e1)[w], np.asarray(f1)[w], \
                                                     np.asarray(g1)[w], np.asarray(h1)[w], np.asarray(i1)[w], \
                                                     np.asarray(j1)[w]

            label1 = []
            for d in range(0, len(a1)):
                label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i1[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(round(j1[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g1[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h1[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                    round(e1[d], 3)) + ' $ \pm $ ' + str(round(f1[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                    round(b1[d], 3)) + ' $ \pm $ ' + str(round(c1[d], 3)))

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['ALL POL'].to_numpy()
            c2 = v_points['ALL UNC'].to_numpy()
            d2 = v_points['ALL DIFF'].to_numpy()
            e2 = v_points['POL OBS'].to_numpy()
            f2 = v_points['POL OBS error'].to_numpy()
            g2 = v_points['THETA MOON'].to_numpy()
            h2 = v_points['PHI MOON'].to_numpy()
            i2 = v_points['THETA FIELD'].to_numpy()
            j2 = v_points['PHI FIELD'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2, g2, h2, i2, j2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], \
                                                     np.asarray(d2)[w], np.asarray(e2)[w], np.asarray(f2)[w], \
                                                     np.asarray(g2)[w], np.asarray(h2)[w], np.asarray(i2)[w], \
                                                     np.asarray(j2)[w]

            label2 = []
            for d in range(0, len(a2)):
                label2.append('Field observed in band V: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i2[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j2[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g2[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h2[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e2[d], 3)) + ' $ \pm $ ' + str(
                    round(f2[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b2[d], 3)) + ' $ \pm $ ' + str(
                    round(c2[d], 3)))

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['ALL POL'].to_numpy()
            c3 = r_points['ALL UNC'].to_numpy()
            d3 = r_points['ALL DIFF'].to_numpy()
            e3 = r_points['POL OBS'].to_numpy()
            f3 = r_points['POL OBS error'].to_numpy()
            g3 = r_points['THETA MOON'].to_numpy()
            h3 = r_points['PHI MOON'].to_numpy()
            i3 = r_points['THETA FIELD'].to_numpy()
            j3 = r_points['PHI FIELD'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3, g3, h3, i3, j3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], \
                                                     np.asarray(d3)[w], np.asarray(e3)[w], np.asarray(f3)[w], \
                                                     np.asarray(g3)[w], np.asarray(h3)[w], np.asarray(i3)[w], \
                                                     np.asarray(j3)[w]

            label3 = []
            for d in range(0, len(a3)):
                label3.append('Field observed in band R: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i3[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j3[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g3[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h3[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e3[d], 3)) + ' $ \pm $ ' + str(
                    round(f3[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b3[d], 3)) + ' $ \pm $ ' + str(
                    round(c3[d], 3)))

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['ALL POL'].to_numpy()
            c4 = i_points['ALL UNC'].to_numpy()
            d4 = i_points['ALL DIFF'].to_numpy()
            e4 = i_points['POL OBS'].to_numpy()
            f4 = i_points['POL OBS error'].to_numpy()
            g4 = i_points['THETA MOON'].to_numpy()
            h4 = i_points['PHI MOON'].to_numpy()
            i4 = i_points['THETA FIELD'].to_numpy()
            j4 = i_points['PHI FIELD'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4, g4, h4, i4, j4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], \
                                                     np.asarray(d4)[w], np.asarray(e4)[w], np.asarray(f4)[w], \
                                                     np.asarray(g4)[w], np.asarray(h4)[w], np.asarray(i4)[w], \
                                                     np.asarray(j4)[w]

            label4 = []
            for d in range(0, len(a4)):
                label4.append('Field observed in band I: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i4[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j4[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g4[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h4[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e4[d], 3)) + ' $ \pm $ ' + str(
                    round(f4[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b4[d], 3)) + ' $ \pm $ ' + str(
                    round(c4[d], 3)))

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            band1 = plt.scatter(a1, e1, color='none')
            mplcursors.cursor(band1, hover=True).connect("add", lambda sel: sel.annotation.set_text(label1[sel.index]))

            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            band2 = plt.scatter(a2, e2, color='none')
            mplcursors.cursor(band2, hover=True).connect("add", lambda sel: sel.annotation.set_text(label2[sel.index]))

            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            band3 = plt.scatter(a3, e3, color='none')
            mplcursors.cursor(band3, hover=True).connect("add", lambda sel: sel.annotation.set_text(label3[sel.index]))

            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')
            band4 = plt.scatter(a4, e4, color='none')
            mplcursors.cursor(band4, hover=True).connect("add", lambda sel: sel.annotation.set_text(label4[sel.index]))

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            k1 = np.add(b2, c2)
            k2 = np.subtract(b2, c2)
            plt.fill_between(a2, k2, k1, where=(k2 < k1), interpolate=True, color='beige')

            l1 = np.add(b3, c3)
            l2 = np.subtract(b3, c3)
            plt.fill_between(a3, l2, l1, where=(l2 < l1), interpolate=True, color='mistyrose')

            m1 = np.add(b4, c4)
            m2 = np.subtract(b4, c4)
            plt.fill_between(a4, m2, m1, where=(m2 < m1), interpolate=True, color='antiquewhite')

            plt.ylim(0, 0.8)
            plt.ylabel('Polarization')
            label_text = 'fit parameters: ' + ' $k_{1}$ = ' + str(
                round(result.params['k'].value, 3)) + '$\pm$' + str(
                round(result.params['k'].stderr, 3)) + ',   $k_{2}$ = ' + str(
                round(result.params['d'].value, 3)) + '$\pm$' + str(
                round(result.params['d'].stderr, 3)) + ',   $N$ = ' + str(
                round(result.params['N'].value, 3)) + '$\pm$' + str(
                round(result.params['N'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                round(result.chisqr, 10)) + 'reduced chi-square: ' + str(
                round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            leg_text = 'fit with all corrections...\n \n$c_{B}$= ' + str(
                round(B_par[0], 3)) + '\n$c_{V}$= ' + str(
                round(V_par[0], 3)) + '\n$c_{R}$= ' + str(
                round(R_par[0], 3)) + '\n$c_{I}$= ' + str(round(I_par[0], 3))
            plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

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
            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_all_ray_' + meto + '.png', bbox_inches='tight')

            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
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
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                barra.update(10 / len(band))

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                band = plt.scatter(a, e, color='none')
                mplcursors.cursor(band, hover=True).connect("add",
                                                            lambda sel: sel.annotation.set_text(label2[sel.index]))

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylim(0, 0.8)
                plt.ylabel('Polarization')
                label_text = 'fit parameters: ' + ' $k_{1}$ = ' + str(
                    round(result.params['k'].value, 3)) + '$\pm$' + str(
                    round(result.params['k'].stderr, 3)) + ',   $k_{2}$ = ' + str(
                    round(result.params['d'].value, 3)) + '$\pm$' + str(
                    round(result.params['d'].stderr, 3)) + ',   $N$ = ' + str(
                    round(result.params['N'].value, 3)) + '$\pm$' + str(
                    round(result.params['N'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                    round(result.chisqr, 10)) + 'reduced chi-square: ' + str(
                    round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                leg_text = 'fit with all corrections...\n \n$c_{B}$= ' + str(
                    round(B_par[0], 3)) + '\n$c_{V}$= ' + str(
                    round(V_par[0], 3)) + '\n$c_{R}$= ' + str(
                    round(R_par[0], 3)) + '\n$c_{I}$= ' + str(round(I_par[0], 3))
                plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
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
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_all_ray_' + meto + '.png', bbox_inches='tight')
            plt.close()

        # ___________________________________________________________________________________________

        model = lmfit.Model(func_simple_DOP, independent_vars=['gamma', 'seeing', 'theta_lua', 'wave_par'])
        model.set_param_hint('k', min=2, max=20)
        model.set_param_hint('d', min=0.1, max=1)
        model.set_param_hint('N', min=0.1, max=20)
        p = model.make_params(N=np.random.rand(), k=np.random.rand(), d=np.random.rand())  # , N=10)
        result_simple = model.fit(data=POL_OBS, params=p, theta_lua=C1lua, gamma=GAMMA, seeing=SEEING,
                                  wave_par=WAV, weights=errPOL_OBS, method=meto)

        results_simple = [result_simple.params['N'].value, result_simple.params['k'].value,
                          result_simple.params['d'].value]
        results_simple = np.asarray(results_simple)

        model_fit_report = result_simple.fit_report()
        TXT.write('*** Fit: all previous corrections in a simplier way *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        model_name = 'MODEL_' + LABEL + '_' + LAB + '_simple_all_multi_' + meto + '.sav'

        lmfit.model.save_modelresult(result_simple, model_name)

        y1 = func_simple_DOP(theta_lua=C1lua, gamma=GAMMA, seeing=SEEING, wave_par=WAV, N=results_simple[0],
                             k=results_simple[1], d=results_simple[2])
        fit_observations_resume.insert(coluna, 'SIM ALL POL', y1)
        coluna += 1

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
        fit_observations_resume.insert(coluna, 'SIM ALL UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])
        fit_observations_resume.insert(coluna, 'SIM ALL DIFF', diff)
        coluna += 1

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['SIM ALL POL'].to_numpy()
            c1 = b_points['SIM ALL UNC'].to_numpy()
            d1 = b_points['SIM ALL DIFF'].to_numpy()
            e1 = b_points['POL OBS'].to_numpy()
            f1 = b_points['POL OBS error'].to_numpy()
            g1 = b_points['THETA MOON'].to_numpy()
            h1 = b_points['PHI MOON'].to_numpy()
            i1 = b_points['THETA FIELD'].to_numpy()
            j1 = b_points['PHI FIELD'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1, g1, h1, i1, j1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], \
                                                     np.asarray(d1)[w], np.asarray(e1)[w], np.asarray(f1)[w], \
                                                     np.asarray(g1)[w], np.asarray(h1)[w], np.asarray(i1)[w], \
                                                     np.asarray(j1)[w]

            label1 = []
            for d in range(0, len(a1)):
                label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i1[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(round(j1[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g1[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h1[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                    round(e1[d], 3)) + ' $ \pm $ ' + str(round(f1[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                    round(b1[d], 3)) + ' $ \pm $ ' + str(round(c1[d], 3)))

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['SIM ALL POL'].to_numpy()
            c2 = v_points['SIM ALL UNC'].to_numpy()
            d2 = v_points['SIM ALL DIFF'].to_numpy()
            e2 = v_points['POL OBS'].to_numpy()
            f2 = v_points['POL OBS error'].to_numpy()
            g2 = v_points['THETA MOON'].to_numpy()
            h2 = v_points['PHI MOON'].to_numpy()
            i2 = v_points['THETA FIELD'].to_numpy()
            j2 = v_points['PHI FIELD'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2, g2, h2, i2, j2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], \
                                                     np.asarray(d2)[w], np.asarray(e2)[w], np.asarray(f2)[w], \
                                                     np.asarray(g2)[w], np.asarray(h2)[w], np.asarray(i2)[w], \
                                                     np.asarray(j2)[w]

            label2 = []
            for d in range(0, len(a2)):
                label2.append('Field observed in band V: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i2[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j2[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g2[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h2[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e2[d], 3)) + ' $ \pm $ ' + str(
                    round(f2[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b2[d], 3)) + ' $ \pm $ ' + str(
                    round(c2[d], 3)))

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['SIM ALL POL'].to_numpy()
            c3 = r_points['SIM ALL UNC'].to_numpy()
            d3 = r_points['SIM ALL DIFF'].to_numpy()
            e3 = r_points['POL OBS'].to_numpy()
            f3 = r_points['POL OBS error'].to_numpy()
            g3 = r_points['THETA MOON'].to_numpy()
            h3 = r_points['PHI MOON'].to_numpy()
            i3 = r_points['THETA FIELD'].to_numpy()
            j3 = r_points['PHI FIELD'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3, g3, h3, i3, j3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], \
                                                     np.asarray(d3)[w], np.asarray(e3)[w], np.asarray(f3)[w], \
                                                     np.asarray(g3)[w], np.asarray(h3)[w], np.asarray(i3)[w], \
                                                     np.asarray(j3)[w]

            label3 = []
            for d in range(0, len(a3)):
                label3.append('Field observed in band R: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i3[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j3[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g3[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h3[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e3[d], 3)) + ' $ \pm $ ' + str(
                    round(f3[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b3[d], 3)) + ' $ \pm $ ' + str(
                    round(c3[d], 3)))

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['SIM ALL POL'].to_numpy()
            c4 = i_points['SIM ALL UNC'].to_numpy()
            d4 = i_points['SIM ALL DIFF'].to_numpy()
            e4 = i_points['POL OBS'].to_numpy()
            f4 = i_points['POL OBS error'].to_numpy()
            g4 = i_points['THETA MOON'].to_numpy()
            h4 = i_points['PHI MOON'].to_numpy()
            i4 = i_points['THETA FIELD'].to_numpy()
            j4 = i_points['PHI FIELD'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4, g4, h4, i4, j4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], \
                                                     np.asarray(d4)[w], np.asarray(e4)[w], np.asarray(f4)[w], \
                                                     np.asarray(g4)[w], np.asarray(h4)[w], np.asarray(i4)[w], \
                                                     np.asarray(j4)[w]

            label4 = []
            for d in range(0, len(a4)):
                label4.append('Field observed in band I: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i4[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j4[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g4[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h4[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e4[d], 3)) + ' $ \pm $ ' + str(
                    round(f4[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b4[d], 3)) + ' $ \pm $ ' + str(
                    round(c4[d], 3)))

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            band1 = plt.scatter(a1, e1, color='none')
            mplcursors.cursor(band1, hover=True).connect("add", lambda sel: sel.annotation.set_text(label1[sel.index]))

            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            band2 = plt.scatter(a2, e2, color='none')
            mplcursors.cursor(band2, hover=True).connect("add", lambda sel: sel.annotation.set_text(label2[sel.index]))

            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            band3 = plt.scatter(a3, e3, color='none')
            mplcursors.cursor(band3, hover=True).connect("add", lambda sel: sel.annotation.set_text(label3[sel.index]))

            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')
            band4 = plt.scatter(a4, e4, color='none')
            mplcursors.cursor(band4, hover=True).connect("add", lambda sel: sel.annotation.set_text(label4[sel.index]))

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            k1 = np.add(b2, c2)
            k2 = np.subtract(b2, c2)
            plt.fill_between(a2, k2, k1, where=(k2 < k1), interpolate=True, color='beige')

            l1 = np.add(b3, c3)
            l2 = np.subtract(b3, c3)
            plt.fill_between(a3, l2, l1, where=(l2 < l1), interpolate=True, color='mistyrose')

            m1 = np.add(b4, c4)
            m2 = np.subtract(b4, c4)
            plt.fill_between(a4, m2, m1, where=(m2 < m1), interpolate=True, color='antiquewhite')

            plt.ylim(0, 0.8)
            plt.ylabel('Polarization')
            label_text = 'fit parameters: ' + ' $k_{1}$ = ' + str(
                round(result_simple.params['k'].value, 3)) + '$\pm$' + str(
                round(result_simple.params['k'].stderr, 3)) + ',   $k_{2}$ = ' + str(
                round(result_simple.params['d'].value, 3)) + '$\pm$' + str(
                round(result_simple.params['d'].stderr, 3)) + ',   $N$ = ' + str(
                round(result_simple.params['N'].value, 3)) + '$\pm$' + str(
                round(result_simple.params['N'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                round(result_simple.chisqr, 10)) + ',  reduced chi-square: ' + str(
                round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                round(result_simple.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            leg_text = 'fit with all corrections...\n \n$c_{B}$= ' + str(
                round(B_par[0], 3)) + '\n$c_{V}$= ' + str(round(V_par[0], 3)) + '\n$c_{R}$= ' + str(
                round(R_par[0], 3)) + '\n$c_{I}$= ' + str(round(I_par[0], 3))
            plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

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
            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_simple_all_multi_' + meto + '.png', bbox_inches='tight')

            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
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
                b = points['SIM ALL POL'].to_numpy()
                c = points['SIM ALL UNC'].to_numpy()
                d = points['SIM ALL DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                barra.update(10 / len(band))

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                band = plt.scatter(a, e, color='none')
                mplcursors.cursor(band, hover=True).connect("add",
                                                            lambda sel: sel.annotation.set_text(label2[sel.index]))

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylim(0, 0.8)
                plt.ylabel('Polarization')
                label_text = 'fit parameters: ' + ' $k_{1}$ = ' + str(
                    round(result_simple.params['k'].value, 3)) + '$\pm$' + str(
                    round(result_simple.params['k'].stderr, 3)) + ',   $k_{2}$ = ' + str(
                    round(result_simple.params['d'].value, 3)) + '$\pm$' + str(
                    round(result_simple.params['d'].stderr, 3)) + ',   $N$ = ' + str(
                    round(result_simple.params['N'].value, 3)) + '$\pm$' + str(
                    round(result_simple.params['N'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                    round(result_simple.chisqr, 10)) + ',  reduced chi-square: ' + str(
                    round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                    round(result_simple.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                leg_text = 'fit with all corrections...\n \n$c_{B}$= ' + str(
                    round(B_par[0], 3)) + '\n$c_{V}$= ' + str(round(V_par[0], 3)) + '\n$c_{R}$= ' + str(
                    round(R_par[0], 3)) + '\n$c_{I}$= ' + str(round(I_par[0], 3))
                plt.annotate(leg_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(1.05, 0.1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
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
                b = points['SIM ALL POL'].to_numpy()
                c = points['SIM ALL UNC'].to_numpy()
                d = points['SIM ALL DIFF'].to_numpy()
                e = points['POL OBS'].to_numpy()
                f = points['POL OBS error'].to_numpy()
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_simple_all_ray_' + meto + '.png', bbox_inches='tight')
            plt.close()
            TXT.close()

    # -------------------------------------------------------------------------------------------

    if command == 'ALL' or command == 'wave aop fit':
        model = lmfit.Model(func_wav_AOP)
        model.set_param_hint('n')
        p = model.make_params(n=np.random.rand())
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee')
        result = model.fit(data=AOP, params=p, allvars=[C1field, C2field, C1lua, C2lua, BAND, AOP_BAND],
                           weights=errAOP,
                           method=meto)

        result_emcee_hor = [result.params['n'].value]
        result_emcee_hor = np.asarray(result_emcee_hor)
        Rpar = result_emcee_hor

        txname = 'REPORT_' + LABEL + '_' + LAB + '_regular_aop_ray_' + meto + '.txt'
        model_name = 'MODEL_' + LABEL + '_' + LAB + '_regular_aop_ray_' + meto + '.sav'

        lmfit.model.save_modelresult(result, model_name)
        TXT = open(txname, "w+")

        model_fit_report = result.fit_report()
        TXT.write('***  Fit aop  *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        y1 = func_wav_AOP([C1field, C2field, C1lua, C2lua, BAND, AOP_BAND], *result_emcee_hor)
        fit_observations_resume.insert(coluna, 'REG AOP', y1)
        coluna += 1

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
        fit_observations_resume.insert(coluna, 'REG AOP UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])
        fit_observations_resume.insert(coluna, 'REG AOP DIFF', diff)
        coluna += 1

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
            g1 = b_points['THETA MOON'].to_numpy()
            h1 = b_points['PHI MOON'].to_numpy()
            i1 = b_points['THETA FIELD'].to_numpy()
            j1 = b_points['PHI FIELD'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1, g1, h1, i1, j1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], \
                                                     np.asarray(d1)[w], np.asarray(e1)[w], np.asarray(f1)[w], \
                                                     np.asarray(g1)[w], np.asarray(h1)[w], np.asarray(i1)[w], \
                                                     np.asarray(j1)[w]

            label1 = []
            for d in range(0, len(a1)):
                label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i1[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(round(j1[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g1[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h1[d], 2)) + ';\n$AoP_{OBS}$ = ' + str(
                    round(e1[d], 3)) + ' $ \pm $ ' + str(round(f1[d], 3)) + ';\n$AoP_{SIM}$ = ' + str(
                    round(b1[d], 3)) + ' $ \pm $ ' + str(round(c1[d], 3)))

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['WAV AOP'].to_numpy()
            c2 = v_points['WAV AOP UNC'].to_numpy()
            d2 = v_points['WAV AOP DIFF'].to_numpy()
            e2 = v_points['AOP'].to_numpy()
            f2 = v_points['AOP error'].to_numpy()
            g2 = v_points['THETA MOON'].to_numpy()
            h2 = v_points['PHI MOON'].to_numpy()
            i2 = v_points['THETA FIELD'].to_numpy()
            j2 = v_points['PHI FIELD'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2, g2, h2, i2, j2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], \
                                                     np.asarray(d2)[w], np.asarray(e2)[w], np.asarray(f2)[w], \
                                                     np.asarray(g2)[w], np.asarray(h2)[w], np.asarray(i2)[w], \
                                                     np.asarray(j2)[w]

            label2 = []
            for d in range(0, len(a2)):
                label2.append('Field observed in band V: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i2[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j2[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g2[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h2[d], 2)) + ';\n$AoP_{OBS}$ = ' + str(round(e2[d], 3)) + ' $ \pm $ ' + str(
                    round(f2[d], 3)) + ';\n$AoP_{SIM}$ = ' + str(round(b2[d], 3)) + ' $ \pm $ ' + str(
                    round(c2[d], 3)))

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['WAV AOP'].to_numpy()
            c3 = r_points['WAV AOP UNC'].to_numpy()
            d3 = r_points['WAV AOP DIFF'].to_numpy()
            e3 = r_points['AOP'].to_numpy()
            f3 = r_points['AOP error'].to_numpy()
            g3 = r_points['THETA MOON'].to_numpy()
            h3 = r_points['PHI MOON'].to_numpy()
            i3 = r_points['THETA FIELD'].to_numpy()
            j3 = r_points['PHI FIELD'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3, g3, h3, i3, j3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], \
                                                     np.asarray(d3)[w], np.asarray(e3)[w], np.asarray(f3)[w], \
                                                     np.asarray(g3)[w], np.asarray(h3)[w], np.asarray(i3)[w], \
                                                     np.asarray(j3)[w]

            label3 = []
            for d in range(0, len(a3)):
                label3.append('Field observed in band R: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i3[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j3[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g3[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h3[d], 2)) + ';\n$AoP_{OBS}$ = ' + str(round(e3[d], 3)) + ' $ \pm $ ' + str(
                    round(f3[d], 3)) + ';\n$AoP_{SIM}$ = ' + str(round(b3[d], 3)) + ' $ \pm $ ' + str(
                    round(c3[d], 3)))

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['WAV AOP'].to_numpy()
            c4 = i_points['WAV AOP UNC'].to_numpy()
            d4 = i_points['WAV AOP DIFF'].to_numpy()
            e4 = i_points['AOP'].to_numpy()
            f4 = i_points['AOP error'].to_numpy()
            g4 = i_points['THETA MOON'].to_numpy()
            h4 = i_points['PHI MOON'].to_numpy()
            i4 = i_points['THETA FIELD'].to_numpy()
            j4 = i_points['PHI FIELD'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4, g4, h4, i4, j4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], \
                                                     np.asarray(d4)[w], np.asarray(e4)[w], np.asarray(f4)[w], \
                                                     np.asarray(g4)[w], np.asarray(h4)[w], np.asarray(i4)[w], \
                                                     np.asarray(j4)[w]

            label4 = []
            for d in range(0, len(a4)):
                label4.append('Field observed in band I: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i4[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j4[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g4[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h4[d], 2)) + ';\n$AoP_{OBS}$ = ' + str(round(e4[d], 3)) + ' $ \pm $ ' + str(
                    round(f4[d], 3)) + ';\n$AoP_{SIM}$ = ' + str(round(b4[d], 3)) + ' $ \pm $ ' + str(
                    round(c4[d], 3)))

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            band1 = plt.scatter(a1, e1, color='none')
            mplcursors.cursor(band1, hover=True).connect("add", lambda sel: sel.annotation.set_text(label1[sel.index]))

            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            band2 = plt.scatter(a2, e2, color='none')
            mplcursors.cursor(band2, hover=True).connect("add", lambda sel: sel.annotation.set_text(label2[sel.index]))

            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            band3 = plt.scatter(a3, e3, color='none')
            mplcursors.cursor(band3, hover=True).connect("add", lambda sel: sel.annotation.set_text(label3[sel.index]))

            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')
            band4 = plt.scatter(a4, e4, color='none')
            mplcursors.cursor(band4, hover=True).connect("add", lambda sel: sel.annotation.set_text(label4[sel.index]))

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            k1 = np.add(b2, c2)
            k2 = np.subtract(b2, c2)
            plt.fill_between(a2, k2, k1, where=(k2 < k1), interpolate=True, color='beige')

            l1 = np.add(b3, c3)
            l2 = np.subtract(b3, c3)
            plt.fill_between(a3, l2, l1, where=(l2 < l1), interpolate=True, color='mistyrose')

            m1 = np.add(b4, c4)
            m2 = np.subtract(b4, c4)
            plt.fill_between(a4, m2, m1, where=(m2 < m1), interpolate=True, color='antiquewhite')

            plt.ylabel('Polarization Angle')
            label_text = 'fit parameters: ' + ' $c$ = ' + str(
                round(result.params['n'].value, 3)) + '$\pm$' + str(
                round(result.params['n'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                round(result.chisqr, 10)) + ',   reduced chi-square: ' + str(
                round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                round(result.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

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
            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_wave_aop_ray_' + meto + '.png', bbox_inches='tight')

            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
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
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$AoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$AoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                barra.update(10 / len(band))

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                band = plt.scatter(a, e, color='none')
                mplcursors.cursor(band, hover=True).connect("add",
                                                            lambda sel: sel.annotation.set_text(label2[sel.index]))

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                label_text = 'fit parameters: ' + ' $c$ = ' + str(
                    round(result.params['n'].value, 3)) + '$\pm$' + str(
                    round(result.params['n'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                    round(result.chisqr, 10)) + ',   reduced chi-square: ' + str(
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
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$AoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$AoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_wave_aop_ray_' + meto + '.png', bbox_inches='tight')
            plt.close()

        # ___________________________________________________________________________________________

        model = lmfit.Model(func_simple_wav_AOP, independent_vars=['phi_obs', 'theta_obs', 'phi_lua', 'theta_lua', 'banda', 'par'])
        # func_simple_wav_AOP(phi_obs, theta_obs, phi_lua, theta_lua, banda, par, n)
        model.set_param_hint('n')
        p = model.make_params(n=np.random.rand())
        result_simple = model.fit(data=AOP, params=p, theta_obs=C1field, phi_obs=C2field, theta_lua=C1lua, phi_lua=C2lua, banda=BAND, par=AOP_BAND,
                                  weights=errAOP, method=meto)

        result_simple_hor = [result_simple.params['n'].value]
        result_simple_hor = np.asarray(result_simple_hor)

        model_fit_report = result_simple.fit_report()
        TXT.write('***  Fit: aop in a simplier way *** \n \n')
        TXT.write('Independent variables: \n')
        TXT.write(str(model.independent_vars))
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        model_name = 'MODEL_' + LABEL + '_' + LAB + '_simple_wave_aop_ray_' + meto + '.sav'
        lmfit.model.save_modelresult(result_simple, model_name)

        y1 = func_simple_wav_AOP(theta_obs=C1field, phi_obs=C2field, theta_lua=C1lua, phi_lua=C2lua, banda=BAND, par=AOP_BAND, n=result_simple_hor[0])
        fit_observations_resume.insert(coluna, 'SIM WAV AOP', y1)
        coluna += 1

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))
        fit_observations_resume.insert(coluna, 'SIM WAV AOP UNC', rsd)
        coluna += 1

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])
        fit_observations_resume.insert(coluna, 'SIM WAV AOP DIFF', diff)
        coluna += 1

        if band is None or band is BAN:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            b_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 437]
            a1 = b_points['GAMMA'].to_numpy()
            b1 = b_points['SIM WAV AOP'].to_numpy()
            c1 = b_points['SIM WAV AOP UNC'].to_numpy()
            d1 = b_points['SIM WAV AOP DIFF'].to_numpy()
            e1 = b_points['AOP'].to_numpy()
            f1 = b_points['AOP error'].to_numpy()
            g1 = b_points['THETA MOON'].to_numpy()
            h1 = b_points['PHI MOON'].to_numpy()
            i1 = b_points['THETA FIELD'].to_numpy()
            j1 = b_points['PHI FIELD'].to_numpy()
            w = np.argsort(a1)
            a1, b1, c1, d1, e1, f1, g1, h1, i1, j1 = np.asarray(a1)[w], np.asarray(b1)[w], np.asarray(c1)[w], \
                                                     np.asarray(d1)[w], np.asarray(e1)[w], np.asarray(f1)[w], \
                                                     np.asarray(g1)[w], np.asarray(h1)[w], np.asarray(i1)[w], \
                                                     np.asarray(j1)[w]

            label1 = []
            for d in range(0, len(a1)):
                label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i1[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(round(j1[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g1[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h1[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                    round(e1[d], 3)) + ' $ \pm $ ' + str(round(f1[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                    round(b1[d], 3)) + ' $ \pm $ ' + str(round(c1[d], 3)))

            v_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 555]
            a2 = v_points['GAMMA'].to_numpy()
            b2 = v_points['SIM WAV AOP'].to_numpy()
            c2 = v_points['SIM WAV AOP UNC'].to_numpy()
            d2 = v_points['SIM WAV AOP DIFF'].to_numpy()
            e2 = v_points['AOP'].to_numpy()
            f2 = v_points['AOP error'].to_numpy()
            g2 = v_points['THETA MOON'].to_numpy()
            h2 = v_points['PHI MOON'].to_numpy()
            i2 = v_points['THETA FIELD'].to_numpy()
            j2 = v_points['PHI FIELD'].to_numpy()
            w = np.argsort(a2)
            a2, b2, c2, d2, e2, f2, g2, h2, i2, j2 = np.asarray(a2)[w], np.asarray(b2)[w], np.asarray(c2)[w], \
                                                     np.asarray(d2)[w], np.asarray(e2)[w], np.asarray(f2)[w], \
                                                     np.asarray(g2)[w], np.asarray(h2)[w], np.asarray(i2)[w], \
                                                     np.asarray(j2)[w]

            label2 = []
            for d in range(0, len(a2)):
                label2.append('Field observed in band V: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i2[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j2[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g2[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h2[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e2[d], 3)) + ' $ \pm $ ' + str(
                    round(f2[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b2[d], 3)) + ' $ \pm $ ' + str(
                    round(c2[d], 3)))

            r_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 655]
            a3 = r_points['GAMMA'].to_numpy()
            b3 = r_points['SIM WAV AOP'].to_numpy()
            c3 = r_points['SIM WAV AOP UNC'].to_numpy()
            d3 = r_points['SIM WAV AOP DIFF'].to_numpy()
            e3 = r_points['AOP'].to_numpy()
            f3 = r_points['AOP error'].to_numpy()
            g3 = r_points['THETA MOON'].to_numpy()
            h3 = r_points['PHI MOON'].to_numpy()
            i3 = r_points['THETA FIELD'].to_numpy()
            j3 = r_points['PHI FIELD'].to_numpy()
            w = np.argsort(a3)
            a3, b3, c3, d3, e3, f3, g3, h3, i3, j3 = np.asarray(a3)[w], np.asarray(b3)[w], np.asarray(c3)[w], \
                                                     np.asarray(d3)[w], np.asarray(e3)[w], np.asarray(f3)[w], \
                                                     np.asarray(g3)[w], np.asarray(h3)[w], np.asarray(i3)[w], \
                                                     np.asarray(j3)[w]

            label3 = []
            for d in range(0, len(a3)):
                label3.append('Field observed in band R: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i3[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j3[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g3[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h3[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e3[d], 3)) + ' $ \pm $ ' + str(
                    round(f3[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b3[d], 3)) + ' $ \pm $ ' + str(
                    round(c3[d], 3)))

            i_points = fit_observations_resume[fit_observations_resume['WAVELENGTH'] == 768]
            a4 = i_points['GAMMA'].to_numpy()
            b4 = i_points['SIM WAV AOP'].to_numpy()
            c4 = i_points['SIM WAV AOP UNC'].to_numpy()
            d4 = i_points['SIM WAV AOP DIFF'].to_numpy()
            e4 = i_points['AOP'].to_numpy()
            f4 = i_points['AOP error'].to_numpy()
            g4 = i_points['THETA MOON'].to_numpy()
            h4 = i_points['PHI MOON'].to_numpy()
            i4 = i_points['THETA FIELD'].to_numpy()
            j4 = i_points['PHI FIELD'].to_numpy()
            w = np.argsort(a4)
            a4, b4, c4, d4, e4, f4, g4, h4, i4, j4 = np.asarray(a4)[w], np.asarray(b4)[w], np.asarray(c4)[w], \
                                                     np.asarray(d4)[w], np.asarray(e4)[w], np.asarray(f4)[w], \
                                                     np.asarray(g4)[w], np.asarray(h4)[w], np.asarray(i4)[w], \
                                                     np.asarray(j4)[w]

            label4 = []
            for d in range(0, len(a4)):
                label4.append('Field observed in band I: \n $\u03B8_{FIELD}$ = ' + str(
                    round(i4[d], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                    round(j4[d], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                    round(g4[d], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(
                    round(h4[d], 2)) + ';\n$DoP_{OBS}$ = ' + str(round(e4[d], 3)) + ' $ \pm $ ' + str(
                    round(f4[d], 3)) + ';\n$DoP_{SIM}$ = ' + str(round(b4[d], 3)) + ' $ \pm $ ' + str(
                    round(c4[d], 3)))

            plt.plot(a1, b1, '-', color='cornflowerblue', markersize=2, label='fit results B band')
            plt.errorbar(a1, e1, yerr=f1, ms=2.0, fmt='o', color='blue', label='data B band')
            band1 = plt.scatter(a1, e1, color='none')
            mplcursors.cursor(band1, hover=True).connect("add", lambda sel: sel.annotation.set_text(label1[sel.index]))

            plt.plot(a2, b2, '-', color='mediumseagreen', markersize=2, label='fit results V band')
            plt.errorbar(a2, e2, yerr=f2, ms=2.0, fmt='o', color='green', label='data V band')
            band2 = plt.scatter(a2, e2, color='none')
            mplcursors.cursor(band2, hover=True).connect("add", lambda sel: sel.annotation.set_text(label2[sel.index]))

            plt.plot(a3, b3, '-', color='indianred', markersize=2, label='fit results R band')
            plt.errorbar(a3, e3, yerr=f3, ms=2.0, fmt='o', color='red', label='data R band')
            band3 = plt.scatter(a3, e3, color='none')
            mplcursors.cursor(band3, hover=True).connect("add", lambda sel: sel.annotation.set_text(label3[sel.index]))

            plt.plot(a4, b4, '-', color='orange', markersize=2, label='fit results I band')
            plt.errorbar(a4, e4, yerr=f4, ms=2.0, fmt='o', color='darkorange', label='data I band')
            band4 = plt.scatter(a4, e4, color='none')
            mplcursors.cursor(band4, hover=True).connect("add", lambda sel: sel.annotation.set_text(label4[sel.index]))

            g1 = np.add(b1, c1)
            g2 = np.subtract(b1, c1)
            plt.fill_between(a1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

            k1 = np.add(b2, c2)
            k2 = np.subtract(b2, c2)
            plt.fill_between(a2, k2, k1, where=(k2 < k1), interpolate=True, color='beige')

            l1 = np.add(b3, c3)
            l2 = np.subtract(b3, c3)
            plt.fill_between(a3, l2, l1, where=(l2 < l1), interpolate=True, color='mistyrose')

            m1 = np.add(b4, c4)
            m2 = np.subtract(b4, c4)
            plt.fill_between(a4, m2, m1, where=(m2 < m1), interpolate=True, color='antiquewhite')

            plt.ylabel('Polarization Angle')
            label_text = 'fit parameters:  ' + ' $c$ = ' + str(
                round(result.params['n'].value, 3)) + '$\pm$' + str(
                round(result.params['n'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                round(result_simple.chisqr, 10)) + ',   reduced chi-square: ' + str(
                round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                round(result_simple.bic, 2))
            plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                         textcoords='axes fraction',
                         horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

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
            plt.savefig('IMAGE_' + LABEL + '_' + LAB + '_simple_wav_aop_ray_' + meto + '.png',
                        bbox_inches='tight')

            plt.pause(2)
            plt.close()

        else:
            fig_x = plt.figure(figsize=(10, 5))
            fig_x.add_axes((.1, .3, .6, .6))

            for item in band:
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
                b = points['SIM WAV AOP'].to_numpy()
                c = points['SIM WAV AOP UNC'].to_numpy()
                d = points['SIM WAV AOP DIFF'].to_numpy()
                e = points['AOP'].to_numpy()
                f = points['AOP error'].to_numpy()
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$AoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$AoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                barra.update(10 / len(band))

                plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
                plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

                band = plt.scatter(a, e, color='none')
                mplcursors.cursor(band, hover=True).connect("add",
                                                            lambda sel: sel.annotation.set_text(label2[sel.index]))

                g1 = np.add(b, c)
                g2 = np.subtract(b, c)
                plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

                plt.ylabel('Polarization Angle')
                label_text = 'fit parameters:  ' + ' $c$ = ' + str(
                    round(result.params['n'].value, 3)) + '$\pm$' + str(
                    round(result.params['n'].stderr, 3)) + '\n' + 'chi-square: ' + str(
                    round(result_simple.chisqr, 10)) + ',   reduced chi-square: ' + str(
                    round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                    round(result_simple.bic, 2))
                plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                             textcoords='axes fraction',
                             horizontalalignment='left', verticalalignment='center',
                             bbox=dict(boxstyle="round", fc="w"))
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

                plt.pause(2)

            fig_x.add_axes((.1, .1, .6, .2))

            for item in band:
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
                b = points['SIM WAV AOP'].to_numpy()
                c = points['SIM WAV AOP UNC'].to_numpy()
                d = points['SIM WAV AOP DIFF'].to_numpy()
                e = points['AOP'].to_numpy()
                f = points['AOP error'].to_numpy()
                g = points['THETA MOON'].to_numpy()
                h = points['PHI MOON'].to_numpy()
                i = points['THETA FIELD'].to_numpy()
                j = points['PHI FIELD'].to_numpy()
                w = np.argsort(a)
                a, b, c, d, e, f, g, h, i, j = np.asarray(a)[w], np.asarray(b)[w], np.asarray(c)[w], \
                                               np.asarray(d)[w], np.asarray(e)[w], np.asarray(f)[w], \
                                               np.asarray(g)[w], np.asarray(h)[w], np.asarray(i)[w], \
                                               np.asarray(j)[w]

                label1 = []
                for z in range(0, len(a)):
                    label1.append('Field observed in band B: \n $\u03B8_{FIELD}$ = ' + str(
                        round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                        round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                        round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$AoP_{OBS}$ = ' + str(
                        round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$AoP_{SIM}$ = ' + str(
                        round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

                plt.errorbar(a, d, yerr=f, ms=2.0, fmt='o', color=cor, label='diff ' + lab_wave + ' band')
                plt.plot(a, c, '-', color=cor_line, markersize=2, label='uncertanties fit')

                plt.xlabel('Scattering Angle (degrees)')
                plt.ylabel('Residual data')
                plt.grid(True)

                plt.pause(2)

            plt.savefig('IMAGE_' + LABEL + '_' +LAB + '_simple_wav_aop_ray_' + meto + '.png',
                        bbox_inches='tight')
            plt.close()
            TXT.close()

    barra.close()

    fit_observations_resume.to_csv('TABLE_results_POLARIZATION_Stokes.csv')
    # result_data.to_excel('TABLE.xlsx', sheet_name='results_POLARIZATION_Stokes')

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
                    map_observations('B', condition, B_par, Rpar, P1=plot, P2=answer_3)

                    print('*** Plotting V band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('V', condition, V_par, Rpar, P1=plot, P2=answer_3)

                    print('*** Plotting R band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('R', condition, R_par, Rpar, P1=plot, P2=answer_3)

                    print('*** Plotting I band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('I', condition, I_par, Rpar, P1=plot, P2=answer_3)

                if answer_3 == 'Mix':
                    print('*** Plotting B band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('B', condition, B_par_mix, Rpar, P1=plot, P2=answer_3)

                    print('*** Plotting V band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('V', condition, V_par_mix, Rpar, P1=plot, P2=answer_3)

                    print('*** Plotting R band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('R', condition, R_par_mix, Rpar, P1=plot, P2=answer_3)

                    print('*** Plotting I band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('I', condition, I_par_mix, Rpar, P1=plot, P2=answer_3)

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

                if answer_3 == 'Wave':
                    print('*** Plotting B band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('B', condition, B_par, Rpar, P1=plot, P2=answer_3)

                    print('*** Plotting V band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('V', condition, V_par, Rpar, P1=plot, P2=answer_3)

                    print('*** Plotting R band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('R', condition, R_par, Rpar, P1=plot, P2=answer_3)

                    print('*** Plotting I band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('I', condition, I_par, Rpar, P1=plot, P2=answer_3)

                else:
                    print('Wrong input!!')

            if answer_2 == 'No':
                plot = 'complex'
                if command != 'fit wavelength':
                    print('*** Plotting B band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('B', condition, B_par, Rpar, P1=plot, P2=command)

                    print('*** Plotting V band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('V', condition, V_par, Rpar, P1=plot, P2=command)

                    print('*** Plotting R band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('R', condition, R_par, Rpar, P1=plot, P2=command)

                    print('*** Plotting I band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('I', condition, I_par, Rpar, P1=plot, P2=command)
                if command == 'fit wavelength':
                    print('*** Plotting B band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('B', condition, B_par_mix, Rpar, P1=plot, P2=command)

                    print('*** Plotting V band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('V', condition, V_par_mix, Rpar, P1=plot, P2=command)

                    print('*** Plotting R band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('R', condition, R_par_mix, Rpar, P1=plot, P2=command)

                    print('*** Plotting I band ***')
                    # band, condition, Par1, Par2, P1='individual', P2='Regular'
                    map_observations('I', condition, I_par_mix, Rpar, P1=plot, P2=command)

            else:
                print('Wrong input!!')

            print('Processo concluído')

        if answer == 'No':
            print('Processo concluído')

    return Rpar, fit_observations_resume



