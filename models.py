import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
import moon_functions


def func_gamma(theta_obs, phi_obs, theta_lua, phi_lua):
    gamma = np.arccos(
        np.sin(theta_lua) * np.sin(theta_obs) * np.cos(phi_obs - phi_lua) + np.cos(theta_lua) * np.cos(theta_obs))

    return gamma


def DoP_simple_map(temp, dh, n=0, source='Moon', ref='polar'):
    D = 1
    R = 1
    tempo = Time(temp)
    tempo_start = tempo.ymdhms
    delta = dh * u.hour

    for a in range(0, n):

        observation = moon_functions.Moon(tempo)
        observation.get_parameters()

        SOL, obs_sol = observation.true_sun()

        k = observation.phase

        alt_lua = observation.alt
        az_lua = observation.az

        alt_sol = observation.my_sun.alt
        az_sol = observation.my_sun.az

        if -90 <= alt_lua <= 90 and 0 <= az_lua <= 360:

            t_sol = observation.my_sun.theta
            phi_sol = observation.my_sun.phi

            t_lua = observation.theta
            phi_lua = observation.phi

            dop = np.zeros((100, 400))

            x = np.zeros((100, 400))
            y = np.zeros((100, 400))

            Alt = np.zeros((100, 400))
            Az = np.zeros((100, 400))

            i, j = 0, 0

            for Eo in np.linspace(0, np.pi / 2, 100, endpoint=True):
                for Azo in np.linspace(0, 2 * np.pi, 400, endpoint=True):
                    to = np.pi / 2 - Eo
                    phio = Azo

                    gamma_lua = func_gamma(to, phio, t_lua, phi_lua)
                    gamma_sol = func_gamma(to, phio, t_sol, phi_sol)

                    # dop: grau de polarização
                    dop_lua = (D * (np.sin(gamma_lua)) ** 2) / (1 + np.cos(gamma_lua) ** 2) * k
                    dop_sol = (D * (np.sin(gamma_sol)) ** 2) / (1 + np.cos(gamma_sol) ** 2)
                    dop_tot = dop_lua + dop_sol

                    if source == 'Moon':
                        dop[i, j] = dop_lua
                    if source == 'Sun':
                        dop[i, j] = dop_sol
                    if source == 'Total':
                        dop[i, j] = dop_tot

                    x[i, j] = R * np.sin(to) * np.cos(phio)
                    y[i, j] = R * np.sin(to) * np.sin(phio)

                    Alt[i, j] = to * 180 / np.pi
                    Az[i, j] = phio

                    j += 1

                i += 1
                j = 0

            if ref == 'polar':

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

                az_label_offset = 0 * u.deg
                theta_labels = []
                for chunk in range(0, 7):
                    label_angle = (az_label_offset * (1 / u.deg)) + (chunk * 45)
                    while label_angle >= 360:
                        label_angle -= 360
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

                if source == 'Moon':
                    fig = plt.figure(figsize=(7, 5))
                    plt.clf()
                    ax = fig.gca(projection='polar')
                    ax.set_theta_zero_location('N')
                    ax.set_rlim(1, 91)
                    plt.pcolormesh(Az, Alt, dop, vmax=1, vmin=0)
                    if 0 <= t_lua <= np.pi / 2:
                        ax.plot(phi_lua, t_lua * 180 / np.pi, 'o', color='white')
                    plt.title(str(tempo), loc='right', fontsize=9, color='black')
                    ax.grid(True, which='major')
                    ax.set_rgrids(range(1, 106, 15), r_labels, angle=-45)
                    ax.set_thetagrids(range(0, 360, 45), theta_labels)
                    plt.colorbar(pad=0.1)
                    nome = 'MAP_dop_rayleigh_polar_Moon_' + str(tempo_start) + '_frame_' + str(a) + '.png'
                    plt.savefig(nome)
                    ax.figure.canvas.draw()
                    plt.pause(0.2)
                    plt.close()

                if source == 'Sun':
                    fig = plt.figure(figsize=(7, 5))
                    plt.clf()
                    ax = fig.gca(projection='polar')
                    ax.set_theta_zero_location('N')
                    ax.set_rlim(1, 91)
                    plt.pcolormesh(Az, Alt, dop, vmax=1, vmin=0)
                    if 0 <= t_sol <= np.pi / 2:
                        ax.plot(phi_sol, t_sol * 180 / np.pi, 'o', color='grey')
                    plt.title(str(tempo), loc='right', fontsize=9, color='black')
                    ax.grid(True, which='major')
                    ax.set_rgrids(range(1, 106, 15), r_labels, angle=-45)
                    ax.set_thetagrids(range(0, 360, 45), theta_labels)
                    plt.colorbar(pad=0.1)
                    nome = 'MAP_dop_rayleigh_polar_Sun_' + str(tempo_start) + '_frame_' + str(a) + '.png'
                    plt.savefig(nome)
                    ax.figure.canvas.draw()
                    plt.pause(0.2)
                    plt.close()

                if source == 'Total':
                    fig = plt.figure(figsize=(7, 5))
                    plt.clf()
                    ax = fig.gca(projection='polar')
                    ax.set_theta_zero_location('N')
                    ax.set_rlim(1, 91)
                    plt.pcolormesh(Az, Alt, dop)
                    if 0 <= t_sol <= np.pi / 2:
                        ax.plot(phi_sol, t_sol * 180 / np.pi, 'o', color='grey')
                    if 0 <= t_lua <= np.pi / 2:
                        ax.plot(phi_lua, t_lua * 180 / np.pi, 'o', color='white')
                    plt.title(str(tempo), loc='right', fontsize=9, color='black')
                    ax.grid(True, which='major')
                    ax.set_rgrids(range(1, 106, 15), r_labels, angle=-45)
                    ax.set_thetagrids(range(0, 360, 45), theta_labels)
                    plt.colorbar(pad=0.1)
                    nome = 'MAP_dop_rayleigh_polar_Total_' + str(tempo_start) + '_frame_' + str(a) + '.png'
                    plt.savefig(nome)
                    ax.figure.canvas.draw()
                    plt.pause(0.2)
                    plt.close()

            if ref == 'XY':
                if source == 'Moon':
                    x_point = observation.x
                    y_point = observation.y

                    fig = plt.figure(figsize=(7, 5))
                    plt.clf()
                    plt.pcolormesh(x, y, dop, vmax=1, vmin=0)
                    if 0 <= alt_lua <= 90:
                        plt.plot(x_point, y_point, 'o', color='white')
                    plt.text(0, 1.2, str(tempo), fontsize=11)
                    plt.text(0.3, 1.1, 'fase da Lua: %.3f ' % round(k, 3), fontsize=9)
                    plt.text(0, 1.1, 'E', fontsize=14)
                    plt.text(0, -1.2, 'O', fontsize=14)
                    plt.text(1.2, 0, 'N', fontsize=14)
                    plt.text(-1.3, 0, 'S', fontsize=14)
                    plt.axis('equal')
                    plt.axis('off')
                    plt.colorbar(pad=0.1)
                    nome = 'MAP_dop_rayleigh_xy_Moon_' + str(tempo_start) + '_frame_' + str(a) + '.png'
                    plt.savefig(nome)
                    plt.pause(0.2)
                    plt.close()

                if source == 'Sun':
                    x_point = observation.my_sun.x
                    y_point = observation.my_sun.y

                    fig = plt.figure(figsize=(7, 5))
                    plt.clf()
                    plt.pcolormesh(x, y, dop, vmax=1, vmin=0)
                    if 0 <= alt_sol <= 90:
                        plt.plot(x_point, y_point, 'o', color='grey')
                    plt.text(0, 1.2, str(tempo), fontsize=9)
                    plt.text(0, 1.1, 'E', fontsize=14)
                    plt.text(0, -1.2, 'O', fontsize=14)
                    plt.text(1.2, 0, 'N', fontsize=14)
                    plt.text(-1.3, 0, 'S', fontsize=14)
                    plt.axis('equal')
                    plt.axis('off')
                    plt.colorbar(pad=0.1)
                    nome = 'MAP_dop_rayleigh_xy_Sun_' + str(tempo_start) + '_frame_' + str(a) + '.png'
                    plt.savefig(nome)
                    plt.pause(0.2)
                    plt.close()

                if source == 'Total':
                    xl = observation.x
                    yl = observation.y

                    xs = observation.my_sun.x
                    ys = observation.my_sun.y

                    fig = plt.figure(figsize=(7, 5))
                    plt.clf()
                    plt.pcolormesh(x, y, dop)
                    if 0 <= alt_lua <= 90:
                        plt.plot(xl, yl, 'o', color='white')
                    if 0 <= alt_sol <= 90:
                        plt.plot(xs, ys, 'o', color='grey')
                    plt.text(0, 1.2, str(tempo), fontsize=9)
                    plt.text(0.3, 1.1, 'fase da Lua: %.3f ' % round(k, 3), fontsize=11)
                    plt.text(0, 1.1, 'E', fontsize=14)
                    plt.text(0, -1.2, 'O', fontsize=14)
                    plt.text(1.2, 0, 'N', fontsize=14)
                    plt.text(-1.3, 0, 'S', fontsize=14)
                    plt.axis('equal')
                    plt.axis('off')
                    plt.colorbar(pad=0.1)
                    nome = 'MAP_dop_rayleigh_xy_Total_' + str(tempo_start) + '_frame_' + str(a) + '.png'
                    plt.savefig(nome)
                    plt.pause(0.2)
                    plt.close()

        else:
            print('Os parâmetros que inseriu não são adequados')

        tempo += delta


def AoP_simple_map(temp, dh, n=0, source='Moon', ref='polar'):
    R = 1
    tempo = Time(temp)
    tempo_start = tempo.ymdhms
    delta = dh * u.hour

    for a in range(0, n):

        observation = moon_functions.Moon(tempo)
        observation.get_parameters()

        SOL, obs_sol = observation.true_sun()

        k = observation.phase

        alt_lua = observation.alt
        az_lua = observation.az

        alt_sol = observation.my_sun.alt
        az_sol = observation.my_sun.az

        if -90 <= alt_lua <= 90 and 0 <= az_lua <= 360:

            t_sol = observation.my_sun.theta
            phi_sol = observation.my_sun.phi

            t_lua = observation.theta
            phi_lua = observation.phi

            aop = np.zeros((100, 400))

            x = np.zeros((100, 400))
            y = np.zeros((100, 400))

            Alt = np.zeros((100, 400))
            Az = np.zeros((100, 400))

            i, j = 0, 0

            for Eo in np.linspace(0, np.pi / 2, 100, endpoint=True):
                for Azo in np.linspace(0, 2 * np.pi, 400, endpoint=True):
                    to = np.pi / 2 - Eo
                    phio = Azo

                    if np.sin(phio - phi_lua) * np.sin(t_lua) == 0:
                        aop_lua = np.pi / 2
                    else:
                        aop_lua = np.arctan(
                            (np.sin(to) * np.cos(t_lua) - np.cos(to) * np.cos(phio - phi_lua) * np.sin(t_lua)) / (
                                    np.sin(phio - phi_lua) * np.sin(t_lua)))

                    if np.sin(phio - phi_sol) * np.sin(t_sol) == 0:
                        aop_sol = np.pi / 2
                    else:
                        aop_sol = np.arctan((np.sin(to) * np.cos(t_sol) - np.cos(to) * np.cos(phio - phi_sol) * np.sin(t_sol)) / (np.sin(phio - phi_sol) * np.sin(t_sol)))

                    if source == 'Moon':
                        aop[i, j] = aop_lua * 180 / np.pi
                    if source == 'Sun':
                        aop[i, j] = aop_sol * 180 / np.pi

                    x[i, j] = R * np.sin(to) * np.cos(phio)
                    y[i, j] = R * np.sin(to) * np.sin(phio)

                    Alt[i, j] = to * 180 / np.pi
                    Az[i, j] = phio

                    j += 1

                i += 1
                j = 0

            if ref == 'polar':

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

                az_label_offset = 0 * u.deg
                theta_labels = []
                for chunk in range(0, 7):
                    label_angle = (az_label_offset * (1 / u.deg)) + (chunk * 45)
                    while label_angle >= 360:
                        label_angle -= 360
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

                if source == 'Moon':
                    fig = plt.figure(figsize=(7, 5))
                    plt.clf()
                    ax = fig.gca(projection='polar')
                    ax.set_theta_zero_location('N')
                    ax.set_rlim(1, 91)
                    plt.pcolormesh(Az, Alt, aop, cmap='twilight')
                    if 0 <= t_lua <= np.pi / 2:
                        ax.plot(phi_lua, t_lua * 180 / np.pi, 'o', color='white')
                    plt.title(str(tempo), loc='right', fontsize=9, color='black')
                    ax.grid(True, which='major')
                    ax.set_rgrids(range(1, 106, 15), r_labels, angle=-45)
                    ax.set_thetagrids(range(0, 360, 45), theta_labels)
                    plt.colorbar(pad=0.1)
                    nome = 'MAP_aop_rayleigh_polar_Moon_' + str(tempo_start) + '_' + str(a) + '.png'
                    plt.savefig(nome)
                    ax.figure.canvas.draw()
                    plt.pause(0.2)
                    plt.close()

                if source == 'Sun':
                    fig = plt.figure(figsize=(7, 5))
                    plt.clf()
                    ax = fig.gca(projection='polar')
                    ax.set_theta_zero_location('N')
                    ax.set_rlim(1, 91)
                    plt.pcolormesh(Az, Alt, aop, cmap='twilight')
                    if 0 <= t_sol <= np.pi / 2:
                        ax.plot(phi_sol, t_sol * 180 / np.pi, 'o', color='grey')
                    plt.title(str(tempo), loc='right', fontsize=9, color='black')
                    ax.grid(True, which='major')
                    ax.set_rgrids(range(1, 106, 15), r_labels, angle=-45)
                    ax.set_thetagrids(range(0, 360, 45), theta_labels)
                    plt.colorbar(pad=0.1)
                    nome = 'MAP_aop_rayleigh_polar_Sun_' + str(tempo_start) + '_' + str(a) + '.png'
                    plt.savefig(nome)
                    ax.figure.canvas.draw()
                    plt.pause(0.2)
                    plt.close()

                if source == 'Total':
                    fig = plt.figure(figsize=(7, 5))
                    plt.clf()
                    ax = fig.gca(projection='polar')
                    ax.set_theta_zero_location('N')
                    ax.set_rlim(1, 91)
                    plt.pcolormesh(Az, Alt, aop, cmap='twilight')
                    if 0 <= t_sol <= np.pi / 2:
                        ax.plot(phi_sol, t_sol * 180 / np.pi, 'o', color='grey')
                    if 0 <= t_lua <= np.pi / 2:
                        ax.plot(phi_lua, t_lua * 180 / np.pi, 'o', color='white')
                    plt.title(str(tempo), loc='right', fontsize=9, color='black')
                    ax.grid(True, which='major')
                    ax.set_rgrids(range(1, 106, 15), r_labels, angle=-45)
                    ax.set_thetagrids(range(0, 360, 45), theta_labels)
                    plt.colorbar(pad=0.1)
                    nome = 'MAP_aop_rayleigh_polar_Total_' + str(tempo_start) + '_' + str(a) + '.png'
                    plt.savefig(nome)
                    ax.figure.canvas.draw()
                    plt.pause(0.2)
                    plt.close()

            if ref == 'XY':
                if source == 'Moon':
                    x_point = observation.x
                    y_point = observation.y

                    meridian_x_point = observation.meridian_x
                    meridian_y_point = observation.meridian_y

                    fig = plt.figure(figsize=(7, 5))
                    plt.clf()
                    plt.pcolormesh(x, y, aop, cmap='twilight')
                    if 0 <= alt_lua <= 90:
                        plt.plot(x_point, y_point, 'o', color='white')
                    plt.text(0, 1.2, str(tempo), fontsize=9)
                    plt.text(0.3, 1.1, 'fase da Lua: %.3f ' % round(k, 3), fontsize=11)
                    plt.text(0, 1.1, 'E', fontsize=14)
                    plt.text(0, -1.2, 'O', fontsize=14)
                    plt.text(1.2, 0, 'N', fontsize=14)
                    plt.text(-1.3, 0, 'S', fontsize=14)
                    plt.axis('equal')
                    plt.axis('off')
                    plt.colorbar(pad=0.1)
                    nome = 'MAP_aop_rayleigh_xy_Moon_' + str(tempo_start) + '_' + str(a) + '.png'
                    plt.savefig(nome)
                    plt.pause(0.2)
                    plt.close()

                if source == 'Sun':
                    x_point = observation.my_sun.x
                    y_point = observation.my_sun.y

                    fig = plt.figure(figsize=(7, 5))
                    plt.clf()
                    plt.pcolormesh(x, y, aop, cmap='twilight')
                    if 0 <= alt_sol <= 90:
                        plt.plot(x_point, y_point, 'o', color='grey')
                    plt.text(0, 1.2, str(tempo), fontsize=9)
                    plt.text(0, 1.1, 'E', fontsize=14)
                    plt.text(0, -1.2, 'O', fontsize=14)
                    plt.text(1.2, 0, 'N', fontsize=14)
                    plt.text(-1.3, 0, 'S', fontsize=14)
                    plt.axis('equal')
                    plt.axis('off')
                    plt.colorbar(pad=0.1)
                    nome = 'MAP_aop_rayleigh_xy_Sun_' + str(tempo_start) + '_' + str(a) + '.png'
                    plt.savefig(nome)
                    plt.pause(0.2)
                    plt.close()

                if source == 'Total':
                    xl = observation.x
                    yl = observation.y

                    meridian_x_lua = observation.meridian_x
                    meridian_y_lua = observation.meridian_y

                    xs = observation.my_sun.x
                    ys = observation.my_sun.y

                    meridian_x_sol = observation.my_sun.meridian_x
                    meridian_y_sol = observation.my_sun.meridian_y

                    fig = plt.figure(figsize=(7, 5))
                    plt.clf()
                    plt.pcolormesh(x, y, aop, cmap='twilight')
                    if 0 <= alt_lua <= 90:
                        plt.plot(xl, yl, 'o', color='white')
                    if 0 <= alt_sol <= 90:
                        plt.plot(xs, ys, 'o', color='grey')
                    plt.text(0, 1.2, str(tempo), fontsize=9)
                    plt.text(0.3, 1.1, 'fase da Lua: %.3f ' % round(k, 3), fontsize=11)
                    plt.text(0, 1.1, 'E', fontsize=14)
                    plt.text(0, -1.2, 'O', fontsize=14)
                    plt.text(1.2, 0, 'N', fontsize=14)
                    plt.text(-1.3, 0, 'S', fontsize=14)
                    plt.axis('equal')
                    plt.axis('off')
                    plt.colorbar(pad=0.1)
                    nome = 'MAP_aop_rayleigh_xy_Total_' + str(tempo_start) + '_' + str(a) + '.png'
                    plt.savefig(nome)
                    plt.pause(0.2)
                    plt.close()

        else:
            print('Os parâmetros que inseriu não são adequados')

        tempo += delta


def DoP_multi_map(temp, L, dh, n=0, source='Moon', ref='polar'):
    D = 1
    R = 1
    tempo = Time(temp)
    tempo_start = tempo.ymdhms
    delta = dh * u.hour

    for a in range(0, n):
        observation = moon_functions.Moon(tempo)
        observation.get_parameters()

        SOL, obs_sol = observation.true_sun()

        k = observation.phase

        alt_sol = observation.my_sun.alt
        az_sol = observation.my_sun.az

        t_sol = observation.my_sun.theta
        phi_sol = observation.my_sun.phi

        alt_lua = observation.alt
        az_lua = observation.az

        t_lua = observation.theta
        phi_lua = observation.phi

        x_lua = np.sin(t_lua) * np.cos(phi_lua)
        y_lua = np.sin(t_lua) * np.sin(phi_lua)
        z_lua = np.cos(t_lua)

        X_lua = x_lua / (1 - z_lua)
        Y_lua = y_lua / (1 - z_lua)

        # Lua
        Lepsip = np.zeros(2, dtype=float)
        Lepsil = np.zeros(2, dtype=float)

        Lepsip[0] = (X_lua + L * np.cos(phi_lua)) / (1 - L * np.cos(phi_lua) * X_lua)
        Lepsip[1] = (Y_lua + L * np.sin(phi_lua)) / (1 - L * np.sin(phi_lua) * Y_lua)
        LEp = Lepsip[0] + 1j * Lepsip[1]

        LEpp = -1 / np.conj(LEp)
        # Lepsipp = np.zeros(2)
        # Lepsipp = xy(LEpp)

        # epsilon- definição
        Lepsil[0] = (X_lua - L * np.cos(phi_lua)) / (1 + L * np.cos(phi_lua) * X_lua)
        Lepsil[1] = (Y_lua - L * np.sin(phi_lua)) / (1 + L * np.sin(phi_lua) * Y_lua)
        LEl = Lepsil[0] + 1j * Lepsil[1]

        LEll = -1 / np.conj(LEl)

        # --------------------------------

        x_sol = np.sin(t_sol) * np.cos(phi_sol)
        y_sol = np.sin(t_sol) * np.sin(phi_sol)
        z_sol = np.cos(t_sol)

        X_sol = x_sol / (1 - z_sol)
        Y_sol = y_sol / (1 - z_sol)

        # sol
        epsip = np.zeros(2, dtype=float)
        epsil = np.zeros(2, dtype=float)

        epsip[0] = (X_sol + L * np.cos(phi_sol)) / (1 - L * np.cos(phi_sol) * X_sol)
        epsip[1] = (Y_sol + L * np.sin(phi_sol)) / (1 - L * np.sin(phi_sol) * Y_sol)
        Ep = epsip[0] + 1j * epsip[1]

        Epp = -1 / np.conj(Ep)
        # Lepsipp = np.zeros(2)
        # Lepsipp = xy(LEpp)

        # epsilon- definição
        epsil[0] = (X_sol - L * np.cos(phi_sol)) / (1 + L * np.cos(phi_sol) * X_sol)
        epsil[1] = (Y_lua - L * np.sin(phi_sol)) / (1 + L * np.sin(phi_sol) * Y_sol)
        El = epsil[0] + 1j * epsil[1]

        Ell = -1 / np.conj(El)

        if -90 <= alt_lua <= 90 and 0 <= az_lua <= 360:

            dop = np.zeros((100, 400))

            x = np.zeros((100, 400))
            y = np.zeros((100, 400))

            Alt = np.zeros((100, 400))
            Az = np.zeros((100, 400))

            i, j = 0, 0

            for Eo in np.linspace(0, np.pi / 2, 100, endpoint=True):
                for Azo in np.linspace(0, 2 * np.pi, 400, endpoint=True):

                    to = np.pi / 2 - Eo
                    phio = Azo

                    Alt[i, j] = to * 180 / np.pi
                    Az[i, j] = phio

                    xo = np.sin(to) * np.cos(phio)
                    yo = np.sin(to) * np.sin(phio)
                    zo = np.cos(to)

                    X = xo / (1 - zo)
                    Y = yo / (1 - zo)

                    x[i, j] = xo
                    y[i, j] = yo

                    xyo = X + 1j * Y

                    wl = ((xyo - LEp) * (xyo - LEl) * (xyo - LEpp) * (xyo - LEll)) / (
                            ((1 + (np.abs(xyo) ** 2)) ** 2) * np.abs(LEp - LEpp) * np.abs(LEl - LEll))

                    w = ((xyo - Ep) * (xyo - El) * (xyo - Epp) * (xyo - Ell)) / (
                            ((1 + (np.abs(xyo) ** 2)) ** 2) * np.abs(Ep - Epp) * np.abs(El - Ell))

                    if source == 'Moon':
                        dop[i, j] = np.abs(wl) * k
                    if source == 'Sun':
                        dop[i, j] = np.abs(w)
                    if source == 'Total':
                        dop[i, j] = np.abs(wl) * k + np.abs(w)

                    j += 1

                i += 1
                j = 0

            if ref == 'polar':

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

                    az_label_offset = 0 * u.deg
                    theta_labels = []
                    for chunk in range(0, 7):
                        label_angle = (az_label_offset * (1 / u.deg)) + (chunk * 45)
                        while label_angle >= 360:
                            label_angle -= 360
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

                    if source == 'Moon':
                        fig = plt.figure(figsize=(7, 5))
                        plt.clf()
                        ax = fig.gca(projection='polar')
                        ax.set_theta_zero_location('N')
                        ax.set_rlim(1, 91)
                        plt.pcolormesh(Az, Alt, dop, vmax=0.25, vmin=0)
                        if 0 <= t_lua <= np.pi / 2:
                            ax.plot(phi_lua, t_lua * 180 / np.pi, 'o', color='white')
                        plt.title(str(tempo), loc='right', fontsize=9, color='black')
                        ax.grid(True, which='major')
                        ax.set_rgrids(range(1, 106, 15), r_labels, angle=-45)
                        ax.set_thetagrids(range(0, 360, 45), theta_labels)
                        plt.colorbar(pad=0.1)
                        nome = 'MAP_dop_multi_polar_Moon_' + str(tempo_start) + '_L_' + str(L) + '_frame_' + str(a) + '.png'
                        plt.savefig(nome)
                        ax.figure.canvas.draw()
                        plt.pause(0.2)
                        plt.close()

                    if source == 'Sun':
                        fig = plt.figure(figsize=(7, 5))
                        plt.clf()
                        ax = fig.gca(projection='polar')
                        ax.set_theta_zero_location('N')
                        ax.set_rlim(1, 91)
                        plt.pcolormesh(Az, Alt, dop, vmax=0.25, vmin=0)
                        if 0 <= t_sol <= np.pi / 2:
                            ax.plot(phi_sol, t_sol * 180 / np.pi, 'o', color='grey')
                        plt.title(str(tempo), loc='right', fontsize=9, color='black')
                        ax.grid(True, which='major')
                        ax.set_rgrids(range(1, 106, 15), r_labels, angle=-45)
                        ax.set_thetagrids(range(0, 360, 45), theta_labels)
                        plt.colorbar(pad=0.1)
                        nome = 'MAP_dop_multi_polar_Sun_' + str(tempo_start) + '_L_' + str(L) +  '_frame_' + str(a) + '.png'
                        plt.savefig(nome)
                        ax.figure.canvas.draw()
                        plt.pause(0.2)
                        plt.close()

                    if source == 'Total':
                        fig = plt.figure(figsize=(7, 5))
                        plt.clf()
                        ax = fig.gca(projection='polar')
                        ax.set_theta_zero_location('N')
                        ax.set_rlim(1, 91)
                        plt.pcolormesh(Az, Alt, dop)
                        if 0 <= t_sol <= np.pi / 2:
                            ax.plot(phi_sol, t_sol * 180 / np.pi, 'o', color='grey')
                        if 0 <= t_lua <= np.pi / 2:
                            ax.plot(phi_lua, t_lua * 180 / np.pi, 'o', color='white')
                        plt.title(str(tempo), loc='right', fontsize=9, color='black')
                        ax.grid(True, which='major')
                        ax.set_rgrids(range(1, 106, 15), r_labels, angle=-45)
                        ax.set_thetagrids(range(0, 360, 45), theta_labels)
                        plt.colorbar(pad=0.1)
                        nome = 'MAP_dop_multi_polar_Total_' + str(tempo_start) + '_L_' + str(L) +  '_frame_' + str(a) + '.png'
                        plt.savefig(nome)
                        ax.figure.canvas.draw()
                        plt.pause(0.2)
                        plt.close()

            if ref == 'XY':
                    if source == 'Moon':
                        x_point = observation.x
                        y_point = observation.y

                        fig = plt.figure(figsize=(7, 5))
                        plt.clf()
                        plt.pcolormesh(x, y, dop, vmax=0.25, vmin=0)
                        if 0 <= alt_lua <= 90:
                            plt.plot(x_point, y_point, 'o', color='white')
                        plt.text(0, 1.2, str(tempo), fontsize=11)
                        plt.text(0.3, 1.1, 'fase da Lua: %.3f ' % round(k, 3), fontsize=9)
                        plt.text(0, 1.1, 'E', fontsize=14)
                        plt.text(0, -1.2, 'O', fontsize=14)
                        plt.text(1.2, 0, 'N', fontsize=14)
                        plt.text(-1.3, 0, 'S', fontsize=14)
                        plt.axis('equal')
                        plt.axis('off')
                        plt.colorbar(pad=0.1)
                        nome = 'MAP_dop_multi_xy_Moon_' + str(tempo_start) + '_L_' + str(L) +  '_frame_' + str(a) + '.png'
                        plt.savefig(nome)
                        plt.pause(0.2)
                        plt.close()

                    if source == 'Sun':
                        x_point = observation.my_sun.x
                        y_point = observation.my_sun.y

                        fig = plt.figure(figsize=(7, 5))
                        plt.clf()
                        plt.pcolormesh(x, y, dop, vmax=0.25, vmin=0)
                        if 0 <= alt_sol <= 90:
                            plt.plot(x_point, y_point, 'o', color='grey')
                        plt.text(0, 1.2, str(tempo), fontsize=9)
                        plt.text(0, 1.1, 'E', fontsize=14)
                        plt.text(0, -1.2, 'O', fontsize=14)
                        plt.text(1.2, 0, 'N', fontsize=14)
                        plt.text(-1.3, 0, 'S', fontsize=14)
                        plt.axis('equal')
                        plt.axis('off')
                        plt.colorbar(pad=0.1)
                        nome = 'MAP_dop_multi_xy_Sun_' + str(tempo_start) + '_L_' + str(L) +  '_frame_' + str(a) + '.png'
                        plt.savefig(nome)
                        plt.pause(0.2)
                        plt.close()

                    if source == 'Total':
                        xl = observation.x
                        yl = observation.y

                        xs = observation.my_sun.x
                        ys = observation.my_sun.y

                        fig = plt.figure(figsize=(7, 5))
                        plt.clf()
                        plt.pcolormesh(x, y, dop)
                        if 0 <= alt_lua <= 90:
                            plt.plot(xl, yl, 'o', color='white')
                        if 0 <= alt_sol <= 90:
                            plt.plot(xs, ys, 'o', color='grey')
                        plt.text(0, 1.2, str(tempo), fontsize=9)
                        plt.text(0.3, 1.1, 'fase da Lua: %.3f ' % round(k, 3), fontsize=11)
                        plt.text(0, 1.1, 'E', fontsize=14)
                        plt.text(0, -1.2, 'O', fontsize=14)
                        plt.text(1.2, 0, 'N', fontsize=14)
                        plt.text(-1.3, 0, 'S', fontsize=14)
                        plt.axis('equal')
                        plt.axis('off')
                        plt.colorbar(pad=0.1)
                        nome = 'MAP_dop_multi_xy_Total_' + str(tempo_start) + '_L_' + str(L) +  '_frame_' + str(a) + '.png'
                        plt.savefig(nome)
                        plt.pause(0.2)
                        plt.close()

        else:
            print('Os parâmetros que inseriu não são adequados')

        tempo += delta

