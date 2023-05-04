import PySimpleGUI as sg
import mplcursors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy.time import Time
import sky
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tables_data as td


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def plo_graph(df, band, condition, label_text):
    global par_wave, cor_line, cor, lab_wave, cor_unc

    if condition is not None:
        if isinstance(condition, str):
            df = df[df['CONDITION'] == condition]
        else:
            df = df[df['CONDITION'].isin(condition)]

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

        points = df[df['WAVELENGTH'] == par_wave]
        a = points['GAMMA'].to_numpy()
        b = points['POL FIT'].to_numpy()
        c = points['RES'].to_numpy()
        d = points['DIFF'].to_numpy()
        e = points['POL OB'].to_numpy()
        f = points['error POL OB'].to_numpy()
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
            label1.append('Field observed in band ' + item + ': \n $\u03B8_{FIELD}$ = ' + str(
                round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

        plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
        plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

        bando = plt.scatter(a, e, color='none')
        mplcursors.cursor(bando, hover=True).connect("add",
                                                     lambda sel: sel.annotation.set_text(label1[sel.index]))

        g1 = np.add(b, c)
        g2 = np.subtract(b, c)
        plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

        plt.ylim(0, 0.8)
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 1),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center',
                     bbox=dict(boxstyle="round", fc="w"))

        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

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

        points = df[df['WAVELENGTH'] == par_wave]
        a = points['GAMMA'].to_numpy()
        b = points['POL FIT'].to_numpy()
        c = points['RES'].to_numpy()
        d = points['DIFF'].to_numpy()
        e = points['POL OB'].to_numpy()
        f = points['error POL OB'].to_numpy()
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
            label1.append('Field observed in band ' + item + '\n $\u03B8_{FIELD}$ = ' + str(
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

    plt.show()


def plot_window(df, result_par, erro, quality, band, condition, model, correction, label):
    _VARS = {'window': False}

    layout = [[sg.Text("Fit Status")], [sg.Canvas(size=(640, 480), key='figCanvas'), sg.Text(label)],
              [sg.Button("SUMMARY"), sg.Button("OK")]]

    # Create the window
    _VARS['window'] = sg.Window("SkyPol", layout, finalize=True, resizable=True)

    global par_wave, cor_line, cor, lab_wave, cor_unc

    if condition is not None:
        if isinstance(condition, str):
            df = df[df['CONDITION'] == condition]
        else:
            df = df[df['CONDITION'].isin(condition)]

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

        points = df[df['WAVELENGTH'] == par_wave]
        a = points['GAMMA'].to_numpy()
        b = points['POL FIT'].to_numpy()
        c = points['RES'].to_numpy()
        d = points['DIFF'].to_numpy()
        e = points['POL OB'].to_numpy()
        f = points['error POL OB'].to_numpy()
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
            label1.append('Field observed in band ' + item + ': \n $\u03B8_{FIELD}$ = ' + str(
                round(i[z], 2)) + ' ; $\u03C6_{FIELD}$ = ' + str(
                round(j[z], 2)) + ';\n$\u03B8_{MOON}$ = ' + str(
                round(g[z], 2)) + ' ; $\u03C6_{MOON}$ = ' + str(round(h[z], 2)) + ';\n$DoP_{OBS}$ = ' + str(
                round(e[z], 3)) + ' $ \pm $ ' + str(round(f[z], 3)) + ';\n$DoP_{SIM}$ = ' + str(
                round(b[z], 3)) + ' $ \pm $ ' + str(round(c[z], 3)))

        plt.plot(a, b, '-', color=cor_line, markersize=2, label='fit results ' + lab_wave + ' band')
        plt.errorbar(a, e, yerr=f, ms=2.0, fmt='o', color=cor, label='data ' + lab_wave + 'band')

        bando = plt.scatter(a, e, color='none')
        mplcursors.cursor(bando, hover=True).connect("add",
                                                     lambda sel: sel.annotation.set_text(label1[sel.index]))

        g1 = np.add(b, c)
        g2 = np.subtract(b, c)
        plt.fill_between(a, g2, g1, where=(g2 < g1), interpolate=True, color=cor_unc)

        plt.ylim(0, 0.8)

        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

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

        points = df[df['WAVELENGTH'] == par_wave]
        a = points['GAMMA'].to_numpy()
        b = points['POL FIT'].to_numpy()
        c = points['RES'].to_numpy()
        d = points['DIFF'].to_numpy()
        e = points['POL OB'].to_numpy()
        f = points['error POL OB'].to_numpy()
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
            label1.append('Field observed in band ' + item + '\n $\u03B8_{FIELD}$ = ' + str(
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

    # Instead of plt.show
    draw_figure(_VARS['window']['figCanvas'].TKCanvas, fig_x)

    # Create an event loop
    while True:
        event, values = _VARS['window'].read()

        # End program if user closes window or
        # presses the OK button
        if event == "OK" or event == sg.WIN_CLOSED:
            break
        if event == "SUMMARY":
            td.window_summary(df, result_par, erro, quality, band, condition, model, correction)

    _VARS['window'].close()


def sim_obs(RA, DEC, Start, n=10, dt=0.25):
    time_start = Time(Start, format='isot', scale='utc')

    # dt = time_end - time_start
    delta = dt * u.hour

    PHI_VEC = []
    THETA_VEC = []

    OBS_TIME_VEC = []

    PHI_VEC_MOON = []
    THETA_VEC_MOON = []

    PHASE = []
    GAMMA = []

    tempo = time_start

    for t in range(0, n):
        observation = sky.Moon(tempo)
        observation.set_parameters()

        PHI_VEC_MOON.append(float(observation.phi))
        THETA_VEC_MOON.append(float(observation.theta))
        PHASE.append(observation.phase)

        coords_obj = sky.coord_radectoaltaz(RA, DEC, tempo)

        theta_obs = np.pi / 2 - coords_obj[0].value * np.pi / 180.0
        phi_obs = coords_obj[1].value * np.pi / 180.0

        PHI_VEC.append(phi_obs)
        THETA_VEC.append(theta_obs)

        GAMMA.append(sky.func_gamma(theta_obs, phi_obs, observation.theta, observation.phi))

        OBS_TIME_VEC.append(tempo)

        tempo += delta

    loc = coord.EarthLocation(lon=-70.404167 * u.deg, lat=-24.627222 * u.deg, height=2635 * u.m)
    AltAz = coord.AltAz(obstime=tempo, location=loc)

    lon = np.linspace(0, 360, 100)
    lat = np.zeros(100)

    ecl = SkyCoord(lon, lat, unit=u.deg, frame='barycentricmeanecliptic')

    ecl_gal = ecl.transform_to(AltAz)
    ALT = ecl_gal.alt.value
    AZ = ecl_gal.az.value

    THETA = np.pi / 2 - ALT * np.pi / 180.0
    PHI = AZ * np.pi / 180.0

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

    fig = plt.figure(figsize=(6, 4))
    plt.clf()
    ax = fig.gca(projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_rlim(1, 91)

    plt.plot(PHI, THETA * 180 / np.pi, ls="--")
    for n in range(0, 10):
        if 0 <= THETA_VEC_MOON[n] <= np.pi / 2:
            plt.plot(PHI_VEC_MOON[n], THETA_VEC_MOON[n] * 180 / np.pi, 'o', color='black', markersize=3,
                     label='Time %s and moon phase %s' % (OBS_TIME_VEC[n], PHASE[n]))
        if 0 <= THETA_VEC[n] <= np.pi / 2:
            plt.plot(PHI_VEC[n], THETA_VEC[n] * 180 / np.pi, 'o', color='blue', markersize=3,
                     label='Target at %s: (%s, %s)\n moon at (%s, %s)\n separation: %s' % (
                         OBS_TIME_VEC[n], THETA_VEC[n] * 180 / np.pi, 180 - PHI_VEC[n] * 180 / np.pi,
                         THETA_VEC_MOON[n] * 180 / np.pi, 180 - PHI_VEC_MOON[n] * 180 / np.pi, GAMMA[n] * 180 / np.pi))
    mplcursors.cursor().connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
    ax.grid(True, which='major')
    ax.set_rgrids(range(1, 106, 15), r_labels, angle=-45)
    ax.set_thetagrids(range(0, 360, 45), theta_labels)
    nome = 'sim.png'
    plt.savefig(nome)
    plt.show()


# Start = '2017-05-24T00:56:57.948'
# End = '2017-05-24T02:18:20.287'
# sim_obs(156.3716667, -39.8277778, Start, End)

# Start = '2017-06-13T00:00:24.662'
# End = '2017-06-13T00:39:02.910'
# sim_obs(156.3716667, -39.8277778, Start, End)
