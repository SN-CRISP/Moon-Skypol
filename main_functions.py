"""
    Decision maker about the method, data analysis and fitting process

"""

import PySimpleGUI as sg
import generic as gen
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy.time import Time
from datetime import datetime
import sky
import mapping as map
import fit_process
import generic
import tables_data as td


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def data_storage():
    global wave
    df = pd.read_csv('data.csv', sep=';')

    DATA = df
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
    BANDA = DATA['BAND'].to_numpy()
    CONDITION = DATA['CONDITION'].to_numpy()

    fit_observations_resume = pd.DataFrame(
        {'FIELD': field, 'RA': RA, 'DEC': DEC, 'OBS TIME MED': MED, 'I': Ival, 'Q': Qval, 'error Q': errQval, 'U': Uval,
         'U error': errUval, 'SEEING': seen, 'BANDA': BANDA, 'CONDITION': CONDITION})

    total = len(RA)

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
    SEEING = []
    WAV = []
    errAOP = []

    for n in range(0, total):
        if BANDA[n] == 'B':
            wave = 437

        if BANDA[n] == 'V':
            wave = 555

        if BANDA[n] == 'R':
            wave = 655

        if BANDA[n] == 'I':
            wave = 768

        observation = sky.Moon(MED[n])
        observation.set_parameters()

        campo = sky.Field(BANDA[n], CONDITION[n], field[n], float(RA[n]), float(DEC[n]))
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
        alb = observation.plot_and_retrive_albedo(wave)
        ALBEDO.append(alb)
        AOP.append(0.5 * np.arctan(float(Uval[n]) / float(Qval[n])) * 180 / np.pi)
        errAOP.append(0.5 * np.sqrt((Qval[n] * errU_OBS[n]) ** 2 + (Uval[n] * errQ_OBS[n]) ** 2) / (
                (1 + (Uval[n] / Qval[n]) ** 2) * Qval[n] ** 2) * 180 / np.pi)
        SEEING.append(seen[n])
        WAV.append(wave)

    C1field, C2field = np.asarray(C1field, dtype=np.float32), np.asarray(C2field, dtype=np.float32)
    C1lua, C2lua = np.asarray(C1lua, dtype=np.float32), np.asarray(C2lua, dtype=np.float32)
    C1sol, C2sol = np.asarray(C1sol, dtype=np.float32), np.asarray(C2sol, dtype=np.float32)
    POL_OBS, errPOL_OBS = np.asarray(POL_OBS, dtype=np.float32), np.asarray(errPOL_OBS, dtype=np.float32)
    GAMMA, AOP = np.asarray(GAMMA, dtype=np.float32), np.asarray(AOP, dtype=np.float32)
    ALBEDO, SEEING = np.asarray(ALBEDO, dtype=np.float32), np.asarray(SEEING, dtype=np.float32)
    GAMMA_SOL, WAV = np.asarray(GAMMA_SOL, dtype=np.float32), np.asarray(WAV, dtype=np.float32)

    fit_observations_resume.insert(12, 'THETA MOON', C1lua)
    fit_observations_resume.insert(13, 'PHI MOON', C2lua)
    fit_observations_resume.insert(14, 'ALBEDO', ALBEDO)
    fit_observations_resume.insert(15, 'WAVELENGTH', WAV)
    fit_observations_resume.insert(16, 'THETA FIELD', C1field)
    fit_observations_resume.insert(17, 'PHI FIELD', C2field)
    fit_observations_resume.insert(18, 'GAMMA', GAMMA)
    fit_observations_resume.insert(19, 'AOP', AOP)
    fit_observations_resume.insert(20, 'AOP error', errAOP)
    fit_observations_resume.insert(21, 'POL OBS', POL_OBS)
    fit_observations_resume.insert(22, 'POL OBS error', errPOL_OBS)
    fit_observations_resume.insert(23, 'THETA SUN', C1sol)
    fit_observations_resume.insert(24, 'PHI SUN', C2sol)

    fit_observations_resume.to_csv("data_output.csv")


def window_status():
    _VARS = {'window': False}

    layout = [[sg.Text("Fit Status")], [sg.Canvas(size=(640, 480), key='figCanvas')], [sg.Button("OK")],
              [sg.Button("SAVE")]]

    # Create the window
    _VARS['window'] = sg.Window("SkyPol", layout, finalize=True, resizable=True)

    dataSize = 1000
    Start = '2017-05-24T00:56:57.948'
    End = '2017-05-24T02:18:20.287'
    RA = 156.3716667
    DEC = -39.8277778

    # make fig and plot
    fig = plt.figure()
    time_start = Time(Start, format='isot', scale='utc')
    time_end = Time(End, format='isot', scale='utc')

    # dt = time_end - time_start
    delta = 0.25 * u.hour

    PHI_VEC = []
    THETA_VEC = []

    OBS_TIME_VEC = []

    PHI_VEC_MOON = []
    THETA_VEC_MOON = []

    PHASE = []
    GAMMA = []

    tempo = time_start

    for t in range(0, 10):
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

    ax = fig.gca(projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_rlim(1, 91)

    fig.canvas.mpl_connect("motion_notify_event", lambda event: fig.canvas.toolbar.set_message(""))
    fig.canvas.draw()

    plt.plot(PHI, THETA * 180 / np.pi, ls="--")
    for n in range(0, 10):
        if 0 <= THETA_VEC_MOON[n] <= np.pi / 2:
            plt.plot(PHI_VEC_MOON[n], THETA_VEC_MOON[n] * 180 / np.pi, 'o', color='black', markersize=3,
                     label='Time %s and moon phase %s' % (OBS_TIME_VEC[n], PHASE[n]))
        if 0 <= THETA_VEC[n] <= np.pi / 2:
            plt.plot(PHI_VEC[n], THETA_VEC[n] * 180 / np.pi, 'o', color='blue', markersize=3,
                     label='Target at %s: (%s, %s)\n moon at (%s, %s)\n separation: %s' % (
                         OBS_TIME_VEC[n], PHI_VEC[n] * 180 / np.pi, THETA_VEC[n] * 180 / np.pi,
                         PHI_VEC_MOON[n] * 180 / np.pi, THETA_VEC_MOON[n] * 180 / np.pi, GAMMA[n] * 180 / np.pi))
    mplcursors.cursor().connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
    ax.grid(True, which='major')
    ax.set_rgrids(range(1, 106, 15), r_labels, angle=-45)
    ax.set_thetagrids(range(0, 360, 45), theta_labels)

    # Instead of plt.show
    draw_figure(_VARS['window']['figCanvas'].TKCanvas, fig)

    # Create an event loop
    while True:
        event, values = _VARS['window'].read()

        # End program if user closes window or
        # presses the OK button
        if event == "OK" or event == sg.WIN_CLOSED:
            break

    _VARS['window'].close()


def window_interface_fit():
    cb = []
    cb.append(sg.Checkbox('B', key='B'))
    cb.append(sg.Checkbox('V', key='V'))
    cb.append(sg.Checkbox('R', key='R'))
    cb.append(sg.Checkbox('I', key='I'))

    rb = []
    rb.append(sg.Checkbox('ok', key='ok'))
    rb.append(sg.Checkbox('clouds', key='clouds'))
    rb.append(sg.Checkbox('sun', key='sun'))

    layout = [[sg.Text("Hello where you can fit the moonlight polarization patterns:")],
              [sg.Text('Choose Model ', size=(20, 1), font='Lucida', justification='left')],
              [sg.Combo(
                  ['Rayleigh single scattering', 'Multiple scattering', 'Stokes method Rayleigh',
                   'Mie single scattering'],
                  key='model')],
              [sg.Text('Choose Correction ', size=(30, 1), font='Lucida', justification='left')],
              [sg.Combo(
                  ['Amplitude empirical parameter', 'Depolarization factor', 'Sun influence', 'Seeing correction',
                   'Mix atmospheric corrections'],
                  key='correction')],
              [sg.Text("Bands:", size=(20, 1), font='Lucida', justification='left')], [cb],
              [sg.Text("Conditions:", size=(20, 1), font='Lucida', justification='left')], [rb],
              [sg.Button('NEXT', font=('Times New Roman', 12)), sg.Button('EXIT', font=('Times New Roman', 12))]]

    # Create the window
    window = sg.Window("SkyPol", layout)

    win2_active = False

    # Create an event loop
    while True:
        event, values = window.read()
        if event == "NEXT" and not win2_active:
            window.Hide()
            win2_active = True

            window['B'].update(text="B")
            window['V'].update(text="V")
            window['R'].update(text="R")
            window['I'].update(text="I")

            window['ok'].update(text="ok")
            window['clouds'].update(text="clouds")
            window['sun'].update(text="sun")

            bands = [x.Text for x in cb if x.get() == True]
            bands_str = gen.listToString(bands)

            cond = [x.Text for x in rb if x.get() == True]
            cond_str = gen.listToString(cond)

            label_fit = 'Fit model from ' + values['model'] + '\n corrected using ' + values[
                'correction'] + '\n for ' + bands_str + ' bands in ' + cond_str + ' conditions of observation.'

            fit_resume, result_par, erro, quality = fit_process.process_data(values['model'], values['correction'],
                                                                             method_fit='leastsq', condition=cond,
                                                                             band=bands)

            td.tab_from_df(fit_resume)

            pieces = []
            label_text = []
            for wave in range(0, len(bands)):
                for i in range(0, len(result_par[wave])):
                    pieces.append(str(result_par[wave][i]) + ' $\pm$ ' + str(erro[wave][i]) + '\n')

                list_par = generic.listToString(pieces)

                label = 'fit parameters band ' + bands[wave] + ':\n' + list_par + 'chi-square: ' + str(
                    round(quality[wave][0], 10)) + ',  reduced chi-square: ' + str(
                    round(quality[wave][1], 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
                    round(quality[wave][2], 10))

                label_text.append(label)

            # text = generic.listToString(label_text)

            # sg.popup(text)

            map.plot_window(fit_resume, result_par, erro, quality, bands, cond_str, values['model'],
                            values['correction'], label_fit)

        if event == "EXIT" or event == sg.WIN_CLOSED:
            break

    window.close()


def window_interface_sim():
    dates = [i for i in range(1, 32)]
    s1 = sg.Spin(dates, initial_value=1, readonly=True, size=3, enable_events=True, key='-DAY-')

    months = [i for i in range(1, 13)]
    s2 = sg.Spin(months, initial_value=1, readonly=True, size=3, enable_events=True, key='-MON-')

    yrs = [i for i in range(1996, 2100)]
    s3 = sg.Spin(yrs, initial_value=2000, readonly=True, size=5, enable_events=True, key='-YR-')

    hours = [i for i in range(0, 24)]
    s4 = sg.Spin(hours, initial_value=0, readonly=True, size=3, enable_events=True, key='-HR-')

    minutes = [i for i in range(0, 60)]
    s5 = sg.Spin(minutes, initial_value=0, readonly=True, size=3, enable_events=True, key='-MIN-')

    seconds = [i for i in range(0, 60)]
    s6 = sg.Spin(seconds, initial_value=0, readonly=True, size=5, enable_events=True, key='-SD-')

    layout = [
        [sg.Text('Date'), s1, sg.Text("Month"), s2, sg.Text("Year"), s3],
        [sg.Text('Hours'), s4, sg.Text("Minutes"), s5, sg.Text("Seconds"), s6],
        [sg.Text('RA'), sg.Input(key='-RA-', size=10), sg.Text("DEC"), sg.Input(key='-DEC-', size=10), sg.Text('Delta'), sg.Input(key='-DT-', size=5), sg.Text("Data Points"), sg.Input(key='-DP-', size=5)],
        [sg.OK(), sg.Text("", key='-OUT-')]
    ]
    window = sg.Window('SkyPol - Simulation', layout, font='_ 18', finalize=True, resizable=True)

    win2_active = False

    while True:
        event, values = window.read()
        if event == 'OK' and not win2_active:
            window.Hide()
            win2_active = True
            t = Time(datetime(values['-YR-'], values['-MON-'], values['-DAY-'], values['-HR-'], values['-MIN-'],
                              values['-SD-']), scale='utc')
            t.format = 'fits'
            tempo = t.value
            ra = float(values['-RA-'])
            dec = float(values['-DEC-'])
            n = int(values['-DP-'])
            delta = float(values['-DT-'])

            map.sim_obs(ra, dec, tempo, n, delta)

        if event == sg.WIN_CLOSED:
            break

    window.close()


def window_first():
    layout = [[sg.Image('bob.gif', size=(5, 5), key="GIF")],
              [sg.Button('FIT', font=('Times New Roman', 12)), sg.Button('SIMULATION', font=('Times New Roman', 12))]]

    window = sg.Window("SkyPol", layout, finalize=True, resizable=True)
    win2_active = False

    while True:
        event, values = window.Read(timeout=100)
        window["GIF"].UpdateAnimation("bob.gif", time_between_frames=100)

        if event == "FIT" and not win2_active:
            window.Hide()
            win2_active = True
            window_interface_fit()

        if event == "SIMULATION" and not win2_active:
            window.Hide()
            win2_active = True
            window_interface_sim()

        if event == sg.WIN_CLOSED:
            break

    window.close()


window_first()

# window_interface_fit()
