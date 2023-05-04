import pandas as pd
import numpy as np
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord
import moon_functions
import lmfit
import multiple_func
import Stokes_func
import Rayleigh_func
import matplotlib.pyplot as plt
from tqdm import tqdm
import Mie_func


def coord_radectoaltaz(RA, DEC, tem):
    observing_location = coord.EarthLocation(lon=-70.404167 * u.deg, lat=-24.627222 * u.deg, height=2635 * u.m)
    observing_time = Time(tem)
    cond = coord.AltAz(location=observing_location, obstime=observing_time)

    CO = SkyCoord(ra=RA * u.deg, dec=DEC * u.deg, frame='icrs')
    CO.transform_to(cond)
    alt = CO.transform_to(cond).alt * u.deg
    az = CO.transform_to(cond).az * u.deg

    # results have the shape: ALT, AZ
    results = [alt, az]

    return results


def plot_data(data, command, method='leastsq'):

    x = data['GAMMA'].to_numpy()
    y = data['FIT IND'].to_numpy()
    rsd = data['FIT IND UNC'].to_numpy()
    diff = data['FIT IND DIFF'].to_numpy()
    pol = data['POL'].to_numpy()
    er = data['ERROR POL'].to_numpy()
    w = np.argsort(x)
    new_x, new_y, er_diff, er_rsd, new_pol, new_er = np.asarray(x)[w], np.asarray(y)[w], np.asarray(diff)[w], np.asarray(rsd)[w], np.asarray(pol)[w], np.asarray(er)[w]
    g1 = np.add(new_y, er_rsd)
    g2 = np.subtract(new_y, er_rsd)

    fig_x = plt.figure(figsize=(10, 5))
    fig_x.add_axes((.1, .3, .8, .6))

    plt.plot(new_x, new_y, '-', color='cornflowerblue', markersize=2, label='fit results')
    plt.errorbar(new_x, pol, yerr=new_er, ms=2.0, fmt='o', color='blue', label='data')
    plt.fill_between(new_x, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

    plt.grid(True)
    plt.ylabel('Polarization')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    fig_x.add_axes((.1, .1, .8, .2))

    plt.errorbar(new_x, er_diff, yerr=new_er, ms=2.0, fmt='o', color='blue', label='diff')
    plt.plot(new_x, er_rsd, '-', color='cornflowerblue', markersize=2, label='uncertanties fit')

    plt.xlabel('Scattering Angle (degrees)')
    plt.ylabel('Residual data')
    plt.grid(True)
    plt.savefig('Plot_' + str(command) + '_method_' + str(method) + '.png', bbox_inches='tight')
    plt.pause(2)
    plt.close()


def plot_all(results, command, method='leastsq'):
    global new_x_b, new_y_b, pol_b, new_er_b, new_x_v, g2_b, g1_b, new_y_v, pol_v, new_er_v, g2_v, g1_v, new_x_r, new_y_r, new_er_r, new_y_i, pol_r, g2_r, g1_r, new_x_i, pol_i, new_er_i, g2_i, g1_i, er_diff_b, er_rsd_b, er_diff_v, er_rsd_v, er_diff_r, er_rsd_r, er_diff_i, er_rsd_i
    observation = ['B', 'V', 'R', 'I']

    for banda in observation:
        data = results[results['BAND'] == banda]
        if banda == 'B':
            x_b = data['GAMMA'].to_numpy()
            y_b = data['FIT IND'].to_numpy()
            rsd_b = data['FIT IND UNC'].to_numpy()
            diff_b = data['FIT IND DIFF'].to_numpy()
            pol_b = data['POL'].to_numpy()
            er_b = data['ERROR POL'].to_numpy()
            w = np.argsort(x_b)
            new_x_b, new_y_b, er_diff_b, er_rsd_b, new_pol_b, new_er_b = np.asarray(x_b)[w], np.asarray(y_b)[w], np.asarray(diff_b)[
                w], np.asarray(rsd_b)[w], np.asarray(pol_b)[w], np.asarray(er_b)[w]
            g1_b = np.add(new_y_b, er_rsd_b)
            g2_b = np.subtract(new_y_b, er_rsd_b)
        if banda == 'V':
            x_v = data['GAMMA'].to_numpy()
            y_v = data['FIT IND'].to_numpy()
            rsd_v = data['FIT IND UNC'].to_numpy()
            diff_v = data['FIT IND DIFF'].to_numpy()
            pol_v = data['POL'].to_numpy()
            er_v = data['ERROR POL'].to_numpy()
            w = np.argsort(x_v)
            new_x_v, new_y_v, er_diff_v, er_rsd_v, new_pol_v, new_er_v = np.asarray(x_v)[w], np.asarray(y_v)[w], np.asarray(diff_v)[
                w], np.asarray(rsd_v)[w], np.asarray(pol_v)[w], np.asarray(er_v)[w]
            g1_v = np.add(new_y_v, er_rsd_v)
            g2_v = np.subtract(new_y_v, er_rsd_v)
        if banda == 'R':
            x_r = data['GAMMA'].to_numpy()
            y_r = data['FIT IND'].to_numpy()
            rsd_r = data['FIT IND UNC'].to_numpy()
            diff_r = data['FIT IND DIFF'].to_numpy()
            pol_r = data['POL'].to_numpy()
            er_r = data['ERROR POL'].to_numpy()
            w = np.argsort(x_r)
            new_x_r, new_y_r, er_diff_r, er_rsd_r, new_pol_r, new_er_r = np.asarray(x_r)[w], np.asarray(y_r)[w], np.asarray(diff_r)[
                w], np.asarray(rsd_r)[w], np.asarray(pol_r)[w], np.asarray(er_r)[w]
            g1_r = np.add(new_y_r, er_rsd_r)
            g2_r = np.subtract(new_y_r, er_rsd_r)
        if banda == 'I':
            x_i = data['GAMMA'].to_numpy()
            y_i = data['FIT IND'].to_numpy()
            rsd_i = data['FIT IND UNC'].to_numpy()
            diff_i = data['FIT IND DIFF'].to_numpy()
            pol_i = data['POL'].to_numpy()
            er_i = data['ERROR POL'].to_numpy()
            w = np.argsort(x_i)
            new_x_i, new_y_i, er_diff_i, er_rsd_i, new_pol_i, new_er_i = np.asarray(x_i)[w], np.asarray(y_i)[w], np.asarray(diff_i)[
                w], np.asarray(rsd_i)[w], np.asarray(pol_i)[w], np.asarray(er_i)[w]
            g1_i = np.add(new_y_i, er_rsd_i)
            g2_i = np.subtract(new_y_i, er_rsd_i)

    fig_x = plt.figure(figsize=(10, 5))
    fig_x.add_axes((.1, .3, .8, .6))
    plt.ylim(0, 0.8)

    plt.plot(new_x_b, new_y_b, '-', color='cornflowerblue', markersize=2, label='fit results B band')
    plt.errorbar(new_x_b, pol_b, yerr=new_er_b, ms=2.0, fmt='o', color='blue', label='data B band')
    plt.fill_between(new_x_b, g2_b, g1_b, where=(g2_b < g1_b), interpolate=True, color='lavender')

    plt.plot(new_x_v, new_y_v, '-', color='mediumseagreen', markersize=2, label='fit results V band')
    plt.errorbar(new_x_v, pol_v, yerr=new_er_v, ms=2.0, fmt='o', color='green', label='data V band')
    plt.fill_between(new_x_v, g2_v, g1_v, where=(g2_v < g1_v), interpolate=True, color='beige')

    plt.plot(new_x_r, new_y_r, '-', color='indianred', markersize=2, label='fit results R band')
    plt.errorbar(new_x_r, pol_r, yerr=new_er_r, ms=2.0, fmt='o', color='red', label='data R band')
    plt.fill_between(new_x_r, g2_r, g1_r, where=(g2_r < g1_r), interpolate=True, color='mistyrose')

    plt.plot(new_x_i, new_y_i, '-', color='orange', markersize=2, label='fit results I band')
    plt.errorbar(new_x_i, pol_i, yerr=new_er_i, ms=2.0, fmt='o', color='darkorange', label='data I band')
    plt.fill_between(new_x_i, g2_i, g1_i, where=(g2_i < g1_i), interpolate=True, color='antiquewhite')

    plt.grid(True)
    plt.ylabel('Polarization')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    fig_x.add_axes((.1, .1, .8, .2))

    plt.errorbar(new_x_b, er_diff_b, yerr=new_er_b, ms=2.0, fmt='o', color='blue', label='diff B band')
    plt.plot(new_x_b, er_rsd_b, '-', color='cornflowerblue', markersize=2, label='uncertanties fit')

    plt.errorbar(new_x_v, er_diff_v, yerr=new_er_v, ms=2.0, fmt='o', color='green', label='diff V band')
    plt.plot(new_x_v, er_rsd_v, '-', color='mediumseagreen', markersize=2, label='uncertanties fit')

    plt.errorbar(new_x_r, er_diff_r, yerr=new_er_r, ms=2.0, fmt='o', color='red', label='diff R band')
    plt.plot(new_x_r, er_rsd_r, '-', color='indianred', markersize=2, label='uncertanties fit')

    plt.errorbar(new_x_i, er_diff_i, yerr=new_er_i, ms=2.0, fmt='o', color='darkorange', label='diff V band')
    plt.plot(new_x_i, er_rsd_i, '-', color='orange', markersize=2, label='uncertanties fit')

    plt.xlabel('Scattering Angle (degrees)')
    plt.ylabel('Residual data')
    plt.grid(True)
    plt.savefig('IMAGE_all_individuals_command_' + str(command) + '_method_' + str(method) + '.png', bbox_inches='tight')
    plt.pause(2)
    plt.close()


def plot_all_bands(band_B, band_v, band_r, band_I, conditc, command):
    df = pd.read_csv('data.csv', sep=';')

    df.isnull().sum()
    df.dropna(axis=1)
    observation = ['B', 'V', 'R', 'I']

    data = pd.DataFrame()

    for banda in observation:

        if banda == 'B':
            parameters = band_B
        if banda == 'V':
            parameters = band_v
        if banda == 'R':
            parameters = band_r
        if banda == 'I':
            parameters = band_I

        DATA = df[df['BAND'] == banda]

        if conditc == 'ok':
            cond = 'ok'
            DATA = DATA[DATA['CONDITION'] == cond]
        if conditc == 'clouds':
            cond = 'clouds'
            DATA = DATA[DATA['CONDITION'] == cond]
        if conditc == 'sun':
            cond = 'sun'
            DATA = DATA[DATA['CONDITION'] == cond]
        if conditc == 'all':
            pass

        field = DATA['FIELD'].to_numpy()
        RA = DATA['RA'].to_numpy()
        DEC = DATA['DEC'].to_numpy()
        MED = DATA['MED OBS'].to_numpy()
        Qval = DATA['Q'].to_numpy()
        errQval = DATA['error Q'].to_numpy()
        Uval = DATA['U'].to_numpy()
        errUval = DATA['error U '].to_numpy()
        seen = DATA['SEEING'].to_numpy()

        total = len(RA)

        C1field = []
        C2field = []
        C1lua = []
        C2lua = []
        C1sol = []
        C2sol = []
        GAMMA = []
        GAMMA_SOL = []
        POL_OBS = []
        errPOL_OBS = []
        ALBEDO = []
        AOP = []
        SEEING = []
        BAND = []

        OBS = []

        for n in range(0, total):
            campo_observado = Field(banda, conditc, field[n], float(RA[n]), float(DEC[n]))
            campo_observado.get_observation(MED[n])

            observation = moon_functions.Moon(MED[n])

            SOL = observation.true_sun()
            t_sol = np.pi / 2 - SOL[0].value * np.pi / 180.0
            phi_sol = SOL[1].value * np.pi / 180.0

            C1lua.append(campo_observado.moon.theta)
            C2lua.append(campo_observado.moon.phi)
            C1sol.append(float(t_sol))
            C2sol.append(float(phi_sol))
            C1field.append(campo_observado.theta)
            C2field.append(campo_observado.phi)
            GAMMA.append(campo_observado.gamma * 180 / np.pi)
            GAMMA_SOL.append(campo_observado.func_gamma(t_sol, phi_sol, units='degrees'))
            pol = np.sqrt(float(Qval[n]) ** 2 + float(Uval[n]) ** 2)
            POL_OBS.append(pol)
            errPOL_OBS.append(np.sqrt(
                float(Qval[n]) ** 2 * float(errQval[n]) ** 2 + float(Uval[n]) ** 2 * float(
                    errUval[n]) ** 2) / pol)
            alb = campo_observado.moon.plot_and_retrive_albedo(campo_observado.wave)
            campo_observado.moon.plot_tempo()
            ALBEDO.append(alb)
            AOP.append(0.5 * np.arctan(float(Uval[n]) / float(Qval[n])) * 180 / np.pi)
            SEEING.append(seen[n])
            BAND.append(campo_observado.wave)
            OBS.append(campo_observado)

        C1field, C2field = np.asarray(C1field, dtype=np.float32), np.asarray(C2field, dtype=np.float32)
        C1lua, C2lua = np.asarray(C1lua, dtype=np.float32), np.asarray(C2lua, dtype=np.float32)
        POL_OBS, errPOL_OBS = np.asarray(POL_OBS, dtype=np.float32), np.asarray(errPOL_OBS, dtype=np.float32)
        GAMMA, AOP = np.asarray(GAMMA, dtype=np.float32), np.asarray(AOP, dtype=np.float32)

        if command == 'regular_multi':
            y1 = multiple_func.func_reg_DOP([C1field, C2field, C1lua, C2lua], *parameters)
        if command == 'simple_regular_multi':
            y1 = multiple_func.func_simple_reg_DOP(theta_field=C1field, phi_field=C2field, theta_lua=C1lua,
                                                   phi_lua=C2lua,
                                                   L=parameters[0], par=parameters[1])
        if command == 'regular_stokes':
            y1 = Stokes_func.func_reg_DOP([C1field, C2field, C1lua, C2lua], *parameters)
        if command == 'regular_ray':
            y1 = Stokes_func.func_reg_DOP([C1field, C2field, C1lua, C2lua], *parameters)
        if command == 'regular_raay_simple':
            y1 = Rayleigh_func.func_simple_reg_DOP(gamma=GAMMA, par=parameters[0])

        data['OBS_DATA_' + banda] = POL_OBS
        data['err_OBS_DATA_' + banda] = errPOL_OBS
        data['GAMMA_' + banda] = GAMMA
        data[banda] = y1

    y_b = data['OBS_DATA_B'].to_numpy()
    err_y_b = data['err_OBS_DATA_B'].to_numpy()
    x_b = data['GAMMA_B'].to_numpy()
    y_v = data['OBS_DATA_V'].to_numpy()
    err_y_v = data['err_OBS_DATA_V'].to_numpy()
    x_v = data['GAMMA_V'].to_numpy()
    y_r = data['OBS_DATA_R'].to_numpy()
    err_y_r = data['err_OBS_DATA_R'].to_numpy()
    x_r = data['GAMMA_R'].to_numpy()
    y_i = data['OBS_DATA_I'].to_numpy()
    err_y_i = data['err_OBS_DATA_I'].to_numpy()
    x_i = data['GAMMA_I'].to_numpy()
    func_b = data['B'].to_numpy()
    func_v = data['V'].to_numpy()
    func_r = data['R'].to_numpy()
    func_i = data['I'].to_numpy()

    plt.figure(figsize=(10, 5))
    plt.errorbar(x_b, y_b, yerr=err_y_b, ms=2.0, fmt='o', color='blue')
    plt.errorbar(x_v, y_v, yerr=err_y_v, ms=2.0, fmt='o', color='green')
    plt.errorbar(x_r, y_r, yerr=err_y_r, ms=2.0, fmt='o', color='red')
    plt.errorbar(x_i, y_i, yerr=err_y_i, ms=2.0, fmt='o', color='darkorange')

    ind1 = np.argsort(x_b)
    x_b, func_b = np.asarray(x_b)[ind1], np.asarray(func_b)[ind1]
    ind2 = np.argsort(x_v)
    x_v, func_v = np.asarray(x_v)[ind2], np.asarray(func_v)[ind2]
    ind3 = np.argsort(x_r)
    x_r, func_r = np.asarray(x_r)[ind3], np.asarray(func_r)[ind3]
    ind4 = np.argsort(x_i)
    x_i, func_i = np.asarray(x_i)[ind4], np.asarray(func_i)[ind4]

    plt.plot(x_b, func_b, '-', color='cornflowerblue', markersize=2, label='fit results B band')
    plt.plot(x_v, func_v, '-', color='mediumseagreen', markersize=2, label='fit results V band')
    plt.plot(x_r, func_r, '-', color='indianred', markersize=2, label='fit results R band')
    plt.plot(x_i, func_i, '-', color='orange', markersize=2, label='fit results I band')

    label = 'IMGAGES_all_bands_' + command + '.png'
    plt.ylim(0, 0.8)
    plt.ylabel('Polarization')
    plt.xlabel('Scattering Angle (degrees)')
    plt.legend()
    plt.grid(True)
    plt.savefig(label)
    plt.pause(0.3)
    plt.close()


def fit_base(banda, conditc, method='leastsq', command='regular_multi'):
    barra = tqdm(total=100, desc='Individual fit band ' + banda + ' processing ' + command)

    df = pd.read_csv('data.csv', sep=';')

    df.isnull().sum()
    df.dropna(axis=1)

    np.seterr(invalid='ignore')

    DATA = df[df['BAND'] == banda]

    if conditc is None:
        pass
    if conditc is not None:
        if isinstance(conditc, str):
            DATA = DATA[DATA['CONDITION'] == conditc]
        else:
            DATA = DATA[DATA['CONDITION'].isin(conditc)]

    field = DATA['FIELD'].to_numpy()
    RA = DATA['RA'].to_numpy()
    DEC = DATA['DEC'].to_numpy()
    MED = DATA['MED OBS'].to_numpy()
    # Ival = DATA['I'].to_numpy()
    Qval = DATA['Q'].to_numpy()
    errQval = DATA['error Q'].to_numpy()
    Uval = DATA['U'].to_numpy()
    errUval = DATA['error U '].to_numpy()
    seen = DATA['SEEING'].to_numpy()

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
    # I_OBS = []
    POL_OBS = []
    errPOL_OBS = []
    ALBEDO = []
    AOP = []
    SEEING = []
    BAND = []

    COND1 = []
    COND2 = []

    OBS = []

    barra.update(10)

    for n in range(0, total):
        campo_observado = Field(banda, conditc, field[n], float(RA[n]), float(DEC[n]))
        campo_observado.get_observation(MED[n])

        observation = moon_functions.Moon(MED[n])

        observation.true_sun()
        t_sol = observation.my_sun.theta
        phi_sol = observation.my_sun.phi

        C1lua.append(campo_observado.moon.theta)
        C2lua.append(campo_observado.moon.phi)
        C1sol.append(float(t_sol))
        C2sol.append(float(phi_sol))
        C1field.append(campo_observado.theta)
        C2field.append(campo_observado.phi)
        GAMMA.append(campo_observado.gamma * 180 / np.pi)
        GAMMA_SOL.append(campo_observado.func_gamma(t_sol, phi_sol, units='degrees'))
        Q_OBS.append(float(Qval[n]))  # * float(Ival[n]))
        errQ_OBS.append(float(errQval[n]))  # * float(Ival[n]))
        U_OBS.append(float(Uval[n]))  # * float(Ival[n]))
        errU_OBS.append(float(errUval[n]))  # * float(Ival[n]))
        # I_OBS.append(float(Ival[n]))
        pol = np.sqrt(float(Qval[n]) ** 2 + float(Uval[n]) ** 2)
        POL_OBS.append(pol)
        errPOL_OBS.append(np.sqrt(
            float(Qval[n]) ** 2 * float(errQval[n]) ** 2 + float(Uval[n]) ** 2 * float(
                errUval[n]) ** 2) / pol)
        alb = campo_observado.moon.plot_and_retrive_albedo(campo_observado.wave)
        # campo_observado.moon.plot_tempo()
        ALBEDO.append(alb)
        AOP.append(0.5 * np.arctan(float(Uval[n]) / float(Qval[n])) * 180 / np.pi)
        SEEING.append(seen[n])
        BAND.append(campo_observado.wave)
        OBS.append(campo_observado)
        COND1.append(banda)
        COND2.append(conditc)

        barra.update(int(50/total))

    C1field, C2field = np.asarray(C1field, dtype=np.float64), np.asarray(C2field, dtype=np.float64)
    C1lua, C2lua = np.asarray(C1lua, dtype=np.float64), np.asarray(C2lua, dtype=np.float64)
    POL_OBS, errPOL_OBS = np.asarray(POL_OBS, dtype=np.float64), np.asarray(errPOL_OBS, dtype=np.float64)
    GAMMA, AOP = np.asarray(GAMMA, dtype=np.float64), np.asarray(AOP, dtype=np.float64)
    SEEING, ALBEDO = np.asarray(SEEING, dtype=np.float64), np.asarray(ALBEDO, dtype=np.float64)

    LABEL = 'method_' + str(method) + '_command_' + str(command)

    txname = 'REPORT_individual_POL_' + banda + '_' + LABEL + '.txt'
    model_name = 'MODEL_individual_' + banda + '_' + LABEL + '.sav'

    TXT = open(txname, "w+")

    if command == 'regular_multi':
        model = lmfit.Model(multiple_func.func_reg_DOP)
        model.set_param_hint('L', min=0, max=0.1)
        model.set_param_hint('par', min=0.0, max=4.0)
        p = model.make_params(L=np.random.rand(), par=np.random.rand())
        model.eval(params=p, allvars=[C1field, C2field, C1lua, C2lua])
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], weights=errPOL_OBS,
                           method=method)

        model_fit_report = result.fit_report()
        lmfit.model.save_modelresult(result, model_name)

        result_reg = [result.params['L'].value, result.params['par'].value]
        result_reg = np.asarray(result_reg)

        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = multiple_func.func_reg_DOP([C1field, C2field, C1lua, C2lua], *result_reg)

        rsd = result.eval_uncertainty()
        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er, 'ALBEDO': ALBEDO,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})

        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações

        barra.update(10)

        if isinstance(result.params['L'].stderr, float):
            L_par = round(result.params['L'].stderr, 5)
        else:
            L_par = result.params['L'].stderr
        if isinstance(result.params['par'].stderr, float):
            par_par = round(result.params['par'].stderr, 5)
        else:
            par_par = result.params['par'].stderr

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        label_text = 'fit parameters: L = ' + str(round(result.params['L'].value, 5)) + '$\pm$' + str(
            L_par) + ',   $A$ = ' + str(round(result.params['par'].value, 5)) + '$\pm$' + str(
            par_par) + '\n' + 'chi-square: ' + str(round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_reg, result.chisqr, result.bic, result_data

    if command == 'wave_multi':
        model = lmfit.Model(multiple_func.func_wav_DOP)
        model.set_param_hint('L', min=0)
        model.set_param_hint('c')
        p = model.make_params(L=np.random.rand(), c=np.random.rand())
        model.eval(params=p, allvars=[C1field, C2field, C1lua, C2lua, BAND])
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, BAND], weights=errPOL_OBS,
                           method=method)

        model_fit_report = result.fit_report()
        lmfit.model.save_modelresult(result, model_name)

        result_reg = [result.params['L'].value, result.params['c'].value]
        result_reg = np.asarray(result_reg)

        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = multiple_func.func_wav_DOP([C1field, C2field, C1lua, C2lua, BAND], *result_reg)

        rsd = result.eval_uncertainty()

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er, 'ALBEDO': ALBEDO,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})

        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações

        barra.update(10)

        if isinstance(result.params['L'].stderr, float):
            L_par = round(result.params['L'].stderr, 5)
        else:
            L_par = result.params['L'].stderr
        if isinstance(result.params['c'].stderr, float):
            c_par = round(result.params['c'].stderr, 5)
        else:
            c_par = result.params['c'].stderr

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        label_text = 'fit parameters: L = ' + str(round(result.params['L'].value, 5)) + '$\pm$' + str(
            L_par) + ',   c = ' + str(round(result.params['c'].value, 5)) + '$\pm$' + str(
            c_par) + '\n' + 'chi-square: ' + str(round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)
        barra.update(10)
        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_reg, result.chisqr, result.bic, result_data

    if command == 'simple_wave_multi':
        model = lmfit.Model(multiple_func.func_simple_wav_DOP,
                            independent_vars=['theta_field', 'phi_field', 'theta_lua', 'phi_lua', 'banda'])
        model.set_param_hint('L', min=0)
        model.set_param_hint('c')
        p = model.make_params(L=np.random.rand(), c=np.random.rand())
        result = model.fit(data=POL_OBS, params=p, theta_field=C1field, phi_field=C2field, theta_lua=C1lua,
                           phi_lua=C2lua, banda=BAND, weights=errPOL_OBS, method=method)

        model_fit_report = result.fit_report()
        lmfit.model.save_modelresult(result, model_name)

        result_reg = [result.params['L'].value, result.params['c'].value]
        result_reg = np.asarray(result_reg)

        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = multiple_func.func_simple_wav_DOP(theta_field=C1field, phi_field=C2field, theta_lua=C1lua, phi_lua=C2lua, banda=BAND, L=result_reg[0], c=result_reg[1])

        rsd = result.eval_uncertainty()

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        # {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': GAMMA, 'POL': POL_OBS, 'ERROR POL': errPOL_OBS, 'FIT IND': y1, 'FIT IND UNC': rsd, 'FIT IND DIFF': diff}

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol, 'ERROR POL': new_pol_er, 'ALBEDO': ALBEDO,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})
        # plot_data(result_data, command='simple_wave_multi')

        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        barra.update(10)

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações

        if isinstance(result.params['L'].stderr, float):
            L_par = round(result.params['L'].stderr, 5)
        else:
            L_par = result.params['L'].stderr
        if isinstance(result.params['c'].stderr, float):
            c_par = round(result.params['c'].stderr, 5)
        else:
            c_par = result.params['c'].stderr

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        label_text = 'fit parameters: L = ' + str(round(result.params['L'].value, 5)) + '$\pm$' + str(
            L_par) + ',   c = ' + str(round(result.params['c'].value, 5)) + '$\pm$' + str(
            c_par) + '\n' + 'chi-square: ' + str(round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)
        barra.update(10)
        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_reg, result.chisqr, result.bic, result_data

    if command == 'simple_regular_multi':
        model = lmfit.Model(multiple_func.func_simple_reg_DOP,
                            independent_vars=['theta_field', 'phi_field', 'theta_lua', 'phi_lua'])

        model.set_param_hint('L', min=0, max=1)
        model.set_param_hint('par')
        p = model.make_params(L=np.random.rand(), par=np.random.rand())
        result_simple = model.fit(data=POL_OBS, params=p, theta_field=C1field, phi_field=C2field, theta_lua=C1lua,
                                  phi_lua=C2lua, weights=errPOL_OBS, method='leastsq')

        result_simple_reg = [result_simple.params['L'].value, result_simple.params['par'].value]
        result_simple_reg = np.asarray(result_simple_reg)

        model_fit_report = result_simple.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        lmfit.model.save_modelresult(result_simple, model_name)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = multiple_func.func_simple_reg_DOP(theta_field=C1field, phi_field=C2field, theta_lua=C1lua,
                                               phi_lua=C2lua,
                                               L=result_simple_reg[0], par=result_simple_reg[1])
        rsd = result_simple.eval_uncertainty()

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er, 'ALBEDO': ALBEDO,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})

        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações

        barra.update(10)

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        if isinstance(result_simple.params['L'].stderr, float):
            L_par = round(result_simple.params['L'].stderr, 3)
        else:
            L_par = result_simple.params['L'].stderr
        if isinstance(result_simple.params['par'].stderr, float):
            par_par = round(result_simple.params['par'].stderr, 3)
        else:
            par_par = result_simple.params['par'].stderr
        label_text = 'fit parameters: L = ' + str(round(result_simple.params['L'].value, 3)) + '$\pm$' + str(
            L_par) + ',   $A$ = ' + str(
            round(result_simple.params['par'].value, 3)) + '$\pm$' + str(
            par_par) + '\n' + 'chi-square: ' + str(
            round(result_simple.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
            round(result_simple.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)
        barra.update(10)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_simple_reg, result_simple.chisqr, result_simple.bic, result_data

    if command == 'simple_hor_multi':
        model = lmfit.Model(multiple_func.func_simple_hor,
                            independent_vars=['theta_field', 'phi_field', 'theta_lua', 'phi_lua'])

        model.set_param_hint('L', min=0)
        model.set_param_hint('N', min=0)
        p = model.make_params(L=np.random.rand(), N=np.random.rand())
        result_simple = model.fit(data=POL_OBS, params=p, theta_field=C1field, phi_field=C2field, theta_lua=C1lua,
                                  phi_lua=C2lua, weights=errPOL_OBS, method='leastsq')

        result_simple_reg = [result_simple.params['L'].value, result_simple.params['N'].value]
        result_simple_reg = np.asarray(result_simple_reg)

        model_fit_report = result_simple.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        lmfit.model.save_modelresult(result_simple, model_name)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = multiple_func.func_simple_hor(theta_field=C1field, phi_field=C2field, theta_lua=C1lua,
                                               phi_lua=C2lua,
                                               L=result_simple_reg[0], N=result_simple_reg[1])
        try:
            rsd = result_simple.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er, 'ALBEDO': ALBEDO,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})

        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações

        barra.update(10)

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        if isinstance(result_simple.params['L'].stderr, float):
            L_par = round(result_simple.params['L'].stderr, 5)
        else:
            L_par = result_simple.params['L'].stderr
        if isinstance(result_simple.params['N'].stderr, float):
            N_par = round(result_simple.params['N'].stderr, 5)
        else:
            N_par = result_simple.params['N'].stderr
        label_text = 'fit parameters: L = ' + str(round(result_simple.params['L'].value, 3)) + '$\pm$' + str(
            L_par) + ',   $N$ = ' + str(
            round(result_simple.params['N'].value, 3)) + '$\pm$' + str(
            N_par) + '\n' + 'chi-square: ' + str(
            round(result_simple.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
            round(result_simple.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)
        barra.update(10)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_simple_reg, result_simple.chisqr, result_simple.bic, result_data

    if command == 'simple_seeing_multi':
        model = lmfit.Model(multiple_func.func_simple_seeing,
                            independent_vars=['theta_field', 'phi_field', 'theta_lua', 'phi_lua', 'seeing'])

        model.set_param_hint('L', min=0, max=1)
        model.set_param_hint('k', min=0, max=30)
        model.set_param_hint('d', min=0, max=2)
        p = model.make_params(L=np.random.rand(), k=np.random.rand(), p=np.random.rand())
        result_simple = model.fit(data=POL_OBS, params=p, theta_field=C1field, phi_field=C2field, theta_lua=C1lua,
                                  phi_lua=C2lua, seeing=SEEING, weights=errPOL_OBS, method='leastsq')

        result_simple_reg = [result_simple.params['L'].value, result_simple.params['k'].value, result_simple.params['p'].value]
        result_simple_reg = np.asarray(result_simple_reg)

        model_fit_report = result_simple.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        lmfit.model.save_modelresult(result_simple, model_name)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = multiple_func.func_simple_seeing(theta_field=C1field, phi_field=C2field, theta_lua=C1lua,
                                               phi_lua=C2lua, seeing=SEEING,
                                               L=result_simple_reg[0], k=result_simple_reg[1], p=result_simple_reg[2])
        try:
            rsd = result_simple.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er, 'ALBEDO': ALBEDO,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})

        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações

        barra.update(10)

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        if isinstance(result_simple.params['L'].stderr, float):
            L_par = round(result_simple.params['L'].stderr, 5)
        else:
            L_par = result_simple.params['L'].stderr
        if isinstance(result_simple.params['k'].stderr, float):
            k_par = round(result_simple.params['k'].stderr, 5)
        else:
            k_par = result_simple.params['k'].stderr
        if isinstance(result_simple.params['p'].stderr, float):
            d_par = round(result_simple.params['p'].stderr, 5)
        else:
            d_par = result_simple.params['p'].stderr
        label_text = 'fit parameters: L = ' + str(round(result_simple.params['L'].value, 3)) + '$\pm$' + str(
            L_par) + ',  $k$ = ' + str(round(result_simple.params['k'].value, 3)) + '$\pm$' + str(
            k_par) + ',   $d$ = ' + str(round(result_simple.params['p'].value, 3)) + '$\pm$' + str(
            d_par) + '\n' + 'chi-square: ' + str(
            round(result_simple.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
            round(result_simple.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)
        barra.update(10)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_simple_reg, result_simple.chisqr, result_simple.bic, result_data

    if command == 'simple_mix_multi':
        model = lmfit.Model(multiple_func.func_simple_mix,
                            independent_vars=['theta_field', 'phi_field', 'theta_lua', 'phi_lua', 'seeing'])

        model.set_param_hint('L', min=0, max=1)
        model.set_param_hint('N', min=0)
        model.set_param_hint('k', min=0, max=15)
        model.set_param_hint('d', min=-1, max=1)
        p = model.make_params(L=np.random.rand(), N=np.random.rand(), k=np.random.rand(), d=np.random.rand())
        result_simple = model.fit(data=POL_OBS, params=p, theta_field=C1field, phi_field=C2field, theta_lua=C1lua,
                                  phi_lua=C2lua, seeing=SEEING, weights=errPOL_OBS, method='leastsq')

        result_simple_reg = [result_simple.params['L'].value, result_simple.params['N'].value, result_simple.params['k'].value, result_simple.params['d'].value]
        result_simple_reg = np.asarray(result_simple_reg)

        model_fit_report = result_simple.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        lmfit.model.save_modelresult(result_simple, model_name)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = multiple_func.func_simple_mix(theta_field=C1field, phi_field=C2field, theta_lua=C1lua,
                                               phi_lua=C2lua, seeing=SEEING,
                                               L=result_simple_reg[0], N=result_simple_reg[1], k=result_simple_reg[2], d=result_simple_reg[3])
        try:
            rsd = result_simple.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er, 'ALBEDO': ALBEDO,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})

        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações

        barra.update(10)

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        if isinstance(result_simple.params['L'].stderr, float):
            L_par = round(result_simple.params['L'].stderr, 5)
        else:
            L_par = result_simple.params['L'].stderr
        if isinstance(result_simple.params['N'].stderr, float):
            N_par = round(result_simple.params['N'].stderr, 5)
        else:
            N_par = result_simple.params['N'].stderr
        if isinstance(result_simple.params['k'].stderr, float):
            k_par = round(result_simple.params['k'].stderr, 5)
        else:
            k_par = result_simple.params['k'].stderr
        if isinstance(result_simple.params['d'].stderr, float):
            d_par = round(result_simple.params['d'].stderr, 5)
        else:
            d_par = result_simple.params['d'].stderr
        label_text = 'fit parameters: L = ' + str(round(result_simple.params['L'].value, 3)) + '$\pm$' + str(
            L_par) + ',   $N$ = ' + str(round(result_simple.params['N'].value, 3)) + '$\pm$' + str(
            N_par) + ',   $k$ = ' + str(round(result_simple.params['k'].value, 3)) + '$\pm$' + str(
            k_par) + ',   $d$ = ' + str(round(result_simple.params['d'].value, 3)) + '$\pm$' + str(
            d_par) + '\n' + 'chi-square: ' + str(
            round(result_simple.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
            round(result_simple.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)
        barra.update(10)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_simple_reg, result_simple.chisqr, result_simple.bic, result_data

    if command == 'regular_stokes':
        model = lmfit.Model(Stokes_func.func_reg_DOP)

        model.set_param_hint('par', min=0.0, max=1)
        p = model.make_params(par=np.random.rand())
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee')
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], weights=errPOL_OBS,
                           method='leastsq')

        result_reg = [result.params['par'].value]
        result_reg = np.asarray(result_reg)

        model_fit_report = result.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        lmfit.model.save_modelresult(result, model_name)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = Stokes_func.func_reg_DOP([C1field, C2field, C1lua, C2lua], *result_reg)

        rsd = result.eval_uncertainty()

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er, 'ALBEDO': ALBEDO,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})

        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        label_text = 'fit parameters:    $A$ = ' + str(
            round(result.params['par'].value, 3)) + '$\pm$' + str(
            round(result.params['par'].stderr, 3)) + '\n' + 'chi-square: ' + str(
            round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)
        barra.update(10)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_reg, result.chisqr, result.bic, result_data

    if command == 'hor_stokes':
        model = lmfit.Model(Stokes_func.func_hor)

        model.set_param_hint('N')
        p = model.make_params(N=np.random.rand())
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee')
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], weights=errPOL_OBS,
                           method='leastsq')

        result_reg = [result.params['N'].value]
        result_reg = np.asarray(result_reg)

        model_fit_report = result.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        lmfit.model.save_modelresult(result, model_name)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = Stokes_func.func_hor([C1field, C2field, C1lua, C2lua], *result_reg)

        rsd = result.eval_uncertainty()

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er, 'ALBEDO': ALBEDO,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})

        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        label_text = 'fit parameters:    $N$ = ' + str(
            round(result.params['N'].value, 3)) + '$\pm$' + str(
            round(result.params['N'].stderr, 3)) + '\n' + 'chi-square: ' + str(
            round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)
        barra.update(10)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_reg, result.chisqr, result.bic,  result_data

    if command == 'regular_ray':
        model = lmfit.Model(Rayleigh_func.func_reg_DOP)

        model.set_param_hint('par', min=0.0, max=1)
        p = model.make_params(par=np.random.rand())
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee')
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], weights=errPOL_OBS,
                           method='leastsq')

        result_reg = [result.params['par'].value]
        result_reg = np.asarray(result_reg)

        model_fit_report = result.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        lmfit.model.save_modelresult(result, model_name)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = Rayleigh_func.func_reg_DOP([C1field, C2field, C1lua, C2lua], *result_reg)

        rsd = result.eval_uncertainty()

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er,  'ALBEDO': ALBEDO,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})
        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações
        barra.update(10)

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        label_text = 'fit parameters:    $A$ = ' + str(round(result.params['par'].value, 3)) + '$\pm$' + str(
            round(result.params['par'].stderr, 3)) + '\n' + 'chi-square: ' + str(
            round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_reg, result.chisqr, result.bic, result_data

    if command == 'hor_simple_ray':
        model = lmfit.Model(Rayleigh_func.func_simple_hor, independent_vars=['theta_lua', 'gamma'])

        model.set_param_hint('N')
        p = model.make_params(N=np.random.rand())
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee')
        result = model.fit(data=POL_OBS, params=p, theta_lua=C1lua, gamma=GAMMA, weights=errPOL_OBS,
                           method='leastsq')

        result_reg = [result.params['N'].value]
        result_reg = np.asarray(result_reg)

        model_fit_report = result.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        lmfit.model.save_modelresult(result, model_name)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = Rayleigh_func.func_simple_hor(theta_lua=C1lua, gamma=GAMMA, N=result_reg[0])

        rsd = result.eval_uncertainty()

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})
        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações
        barra.update(10)

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        if isinstance(result.params['N'].stderr, float):
            N_par = round(result.params['N'].stderr, 3)
        else:
            N_par = result.params['N'].stderr
        label_text = 'fit parameters:    $N$ = ' + str(round(result.params['N'].value, 3)) + '$\pm$' + str(
            N_par) + '\n' + 'chi-square: ' + str(
            round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_reg, result.chisqr, result.bic, result_data

    if command == 'seeing_simple_ray':
        model = lmfit.Model(Rayleigh_func.func_simple_seeing, independent_vars=['seeing', 'gamma'])

        model.set_param_hint('k')
        model.set_param_hint('d')
        p = model.make_params(k=np.random.rand(), d=np.random.rand())
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee')
        result = model.fit(data=POL_OBS, params=p, seeing=SEEING, gamma=GAMMA, weights=errPOL_OBS,
                           method='leastsq')

        result_reg = [result.params['k'].value, result.params['d'].value]
        result_reg = np.asarray(result_reg)

        model_fit_report = result.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        lmfit.model.save_modelresult(result, model_name)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = Rayleigh_func.func_simple_seeing(seeing=SEEING, gamma=GAMMA, k=result_reg[0], d=result_reg[1])

        rsd = result.eval_uncertainty()

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})
        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações
        barra.update(10)

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        if isinstance(result.params['k'].stderr, float):
            k_par = round(result.params['k'].stderr, 5)
        else:
            k_par = result.params['k'].stderr
        if isinstance(result.params['d'].stderr, float):
            d_par = round(result.params['d'].stderr, 5)
        else:
            d_par = result.params['d'].stderr
        label_text = 'fit parameters:     $k$ = ' + str(round(result.params['k'].value, 3)) + '$\pm$' + str(
            k_par) + ',   $d$ = ' + str(round(result.params['d'].value, 3)) + '$\pm$' + str(
            d_par) + '\n' + 'chi-square: ' + str(
            round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_reg, result.chisqr, result.bic, result_data

    if command == 'regular_ray_simple':
        model = lmfit.Model(Rayleigh_func.func_simple_reg_DOP, independent_vars=['gamma'])

        model.set_param_hint('par', min=0.0, max=4.0)
        p = model.make_params(par=np.random.rand())
        result_simple = model.fit(data=POL_OBS, params=p, gamma=GAMMA, weights=errPOL_OBS, method='leastsq')

        result_simple_reg = [result_simple.params['par'].value]
        result_simple_reg = np.asarray(result_simple_reg)

        model_fit_report = result_simple.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        lmfit.model.save_modelresult(result_simple, model_name)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = Rayleigh_func.func_simple_reg_DOP(gamma=GAMMA, par=result_simple_reg[0])

        rsd = result_simple.eval_uncertainty()

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er, 'ALBEDO': ALBEDO,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})
        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações

        barra.update(10)

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        label_text = 'fit parameters:   $A$ = ' + str(
            round(result_simple.params['par'].value, 3)) + '$\pm$' + str(
            round(result_simple.params['par'].stderr, 3)) + '\n' + 'chi-square: ' + str(
            round(result_simple.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
            round(result_simple.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        barra.update(5)

        return result_simple_reg, result_simple.chisqr, result_simple.bic, result_data

    if command == 'dep_stokes':
        model = lmfit.Model(Stokes_func.func_dep_DOP)

        model.set_param_hint('P')
        p = model.make_params(P=np.random.rand())
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee')
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], weights=errPOL_OBS,
                           method='leastsq')

        result_reg = [result.params['P'].value]
        result_reg = np.asarray(result_reg)

        model_fit_report = result.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        lmfit.model.save_modelresult(result, model_name)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = Stokes_func.func_dep_DOP([C1field, C2field, C1lua, C2lua], *result_reg)

        try:
            rsd = result.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er, 'ALBEDO': ALBEDO,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})

        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        label_text = 'fit parameters:    $\u03C1$ = ' + str(
            round(result.params['P'].value, 3)) + '$\pm$' + str(
            round(result.params['P'].stderr, 3)) + '\n' + 'chi-square: ' + str(
            round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)
        barra.update(10)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_reg, result.chisqr, result.bic, result_data

    if command == 'dep_ray':
        model = lmfit.Model(Rayleigh_func.func_dep_DOP)

        model.set_param_hint('P', min=0.0, max=1)
        p = model.make_params(P=np.random.rand())
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee')
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], weights=errPOL_OBS,
                           method='leastsq')

        result_reg = [result.params['P'].value]
        result_reg = np.asarray(result_reg)

        model_fit_report = result.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        lmfit.model.save_modelresult(result, model_name)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = Rayleigh_func.func_dep_DOP([C1field, C2field, C1lua, C2lua], *result_reg)

        rsd = result.eval_uncertainty()

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er, 'ALBEDO': ALBEDO,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})
        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações
        barra.update(10)

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        label_text = 'fit parameters:    $\u03C1$ = ' + str(round(result.params['P'].value, 3)) + '$\pm$' + str(
            round(result.params['P'].stderr, 3)) + '\n' + 'chi-square: ' + str(
            round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_reg, result.chisqr, result.bic, result_data

    if command == 'dep_ray_simpleNo':
        model = lmfit.Model(Rayleigh_func.func_simple_dep_DOP, independent_vars=['gamma'])

        model.set_param_hint('P', min=0.0, max=1.0)
        p = model.make_params(P=np.random.rand())
        result_simple = model.fit(data=POL_OBS, params=p, gamma=GAMMA, weights=errPOL_OBS, method='leastsq')

        result_simple_reg = [result_simple.params['P'].value]
        result_simple_reg = np.asarray(result_simple_reg)

        model_fit_report = result_simple.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        lmfit.model.save_modelresult(result_simple, model_name)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = Rayleigh_func.func_simple_dep_DOP(gamma=GAMMA, P=result_simple_reg[0])

        rsd = result_simple.eval_uncertainty()

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er, 'ALBEDO': ALBEDO,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})
        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações

        barra.update(10)

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        label_text = 'fit parameters:   $\u03C1$ = ' + str(
            round(result_simple.params['P'].value, 3)) + '$\pm$' + str(
            round(result_simple.params['P'].stderr, 3)) + '\n' + 'chi-square: ' + str(
            round(result_simple.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
            round(result_simple.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_simple_reg, result_simple.chisqr, result_simple.bic, result_data

    if command == 'mix_ray_simple':
        model = lmfit.Model(Rayleigh_func.func_simple_mix, independent_vars=['gamma', 'theta_lua', 'seeing'])
        model.set_param_hint('N', max=100)
        model.set_param_hint('k', min=0, max=15)
        model.set_param_hint('d', min=-1, max=1)
        p = model.make_params(N=np.random.rand(), k=np.random.rand(), d=np.random.rand())
        result_simple = model.fit(data=POL_OBS, params=p, gamma=GAMMA, theta_lua=C1lua, seeing=SEEING, weights=errPOL_OBS, method='leastsq')

        result_simple_reg = [result_simple.params['N'].value, result_simple.params['k'].value, result_simple.params['d'].value]
        result_simple_reg = np.asarray(result_simple_reg)

        model_fit_report = result_simple.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        lmfit.model.save_modelresult(result_simple, model_name)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = Rayleigh_func.func_simple_mix(gamma=GAMMA, theta_lua=C1lua, seeing=SEEING, N=result_simple_reg[0], k=result_simple_reg[1], d=result_simple_reg[2])

        try:
            rsd = result_simple.eval_uncertainty()
        except ZeroDivisionError:
            rsd = np.zeros(len(POL_OBS))

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er, 'ALBEDO': ALBEDO,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})
        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações

        barra.update(10)

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        if isinstance(result_simple.params['N'].stderr, float):
            N_par = round(result_simple.params['N'].stderr, 3)
        else:
            N_par = result_simple.params['N'].stderr
        if isinstance(result_simple.params['k'].stderr, float):
            k_par = round(result_simple.params['k'].stderr, 3)
        else:
            k_par = result_simple.params['k'].stderr
        if isinstance(result_simple.params['d'].stderr, float):
            d_par = round(result_simple.params['d'].stderr, 3)
        else:
            d_par = result_simple.params['d'].stderr
        label_text = 'fit parameters:   $N$ = ' + str(
            round(result_simple.params['N'].value, 3)) + '$\pm$' + str(
            N_par) + '$k$ = ' + str(
            round(result_simple.params['k'].value, 3)) + '$\pm$' + str(
            k_par) + '$d$ = ' + str(round(result_simple.params['d'].value, 3)) + '$\pm$' + str(
            d_par) + '\n' + 'chi-square: ' + str(
            round(result_simple.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result_simple.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
            round(result_simple.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_simple_reg, result_simple.chisqr, result_simple.bic, result_data

    if command == 'regular_aop':
        model = lmfit.Model(Stokes_func.func_reg_AOP)
        model.set_param_hint('par', min=0.0, max=1.0)
        p = model.make_params(par=np.random.rand())  # , beta1=np.random.rand(), beta2=np.random.rand())
        model.eval(params=p, allvars=[C1field, C2field, C1lua, C2lua])
        result = model.fit(data=AOP, params=p, allvars=[C1field, C2field, C1lua, C2lua], method=method)

        lmfit.model.save_modelresult(result, model_name)

        model_fit_report = result.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        result_reg = [result.params['par'].value]  # , result.params['beta1'].value, result.params['beta2'].value]  # , result.params['N'].value]
        result_reg = np.asarray(result_reg)
        barra.update(10)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = Stokes_func.func_reg_AOP([C1field, C2field, C1lua, C2lua], *result_reg)

        rsd = result.eval_uncertainty()

        diff = []
        for i in range(0, len(AOP)):
            diff.append(AOP[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_aop = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], np.asarray(AOP)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'AOP': new_aop,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})
        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.plot(GAMMA, AOP, 'o', color='black')
        # plt.errorbar(GAMMA, AOP, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')

        if isinstance(result.params['par'].stderr, float):
            par_par = round(result.params['par'].stderr, 3)
        else:
            par_par = result.params['par'].stderr

        '''
        if isinstance(result.params['beta1'].stderr, float):
            beta1_par = round(result.params['beta1'].stderr, 3)
        else:
            beta1_par = result.params['beta1'].stderr
        if isinstance(result.params['beta2'].stderr, float):
            beta2_par = round(result.params['beta2'].stderr, 3)
        else:
            beta2_par = result.params['beta2'].stderr '''

        plt.ylabel('Angle of Polarization')
        label_text = 'fit parameters:   $A$ = ' + str(round(result.params['par'].value, 3)) + '$\pm$' + str(
            par_par) + '\n' + 'chi-square: ' + str(round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
        # + '$beta_{1}$ = ' + str(round(result.params['beta1'].value, 3)) + ',  $beta_{2}$ = ' + str(round(result.params['beta2'].value, 3))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.stem(new_x1, er_diff, use_line_collection=True)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_reg, result.chisqr, result.bic, result_data

    if command == 'regular_ray_simple_aop':
        model = lmfit.Model(Rayleigh_func.func_simple_reg_AOP, independent_vars=['phi_obs', 'theta_obs', 'phi_lua', 'theta_lua'])
        model.set_param_hint('par', min=0.0, max=1.0)
        p = model.make_params(par=np.random.rand())  # , beta1=np.random.rand(), beta2=np.random.rand())

        result = model.fit(data=AOP, params=p, phi_obs=C2field, theta_obs=C1field, theta_lua=C1lua, phi_lua=C2lua, method=method)

        lmfit.model.save_modelresult(result, model_name)

        model_fit_report = result.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        result_reg = [result.params['par'].value]  # , result.params['beta1'].value, result.params['beta2'].value]  # , result.params['N'].value]
        result_reg = np.asarray(result_reg)
        barra.update(10)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = Rayleigh_func.func_simple_reg_AOP(phi_obs=C2field, theta_obs=C1field, theta_lua=C1lua, phi_lua=C2lua, par=result_reg[0])

        rsd = result.eval_uncertainty()

        diff = []
        for i in range(0, len(AOP)):
            diff.append(AOP[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_aop = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], np.asarray(AOP)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'AOP': new_aop,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})
        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.plot(GAMMA, AOP, 'o', color='black')
        # plt.errorbar(GAMMA, AOP, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')

        if isinstance(result.params['par'].stderr, float):
            par_par = round(result.params['par'].stderr, 3)
        else:
            par_par = result.params['par'].stderr

        plt.ylabel('Angle of Polarization')
        label_text = 'fit parameters:   $A$ = ' + str(round(result.params['par'].value, 3)) + '$\pm$' + str(
            par_par) + '\n' + 'chi-square: ' + str(round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
        # + '$beta_{1}$ = ' + str(round(result.params['beta1'].value, 3)) + ',  $beta_{2}$ = ' + str(round(result.params['beta2'].value, 3))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.stem(new_x1, er_diff, use_line_collection=True)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_reg, result.chisqr, result.bic, result_data

    if command == 'regular_q':
        model = lmfit.Model(Stokes_func.func_reg_Q)
        model.set_param_hint('par', min=0.0, max=1.0)
        p = model.make_params(par=np.random.rand())  # , beta1=np.random.rand(), beta2=np.random.rand())
        model.eval(params=p, allvars=[C1field, C2field, C1lua, C2lua])
        result = model.fit(data=Q_OBS, weights=errQ_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua],
                           method=method)

        lmfit.model.save_modelresult(result, model_name)
        barra.update(10)

        model_fit_report = result.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        result_reg = [result.params['par'].value]  # , result.params['beta1'].value, result.params['beta2'].value]  # , result.params['N'].value]
        result_reg = np.asarray(result_reg)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = Stokes_func.func_reg_Q([C1field, C2field, C1lua, C2lua], *result_reg)

        rsd = result.eval_uncertainty()

        diff = []
        for i in range(0, len(Q_OBS)):
            diff.append(Q_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_q, new_q_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                           np.asarray(Q_OBS)[w], np.asarray(errQ_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'Q': new_q, 'ERROR Q': new_q_er,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})
        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, Q_OBS, yerr=errQ_OBS, ms=2.0, fmt='o', color='black')

        if isinstance(result.params['par'].stderr, float):
            par_par = round(result.params['par'].stderr, 3)
        else:
            par_par = result.params['par'].stderr
        '''
        if isinstance(result.params['beta1'].stderr, float):
            beta1_par = round(result.params['beta1'].stderr, 3)
        else:
            beta1_par = result.params['beta1'].stderr
        if isinstance(result.params['beta2'].stderr, float):
            beta2_par = round(result.params['beta2'].stderr, 3)
        else:
            beta2_par = result.params['beta2'].stderr '''

        plt.ylabel('Q parameter')
        label_text = 'fit parameters:  $A$ = ' + str(round(result.params['par'].value, 3)) + '$\pm$' + str(
            par_par) + '\n' + 'chi-square: ' + str(round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_q_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_reg, result.chisqr, result.bic, result_data

    if command == 'regular_u':
        model = lmfit.Model(Stokes_func.func_reg_U)
        model.set_param_hint('par', min=0.0, max=1.0)
        p = model.make_params(par=np.random.rand())  # , beta1=np.random.rand(), beta2=np.random.rand())
        model.eval(params=p, allvars=[C1field, C2field, C1lua, C2lua])
        result = model.fit(data=U_OBS, weights=errU_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua],
                           method=method)

        lmfit.model.save_modelresult(result, model_name)
        barra.update(10)

        model_fit_report = result.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        result_reg = [result.params['par'].value]  # , result.params['beta1'].value, result.params['beta2'].value]  # , result.params['N'].value]
        result_reg = np.asarray(result_reg)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = Stokes_func.func_reg_U([C1field, C2field, C1lua, C2lua], *result_reg)

        rsd = result.eval_uncertainty()

        diff = []
        for i in range(0, len(U_OBS)):
            diff.append(U_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_u, new_u_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                           np.asarray(U_OBS)[w], np.asarray(errU_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'U': new_u, 'ERROR U': new_u_er,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})
        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, U_OBS, yerr=errU_OBS, ms=2.0, fmt='o', color='black')

        if isinstance(result.params['par'].stderr, float):
            par_par = round(result.params['par'].stderr, 3)
        else:
            par_par = result.params['par'].stderr

        '''
        if isinstance(result.params['beta1'].stderr, float):
            beta1_par = round(result.params['beta1'].stderr, 3)
        else:
            beta1_par = result.params['beta1'].stderr
        if isinstance(result.params['beta2'].stderr, float):
            beta2_par = round(result.params['beta2'].stderr, 3)
        else:
            beta2_par = result.params['beta2'].stderr '''

        plt.ylabel('U parameter')
        label_text = 'fit parameters:  $A$ = ' + str(round(result.params['par'].value, 3)) + '$\pm$' + str(
            par_par) + '\n' + 'chi-square: ' + str(round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_u_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_reg, result.chisqr, result.bic, result_data

    if command == 'wave_ray':
        model = lmfit.Model(Rayleigh_func.func_wav_DOP)

        model.set_param_hint('c')
        p = model.make_params(c=np.random.rand())
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee')
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, BAND], weights=errPOL_OBS, method='leastsq')

        result_reg = [result.params['c'].value]
        result_reg = np.asarray(result_reg)
        barra.update(10)

        model_fit_report = result.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        lmfit.model.save_modelresult(result, model_name)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = Rayleigh_func.func_wav_DOP([C1field, C2field, C1lua, C2lua, BAND], *result_reg)

        rsd = result.eval_uncertainty()

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})
        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        if isinstance(result.params['c'].stderr, float):
            c_par = round(result.params['c'].stderr, 3)
        else:
            c_par = result.params['c'].stderr
        label_text = 'fit parameters: ' + '  $c$ = ' + str(round(result.params['c'].value, 3)) + '$\pm$' + str(
            c_par) + '\n' + 'chi-square: ' + str(
            round(result.chisqr, 10)) + ',   reduced chi-square: ' + str(
            round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
            round(result.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_reg, result.chisqr, result.bic, result_data

    if command == 'simple_wave_ray':
        model = lmfit.Model(Rayleigh_func.func_simple_wav_DOP, independent_vars=['gamma', 'wavel'])
        model.set_param_hint('c')
        p = model.make_params(c=np.random.rand())
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee')
        result = model.fit(data=POL_OBS, params=p, gamma=GAMMA, wavel=BAND, weights=errPOL_OBS, method='leastsq')

        result_reg = [result.params['c'].value]
        result_reg = np.asarray(result_reg)
        barra.update(10)

        model_fit_report = result.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        lmfit.model.save_modelresult(result, model_name)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = Rayleigh_func.func_simple_wav_DOP(gamma=GAMMA, wavel=BAND, c=result_reg[0])

        rsd = result.eval_uncertainty()

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})
        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        if isinstance(result.params['c'].stderr, float):
            c_par = round(result.params['c'].stderr, 3)
        else:
            c_par = result.params['c'].stderr
        label_text = 'fit parameters: ' + '  $c$ = ' + str(round(result.params['c'].value, 3)) + '$\pm$' + str(
            c_par) + '\n' + 'chi-square: ' + str(
            round(result.chisqr, 10)) + ',   reduced chi-square: ' + str(
            round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
            round(result.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_reg, result.chisqr, result.bic, result_data

    if command == 'wave_stokes':
        model = lmfit.Model(Stokes_func.func_wav_DOP)

        model.set_param_hint('c')
        p = model.make_params(c=np.random.rand())
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee')
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua, BAND], weights=errPOL_OBS, method='leastsq')

        result_reg = [result.params['c'].value]
        result_reg = np.asarray(result_reg)
        barra.update(10)

        model_fit_report = result.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        lmfit.model.save_modelresult(result, model_name)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = Stokes_func.func_wav_DOP([C1field, C2field, C1lua, C2lua, BAND], *result_reg)

        rsd = result.eval_uncertainty()

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações

        barra.update(10)

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        if isinstance(result.params['c'].stderr, float):
            c_par = round(result.params['c'].stderr, 3)
        else:
            c_par = result.params['c'].stderr
        label_text = 'fit parameters: ' + '  $c$ = ' + str(round(result.params['c'].value, 3)) + '$\pm$' + str(
            c_par) + '\n' + 'chi-square: ' + str(
            round(result.chisqr, 10)) + ',   reduced chi-square: ' + str(
            round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(
            round(result.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_reg, result.chisqr, result.bic, result_data

    if command == 'regular_mie':
        model = lmfit.Model(Mie_func.func_reg_DOP)

        model.set_param_hint('A', min=0, max=2)
        model.set_param_hint('x', min=0, max=1)
        model.set_param_hint('m_part', min=0, max=4)
        p = model.make_params(A=np.random.rand(), x=np.random.rand(), m_part=np.random.rand())
        # result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], method='emcee')
        result = model.fit(data=POL_OBS, params=p, allvars=[C1field, C2field, C1lua, C2lua], weights=errPOL_OBS,
                           method='leastsq')

        result_reg = [result.params['A'].value, result.params['x'].value, result.params['m_part'].value]
        result_reg = np.asarray(result_reg)

        model_fit_report = result.fit_report()
        TXT.write('[independent variables]')
        TXT.write(str(model.independent_vars))
        TXT.write('\n \n')
        TXT.write(model_fit_report)
        TXT.write('\n \n')

        lmfit.model.save_modelresult(result, model_name)

        fig_x = plt.figure(figsize=(10, 5))
        fig_x.add_axes((.1, .3, .8, .6))

        x1 = GAMMA
        y1 = Mie_func.func_reg_DOP([C1field, C2field, C1lua, C2lua], *result_reg)

        rsd = result.eval_uncertainty()

        diff = []
        for i in range(0, len(POL_OBS)):
            diff.append(POL_OBS[i] - y1[i])

        w = np.argsort(x1)
        new_x1, new_y1, er_diff, er_rsd, new_pol, new_pol_er = np.asarray(x1)[w], np.asarray(y1)[w], np.asarray(diff)[
            w], np.asarray(rsd)[w], \
                                                               np.asarray(POL_OBS)[w], np.asarray(errPOL_OBS)[w]
        result_data = pd.DataFrame(
            {'FIELD': field, 'BAND': COND1, 'CONDITION': COND2, 'GAMMA': new_x1, 'POL': new_pol,
             'ERROR POL': new_pol_er, 'ALBEDO': ALBEDO,
             'FIT IND': new_y1, 'FIT IND UNC': er_rsd, 'FIT IND DIFF': er_diff})

        barra.update(10)

        plt.plot(new_x1, new_y1, 'r-', markersize=2)  # GAMMA e resultados fit
        plt.plot(new_x1, new_y1, 'ro', markersize=4)  # GAMMA e resultados fit

        g1 = np.add(new_y1, er_rsd)
        g2 = np.subtract(new_y1, er_rsd)
        plt.fill_between(new_x1, g2, g1, where=(g2 < g1), interpolate=True, color='lavender')

        plt.errorbar(GAMMA, POL_OBS, yerr=errPOL_OBS, ms=2.0, fmt='o', color='black')  # GAMMA e observações

        plt.ylim(0, 0.8)
        plt.ylabel('Polarization')
        label_text = 'fit parameters:    $A$ = ' + str(
            round(result.params['A'].value, 3)) + '$\pm$' + str(
            round(result.params['A'].stderr, 3)) + ' ' + str(
            round(result.params['x'].value, 3)) + '$\pm$' + str(
            round(result.params['x'].stderr, 3)) + ' ' + str(
            round(result.params['m_part'].value, 3)) + '$\pm$' + str(
            round(result.params['m_part'].stderr, 3)) + '\n' + 'chi-square: ' + str(
            round(result.chisqr, 10)) + ',  reduced chi-square: ' + str(
            round(result.redchi, 10)) + '\n' + 'Bayesian Information Criterion: ' + str(round(result.bic, 2))
        plt.annotate(label_text, xy=(0.1, 0.2), xycoords='axes fraction', xytext=(0.1, 0.9),
                     textcoords='axes fraction',
                     horizontalalignment='left', verticalalignment='center', bbox=dict(boxstyle="round", fc="w"))
        plt.grid(True)
        barra.update(10)

        fig_x.add_axes((.1, .1, .8, .2))

        plt.errorbar(new_x1, er_diff, yerr=new_pol_er)
        plt.plot(new_x1, er_rsd, 'g-', markersize=2)
        barra.update(10)

        plt.xlabel('Scattering Angle (degrees)')
        plt.ylabel('Residual data')
        plt.grid(True)
        plt.savefig('IMAGE_individual_' + str(banda) + '_' + LABEL + '.png')
        barra.update(10)
        plt.pause(0.3)
        plt.close()
        barra.close()

        return result_reg, result.chisqr, result.bic, result_data

    else:
        print('Wrong command input.')
        pass


def comparing_methods(df_multi, df_stokes, df_ray):
    print('Comparing methods')
    # print(df_multi)
    # for col in df_multi.columns:
    #     print(col)

    # OPTIONS: All TOGETHER, SEPARATE FOR BANDS, JUST ONE BAND

    band_list = df_multi['BAND'].unique()
    colors_points = ['blue', 'green', 'red', 'darkorange']
    colors_lines = ['cornflowerblue', 'mediumseagreen', 'indianred', 'orange']
    colors_errors = ['lavender', 'beige', 'mistyrose', 'antiquewhite']

    y_list = ['HOR POL', 'SEEING POL', 'WAVE POL', 'ALL POL']
    yerror_list = ['HOR UNC', 'SEEING UNC', 'WAVE UNC', 'ALL UNC']

    fig = plt.figure(figsize=(20, 10))
    st = fig.suptitle("Separate for bands", fontsize=18, y=0.95)
    s = 0

    for n in range(0, len(band_list)):
        data_multi = df_multi[df_multi['BAND'] == band_list[n]]
        data_stokes = df_stokes[df_stokes['BAND'] == band_list[n]]
        data_ray = df_ray[df_ray['BAND'] == band_list[n]]

        for z in range(0, len(y_list)):
            plt.subplot(4, 4, s + 1)
            plt.grid(True)
            plt.ylim(0, 0.8)
            plt.errorbar(data_multi['GAMMA'].to_numpy(), data_multi['POL OBS'].to_numpy(), ms=1.0,
                         yerr=data_multi['POL OBS error'], fmt='o', color=colors_points[n])

            # data_multi.plot(x='GAMMA', y=y_list[z], style='-', color=colors_lines[n])
            w = np.argsort(data_multi['GAMMA'].to_numpy())
            a, b, c, d, e, f, g = np.asarray(data_multi['GAMMA'].to_numpy())[w], \
                                  np.asarray(data_multi[y_list[z]].to_numpy())[w], \
                                  np.asarray(data_multi[yerror_list[z]].to_numpy())[w], \
                                  np.asarray(data_stokes[y_list[z]].to_numpy())[w], \
                                  np.asarray(data_stokes[yerror_list[z]].to_numpy())[w], \
                                  np.asarray(data_ray[y_list[z]].to_numpy())[w], \
                                  np.asarray(data_ray[yerror_list[z]].to_numpy())[w]
            h1 = np.add(b, c)
            h2 = np.subtract(b, c)
            plt.fill_between(a, h2, h1, where=(h2 < h1), interpolate=True, color=colors_errors[n])
            plt.plot(a, b, '-', color=colors_lines[n])
            h1 = np.add(d, e)
            h2 = np.subtract(d, e)
            plt.fill_between(a, h2, h1, where=(h2 < h1), interpolate=True, color=colors_errors[n])
            plt.plot(a, d, '--', color=colors_lines[n])
            h1 = np.add(f, g)
            h2 = np.subtract(f, g)
            plt.fill_between(a, h2, h1, where=(h2 < h1), interpolate=True, color=colors_errors[n])
            plt.plot(a, f, ':', color=colors_lines[n])
            plt.xlabel('Scattering Angle (degrees)')
            plt.ylabel('Polarization')
            plt.title(y_list[z])
            s += 1

    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    plt.savefig('FIGURE_separate_bands_comparing_methods.png')
    plt.pause(3)
    plt.close()

    fig = plt.figure(figsize=(20, 10))
    st = fig.suptitle("All together", fontsize=18, y=0.95)
    s = 0

    for z in range(0, len(y_list)):
        plt.subplot(3, 2, s + 1)
        plt.ylim(0, 0.8)
        plt.grid(True)

        for n in range(0, len(band_list)):
            data_multi = df_multi[df_multi['BAND'] == band_list[n]]
            data_stokes = df_stokes[df_stokes['BAND'] == band_list[n]]
            data_ray = df_ray[df_ray['BAND'] == band_list[n]]

            plt.errorbar(data_multi['GAMMA'].to_numpy(), data_multi['POL OBS'].to_numpy(), ms=1.0,
                         yerr=data_multi['POL OBS error'], fmt='o', color=colors_points[n])

            # data_multi.plot(x='GAMMA', y=y_list[z], style='-', color=colors_lines[n])
            w = np.argsort(data_multi['GAMMA'].to_numpy())
            a, b, c, d, e, f, g = np.asarray(data_multi['GAMMA'].to_numpy())[w], \
                                  np.asarray(data_multi[y_list[z]].to_numpy())[w], \
                                  np.asarray(data_multi[yerror_list[z]].to_numpy())[w], \
                                  np.asarray(data_stokes[y_list[z]].to_numpy())[w], \
                                  np.asarray(data_stokes[yerror_list[z]].to_numpy())[w], \
                                  np.asarray(data_ray[y_list[z]].to_numpy())[w], \
                                  np.asarray(data_ray[yerror_list[z]].to_numpy())[w]
            h1 = np.add(b, c)
            h2 = np.subtract(b, c)
            plt.fill_between(a, h2, h1, where=(h2 < h1), interpolate=True, color=colors_errors[n])
            plt.plot(a, b, '-', color=colors_lines[n])
            h1 = np.add(d, e)
            h2 = np.subtract(d, e)
            plt.fill_between(a, h2, h1, where=(h2 < h1), interpolate=True, color=colors_errors[n])
            plt.plot(a, d, '--', color=colors_lines[n])
            h1 = np.add(f, g)
            h2 = np.subtract(f, g)
            plt.fill_between(a, h2, h1, where=(h2 < h1), interpolate=True, color=colors_errors[n])
            plt.plot(a, f, ':', color=colors_lines[n])
            plt.xlabel('Scattering Angle (degrees)')
            plt.ylabel('Polarization')
            plt.title(y_list[z])
        s += 1

    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    plt.savefig('FIGURE_all_together_comparing_methods.png')
    plt.pause(3)
    plt.close()

    v_list = ['SIM HOR POL', 'SIM SEEING POL', 'SIM WAVE POL', 'SIM ALL POL']
    verror_list = ['SIM HOR UNC', 'SIM SEEING UNC', 'SIM WAVE UNC', 'SIM ALL UNC']

    fig = plt.figure(figsize=(20, 10))
    st = fig.suptitle("Separate for bands - Simple models", fontsize=18, y=0.95)
    s = 0

    for n in range(0, len(band_list)):
        data_multi = df_multi[df_multi['BAND'] == band_list[n]]
        data_ray = df_ray[df_ray['BAND'] == band_list[n]]

        for z in range(0, len(v_list)):
            plt.subplot(4, 4, s + 1)
            plt.ylim(0, 0.8)
            plt.grid(True)
            plt.errorbar(data_multi['GAMMA'].to_numpy(), data_multi['POL OBS'].to_numpy(), ms=1.0,
                         yerr=data_multi['POL OBS error'], fmt='o', color=colors_points[n])

            # data_multi.plot(x='GAMMA', y=y_list[z], style='-', color=colors_lines[n])
            w = np.argsort(data_multi['GAMMA'].to_numpy())
            a, b, c, f, g = np.asarray(data_multi['GAMMA'].to_numpy())[w], \
                            np.asarray(data_multi[v_list[z]].to_numpy())[w], \
                            np.asarray(data_multi[verror_list[z]].to_numpy())[w], \
                            np.asarray(data_ray[v_list[z]].to_numpy())[w], \
                            np.asarray(data_ray[verror_list[z]].to_numpy())[w]
            h1 = np.add(b, c)
            h2 = np.subtract(b, c)
            plt.fill_between(a, h2, h1, where=(h2 < h1), interpolate=True, color=colors_errors[n])
            plt.plot(a, b, '-', color=colors_lines[n])
            h1 = np.add(f, g)
            h2 = np.subtract(f, g)
            plt.fill_between(a, h2, h1, where=(h2 < h1), interpolate=True, color=colors_errors[n])
            plt.plot(a, f, ':', color=colors_lines[n])
            plt.xlabel('Scattering Angle (degrees)')
            plt.ylabel('Polarization')
            plt.title(y_list[z])
            s += 1

    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    plt.savefig('FIGURE_separate_bands_comparing_methods_simple.png')
    plt.pause(3)
    plt.close()

    fig = plt.figure(figsize=(20, 10))
    st = fig.suptitle("All together - Simple models", fontsize=18, y=0.95)
    s = 0

    for z in range(0, len(v_list)):
        plt.subplot(3, 2, s + 1)
        plt.ylim(0, 0.8)
        plt.grid(True)

        for n in range(0, len(band_list)):
            data_multi = df_multi[df_multi['BAND'] == band_list[n]]
            data_ray = df_ray[df_ray['BAND'] == band_list[n]]

            plt.errorbar(data_multi['GAMMA'].to_numpy(), data_multi['POL OBS'].to_numpy(), ms=1.0,
                         yerr=data_multi['POL OBS error'], fmt='o', color=colors_points[n])

            # data_multi.plot(x='GAMMA', y=y_list[z], style='-', color=colors_lines[n])
            w = np.argsort(data_multi['GAMMA'].to_numpy())
            a, b, c, f, g = np.asarray(data_multi['GAMMA'].to_numpy())[w], \
                            np.asarray(data_multi[v_list[z]].to_numpy())[w], \
                            np.asarray(data_multi[verror_list[z]].to_numpy())[w], \
                            np.asarray(data_ray[v_list[z]].to_numpy())[w], \
                            np.asarray(data_ray[verror_list[z]].to_numpy())[w]
            h1 = np.add(b, c)
            h2 = np.subtract(b, c)
            plt.fill_between(a, h2, h1, where=(h2 < h1), interpolate=True, color=colors_errors[n])
            plt.plot(a, b, '-', color=colors_lines[n])
            h1 = np.add(f, g)
            h2 = np.subtract(f, g)
            plt.fill_between(a, h2, h1, where=(h2 < h1), interpolate=True, color=colors_errors[n])
            plt.plot(a, f, ':', color=colors_lines[n])
            plt.xlabel('Scattering Angle (degrees)')
            plt.ylabel('Polarization')
            plt.title(y_list[z])
        s += 1

    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    plt.savefig('FIGURE_all_together_comparing_methods_simple.png')
    plt.pause(3)
    plt.close()


class Field:
    def __init__(self, band, conditions, name, ra, dec):
        self.wave = None
        self.alt = None
        self.az = None
        self.theta = None
        self.phi = None
        self.gamma = None
        self.time = None
        self.moon = None
        self.ra = ra
        self.dec = dec
        self.name = name
        self.band = band
        self.conditions = conditions

    def get_observation(self, tem):
        self.time = tem
        ra = self.ra
        dec = self.dec

        coords_field = coord_radectoaltaz(ra, dec, tem)
        self.alt = coords_field[0].value
        self.az = coords_field[1].value
        theta_obs = np.pi / 2 - coords_field[0].value * np.pi / 180.0
        phi_obs = coords_field[1].value * np.pi / 180.0
        self.theta = theta_obs
        self.phi = phi_obs

        LUA = moon_functions.Moon(tem)
        self.moon = LUA
        lua_coords = LUA.get_parameters(tem)

        theta_lua = np.pi / 2 - lua_coords[0] * np.pi / 180.0
        phi_lua = lua_coords[1] * np.pi / 180.0

        self.gamma = np.arccos(
            np.sin(theta_lua) * np.sin(theta_obs) * np.cos(phi_obs - phi_lua) + np.cos(theta_lua) * np.cos(theta_obs))

        if self.band == 'B':
            self.wave = 437
        if self.band == 'V':
            self.wave = 555
        if self.band == 'R':
            self.wave = 655
        if self.band == 'I':
            self.wave = 768

    def func_gamma(self, theta_source, phi_source, units='radians'):
        gamma = np.arccos(
            np.sin(theta_source) * np.sin(self.theta) * np.cos(self.phi - phi_source) + np.cos(theta_source) * np.cos(
                self.theta))

        if units == 'radians':
            gamma = gamma * 1
        if units == 'degrees':
            gamma = gamma * 180 / np.pi

        return gamma

    def multiple_scattering(self):

        print('Calculating polarization by multiple scattering method...')

    def rayleigh_scattering(self):

        print('Calculating polarization by rayleigh model...')

    def stokes(self):

        print('Calculating Stokes parameters...')


class Target:
    def __init__(self):
        self.wave = None
        self.alt = None
        self.az = None
        self.theta = None
        self.phi = None
        self.gamma = None
        self.time = None
        self.moon = None
        self.ra = None
        self.dec = None

    def get_observation(self, tem, alt, az):
        self.time = tem
        self.alt = alt
        self.az = az
        theta_obs = np.pi / 2 - alt * np.pi / 180.0
        phi_obs = az * np.pi / 180.0
        self.theta = theta_obs
        self.phi = phi_obs

        LUA = moon_functions.Moon(tem)
        self.moon = LUA
        lua_coords = LUA.get_parameters(tem)

        theta_lua = np.pi / 2 - lua_coords[0] * np.pi / 180.0
        phi_lua = lua_coords[1] * np.pi / 180.0

        self.gamma = np.arccos(
            np.sin(theta_lua) * np.sin(theta_obs) * np.cos(phi_obs - phi_lua) + np.cos(theta_lua) * np.cos(theta_obs))

    def func_gamma(self, theta_source, phi_source, units='radians'):
        gamma = np.arccos(
            np.sin(theta_source) * np.sin(self.theta) * np.cos(self.phi - phi_source) + np.cos(theta_source) * np.cos(
                self.theta))

        if units == 'radians':
            gamma = gamma * 1
        if units == 'degrees':
            gamma = gamma * 180 / np.pi

        return gamma


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

        observation = moon_functions.Moon(MED[n])
        observation.get_parameters()

        campo = Field(BANDA[n], CONDITION[n], field[n], float(RA[n]), float(DEC[n]))
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
