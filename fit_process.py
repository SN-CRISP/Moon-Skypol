import lmfit
import pandas as pd
import generic
import sky
import numpy as np
import Rayleigh as ray


def fit(function_name, model_name, par_names, allvariables, method_name, Y_pol, erro_Y, minimum, maximum):
    model = lmfit.Model(function_name, param_names=par_names)
    for i in range(0, len(par_names)):
        model.set_param_hint(par_names[i], value=np.random.rand(), min=minimum[i], max=maximum[i])

    params = model.make_params()

    print(f'independent variables: {model.independent_vars}')

    result = model.fit(data=Y_pol, params=params, allvars=allvariables, weights=erro_Y, method=method_name)

    erro = []
    print('\nParameters:')
    for pname, par in params.items():
        print(pname, par)
        print(result.params.valuesdict())
        erro.append(result.params[pname].stderr)

    results = list(result.params.valuesdict().values())
    quality = [result.chisqr, result.redchi, result.bic]

    try:
        rsd = result.eval_uncertainty()
    except ZeroDivisionError:
        rsd = np.zeros(len(Y_pol))

    txname = 'FIT_' + model_name + '.txt'

    TXT = open(txname, "w+")

    model_fit_report = result.fit_report()
    TXT.write('Independent variables: \n')
    TXT.write(str(model.independent_vars))
    TXT.write('\n')
    TXT.write(model_fit_report)

    return results, erro, rsd, quality


def data_select(model, correction, condition=None, band=None):
    global values, CONDT, BANDA
    df = pd.read_csv('data_output.csv', sep=',')

    DATA = df

    COND = ['ok', 'clouds', 'sun']

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

    conjunto = ['B', 'V', 'R', 'I']

    if band is None:
        BANDA = DATA['BANDA'].to_numpy()
        conjunto = ['B', 'V', 'R', 'I']

    if band is not None and condition is not None:
        conjunto = band
        if isinstance(band, str):
            DATA = DATA[DATA['BANDA'] == band]
            BANDA = DATA['BANDA'].to_numpy()
            CONDT = DATA['CONDITION'].to_numpy()
        else:
            DATA = DATA[DATA['BANDA'].isin(band)]
            BANDA = DATA['BANDA'].to_numpy()
            CONDT = DATA['CONDITION'].to_numpy()

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
    aop = DATA['AOP'].to_numpy()
    erraop = DATA['AOP error'].to_numpy()
    pol = DATA['POL OBS'].to_numpy()
    errpol = DATA['POL OBS error'].to_numpy()
    theta_sun = DATA['THETA SUN'].to_numpy()
    phi_sun = DATA['PHI SUN'].to_numpy()

    fit_resume = pd.DataFrame(
        {'FIELD': field, 'RA': RA, 'DEC': DEC, 'THETA FIELD': theta_field, 'PHI FIELD': phi_field,
         'THETA SUN': theta_sun, 'PHI SUN': phi_sun, 'THETA MOON': theta_moon,
         'PHI MOON': phi_moon, 'OB TIME MED': MED, 'I': Ival, 'Q': Qval, 'error Q': errQval, 'U': Uval,
         'error U': errUval, 'POL OB': pol, 'error POL OB': errpol, 'AOP OB': aop, 'error AOP OB': erraop,
         'GAMMA': gamma, 'SEEING': seen, 'WAVELENGTH': wave, 'ALBEDO': albedo, 'CONDITION': CONDT, 'BANDA': BANDA})

    if model == 'Rayleigh single scattering' and correction == 'Depolarization factor':
        values = [theta_field, phi_field, theta_moon, phi_moon]

    if model == 'Rayleigh single scattering' and correction == 'Amplitude empirical parameter':
        values = [theta_field, phi_field, theta_moon, phi_moon]

    return values, pol, errpol, fit_resume


def process_data(model, correction, method_fit='leastsq', condition=None, band=None):
    global conjunto, BANDA, y, residuals, erro, result_par, quality, CONDT, fit_resume
    df = pd.read_csv('data_output.csv', sep=',')

    DATA = df

    COND = ['ok', 'clouds', 'sun']

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

    conjunto = ['B', 'V', 'R', 'I']

    if band is None:
        BANDA = DATA['BANDA'].to_numpy()
        conjunto = ['B', 'V', 'R', 'I']

    if band is not None and condition is not None:
        conjunto = band
        if isinstance(band, str):
            DATA = DATA[DATA['BANDA'] == band]
            BANDA = DATA['BANDA'].to_numpy()
            CONDT = DATA['CONDITION'].to_numpy()
        else:
            DATA = DATA[DATA['BANDA'].isin(band)]
            BANDA = DATA['BANDA'].to_numpy()
            CONDT = DATA['CONDITION'].to_numpy()

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
    aop = DATA['AOP'].to_numpy()
    erraop = DATA['AOP error'].to_numpy()
    pol = DATA['POL OBS'].to_numpy()
    errpol = DATA['POL OBS error'].to_numpy()
    theta_sun = DATA['THETA SUN'].to_numpy()
    phi_sun = DATA['PHI SUN'].to_numpy()

    LABEL = generic.listToString(conjunto)
    CC = generic.listToString(COND)

    diff = []

    if len(band) == 1:
        if model == 'Rayleigh single scattering':
            if correction == 'Depolarization factor':
                label_fit = 'rayleigh_dop_depolarization_' + LABEL + '_' + CC
                m = [0.0]
                M = [1.0]

                allvars = [theta_field, phi_field, theta_moon, phi_moon]
                result_par, erro, residuals, quality = fit(ray.func_dep_DOP, label_fit, par_names=['P'],
                                                           allvariables=allvars,
                                                           method_name=method_fit, Y_pol=pol, erro_Y=errpol, minimum=m,
                                                           maximum=M)

                y = ray.func_dep_DOP(allvars, *result_par)

            if correction == 'Amplitude empirical parameter':
                label_fit = 'rayleigh_dop_amplitude_' + LABEL + '_' + CC
                allvars = [theta_field, phi_field, theta_moon, phi_moon]
                m = [0.0]
                M = [1.0]

                result_par, erro, residuals, quality = fit(ray.func_reg_DOP, label_fit, par_names=['A'],
                                                           allvariables=allvars,
                                                           method_name=method_fit, Y_pol=pol, erro_Y=errpol, minimum=m,
                                                           maximum=M)

                y = ray.func_reg_DOP(allvars, *result_par)

        for i in range(0, len(pol)):
            diff.append(pol[i] - y[i])

        fit_resume = pd.DataFrame(
            {'FIELD': field, 'RA': RA, 'DEC': DEC, 'THETA FIELD': theta_field, 'PHI FIELD': phi_field,
             'THETA SUN': theta_sun, 'PHI SUN': phi_sun, 'THETA MOON': theta_moon,
             'PHI MOON': phi_moon, 'OB TIME MED': MED, 'I': Ival, 'Q': Qval, 'error Q': errQval, 'U': Uval,
             'error U': errUval, 'POL OB': pol, 'error POL OB': errpol, 'AOP OB': aop, 'error AOP OB': erraop,
             'GAMMA': gamma, 'SEEING': seen, 'WAVELENGTH': wave, 'ALBEDO': albedo, 'POL FIT': y, 'DIFF': diff,
             'RES': residuals, 'CONDITION': CONDT, 'BANDA': BANDA})

    else:
        if model == 'Rayleigh single scattering':
            if correction == 'Depolarization factor':
                label_fit = 'rayleigh_dop_depolarization_' + LABEL + '_' + CC
                m = [0.0]
                M = [1.0]
                par_band = []
                par_erro = []
                res_band = []
                fit_qual = []
                y_band = []
                pol_band = []
                df_band = []
                for wave in range(0, len(band)):
                    allvars, p, er_p, df = data_select(model, correction, condition, band[wave])
                    pol_band.append(p)

                    a, b, c, d = fit(ray.func_dep_DOP, label_fit,
                                     par_names=['P'],
                                     allvariables=allvars,
                                     method_name=method_fit,
                                     Y_pol=p, erro_Y=er_p,
                                     minimum=m,
                                     maximum=M)

                    par_band.append(a)
                    par_erro.append(b)
                    res_band.append(c)
                    fit_qual.append(d)

                    z = ray.func_dep_DOP(allvars, *a)

                    diff_band = []
                    for i in range(0, len(p)):
                        diff_band.append(float(p[i] - z[i]))

                    y_band.append(z)
                    df['POL FIT'] = z
                    df['RES'] = c
                    df['DIFF'] = diff_band
                    df_band.append(df)

                result_par, erro, residuals, quality = par_band, par_erro, res_band, fit_qual
                y = y_band
                pol = pol_band

                for wave in range(0, len(band)):
                    for i in range(0, len(pol)):
                        diff.append(pol[wave][i] - y[wave][i])

                fit_resume = pd.concat(df_band, ignore_index=True)

                fit_resume.to_csv("fit_summary.csv")

            if correction == 'Amplitude empirical parameter':
                label_fit = 'rayleigh_dop_amplitude_' + LABEL + '_' + CC
                m = [0.0]
                M = [1.0]
                par_band = []
                par_erro = []
                res_band = []
                fit_qual = []
                y_band = []
                pol_band = []
                df_band = []
                for wave in range(0, len(band)):
                    allvars, p, er_p, df = data_select(model, correction, condition, band[wave])
                    pol_band.append(p)

                    a, b, c, d = fit(ray.func_reg_DOP, label_fit,
                                     par_names=['A'],
                                     allvariables=allvars,
                                     method_name=method_fit,
                                     Y_pol=p, erro_Y=er_p,
                                     minimum=m,
                                     maximum=M)

                    par_band.append(a)
                    par_erro.append(b)
                    res_band.append(c)
                    fit_qual.append(d)

                    z = ray.func_reg_DOP(allvars, *a)

                    diff_band = []
                    for i in range(0, len(p)):
                        diff_band.append(float(p[i] - z[i]))

                    y_band.append(z)
                    df['POL FIT'] = z
                    df['RES'] = c
                    df['DIFF'] = diff_band
                    df_band.append(df)

                result_par, erro, residuals, quality = par_band, par_erro, res_band, fit_qual
                y = y_band
                pol = pol_band

                for wave in range(0, len(band)):
                    for i in range(0, len(pol)):
                        diff.append(pol[wave][i] - y[wave][i])

                fit_resume = pd.concat(df_band, ignore_index=True)

                fit_resume.to_csv("fit_summary.csv")

    return fit_resume, result_par, erro, quality
