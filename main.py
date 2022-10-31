import multiple_func as multi
import Stokes_func as St
import Rayleigh_func as Ray
import field_functions

# field_functions.data_storage()

print('Choose the model for the fit:\n')
model = str(input('Your options are multiple scattering model from Berry - write multiple;\n Rayleigh scattering based on its equation - write rayleigh and \n Rayleigh scattering calculated by the Stokes Parameters - write stokes.\n'))

print('\n')

band_obs = []

int_band_obs = input('Do you have any restrictions in the band on the observations? (None or Yes or choose one band)\n')

if int_band_obs == 'None':
    band_obs = None

if int_band_obs == 'B':
    band_obs = 'B'

if int_band_obs == 'V':
    band_obs = 'V'

if int_band_obs == 'R':
    band_obs = 'R'

if int_band_obs == 'I':
    band_obs = 'I'

if int_band_obs == 'Yes':
    n = int(input('First enter number of elements.\n'))
    print('\n')
    print('Options for band elements: B, V, R, I\n')
    for i in range(0, n):
        ele = str(input('elemento ' + str(i) + ':   '))
        band_obs.append(ele)

print('\n')

condition_obs = []

int_condition_obs = input('Do you have any restrictions in the conditions of observations? (None or Yes or chose one condition)\n')

if int_condition_obs == 'None':
    condition_obs = None

if int_condition_obs == 'ok':
    condition_obs = 'ok'

if int_condition_obs == 'clouds':
    condition_obs = 'clouds'

if int_condition_obs == 'sun':
    condition_obs = 'sun'

if int_condition_obs == 'Yes':
    n = int(input('First enter number of elements.\n'))
    print('\n')
    print('Options for band elements:  ok, clouds or sun\n')
    for i in range(0, n):
        ele = str(input('elemento ' + str(i) + ':   '))
        print('\n')
        condition_obs.append(ele)

print('\n')

type_fit = input('Which type of fit you want? \nThe options are: horizon fit, fit Sun, fit seeing, fit wavelength, depolarization fit, wave aop fit, fit all or ALL (to run all fits).\n')

print('\n')

if model == 'multiple':
    parameters, df = multi.FIT(band=band_obs, condition=condition_obs, command=type_fit)
if model == 'stokes':
    parameters, df = St.FIT(band=band_obs, condition=condition_obs, command=type_fit)
if model == 'rayleigh':
    parameters, df = Ray.FIT(band=band_obs, condition=condition_obs, command=type_fit)
