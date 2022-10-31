import numpy as np
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
import pandas as pd
import seaborn as sns
import ephem
import matplotlib.pyplot as plt


class Sun:
    def __init__(self, time):
        self.theta = None
        self.phi = None
        self.az = None
        self.alt = None
        self.time = time
        self.x = None
        self.y = None
        self.meridian_x = None
        self.meridian_y = None

    def set_parameters(self, tem=None):
        tem = Time(self.time, scale='utc')
        loc = coord.EarthLocation(lon=-70.404167 * u.deg, lat=-24.627222 * u.deg, height=2635 * u.m)

        AltAz = coord.AltAz(obstime=tem, location=loc)

        sol = coord.get_sun(tem)

        alt = sol.transform_to(AltAz).alt * u.deg
        az = sol.transform_to(AltAz).az * u.deg

        self.alt = alt.value
        self.az = az.value

        self.theta = np.pi / 2 - alt.value * np.pi / 180.0
        self.phi = az.value * np.pi / 180.0

        theta = np.pi / 2 - alt.value * np.pi / 180.0
        phi = az.value * np.pi / 180.0

        self.x = np.sin(theta) * np.cos(phi)
        self.y = np.sin(theta) * np.sin(phi)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)

        meridian_x_sol = np.zeros(2)
        meridian_y_sol = np.zeros(2)

        meridian_x_sol[0] = x / (np.sqrt(x ** 2 + y ** 2))
        meridian_x_sol[1] = - meridian_x_sol[0]

        self.meridian_x = meridian_x_sol

        meridian_y_sol[0] = y / (np.sqrt(x ** 2 + y ** 2))
        meridian_y_sol[1] = -meridian_y_sol[0]

        self.meridian_y = meridian_y_sol


class Moon:
    def __init__(self, time):
        self.theta = None
        self.phi = None
        self.az = None
        self.alt = None
        self.time = time
        self.phase = None
        self.albedo = None
        self.my_sun = None
        self.x = None
        self.y = None
        self.meridian_x = None
        self.meridian_y = None

    def get_parameters(self, tem=None):
        global tempo
        if tem is None and self.time is not None:
            # print('Calculating Moon parameter...')
            tempo = self.time
        if tem is None and self.time is None:
            print('Time input missing! Aceptable input may be for example: 2021-1-19 18:00:00.000')
            exit()
        if tem is not None:
            tempo = Time(tem, scale='utc')
            self.time = tempo

        tempo = Time(self.time, scale='utc')
        loc = coord.EarthLocation(lon=-70.404167 * u.deg, lat=-24.627222 * u.deg, height=2635 * u.m)

        AltAz = coord.AltAz(obstime=tempo, location=loc)

        lua = coord.get_moon(tempo)
        sol = coord.get_sun(tempo)

        elongation = sol.separation(lua)
        f_lua = np.arctan2(sol.distance * np.sin(elongation), lua.distance - sol.distance * np.cos(elongation))
        k = (1 + np.cos(f_lua)) / 2.0

        alt = lua.transform_to(AltAz).alt * u.deg
        az = lua.transform_to(AltAz).az * u.deg

        self.alt = alt.value
        self.az = az.value
        self.phase = k

        self.theta = np.pi / 2 - alt.value * np.pi / 180.0
        self.phi = az.value * np.pi / 180.0

        theta = np.pi / 2 - alt.value * np.pi / 180.0
        phi = az.value * np.pi / 180.0

        self.x = np.sin(theta) * np.cos(phi)
        self.y = np.sin(theta) * np.sin(phi)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)

        meridian_x_lua = np.zeros(2)
        meridian_y_lua = np.zeros(2)

        meridian_x_lua[0] = x / (np.sqrt(x ** 2 + y ** 2))
        meridian_x_lua[1] = - meridian_x_lua[0]

        self.meridian_x = meridian_x_lua

        meridian_y_lua[0] = y / (np.sqrt(x ** 2 + y ** 2))
        meridian_y_lua[1] = -meridian_y_lua[0]

        self.meridian_y = meridian_y_lua

        results = [self.alt, self.az, self.time, self.phase]

        return results

    def get_albedo(self, ind):
        # url = 'https://drive.google.com/file/d/1dFfWXpk50JhGeHL3s8GPnSBqVoboC90Z/view?usp=sharing'
        # path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
        df = pd.read_csv('par_albedo.csv', sep=';')

        par_moon = self.get_parameters()

        df.isnull().sum()
        df.dropna(axis=1)

        wavelength = df['Wave'].to_numpy()
        a0 = df['A_0'].to_numpy()
        a1 = df['A_1'].to_numpy()
        a2 = df['A_2'].to_numpy()
        a3 = df['A_3'].to_numpy()
        b1 = df['B_1'].to_numpy()
        b2 = df['B_2'].to_numpy()
        b3 = df['B_3'].to_numpy()
        d1 = df['D_1'].to_numpy()
        d2 = df['D_2'].to_numpy()
        d3 = df['D_3'].to_numpy()

        # constantes
        c1 = 0.00034115
        c2 = -0.0013425
        c3 = 0.000095906
        c4 = 0.00066229
        p1 = 4.06054
        p2 = 12.8802
        p3 = -30.5858
        p4 = 16.7498

        tim = Time(self.time, scale='utc')
        tim.format = 'iso'

        paranal = ephem.Observer()
        paranal.lat = '-24.627222'
        paranal.lon = '-70.404167'
        paranal.elevation = 2635.43
        paranal.date = str(tim)

        m = ephem.Moon()
        m.compute(paranal)

        phase = float(repr(m.moon_phase))
        g = (1 - phase) / 1 * np.pi

        SC = float(repr(m.colong))
        phis = 0

        if SC < 3 * np.pi / 2:
            phis = np.pi / 2 - SC
        if SC > 3 * np.pi / 2:
            phis = 3 * np.pi / 2 - SC

        theo = float(repr(m.libration_lat)) * 180 / np.pi
        phio = float(repr(m.libration_long)) * 180 / np.pi

        # formula 10 Kieffer
        tim.format = 'mjd'
        x = tim.value
        tim.format = 'iso'
        func = np.exp(
            a0[ind] + a1[ind] * g + a2[ind] * g ** 2 + a3[ind] * g ** 3 + b1[ind] * phis + b2[ind] * phis ** 3 + b3[
                ind] * phis ** 5 + c1 * theo + c2 * phio + c3 * phis * theo + c4 * phis * theo + d1[
                ind] * np.exp(-g / p1) + d2[ind] * np.exp(-g / p2) + d3[ind] * np.cos((g - p3) / p4))

        result = [x, func, g, wavelength[ind], tim, phase, par_moon[0], par_moon[1], self.phase]
        # print(result)

        return result

    def plot_and_retrive_albedo(self, wave):
        # print('Plotting...')
        wave_conj = []
        func_albedo = []

        for j in range(0, 32):
            data = self.get_albedo(j)
            wave_conj.append(data[3])
            func_albedo.append(data[1])

        x = np.array(wave_conj)
        y = np.array(func_albedo)

        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)

        xp = np.linspace(300, 2500, 100)

        plt.plot(x, y, '.', xp, p(xp))
        plt.pause(0.1)
        plt.close()

        df = pd.DataFrame({'Wavelength (nm)': x, 'Lunar Albedo': y})
        sns.lmplot(x='Wavelength (nm)', y='Lunar Albedo', data=df, order=2)
        albedo = np.interp(wave, xp, p(xp))
        self.albedo = albedo
        plt.plot(wave, albedo, 'ro')
        plt.text(wave+0.02, albedo+0.02, str([wave, round(albedo, 3)]))
        plt.pause(0.1)
        plt.close()

        return albedo

    def plot_tempo(self):
        # url = 'https://drive.google.com/file/d/1dFfWXpk50JhGeHL3s8GPnSBqVoboC90Z/view?usp=sharing'
        # path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
        df = pd.read_csv('par_albedo.csv', sep=';')

        # par_moon = self.get_parameters()

        df.isnull().sum()
        df.dropna(axis=1)

        wavelength = df['Wave'].to_numpy()
        a0 = df['A_0'].to_numpy()
        a1 = df['A_1'].to_numpy()
        a2 = df['A_2'].to_numpy()
        a3 = df['A_3'].to_numpy()
        b1 = df['B_1'].to_numpy()
        b2 = df['B_2'].to_numpy()
        b3 = df['B_3'].to_numpy()
        d1 = df['D_1'].to_numpy()
        d2 = df['D_2'].to_numpy()
        d3 = df['D_3'].to_numpy()

        # constantes
        c1 = 0.00034115
        c2 = -0.0013425
        c3 = 0.000095906
        c4 = 0.00066229
        p1 = 4.06054
        p2 = 12.8802
        p3 = -30.5858
        p4 = 16.7498

        G = []
        fase = []
        X = []
        Y1 = []
        Y2 = []
        Y3 = []

        time_obs = Time(self.time, scale='utc')
        delta = 5 * u.hour

        for i in range(0, 200):
            time_obs.format = 'iso'

            paranal = ephem.Observer()
            paranal.lat = '-24.627222'
            paranal.lon = '-70.404167'
            paranal.elevation = 2635.43
            paranal.date = str(time_obs)

            m = ephem.Moon()
            m.compute(paranal)

            phase = float(repr(m.moon_phase))
            g = (1 - phase) / 1 * np.pi

            SC = float(repr(m.colong))
            phis = 0

            if SC < 3 * np.pi / 2:
                phis = np.pi / 2 - SC
            if SC > 3 * np.pi / 2:
                phis = 3 * np.pi / 2 - SC

            theo = float(repr(m.libration_lat)) * 180 / np.pi
            phio = float(repr(m.libration_long)) * 180 / np.pi

            G.append(g * 180 / np.pi)
            fase.append(phase)
            time_obs.format = 'mjd'
            X.append(time_obs.value)
            time_obs.format = 'iso'

            # formula 10 Kieffer
            func = np.exp(a0[0] + a1[0] * g + a2[0] * g ** 2 + a3[0] * g ** 3 + b1[0] * phis + b2[0] * phis ** 3 + b3[0] * phis ** 5 + c1 * theo + c2 * phio + c3 * phis * theo + c4 * phis * theo + d1[0] * np.exp(-g / p1) + d2[0] * np.exp(-g / p2) + d3[0] * np.cos((g - p3) / p4))
            Y1.append(func)

            func = np.exp(a0[7] + a1[7] * g + a2[7] * g ** 2 + a3[7] * g ** 3 + b1[7] * phis + b2[7] * phis ** 3 + b3[
                7] * phis ** 5 + c1 * theo + c2 * phio + c3 * phis * theo + c4 * phis * theo + d1[7] * np.exp(-g / p1) +
                          d2[7] * np.exp(-g / p2) + d3[7] * np.cos((g - p3) / p4))
            Y2.append(func)

            func = np.exp(a0[14] + a1[14] * g + a2[14] * g ** 2 + a3[14] * g ** 3 + b1[14] * phis + b2[14] * phis ** 3 + b3[
                14] * phis ** 5 + c1 * theo + c2 * phio + c3 * phis * theo + c4 * phis * theo + d1[14] * np.exp(-g / p1) +
                          d2[14] * np.exp(-g / p2) + d3[14] * np.cos((g - p3) / p4))
            Y3.append(func)

            time_obs += delta

        plt.plot(G, Y1, 'b-', markersize=0.7, label=str(wavelength[0]))
        plt.plot(G, Y2, 'k-', markersize=0.7, label=str(wavelength[7]))
        plt.plot(G, Y3, 'r-', markersize=0.7, label=str(wavelength[14]))
        plt.legend()
        plt.grid(True)
        plt.title('Lunar Albedo versus Phase')
        plt.xlabel('phase angle g (deg)')
        plt.ylabel('albedo')
        plt.show()

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('time (MJD Days)')
        ax1.set_ylabel('albedo')
        ax1.plot(X, Y1, 'b-', markersize=0.7, label=str(wavelength[0]))
        ax1.plot(X, Y2, 'k-', markersize=0.7, label=str(wavelength[7]))
        ax1.plot(X, Y3, 'r-', markersize=0.7, label=str(wavelength[14]))
        ax1.tick_params(axis='y')
        plt.legend(loc='upper left')

        ax2 = ax1.twinx()

        ax2.set_ylabel('phase (%)')
        ax2.plot(X, fase, 'g--', markersize=0.7, label='phase')
        ax2.tick_params(axis='y')

        fig.tight_layout()
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.title('Albedo')
        plt.show()

    def true_sun(self):
        tem = Time(self.time, scale='utc')
        loc = coord.EarthLocation(lon=-70.404167 * u.deg, lat=-24.627222 * u.deg, height=2635 * u.m)

        AltAz = coord.AltAz(obstime=tem, location=loc)

        sol = coord.get_sun(tem)

        alt_sol = sol.transform_to(AltAz).alt * u.deg
        az_sol = sol.transform_to(AltAz).az * u.deg

        SUN = Sun(tem)
        SUN.set_parameters()

        self.my_sun = SUN

        # results have the shape: ALT, AZ, MOON PHASE
        results = [alt_sol, az_sol]

        return results, SUN
