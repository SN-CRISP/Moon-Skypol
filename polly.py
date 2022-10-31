import pandas as pd
import numpy as np
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord


def data(url=None):
    if url is None:
        url = 'https://drive.google.com/file/d/15FlGlJBC-c3Wh1e8bEv6fE8nycqUxebc/view?usp=sharing'
        path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
        df = pd.read_csv(path, sep=';')
        print(df)
    if url is not None:
        df = pd.read_csv(url)
        print(df)


class target:
    def __init__(self, name, ra, dec):
        self.VBAND_obs = None
        self.BBAND_obs = None
        self.RBAND_obs = None
        self.IBAND_obs = None
        self.name = name
        self.ra = ra
        self.dec = dec

    def display(self):
        print('FIELD ' + self.name + ': ' + '[' + self.ra + ', ' + self.dec)

    def setBBAND_obs(self, allvars):
        begin, end, time, iobs, qobs_value, qobs_error, uobs_value, uobs_error = allvars

        self.BBAND_obs.begin = begin
        self.BBAND_obs.end = end
        self.BBAND_obs.time = time
        self.BBAND_obs.i_value = iobs
        self.BBAND_obs.q_value = qobs_value
        self.BBAND_obs.q_error = qobs_error
        self.BBAND_obs.u_value = uobs_value
        self.BBAND_obs.u_error = uobs_error

        polarization = np.sqrt(qobs_value ** 2 + uobs_value ** 2)
        error_polarization = np.sqrt(
            qobs_value ** 2 * qobs_error ** 2 + uobs_value ** 2 * uobs_error ** 2) / polarization

        self.BBAND_obs.polarization_value = polarization
        self.BBAND_obs.polarization_error = error_polarization

        angle_polarization = 0.5 * np.arctan(uobs_value / qobs_value)
        error_angle_polarization = 0.5 * np.sqrt(
            qobs_value ** 2 * uobs_error ** 2 + uobs_value ** 2 * qobs_error ** 2) / (
                                           (1 + (uobs_value / qobs_value) ** 2) * qobs_value ** 2)

        self.BBAND_obs.angle_polarization = angle_polarization
        self.BBAND_obs.angle_polarization_error = error_angle_polarization

    def setVBAND_obs(self, allvars):
        begin, end, time, iobs, qobs_value, qobs_error, uobs_value, uobs_error = allvars

        self.VBAND_obs.begin = begin
        self.VBAND_obs.end = end
        self.VBAND_obs.time = time
        self.VBAND_obs.i_value = iobs
        self.VBAND_obs.q_value = qobs_value
        self.VBAND_obs.q_error = qobs_error
        self.VBAND_obs.u_value = uobs_value
        self.VBAND_obs.u_error = uobs_error

        polarization = np.sqrt(qobs_value ** 2 + uobs_value ** 2)
        error_polarization = np.sqrt(
            qobs_value ** 2 * qobs_error ** 2 + uobs_value ** 2 * uobs_error ** 2) / polarization

        self.BBAND_obs.polarization_value = polarization
        self.BBAND_obs.polarization_error = error_polarization

        angle_polarization = 0.5 * np.arctan(uobs_value / qobs_value)
        error_angle_polarization = 0.5 * np.sqrt(
            qobs_value ** 2 * uobs_error ** 2 + uobs_value ** 2 * qobs_error ** 2) / (
                                           (1 + (uobs_value / qobs_value) ** 2) * qobs_value ** 2)

        self.BBAND_obs.angle_polarization = angle_polarization
        self.BBAND_obs.angle_polarization_error = error_angle_polarization

    def setRBAND_obs(self, allvars):
        begin, end, time, iobs, qobs_value, qobs_error, uobs_value, uobs_error = allvars

        self.RBAND_obs.begin = begin
        self.RBAND_obs.end = end
        self.RBAND_obs.time = time
        self.RBAND_obs.i_value = iobs
        self.RBAND_obs.q_value = qobs_value
        self.RBAND_obs.q_error = qobs_error
        self.RBAND_obs.u_value = uobs_value
        self.RBAND_obs.u_error = uobs_error

        polarization = np.sqrt(qobs_value ** 2 + uobs_value ** 2)
        error_polarization = np.sqrt(
            qobs_value ** 2 * qobs_error ** 2 + uobs_value ** 2 * uobs_error ** 2) / polarization

        self.BBAND_obs.polarization_value = polarization
        self.BBAND_obs.polarization_error = error_polarization

        angle_polarization = 0.5 * np.arctan(uobs_value / qobs_value)
        error_angle_polarization = 0.5 * np.sqrt(
            qobs_value ** 2 * uobs_error ** 2 + uobs_value ** 2 * qobs_error ** 2) / (
                                           (1 + (uobs_value / qobs_value) ** 2) * qobs_value ** 2)

        self.BBAND_obs.angle_polarization = angle_polarization
        self.BBAND_obs.angle_polarization_error = error_angle_polarization

    def setIBAND_obs(self, allvars):
        begin, end, time, iobs, qobs_value, qobs_error, uobs_value, uobs_error = allvars

        self.IBAND_obs.begin = begin
        self.IBAND_obs.end = end
        self.IBAND_obs.time = time
        self.IBAND_obs.i_value = iobs
        self.IBAND_obs.q_value = qobs_value
        self.IBAND_obs.q_error = qobs_error
        self.IBAND_obs.u_value = uobs_value
        self.IBAND_obs.u_error = uobs_error

        polarization = np.sqrt(qobs_value ** 2 + uobs_value ** 2)
        error_polarization = np.sqrt(
            qobs_value ** 2 * qobs_error ** 2 + uobs_value ** 2 * uobs_error ** 2) / polarization

        self.BBAND_obs.polarization_value = polarization
        self.BBAND_obs.polarization_error = error_polarization

        angle_polarization = 0.5 * np.arctan(uobs_value / qobs_value)
        error_angle_polarization = 0.5 * np.sqrt(
            qobs_value ** 2 * uobs_error ** 2 + uobs_value ** 2 * qobs_error ** 2) / (
                                           (1 + (uobs_value / qobs_value) ** 2) * qobs_value ** 2)

        self.BBAND_obs.angle_polarization = angle_polarization
        self.BBAND_obs.angle_polarization_error = error_angle_polarization

    def trans_coordinates(self, TYPE):
        coordinates = []

        observing_location = coord.EarthLocation(lon=-70.404167 * u.deg, lat=-24.627222 * u.deg, height=2635 * u.m)
        observing_time = Time(self.BBAND_obs.time)
        cond = coord.AltAz(location=observing_location, obstime=observing_time)

        CO = SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg, frame='icrs')
        CO.transform_to(cond)
        alt = CO.transform_to(cond).alt * u.deg
        az = CO.transform_to(cond).az * u.deg

        if TYPE == 'AltAz':
            coordinates.append(alt)
            coordinates.append(az)
        if TYPE == 'Polar':
            theta = np.pi / 2 - alt * np.pi / 180.0
            phi = az * np.pi / 180.0
            coordinates.append(theta)
            coordinates.append(phi)

        return coord

    def moon_pair(self):
        results = []
        observing_time = Time(self.BBAND_obs.time, scale='utc')
        loc = coord.EarthLocation(lon=-70.404167 * u.deg, lat=-24.627222 * u.deg, height=2635 * u.m)

        AltAz = coord.AltAz(obstime=observing_time, location=loc)

        lua = coord.get_moon(observing_time)
        sol = coord.get_sun(observing_time)

        elongation = sol.separation(lua)
        f_lua = np.arctan2(sol.distance * np.sin(elongation), lua.distance - sol.distance * np.cos(elongation))
        k = (1 + np.cos(f_lua)) / 2.0

        alt_lua = lua.transform_to(AltAz).alt * u.deg
        az_lua = lua.transform_to(AltAz).az * u.deg

        # results have the shape: ALT, AZ, MOON PHASE
        results.append(alt_lua)
        results.append(az_lua)
        results.append(k)


class Moon:
    def __init__(self, time):
        self.time = time
        self.phase = None
        self.albedo = None

    # def visualization()
