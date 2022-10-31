import numpy as np
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord


def truemoon(tem):
    results = []
    tempo = Time(tem, scale='utc')
    loc = coord.EarthLocation(lon=-70.404167 * u.deg, lat=-24.627222 * u.deg, height=2635 * u.m)

    AltAz = coord.AltAz(obstime=tempo, location=loc)

    lua = coord.get_moon(tempo)
    sol = coord.get_sun(tempo)

    elongation = sol.separation(lua)
    f_lua = np.arctan2(sol.distance * np.sin(elongation), lua.distance - sol.distance * np.cos(elongation))
    k = (1 + np.cos(f_lua)) / 2.0

    alt_lua = lua.transform_to(AltAz).alt * u.deg
    az_lua = lua.transform_to(AltAz).az * u.deg

    # results have the shape: ALT, AZ, MOON PHASE
    results.append(alt_lua)
    results.append(az_lua)
    results.append(k)

    return results


def true_sun(tem):
    results = []
    tempo = Time(tem, scale='utc')
    loc = coord.EarthLocation(lon=-70.404167 * u.deg, lat=-24.627222 * u.deg, height=2635 * u.m)

    AltAz = coord.AltAz(obstime=tempo, location=loc)

    sol = coord.get_sun(tempo)

    alt_sol = sol.transform_to(AltAz).alt * u.deg
    az_sol = sol.transform_to(AltAz).az * u.deg

    # results have the shape: ALT, AZ, MOON PHASE
    results.append(alt_sol)
    results.append(az_sol)

    return results


def coord_mappol(alt, az):
    # as coordenadas de alt az estão em graus
    results = []
    phi = az * np.pi / 180
    theta = 90 - alt

    # results have the shape: PHI, THETA
    results.append(phi)
    results.append(theta)

    return results


def coord_mapxy(alt, az):
    results = []
    t_lua = np.pi / 2 - alt * np.pi / 180.0
    Al = az * np.pi / 180.0
    phi_lua = Al

    x = np.sin(t_lua) * np.cos(phi_lua)
    y = np.sin(t_lua) * np.sin(phi_lua)

    # results have the shape: X, Y
    results.append(x)
    results.append(y)

    return results


def coord_pol(alt, az):
    results = []
    theta = np.pi / 2 - alt * np.pi / 180.0
    phi = az * np.pi / 180.0

    # results have the shape: THETA, PHI
    results.append(theta)
    results.append(phi)

    return results


def coord_radectoaltaz(RA, DEC, tem):
    results = []

    observing_location = coord.EarthLocation(lon=-70.404167 * u.deg, lat=-24.627222 * u.deg, height=2635 * u.m)
    observing_time = Time(tem)
    cond = coord.AltAz(location=observing_location, obstime=observing_time)

    CO = SkyCoord(ra=RA * u.deg, dec=DEC * u.deg, frame='icrs')
    CO.transform_to(cond)
    alt = CO.transform_to(cond).alt * u.deg
    az = CO.transform_to(cond).az * u.deg

    # results have the shape: ALT, AZ
    results.append(alt)
    results.append(az)

    return results
