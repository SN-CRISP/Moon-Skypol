import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
import ephem


# m = ephem.Moon()
# m.compute('2012-7-12 12:00:00.000')
# t = Time('2012-7-12 12:00:00.000')
# t.format = 'iso'
# print(t)
# print(repr(m.libration_lat))
# print(m.libration_lat)

# a = -2 + float(repr(m.libration_lat))
# print(a)

def find_nearest_ind(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def albedo():
    # file = open('par.txt', 'r')
    # print(file.read())

    # dados da tabela 4
    data = np.loadtxt("par.txt")
    lamb = data[:, 0]
    ca0 = data[:, 1]
    ca1 = data[:, 2]
    ca2 = data[:, 3]
    ca3 = data[:, 4]
    cb1 = data[:, 5]
    cb2 = data[:, 6]
    cb3 = data[:, 7]
    cd1 = data[:, 8]
    cd2 = data[:, 9]
    cd3 = data[:, 10]

    # constantes
    c1 = 0.00034115
    c2 = -0.0013425
    c3 = 0.000095906
    c4 = 0.00066229
    p1 = 4.06054
    p2 = 12.8802
    p3 = -30.5858
    p4 = 16.7498

    print('insira uma data e hora, com o formato (UTC): 2012-7-12 12:00:00.000')
    temp = input()

    tempo = Time(temp)

    print('comprimento de onda')
    L = float(input())

    # descobrir qual o comprimento de onda mais próximo

    indlam = find_nearest_ind(lamb, L)
    # print('o índice k é:')
    # print(indlam)

    a0 = data[indlam][1]
    a1 = data[indlam][2]
    a2 = data[indlam][3]
    a3 = data[indlam][4]
    b1 = data[indlam][5]
    b2 = data[indlam][6]
    b3 = data[indlam][7]
    d1 = data[indlam][8]
    d2 = data[indlam][9]
    d3 = data[indlam][10]

    X = []
    Y = []
    W = []
    K = []
    delta = 10 * u.hour

    paranal = ephem.Observer()
    paranal.lat = '-24.627222'
    paranal.lon = '-70.404167'
    paranal.elevation = 2635.43

    for i in range(0, 100):
        tempo.format = 'iso'

        paranal = ephem.Observer()
        paranal.lat = '-24.627222'
        paranal.lon = '-70.404167'
        paranal.elevation = 2635.43
        paranal.date = str(tempo)

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

        # paranal = ephem.Observer()
        # paranal.lat = '-24.627222 '
        # paranal.lon = '-70.404167'
        # paranal.elevation = 2635.43

        theo = float(repr(m.libration_lat)) * 180 / np.pi
        phio = float(repr(m.libration_long)) * 180 / np.pi

        # phase += 0.1 * 180 / 100 * np.pi / 180
        # fase += 0.1

        K.append(g)
        W.append(phase)
        tempo.format = 'mjd'
        X.append(tempo.value)
        tempo.format = 'iso'

        # formula 10 Kieffer
        func = np.exp(
            a0 + a1 * g + a2 * g ** 2 + a3 * g ** 3 + b1 * phis + b2 * phis ** 3 + b3 * phis ** 5 + c1 * theo + c2 * phio + c3 * phis * theo + c4 * phis * theo + d1 * np.exp(
                -g / p1) + d2 * np.exp(-g / p2) + d3 * np.cos((g - p3) / p4))
        Y.append(func)

        tempo += delta

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time (MJD Days)')
    ax1.set_ylabel('albedo')
    ax1.plot(X, Y, 'ro', markersize=0.7)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()

    ax2.set_ylabel('phase (%)')
    ax2.plot(X, W, 'ko', markersize=0.7)
    ax2.tick_params(axis='y')

    fig.tight_layout()
    plt.grid(True)
    plt.title('Albedo')
    plt.show()


# comparação de 3 comprimentos de onda

def albedo_pha():
    # dados da tabela 4
    DATA = np.loadtxt("par.txt")
    WL = DATA[:, 0]

    # constantes
    C1 = 0.00034115
    C2 = -0.0013425
    C3 = 0.000095906
    C4 = 0.00066229
    P1 = 4.06054
    P2 = 12.8802
    P3 = -30.5858
    P4 = 16.7498

    print('insira uma data e hora, com o formato (UTC): 2012-7-12 12:00:00.000')
    cons_tempo = input()
    TEMPO = Time(cons_tempo)

    print('insira o primeiro comprimento de onda:')
    L1 = float(input())
    ind1 = find_nearest_ind(WL, L1)

    a01 = DATA[ind1][1]
    a11 = DATA[ind1][2]
    a21 = DATA[ind1][3]
    a31 = DATA[ind1][4]
    b11 = DATA[ind1][5]
    b21 = DATA[ind1][6]
    b31 = DATA[ind1][7]
    d11 = DATA[ind1][8]
    d21 = DATA[ind1][9]
    d31 = DATA[ind1][10]

    print('insira o segundo comprimento de onda:')
    L2 = float(input())
    ind2 = find_nearest_ind(WL, L2)

    a02 = DATA[ind2][1]
    a12 = DATA[ind2][2]
    a22 = DATA[ind2][3]
    a32 = DATA[ind2][4]
    b12 = DATA[ind2][5]
    b22 = DATA[ind2][6]
    b32 = DATA[ind2][7]
    d12 = DATA[ind2][8]
    d22 = DATA[ind2][9]
    d32 = DATA[ind2][10]

    print('insira o terceiro comprimento de onda:')
    L3 = float(input())
    ind3 = find_nearest_ind(WL, L3)

    a03 = DATA[ind3][1]
    a13 = DATA[ind3][2]
    a23 = DATA[ind3][3]
    a33 = DATA[ind3][4]
    b13 = DATA[ind3][5]
    b23 = DATA[ind3][6]
    b33 = DATA[ind3][7]
    d13 = DATA[ind3][8]
    d23 = DATA[ind3][9]
    d33 = DATA[ind3][10]

    XX = []
    Y1 = []
    Y2 = []
    Y3 = []
    DELTA = 10 * u.hour

    for j in range(0, 100):
        TEMPO.format = 'iso'

        PARANAL = ephem.Observer()
        PARANAL.lat = '-24.627222'
        PARANAL.lon = '-70.404167'
        PARANAL.elevation = 2635.43
        PARANAL.date = str(TEMPO)

        Lua = ephem.Moon()
        Lua.compute(PARANAL)

        PHASE = float(repr(Lua.moon_phase))
        G = (1 - PHASE) / 1 * np.pi

        sc = float(repr(Lua.colong))
        PHIS = 0

        if sc < 3 * np.pi / 2:
            PHIS = np.pi / 2 - sc
        if sc > 3 * np.pi / 2:
            PHIS = 3 * np.pi / 2 - sc

        THEO = float(repr(Lua.libration_lat)) * 180 / np.pi
        PHIO = float(repr(Lua.libration_long)) * 180 / np.pi

        XX.append(G * 180 / np.pi)

        # formula 10 Kieffer
        func1 = np.exp(
            a01 + a11 * G + a21 * G ** 2 + a31 * G ** 3 + b11 * PHIS + b21 * PHIS ** 3 + b31 * PHIS ** 5 + C1 * THEO + C2 * PHIO + C3 * PHIS * THEO + C4 * PHIS * THEO + d11 * np.exp(
                -G / P1) + d21 * np.exp(-G / P2) + d31 * np.cos((G - P3) / P4))
        Y1.append(func1)

        func2 = np.exp(
            a02 + a12 * G + a22 * G ** 2 + a32 * G ** 3 + b12 * PHIS + b22 * PHIS ** 3 + b32 * PHIS ** 5 + C1 * THEO + C2 * PHIO + C3 * PHIS * THEO + C4 * PHIS * THEO + d12 * np.exp(
                -G / P1) + d22 * np.exp(-G / P2) + d32 * np.cos((G - P3) / P4))
        Y2.append(func2)

        func3 = np.exp(
            a03 + a13 * G + a23 * G ** 2 + a33 * G ** 3 + b13 * PHIS + b23 * PHIS ** 3 + b33 * PHIS ** 5 + C1 * THEO + C2 * PHIO + C3 * PHIS * THEO + C4 * PHIS * THEO + d13 * np.exp(
                -G / P1) + d23 * np.exp(-G / P2) + d33 * np.cos((G - P3) / P4))
        Y3.append(func3)

        TEMPO += DELTA

    plt.plot(XX, Y1, 'ro', markersize=0.7, label=str(L1))
    plt.plot(XX, Y2, 'ko', markersize=0.7, label=str(L2))
    plt.plot(XX, Y3, 'bo', markersize=0.7, label=str(L3))
    plt.legend()
    plt.grid(True)
    plt.title('Lunar Albedo versus Phase')
    plt.xlabel('phase angle g (deg)')
    plt.ylabel('albedo')
    plt.show()


# comparação entre as fases e comprimentos de onda
def albedo_wvl():
    # dados da tabela 4

    DATA = np.loadtxt("par.txt")

    # constantes
    C1 = 0.00034115
    C2 = -0.0013425
    C3 = 0.000095906
    C4 = 0.00066229
    P1 = 4.06054
    P2 = 12.8802
    P3 = -30.5858
    P4 = 16.7498

    TEMPO = []

    print('insira a primeira data e hora, com o formato (UTC): 2012-7-12 12:00:00.000')
    temp1 = input()
    tempo1 = Time(temp1)
    TEMPO.append(tempo1)

    print('insira a segunda data e hora, com o formato (UTC): 2012-7-12 12:00:00.000')
    temp2 = input()
    tempo2 = Time(temp2)
    TEMPO.append(tempo2)

    print('insira a segunda data e hora, com o formato (UTC): 2012-7-12 12:00:00.000')
    temp3 = input()
    tempo3 = Time(temp3)
    TEMPO.append(tempo3)

    XX = []
    YY = []
    FF = []

    for t in range(0, 3):
        KK = []
        WW = []

        TEMPO[t].format = 'iso'

        PARANAL = ephem.Observer()
        PARANAL.lat = '-24.627222'
        PARANAL.lon = '-70.404167'
        PARANAL.elevation = 2635.43
        PARANAL.date = str(TEMPO[t])

        Lua = ephem.Moon()
        Lua.compute(PARANAL)

        PHASE = float(repr(Lua.moon_phase))
        G = (1 - PHASE) / 1 * np.pi

        sc = float(repr(Lua.colong))
        PHIS = 0

        if sc < 3 * np.pi / 2:
            PHIS = np.pi / 2 - sc
        if sc > 3 * np.pi / 2:
            PHIS = 3 * np.pi / 2 - sc

        THEO = float(repr(Lua.libration_lat)) * 180 / np.pi
        PHIO = float(repr(Lua.libration_long)) * 180 / np.pi

        FF.append(PHASE)

        for j in range(0, 32):
            LAMB = DATA[j][0]
            A0 = DATA[j][1]
            A1 = DATA[j][2]
            A2 = DATA[j][3]
            A3 = DATA[j][4]
            B1 = DATA[j][5]
            B2 = DATA[j][6]
            B3 = DATA[j][7]
            D1 = DATA[j][8]
            D2 = DATA[j][9]
            D3 = DATA[j][10]

            KK.append(LAMB)

            # formula 10 Kieffer
            FUNC = np.exp(
                A0 + A1 * G + A2 * G ** 2 + A3 * G ** 3 + B1 * PHIS + B2 * PHIS ** 3 + B3 * PHIS ** 5 + C1 * THEO + C2 * PHIO + C3 * PHIS * THEO + C4 * PHIS * THEO + D1 * np.exp(
                    -G / P1) + D2 * np.exp(-G / P2) + D3 * np.cos((G - P3) / P4))

            WW.append(FUNC)

            j += 1

        XX.append(KK)
        YY.append(WW)
        t += 1

    plt.plot(XX[0], YY[0], 'ro', markersize=0.7, label='g = %.3f' % round(FF[0], 3))
    plt.plot(XX[1], YY[1], 'ko', markersize=0.7, label='g = %.3f' % round(FF[1], 3))
    plt.plot(XX[2], YY[2], 'bo', markersize=0.7, label='g = %.3f' % round(FF[2], 3))
    plt.legend()
    plt.grid(True)
    plt.title('Lunar Albedo')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('albedo')
    plt.show()


def albedo_moon(wave, tem):
    # file = open('par.txt', 'r')
    # print(file.read())

    # dados da tabela 4
    data = np.loadtxt("par.txt")
    lamb = data[:, 0]
    ca0 = data[:, 1]
    ca1 = data[:, 2]
    ca2 = data[:, 3]
    ca3 = data[:, 4]
    cb1 = data[:, 5]
    cb2 = data[:, 6]
    cb3 = data[:, 7]
    cd1 = data[:, 8]
    cd2 = data[:, 9]
    cd3 = data[:, 10]

    # constantes
    c1 = 0.00034115
    c2 = -0.0013425
    c3 = 0.000095906
    c4 = 0.00066229
    p1 = 4.06054
    p2 = 12.8802
    p3 = -30.5858
    p4 = 16.7498

    tempo = Time(tem)

    # descobrir qual o comprimento de onda mais próximo

    indlam = find_nearest_ind(lamb, wave)
    # print('o índice k é:')
    # print(indlam)

    a0 = data[indlam][1]
    a1 = data[indlam][2]
    a2 = data[indlam][3]
    a3 = data[indlam][4]
    b1 = data[indlam][5]
    b2 = data[indlam][6]
    b3 = data[indlam][7]
    d1 = data[indlam][8]
    d2 = data[indlam][9]
    d3 = data[indlam][10]

    X = []
    Y = []
    W = []
    K = []
    delta = 10 * u.hour

    paranal = ephem.Observer()
    paranal.lat = '-24.627222'
    paranal.lon = '-70.404167'
    paranal.elevation = 2635.43

    tempo.format = 'iso'

    paranal = ephem.Observer()
    paranal.lat = '-24.627222'
    paranal.lon = '-70.404167'
    paranal.elevation = 2635.43
    paranal.date = str(tempo)

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

    # paranal = ephem.Observer()
    # paranal.lat = '-24.627222 '
    # paranal.lon = '-70.404167'
    # paranal.elevation = 2635.43

    theo = float(repr(m.libration_lat)) * 180 / np.pi
    phio = float(repr(m.libration_long)) * 180 / np.pi

    # phase += 0.1 * 180 / 100 * np.pi / 180
    # fase += 0.1

    K.append(g)
    W.append(phase)
    tempo.format = 'mjd'
    X.append(tempo.value)
    tempo.format = 'iso'

    # formula 10 Kieffer
    func = np.exp(a0 + a1 * g + a2 * g ** 2 + a3 * g ** 3 + b1 * phis + b2 * phis ** 3 + b3 * phis ** 5 + c1 * theo + c2 * phio + c3 * phis * theo + c4 * phis * theo + d1 * np.exp(-g / p1) + d2 * np.exp(-g / p2) + d3 * np.cos((g - p3) / p4))

    g = 0

    func_max = np.exp(a0 + a1 * g + a2 * g ** 2 + a3 * g ** 3 + b1 * phis + b2 * phis ** 3 + b3 * phis ** 5 + c1 * theo + c2 * phio + c3 * phis * theo + c4 * phis * theo + d1 * np.exp(-g / p1) + d2 * np.exp(-g / p2) + d3 * np.cos((g - p3) / p4))

    ALBEDO = []

    ALBEDO.append(func)
    ALBEDO.append(func_max)

    return ALBEDO
