import models
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mplcursors


tempo = '2021-01-26T00:37:13'
dh = 0.5

# models.DoP_simple_map(tempo, dh, n=80, source='Moon', ref='polar')

# models.AoP_simple_map(tempo, dh, n=80, source='Moon', ref='polar')

# models.DoP_multi_map(tempo, L=0.3, dh=0.5, n=80, source='Moon', ref='polar')

# models.DoP_multi_map(tempo, L=0.1, dh=0.5, n=80, source='Moon', ref='polar')

'''

def main():
    fig, axes = plt.subplots(ncols=2)
    num = 5
    xy = np.random.random((num, 2))

    lines = []
    for i in range(num):
        line, = axes[0].plot((i + 1) * np.arange(10))
        lines.append(line)

    points = []
    for x, y in xy:
        point, = axes[1].plot([x], [y], linestyle="none", marker="o")
        points.append(point)

    cursor = mplcursors.cursor(points + lines, highlight=True)
    pairs = dict(zip(points, lines))
    pairs.update(zip(lines, points))

    @cursor.connect("add")
    def on_add(sel):
        sel.extras.append(cursor.add_highlight(pairs[sel.artist]))

    plt.show()


# if __name__ == "__main__":
#    main()

labels = ["a", "b", "c", "d", "e"]
x = np.array([0, 1, 2, 3, 4])

fig, ax = plt.subplots()
line, = ax.plot(x, x, "ro", label='legend')
mplcursors.cursor(ax).connect(
    "add", lambda sel: sel.annotation.set_text(labels[sel.index]))
plt.legend()

plt.show()

'''


def func(x, a, b):
    return a * x ** (-b)


xdata = np.array([437, 555, 655, 768])
ydata_ray = np.array([0.64, 0.614, 0.555, 0.315])
ydata_ray_error = np.array([0.010, 0.009, 0.006, 0.018])
ydata_multi = np.array([2.425, 2.303, 2.028, 1.121])
ydata_multi_error = np.array([0.121, 0.122, 0.106, 0.043])
BIC_ray = np.array([-177.53, -183.04, -197.04, -177.9])
BIC_multi = np.array([-163.97, -164.58, -169.67, -189.97])

# fig_x = plt.figure(figsize=(10, 5))
# plt.plot(xdata, func(xdata, *popt), 'g--', label='fit Rayleigh: C=%5.3f, n=%5.3f' % tuple(popt))
# plt.plot(xdata, ydata_ray, 'o', color='black', label='data Rayleigh A(\u03BB)')

# plt.plot(xdata, func(xdata, *popt_m), 'g--', label='fit M.S.: C=%5.3f, n=%5.3f' % tuple(popt_m))
# plt.plot(xdata, ydata_ray, 'o', color='black', label='data M.S. A(\u03BB)')

# plt.xlabel('\u03BB (nm)')
# plt.ylabel('A(\u03BB)')
# plt.legend()


fig, ax = plt.subplots()
popt, pcov = curve_fit(func, xdata, ydata_ray)
print(popt)
ax.plot(xdata, func(xdata, *popt), 'g--', label='fit Rayleigh: C=%5.3f, n=%5.3f' % tuple(popt))
ax.plot(xdata, ydata_ray, 'g^', label='data Rayleigh A(\u03BB)')
plt.errorbar(xdata, ydata_ray, yerr=ydata_ray_error, marker=".", linestyle="none", ecolor='green')
popt, pcov = curve_fit(func, xdata, ydata_multi)
print(popt)
ax.plot(xdata, func(xdata, *popt), 'b--', label='fit M.S.: C=%5.3f, n=%5.3f' % tuple(popt))
ax.plot(xdata, ydata_multi, 'b^', label='data M.S. A(\u03BB)')
plt.errorbar(xdata, ydata_multi, yerr=ydata_multi_error, marker=".", linestyle="none")
ax.set_xlabel('\u03BB (nm)')
ax.set_ylabel('A(\u03BB)')

ax2 = ax.twinx()
ax2.plot(xdata, BIC_ray, 'gx', label='BIC Rayleigh')
ax2.plot(xdata, BIC_multi, 'bx', label='BIC M.S.')
ax2.set_ylabel('BIC')
fig.legend()
plt.show()


# tempo = '2021-01-26T00:37:13'
# dh = 0.5

# models.DoP_simple_map(tempo, dh, n=80, source='Moon', ref='polar')

# models.AoP_simple_map(tempo, dh, n=80, source='Moon', ref='polar')
