import numpy as np
import pylab as plt


def logo(seed=46):
    text_color = "#002288"
    trace_color = "#cc2222"
    ndets = 15
    rise_time = 3.0
    fall_time = 11.0
    ph_other = 0.4
    t = np.arange(-10, 115, 0.1)
    x = np.exp(-t / fall_time) - np.exp(-t / rise_time)
    normalize = x.max()
    x[t <= 0] = 0

    # Second pulse
    t2 = 80
    x[t > t2] += 0.35 * (np.exp(-(t[t > t2] - t2) / fall_time) - np.exp(-(t[t > t2] - t2) / rise_time))
    x /= normalize

    fig = plt.figure(9, figsize=(1.28, 1.28), dpi=100)
    margin = 0.08
    tmargin = 1 - margin
    fig.subplots_adjust(bottom=margin, top=tmargin, left=margin, right=tmargin)
    plt.clf()
    plt.plot(t, x, color=trace_color, lw=2)
    plt.xticks([])
    plt.yticks([])
    plt.text(110, 0.75, "Mass2", ha="right", size=17, color=text_color)

    rg = np.random.default_rng(seed)

    # Other dets
    cm = plt.cm.Spectral
    for i in range(ndets):
        n = rg.poisson(0.8, size=1)
        x = np.zeros_like(t)
        for _ in range(int(n)):
            t0 = rg.uniform(-25, 110)
            print(
                t0,
            )
            x[t > t0] += np.exp(-(t[t > t0] - t0) / fall_time) - np.exp(-(t[t > t0] - t0) / rise_time)
        x *= ph_other / normalize
        plt.plot(t, x - 0.1 * i - 0.35, color=cm(i / (ndets - 0.5)))
        print

    plt.ylim([-0.4 - 0.1 * ndets, 1.2 + 0.03 * ndets])
