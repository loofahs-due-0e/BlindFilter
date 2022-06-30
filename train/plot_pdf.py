import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import invgauss

random_scale = True

if len(sys.argv) != 2:
    print('usage : python3 plot_pdf.py <filename>')
    exit()
filename = sys.argv[1]

f = open('{}'.format(filename), 'r')
lines = f.readlines()
a = []


def in_range(x, y, eps):
    return abs(x - y) < eps


for line in lines:
    val = float(line)

    if random_scale:
        if not in_range(val, 0.0, 5e-3):
            r = invgauss.rvs(1, loc=0, scale=5)
            while r < 0.5 or r > 2:
                r = invgauss.rvs(1, loc=0, scale=5)
            a.append(val * r)
    else:
        a.append(val)

f.close()

print("{} ~ {}".format(np.min(a), np.max(a)))
print("mean = {}".format(np.mean(a)))

k = np.mean(((a - np.mean(a)) / np.std(a)) ** 4)
print("kurtosis = {}".format(k))

hist, bins = np.histogram(a, bins=1000)
bin_centers = (bins[1:]+bins[:-1])*0.5
matplotlib.rcParams.update({'font.size': 16})
plt.clf()
plt.xticks(np.arange(-200, 200+1, 50))
plt.yticks(np.arange(0, 140+1, 25))
plt.grid()
plt.xlim([-200, 200])
plt.ylim([0, 140])
plt.xlabel('model output', fontsize=18)
plt.ylabel('occurrence', fontsize=18)
plt.plot(bin_centers, hist)
plt.savefig('{}.png'.format(filename))
