import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# LINEAR ACTIVATION FUNCTION
fig, ax = plt.subplots()
ax.plot([-8,0,8], [-8,0,8])
# ax.spines[["bottom","left"]].set_visible(False)
ax.spines[['right', 'top']].set_position('zero')
ax.set_xticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.set_title("Linear activation function")
ax.grid(visible=True, which='both', axis='both',
        linestyle =':')
plt.savefig("lab3/linear_activation.svg")
plt.show()

# BINARY FUNCTION

fig, ax = plt.subplots()

ax.plot((-8,0.5),(-1,-1),'ro-',color='orange')
ax.plot((0.5,8),(2,2),'ro-',color='orange')

# ax.spines[["bottom","left"]].set_visible(False)
ax.spines[['right', 'top']].set_position('zero')

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.set_title("Binary activation function")
ax.grid(visible=True, which='both', axis='both',
        linestyle =':')

ax.set_xticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
ax.set_yticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
plt.savefig("lab3/binary_activation.svg")
plt.show()