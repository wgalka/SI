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

# 