import numpy as np
import matplotlib.pyplot as plt
x, y = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))

contor = np.sqrt(x ** 2 + y ** 2)
plt.imshow(contor)
plt.colorbar()
plt.show()