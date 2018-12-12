import matplotlib.pyplot as plt
import numpy as np
# #
# for inx, color in enumerate("rgbyck"):
#     plt.subplot(320+inx+1, axisbg=color)
# plt.subplot(321,'r')
w=np.linspace(0.1,1000,1000)
p=np.abs(1/(1+0.1j*w))
plt.subplot(321)
plt.plot(w,p,linewidth=2)

plt.show()
