import numpy as np
import matplotlib.pyplot as plt
new = np.load("/home/eddie/Documents/GitHub/cryojax-ensemble-optimization/ACCESS/mrc_images/ctf_grids20.npy")
old = np.load("/home/eddie/Documents/GitHub/cryojax-ensemble-optimization/ACCESS/mrc_images1/expt_ctfs20.npy")

plt.imshow(new[0])
plt.show()

plt.imshow(old[0])
plt.show()
abs_diff = abs(new - old)
print(abs_diff)
assert np.allclose(old, new, rtol=1e-5), "two methods generate diff ctf grids"


