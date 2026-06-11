import numpy as np
import matplotlib.pyplot as plt
new = np.load("/home/eddie/Documents/GitHub/cryojax-ensemble-optimization/ACCESS/half1Images/expt_ctfs35.npy")
old = np.load("/home/eddie/Documents/GitHub/cryojax-ensemble-optimization/ACCESS/half1ImagesOld/expt_ctfs35.npy")

plt.imshow(new[0])
plt.show()

plt.imshow(old[0])
plt.show()
abs_diff = abs(new - old)
print(abs_diff)
assert np.allclose(old, new, rtol=1e-5), "two methods generate diff ctf grids"


