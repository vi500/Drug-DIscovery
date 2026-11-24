
import numpy as np
def test_bounds(): assert np.all(np.clip(np.array([-1,0,.5,1,2]),0,1)>=0)
