import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

# Fixed symmetrical log norm so that the colorbar is centered at zero

class MidpointLogNorm(colors.SymLogNorm):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))

    All arguments are the same as SymLogNorm, except for midpoint    
    """
    def __init__(self, lin_thres, lin_scale, midpoint=None, vmin=None, vmax=None):
        self.midpoint = midpoint
        self.lin_thres = lin_thres
        self.lin_scale = lin_scale
        #fraction of the cmap that the linear component occupies
        self.linear_proportion = (lin_scale / (lin_scale + 1)) * 0.5
        #print(self.linear_proportion)

        colors.SymLogNorm.__init__(self, lin_thres, lin_scale, vmin, vmax)

    def __get_value__(self, v, log_val, clip=None):
        if v < -self.lin_thres or v > self.lin_thres:
            return log_val
        
        x = [-self.lin_thres, self.midpoint, self.lin_thres]
        y = [0.5 - self.linear_proportion, 0.5, 0.5 + self.linear_proportion]
        interpol = np.interp(v, x, y)
        return interpol

    def __call__(self, value, clip=None):
        log_val = colors.SymLogNorm.__call__(self, value)

        out = [0] * len(value)
        for i, v in enumerate(value):
            out[i] = self.__get_value__(v, log_val[i])

        return np.ma.masked_array(out)