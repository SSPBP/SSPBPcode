import numpy as np
import utm
def get_dataset(name):
    '''
    Read the appropriate dataset. Although this code is pretty repetitive, it allows for
    customization of the raw dataset format.

    Returns: tuple of (X, y) with X as an Nxp matrix and y as an n-vector
    '''
    if name == 'Boston':
        data = np.loadtxt('data/Boston/regression-datasets-housing.csv', delimiter=",")
        X = data[ :, range(data.shape[ 1 ] - 1) ]
        y = data[ :, data.shape[ 1 ] - 1 ]
    elif name == 'Concrete':
        data = np.loadtxt('data/Concrete/Concrete_Data.csv', delimiter=",", skiprows=1)
        X = data[ :, range(data.shape[ 1 ] - 1) ]
        y = data[ :, data.shape[ 1 ] - 1 ]
    elif name == 'Energy_Efficiency':
        data = np.loadtxt('data/Energy_Efficiency/ENB2012_data.csv', delimiter=",", skiprows=1)
        # There are two regression problems: Y1 and Y2 (heating and cooling)
        X = data[ :, range(data.shape[ 1 ] - 1) ]
        y = data[ :, data.shape[ 1 ] - 1 ]
    elif name == 'Kin8nm':
        data = np.loadtxt('data/Kin8nm/regression-datasets-kin8nm.csv', delimiter=",", skiprows=0)
        X = data[ :, range(data.shape[ 1 ] - 1) ]
        y = data[ :, data.shape[ 1 ] - 1 ]
    elif name == 'KriSp_Precip':
        data = np.loadtxt('data/KriSP_Precip/krisp_precip.csv', skiprows=0, delimiter=",")
        X = data[ :, range(data.shape[ 1 ] - 1) ]
        def to_utm(t):
            a = utm.from_latlon(t[0], t[1], force_zone_number=14)
            return [a[0] / 1000.0, a[1] / 1000.0]
        X = np.array([to_utm(t) for t in X])
        y = data[ :, data.shape[ 1 ] - 1 ]
    elif name == 'Naval':
        data = np.loadtxt('data/Naval/data.txt', skiprows=0)
        X = data[ :, range(data.shape[ 1 ] - 2) ]
        y = data[ :, data.shape[ 1 ] - 2 ]
    elif name == 'Power':
        data = np.loadtxt('data/Power/data.csv', delimiter=",", skiprows=0)
        X = data[ :, range(data.shape[ 1 ] - 1) ]
        y = data[ :, data.shape[ 1 ] - 1 ]
    elif name == 'Wine':
        data = np.loadtxt('data/Wine/winequality-red.csv', delimiter=";", skiprows=0)
        X = data[ :, range(data.shape[ 1 ] - 1) ]
        y = data[ :, data.shape[ 1 ] - 1 ]
    elif name == 'Yacht':
        data = np.loadtxt('data/Yacht/yacht_hydrodynamics.data.txt', skiprows=0)
        X = data[ :, range(data.shape[ 1 ] - 1) ]
        y = data[ :, data.shape[ 1 ] - 1 ]
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y
