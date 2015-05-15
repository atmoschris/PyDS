import numpy as np


class Models:
    '''
    Collection of dynamical models.
    '''

    def Lorenz63(
        sigma=10, r=24.74, b=8/3,
        x0=6*np.sqrt(2), y0=6*np.sqrt(2), z0=27,
        dt=0.01, nt=1000,
        int_method='RK4'
    ):
        '''

        Equations:
            X' = -sigma*X + sigma*Y
            Y' = -X*Z + r*X - Y
            Z' = X*Y - b*Z

        Parameters:
            - sigma, r, b: coefficients in the equations above;
            - x0, y0, z0: initial conditions;
            - dt: time interval for integration (delta t)
            - nt: time steps
            - int_method: the integration method used to run this model

            NOTE: The defaults are documented in Lorenz (1963).

        Reference:
            Lorenz, E. N., 1963: Deterministic Nonperiodic Flow.
            J. Atmos. Sci., 20, 130–141.
            doi:10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2.

        '''

        print('')
        print('>>>>>> Lorenz63 ...')
        print('')
        print('     sigma:', sigma)
        print('         r:', r)
        print('         b:', b)
        print('        x0:', x0)
        print('        y0:', y0)
        print('        z0:', z0)
        print('        dt:', dt)
        print('        dt:', nt)
        print('int_method:', int_method)
        print('...................')
        print('')


class Plot:
    '''
    Plot results.
    '''
