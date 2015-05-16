import numpy as np


class NumericalMehtods:
    '''
    Numerical methods such as temperal integration.
    '''

    def forward_int(x0, F, dt):
        '''
        Equation:
            x' = F,
            x(n+1) = x(n) + dt*F(n)

        Parameters:
            - x0: initial condition, now
            - F: forcing
            - dt: time increment every step (delta t)

        Return:
            - x: forecast, 1 step after now
        '''

        x = x0 + F*dt

        return x

    def centered_int(x1, F, dt, nt):
        '''
        Equation:
            x' = F,
            x(n+1) = x(n-1) + 2*dt*F(n)

        Parameters:
            - x1: initial condition, 1 step before now
            - F: forcing, now
            - dt: time increment every step (delta t)

        Return:
            - x: forecast, 1 step after now
        '''

        x = 2*F*dt + x1

        return x


class Model:
    '''
    Collection of dynamical systems.
    '''

    def Lorenz63(
        sigma=10, r=28, b=8/3,
        x0=0, y0=1, z0=0,
        dt=0.01, nt=1000,
        int_method='center'
    ):
        '''

        Equations:
            X' = -sigma*X + sigma*Y
            Y' = -X*Z + r*X - Y
            Z' = X*Y - b*Z

        Parameters:
            - sigma, r, b: coefficients in the equations above;
            - x0, y0, z0: initial conditions;
            - dt: time increment every step (delta t)
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
        print('       sigma:', sigma)
        print('           r:', r)
        print('           b:', b)
        print('(x0, y0, z0):', (x0, y0, z0))
        print('          dt:', dt)
        print('          nt:', nt)
        print('  int_method:', int_method)
        print('')
        print('...................')
        print('')

        x = x0
        y = y0
        z = z0

        for i in np.arange(nt):

            Fx = - sigma*x + sigma*y
            Fy = - x*z + r*x - y
            Fz = x*y - b*z

            x = NumericalMehtods.forward_int(x, Fx, dt)
            y = NumericalMehtods.forward_int(y, Fy, dt)
            z = NumericalMehtods.forward_int(z, Fz, dt)

            print(
                '{0:05}'.format(i+1),
                '{0:20.10f}'.format(x),
                '{0:20.10f}'.format(y),
                '{0:20.10f}'.format(z)
            )


class Plot:
    '''
    Plot results.
    '''
