import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

    def centered_int(x1, F, dt):
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

    def RK4(x0, F, dt):
        '''
        Equation:
            x' = F,
            k1 = dt * F(x(n))
            k2 = dt * F(x(n)+k1/2)
            k3 = dt * F(x(n)+k2/2)
            k4 = dt * F(x(n)+k3)
            x(n+1) = x(n) + k1/6 + k2/3 + k3/3 + k4/6
        '''
        pass


class Model:
    '''
    Collection of dynamical systems.
    '''

    def Lorenz63(
        sigma=10, r=28, b=8/3,
        x0=0, y0=1, z0=0,
        dt=0.01, nt=10,
        int_method='double-approx',
        prt=False
    ):
        '''

        Equations:
            X' = -sigma*X + sigma*Y
            Y' = -X*Z + r*X - Y
            Z' = X*Y - b*Z

        Parameters:
            + sigma, r, b: coefficients in the equations above;
            + x0, y0, z0: initial conditions;
            + dt: time increment every step (delta t)
            + nt: time steps
            + int_method: the integration method used to run this model
                - double-approx: double-approximation procedure [default]
                - RK4: 4th order Runge-Kutta

            NOTE: The defaults are documented in Lorenz (1963).

        Reference:
            Lorenz, E. N., 1963: Deterministic Nonperiodic Flow.
            J. Atmos. Sci., 20, 130–141.
            doi:10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2.

        '''

        def Px(x, y, sigma):
            P = - sigma*x + sigma*y
            return P

        def Py(x, y, z, r):
            P = - x*z + r*x - y
            return P

        def Pz(x, y, z, b):
            P = x*y - b*z
            return P

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

        '''
        Init arrays with size of nt+2 (index from 0 to nt+1).
            + 0: initial condition
            + 1 to nt: nt integration results
            + nt+1: the boundary for calculation of nt
        '''
        x = np.empty([nt+2])
        y = np.empty([nt+2])
        z = np.empty([nt+2])

        i = 0
        x[i] = x0
        y[i] = y0
        z[i] = z0

        if prt is True:
            print(
                '{0:5d}'.format(i),
                '{0:20.10f}'.format(x[i]),
                '{0:20.10f}'.format(y[i]),
                '{0:20.10f}'.format(z[i])
            )

        if int_method == 'RK4':

            for i in np.arange(nt):

                k1_x = dt * Px(x[i], y[i], sigma)
                k1_y = dt * Py(x[i], y[i], z[i], r)
                k1_z = dt * Pz(x[i], y[i], z[i], b)

                k2_x = dt * Px(x[i]+k1_x/2.0, y[i]+k1_y/2.0, sigma)
                k2_y = dt * Py(x[i]+k1_x/2.0, y[i]+k1_y/2.0, z[i]+k1_z/2.0, r)
                k2_z = dt * Pz(x[i]+k1_x/2.0, y[i]+k1_y/2.0, z[i]+k1_z/2.0, b)

                k3_x = dt * Px(x[i]+k2_x/2.0, y[i]+k2_y/2.0, sigma)
                k3_y = dt * Py(x[i]+k2_x/2.0, y[i]+k2_y/2.0, z[i]+k2_z/2.0, r)
                k3_z = dt * Pz(x[i]+k2_x/2.0, y[i]+k2_y/2.0, z[i]+k2_z/2.0, b)

                k4_x = dt * Px(x[i]+k3_x, y[i]+k3_y, sigma)
                k4_y = dt * Py(x[i]+k3_x, y[i]+k3_y, z[i]+k3_z, r)
                k4_z = dt * Pz(x[i]+k3_x, y[i]+k3_y, z[i]+k3_z, b)

                x[i+1] = x[i] + (k1_x+k4_x)/6.0 + (k2_x+k3_x)/3.0
                y[i+1] = y[i] + (k1_y+k4_y)/6.0 + (k2_y+k3_y)/3.0
                z[i+1] = z[i] + (k1_z+k4_z)/6.0 + (k2_z+k3_z)/3.0

                if prt is True:
                    print(
                        '{0:5d}'.format(i+1),
                        '{0:20.10f}'.format(x[i+1]),
                        '{0:20.10f}'.format(y[i+1]),
                        '{0:20.10f}'.format(z[i+1])
                    )

        else:

            for i in np.arange(nt):

                Fx = Px(x[i], y[i], sigma)
                Fy = Py(x[i], y[i], z[i], r)
                Fz = Pz(x[i], y[i], z[i], b)

                x[i+1] = NumericalMehtods.forward_int(x[i], Fx, dt)
                y[i+1] = NumericalMehtods.forward_int(y[i], Fy, dt)
                z[i+1] = NumericalMehtods.forward_int(z[i], Fz, dt)

                Fx = Px(x[i+1], y[i+1], sigma)
                Fy = Py(x[i+1], y[i+1], z[i+1], r)
                Fz = Pz(x[i+1], y[i+1], z[i+1], b)

                x[i+2] = NumericalMehtods.forward_int(x[i+1], Fx, dt)
                y[i+2] = NumericalMehtods.forward_int(y[i+1], Fy, dt)
                z[i+2] = NumericalMehtods.forward_int(z[i+1], Fz, dt)

                x[i+1] = 1/2 * (x[i]+x[i+2])
                y[i+1] = 1/2 * (y[i]+y[i+2])
                z[i+1] = 1/2 * (z[i]+z[i+2])

                if prt is True:
                    print(
                        '{0:5d}'.format(i+1),
                        '{0:20.10f}'.format(x[i+1]),
                        '{0:20.10f}'.format(y[i+1]),
                        '{0:20.10f}'.format(z[i+1])
                    )

        return x, y, z


class Plot:

    '''
    Plot results.
    '''

    def trajectory_1d(data, std_y=0):
        size = len(data)
        x = np.arange(size)
        plt.axhline(y=std_y)
        plt.plot(x, data[:-1])
        plt.show()

    def trajectory_2d(x, y, std_x=0, std_y=0):
        plt.axvline(x=std_x)
        plt.axhline(y=std_y)
        plt.plot(x[:-1], y[:-1])
        plt.show()

    def trajectory_3d(x, y, z, animation=False, ani_interval=1, lag=1e-20):
        '''
        NOTE: Animation should be used without inline displaying.
        '''

        fig = plt.figure()

        ax = fig.gca(projection='3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if animation is False:
            ax.plot(x[:-1], y[:-1], z[:-1])
            plt.show()
        else:
            size = len(x)
            ax.set_xlim([-20, 20])
            ax.set_ylim([-30, 30])
            ax.set_zlim([0, 50])

            frame = None

            for i in np.arange(0, size-1, ani_interval):
                oldcol = frame

                frame = ax.scatter(x[i], y[i], z[i])

                if oldcol is not None:
                    ax.collections.remove(oldcol)

                plt.pause(lag)
