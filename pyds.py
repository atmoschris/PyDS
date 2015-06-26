import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


class NumericalMehtod:
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
            x' = F(x, t),
            k1 = F(x(n), t(n))
            k2 = F(x(n)+k1/2*dt, t(n)+dt/2)
            k3 = F(x(n)+k2/2*dt, t(n)+dt/2)
            k4 = F(x(n)+k3*dt, t(n)+dt)
            x(n+1) = x(n) + (k1 + 2*k2 + 2*k3 + k4)/6
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

        if prt is True:
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
        x = np.empty([nt+1])
        y = np.empty([nt+1])
        z = np.empty([nt+1])

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

        elif int_method == 'RK2':
            # Heun Method with a Single Corrector (a2 = 1/2)

            for i in np.arange(nt):

                k1_x = dt * Px(x[i], y[i], sigma)
                k1_y = dt * Py(x[i], y[i], z[i], r)
                k1_z = dt * Pz(x[i], y[i], z[i], b)

                k2_x = dt * Px(x[i]+k1_x/2.0, y[i]+k1_y/2.0, sigma)
                k2_y = dt * Py(x[i]+k1_x/2.0, y[i]+k1_y/2.0, z[i]+k1_z/2.0, r)
                k2_z = dt * Pz(x[i]+k1_x/2.0, y[i]+k1_y/2.0, z[i]+k1_z/2.0, b)

                x[i+1] = x[i] + k2_x
                y[i+1] = y[i] + k2_y
                z[i+1] = z[i] + k2_z

                if prt is True:
                    print(
                        '{0:5d}'.format(i+1),
                        '{0:20.10f}'.format(x[i+1]),
                        '{0:20.10f}'.format(y[i+1]),
                        '{0:20.10f}'.format(z[i+1])
                    )

        elif int_method == 'reverse_double_approx':
            for i in np.arange(nt):

                Fx = Px(x[i], y[i], sigma)
                Fy = Py(x[i], y[i], z[i], r)
                Fz = Pz(x[i], y[i], z[i], b)

                x[i+1] = x[i] - Fx*dt
                y[i+1] = y[i] - Fy*dt
                z[i+1] = z[i] - Fz*dt

                Fx = Px(x[i+1], y[i+1], sigma)
                Fy = Py(x[i+1], y[i+1], z[i+1], r)
                Fz = Pz(x[i+1], y[i+1], z[i+1], b)

                x_tmp = x[i+1] - Fx*dt
                y_tmp = y[i+1] - Fy*dt
                z_tmp = z[i+1] - Fz*dt

                x[i+1] = 1/2 * (x[i]+x_tmp)
                y[i+1] = 1/2 * (y[i]+y_tmp)
                z[i+1] = 1/2 * (z[i]+z_tmp)

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

                x[i+1] = x[i] + Fx*dt
                y[i+1] = y[i] + Fy*dt
                z[i+1] = z[i] + Fz*dt

                Fx = Px(x[i+1], y[i+1], sigma)
                Fy = Py(x[i+1], y[i+1], z[i+1], r)
                Fz = Pz(x[i+1], y[i+1], z[i+1], b)

                x_tmp = x[i+1] + Fx*dt
                y_tmp = y[i+1] + Fy*dt
                z_tmp = z[i+1] + Fz*dt

                x[i+1] = 1/2 * (x[i]+x_tmp)
                y[i+1] = 1/2 * (y[i]+y_tmp)
                z[i+1] = 1/2 * (z[i]+z_tmp)

                if prt is True:
                    print(
                        '{0:5d}'.format(i+1),
                        '{0:20.10f}'.format(x[i+1]),
                        '{0:20.10f}'.format(y[i+1]),
                        '{0:20.10f}'.format(z[i+1])
                    )

        return x, y, z

    def Lorenz63_odeint(initial, t):
        x = initial[0]
        y = initial[1]
        z = initial[2]
        sigma = 10
        rho = 28
        beta = 8.0/3
        x_dot = sigma * (y - x)
        y_dot = x * (rho - z) - y
        z_dot = x * y - beta * z
        return [x_dot, y_dot, z_dot]

    def Lorenz96(
        initial,
        F=8,
        dt=0.05, nt=10,
        int_method='RK4',
        prt=False, show_index=0
    ):
        '''
        Equations:
            X(j)' = [X(j+1)-X(j-2)]X(j-1) - X(j) + F

        Parameters:
            + initial: array, initial condition
            + F: forcing
            + j: index of grid points that  create a loop
            + dt: time increment every step (delta t)
            + nt: time steps
            + int_method: the integration method used to run this model
                - RK4: 4th order Runge-Kutta [default]

            NOTE: The defaults are documented in Lorenz (1997).

        References:

            Edward N. Lorenz, 1996: Predictability – A problem partly solved
                Seminar on Predictability, Vol. I, ECMWF,
                book Predictability of Weather and Climate, Chapter 3, 40-58.
                doi: http://dx.doi.org/10.1017/CBO9780511617652.004

            Edward N. Lorenz and Kerry A. Emanuel, 1998:
                Optimal Sites for Supplementary Weather Observations:
                Simulation with a Small Model. J. Atmos. Sci., 55, 399-414.
                doi: http://dx.doi.org/10.1175/
                1520-0469(1998)055<0399:OSFSWO>2.0.CO;2

        '''
        if prt is True:
            print('')
            print('>>>>>> Lorenz96 ...')
            print('')
            print('           F:', F)
            print('     initial:', initial)
            print('          dt:', dt)
            print('          nt:', nt)
            print('  int_method:', int_method)
            print('')
            print('...................')
            print('')

        def Eqn(field, j, F):
            N = np.size(field)
            if j+1 >= N:
                result = (field[j+1-N]-field[j-2]) * field[j-1] - field[j] + F
            else:
                result = (field[j+1]-field[j-2]) * field[j-1] - field[j] + F
            return result

        N = np.size(initial)  # K equations, number of grid points
        old_values = np.copy(initial)
        new_values = np.copy(initial)
        results = np.ndarray(shape=(nt+1, N))
        results[0, :] = initial

        if int_method == 'RK2':

            # Heun Method with a Single Corrector (a2 = 1/2)

            k1 = np.empty(N)
            k2 = np.empty(N)

            for i in np.arange(nt):

                for j in np.arange(N):
                    k1[j] = Eqn(old_values, j, F)
                    new_values[j] = old_values[j] + k1[j]/2*dt

                for j in np.arange(N):
                    k2[j] = Eqn(new_values, j, F)
                    new_values[j] = old_values[j] + k2[j]/2*dt

                new_values = old_values + k2*dt
                old_values = np.copy(new_values)

                results[i+1, :] = new_values

                if prt is True:
                    print(
                        '{0:5d}'.format(i),
                        '{0:20.10f}'.format(new_values[show_index])
                    )

        else:

            k1 = np.empty(N)
            k2 = np.empty(N)
            k3 = np.empty(N)
            k4 = np.empty(N)

            for i in np.arange(nt):

                for j in np.arange(N):
                    k1[j] = Eqn(old_values, j, F)
                    new_values[j] = old_values[j] + k1[j]/2*dt

                for j in np.arange(N):
                    k2[j] = Eqn(new_values, j, F)
                    new_values[j] = old_values[j] + k2[j]/2*dt

                for j in np.arange(N):
                    k3[j] = Eqn(new_values, j, F)
                    new_values[j] = old_values[j] + k3[j]*dt

                for j in np.arange(N):
                    k4[j] = Eqn(new_values, j, F)

                new_values = old_values + (k1+2*k2+2*k3+k4)/6*dt
                old_values = np.copy(new_values)

                results[i+1, :] = np.copy(new_values)

                if prt is True:
                    print(
                        '{0:5d}'.format(i),
                        '{0:20.10f}'.format(new_values[show_index])
                    )

        return results


class Plot:

    '''
    Plot results.
    '''

    def trajectory_1d(data, std_y=0):
        if type(data) is tuple:
            dims = np.shape(data[0])
        else:
            dims = np.shape(data)

        x = np.arange(dims[0])

        plt.axhline(y=std_y, color='black')

        if type(data) is tuple:
            for d in data:
                plt.plot(x, d)
        else:
            plt.plot(x, data)

        plt.show()

    def trajectory_2d(x, y, std_x=0, std_y=0):
        plt.axvline(x=std_x, color='black')
        plt.axhline(y=std_y, color='black')
        plt.plot(x, y)
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
            ax.plot(x, y, z)
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

    def field_plot_with_time(
        data,
        time_interval=1,
        std_y=0, range_y=1,
        fig_size=(12, 1),
    ):
        '''
        For Lorenz96.
        data is a 2-D array.
        '''
        if type(data) is tuple:
            dims = np.shape(data[0])
        else:
            dims = np.shape(data)

        nt = dims[0]
        x = np.arange(dims[1])

        plot_num = np.arange(0, nt, time_interval)
        nplot = np.size(plot_num)
        print('Time slots:', plot_num)

        fig, axs = plt.subplots(
            nplot, 1, figsize=(fig_size[0], fig_size[1]*nplot),
            facecolor='w', edgecolor='k'
        )

        plt.subplots_adjust(hspace=0, wspace=.5)

        axs = axs.ravel()

        if type(data) is tuple:
            for i in np.arange(nplot):
                for d in data:
                    axs[i].plot(x, d[plot_num[i], :])
        else:
            for i in np.arange(nplot):
                axs[i].plot(x, data[plot_num[i], :])

        for ax in axs:
            ax.set_ylim([std_y-range_y, std_y+range_y])
            ax.set_xlim([x.min(), x.max()])
            ax.axhline(y=std_y, color='black')
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)

        axs[nplot//2].set_ylabel("time")
        axs[-1].set_xlabel("site number")
        plt.setp(axs[-1].get_xticklabels(), visible=True)

        plt.show()


class Stat:

    def RMSE(data_std, data_exp, target_dim=-1):
        '''
        data_std: 1-D or 2-D standard data set
        data_exp: 1-D or 2-D experimental data set that to be verified
        '''

        if np.shape(data_exp) != np.shape(data_std):
            sys.exit('ERROR: Data sets with different shapes!')

        dim = len(np.shape(data_exp))

        if dim == 1:
            diff_square = (data_std - data_exp)**2
            avg = np.average(diff_square)
            root = np.sqrt(avg)

        else:
            diff_square = (data_std[target_dim] - data_exp[target_dim])**2
            nt = np.shape(data_std)[0]
            root = np.empty(nt)
            for i in np.arange(nt):
                diff_square = (data_std[i] - data_exp[i])**2
                avg = np.average(diff_square)
                root[i] = np.sqrt(avg)

        return root
