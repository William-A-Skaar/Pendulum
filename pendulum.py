import numpy as np
import scipy.integrate as scp_i
import matplotlib.pyplot as plt

class Pendulum:

    def __init__(self, L = 1, M = 1, g = 9.81):
        self.L = L
        self.M = M
        self.g = g
        self._t = None
        self._theta = None
        self._omega = None
        self._x = None
        self._u = None
        self._potential = None
        self._vx = None
        self._vy = None
        self._kinetic = None

    def __call__(self, t, y):
        dy_dt = tuple([y[1], -(self.g/self.L)*np.sin(y[0])])
        return dy_dt

    def solve(self, y0, T, dt, angles = "rad"):
        if angles == "deg":
            y0[0] = y0[0]*np.pi/180
        t_span = np.linspace(0, T, (int(T/dt) + 1))
        solver = scp_i.solve_ivp(self, [0,T], y0, t_eval = t_span)
        self._t, self.y = solver.t, solver.y
        self._theta = self.y[0]; self._omega = self.y[1]

    def solveexce(self, var):
        if var is None:
            raise Exception("Solve has not been called")
        else:
            return var

    @property
    def t(self):
        return self.solveexce(self._t)
    @property
    def theta(self):
        return self.solveexce(self._theta)
    @property
    def omega(self):
        return self.solveexce(self._omega)
    @property
    def x(self):
        return self.L*np.sin(self._theta)
    @property
    def u(self):
        return -self.L*np.cos(self._theta)
    @property
    def potential(self):
        P = self.M*self.g*(self.u + self.L)
        return P
    @property
    def vx(self):
            return np.gradient(self.x, self.t)
    @property
    def vy(self):
            return np.gradient(self.u, self.t)
    @property
    def kinetic(self):
        K = (1/2)*self.M*((self.vx)**2 + (self.vy)**2)
        return K

class DampenedPendulum(Pendulum):
    def __init__(self, L = 1, M = 1, g = 9.81, B = 0.3):
        Pendulum.__init__(self, L = 1, M = 1, g = 9.81)
        self.B = B
    def __call__(self, t, y):
        dy_dt = tuple([y[1], -(self.g/self.L)*np.sin(y[0]) - (self.B/self.M)*y[1]])
        return dy_dt

def example_run():
    pend = Pendulum()
    pend.solve((np.pi/2,0),10,0.1)

    plt.plot(pend.t, pend.theta)
    plt.title("Plot of theta(t)")
    plt.xlabel("Time"); plt.ylabel("Theta")
    plt.show()

    tot = pend.potential + pend.kinetic
    plt.figure()
    for i, j, k, in zip([221,222,223], [pend.kinetic, pend.potential, tot], ["Kinetic","Potential","Total"]):
        plt.subplot(i)
        plt.plot(pend.t, j)
        plt.xlabel("Time"); plt.ylabel(k)
    plt.show()

def tot_decay_plot():
    damp_pend = DampenedPendulum()
    damp_pend.solve((np.pi/2,0),10,0.1)

    tot = damp_pend.kinetic + damp_pend.potential
    plt.plot(damp_pend.t, tot)
    plt.title("Total energy decay when dampened")
    plt.xlabel("Time")
    plt.ylabel("Total energy")
    plt.show()

if __name__ == "__main__":
    example_run()
    tot_decay_plot()
