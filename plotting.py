from simulations import det_solve, Tsit5
import matplotlib.pyplot as plt

def plot_det_soln(t0, tf, state0, arg, i_species, style='tableau-colorblind10',solver=Tsit5, dt=0.1):
    plt.style.use(style)
    ts, ys = det_solve(t0, tf, state0, arg, i_species, solver=solver, dt=dt)
    if i_species == None:
        for i in range(len(ys)):
            plt.plot(ts, ys[i])
    else:
        plt.plot(ts, ys, label=f"M={arg[-1]}")


if __name__ == "__main__":
    ODE_params = (0.08, 0.06, 0.04, 0.04, 0.04, 0.05, 0.12)
    MOIs = [1, 2, 4]
    t0 = 0.0
    tf = 100.0
    state0 = (0.0, 0.0, 0.0)
    for M in MOIs:
        plot_det_soln(t0, tf, state0, (*ODE_params, M), 2)
    plt.legend(loc='upper right', fontsize='large')
    plt.ylabel("Rescaled [CII]")
    plt.xlabel("Time (minutes)")
    plt.vlines(60.0, 0.0, 0.64, 'red', '--')
    plt.savefig("det_M_vary.png", transparent=True)
    plt.show()
