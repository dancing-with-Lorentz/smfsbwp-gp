from simulations import v_simulate
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.scipy.stats as jsps
import jsmfsb

def unpack(file_str):
    shorthand = open(file_str, "r").read()
    pn_model = jsmfsb.shorthand_to_spn(shorthand)
    initial = pn_model.m
    step = pn_model.step_gillespie()
    return initial, step

def get_step(t0, tf, dt=0.5):
    nstep = int((tf - t0)/dt) + 1
    stoch_t = jnp.linspace(t0, tf, nstep)
    return stoch_t

def plot_stoch_traj(key, MOI, N, init, t0, tf, step, i_species, dt=0.5, dimers=0, style='tableau-colorblind10'):
    if dimers not in [0, 1]:
        raise ValueError(f"Dimer plot must be 0 or 1, given {dimers}")
    plt.figure(style)
    stoch_t = get_step(t0, tf, dt=dt)
    sims = v_simulate(key, MOI, N, init, t0, tf, dt, step)
    if i_species == None:
        labels = ["CI", 'CI2', 'Cro', 'Cro2', 'CII', 'CII2']
        for i in [0+dimers, 2+dimers, 4+dimers]:
            mean_i = jnp.mean(sims[:,:,i], axis=0)
            std_i = jnp.std(sims[:,:,i], axis=0, ddof=1)
            plt.plot(stoch_t, mean_i, label=labels[i])
            plt.fill_between(stoch_t, mean_i-std_i, mean_i+std_i, alpha=0.3)
    else:
        mean_i = jnp.mean(sims[:,:,i_species], axis=0)
        std_i = jnp.std(sims[:,:,i_species], axis=0, ddof=1)
        plt.plot(stoch_t, mean_i, label=f"M = {MOI}")
        plt.fill_between(stoch_t, mean_i-std_i, mean_i+std_i, alpha=0.3)

def conc_dist(key, M, V, N, init, t_conc, step, t0=0.0, dt=0.5, i_mol = 4, showmean=True, meancol='red', vscale=1e-15):
    if t0 >= t_conc:
        raise ValueError("Start time after or at analysis time!")
    tf = jnp.ceil(t_conc) + 1
    stoch_t = get_step(t0, tf, dt=dt)
    i_conc = jnp.argmin(abs(stoch_t - t_conc))
    sims = v_simulate(key, M, N, init, t0, tf, dt, step)
    sim = sims[:,i_conc,i_mol]/(V*vscale)
    mu = jnp.mean(sim)
    s = jnp.std(sim, ddof=1)
    xs = jnp.linspace(-4, 4, 800)*s + mu
    ps = jsps.norm.pdf(xs, loc=mu, scale=s)
    plt.plot(xs, ps, label=f"M = {M}, V={V}")
    if showmean:
        plt.axvline(mu, color=meancol, linestyle = '--')

if __name__ == "__main__": #Distribution plots
    import jax
    plt.figure(figsize=(10,5))
    tc = 60
    MOIs = [1, 2, 4]
    N_MC = 5000
    t0 = 0
    tf = 100
    k0 = jax.random.key(42)
    models = ["V_1_phage.txt", "V_2_phage.txt"]
    for M in MOIs:
        plt.clf()
        for i in range(len(models)):
            k0, k1 = jax.random.split(k0)
            initial, step = unpack(models[i])
            conc_dist(k1, M*(i+1), i+1, N_MC, initial, tc, step, showmean=False)
        plt.gca().set_yticks([])
        plt.legend(loc="upper right", fontsize="large")
        plt.xlabel("[CII] at t=60")
        plt.ylabel("Frequency")
        plt.savefig(f"concentration_comparison_M{M}.png", transparent=True)

''' Stochastic Trajectories
if __name__ == "__main__":
    import jax
    plt.figure(figsize=(10,5))
    N_MC = 20000
    MOIs = [1, 4]
    t0 = 0
    tf = 100
    k0 = jax.random.key(42)
    initial, step = unpack("V_1_phage.txt")
    for M in MOIs:
        k0, k1 = jax.random.split(k0)
        plot_stoch_traj(k1, M, N_MC, initial, t0, tf, step, 4)
    plt.legend(loc='upper right', fontsize='large')
    plt.title("Mean Trajectories with Standard Deviations",loc="left")
    plt.ylabel("CII count")
    plt.xlabel("Time (minutes)")
    plt.vlines(60.0, 0.0, 65.0, 'red', '--')
    plt.savefig("stoch_M_vary.png", transparent=True)
    plt.show()'''
