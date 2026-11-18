from simulations import v_simulate
import matplotlib.pyplot as plt
import jax.numpy as jnp
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
    #i60 = jnp.argmin(jnp.abs(stoch_t-60))
    return stoch_t

def plot_stoch_traj(key, MOI, N, init, t0, tf, step, i_species, dt=0.5, dimers=0, style='tableau-colorblind10'):
    if dimers not in [0, 1]:
        raise ValueError(f"Dimer plot must be 0 or 1, given {dimers}")
    plt.figure(style)
    stoch_t = get_step(t0, tf, dt=dt)
    sims = v_simulate(key, MOI, N, init, t0, tf, dt, step)
    if i_species == None: #Plot proteins
        for i in [0+dimers, 2+dimers, 4+dimers]:
            mean_i = jnp.mean(sims[:,:,i], axis=0)
            std_i = jnp.std(sims[:,:,i], axis=0, ddof=1)
            plt.plot(stoch_t, mean_i)
            plt.fill_between(stoch_t, mean_i-std_i, mean_i+std_i, alpha=0.3)
    else:
        mean_i = jnp.mean(sims[:,:,i_species], axis=0)
        std_i = jnp.std(sims[:,:,i_species], axis=0, ddof=1)
        plt.plot(stoch_t, mean_i, label=f"M = {MOI}")
        plt.fill_between(stoch_t, mean_i-std_i, mean_i+std_i, alpha=0.3)

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
    plt.ylabel("CII count")
    plt.xlabel("Time (minutes)")
    plt.vlines(60.0, 0.0, 65.0, 'red', '--')
    plt.savefig("stoch_M_vary.png", transparent=True)
    plt.show()
