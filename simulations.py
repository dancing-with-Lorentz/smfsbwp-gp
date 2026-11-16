import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
import jsmfsb

@jax.jit
def run_sim(key, MOI, init, t0, tf, stoch_dt, step):
    return jsmfsb.sim_time_series(key, init*MOI, t0, tf, stoch_dt, step)

@jax.jit
def v_simulate(key, MOI, N, init, t0, tf, stoch_dt, step):
    key, k0 = jax.random.split(key)
    keys = jax.random.split(k0, N)
    v_sim =  jax.vmap(run_sim, (0, None))
    return v_sim(keys, MOI, init, t0, tf, stoch_dt, step)

@jax.jit
def simp_det(t, state, args):
    u, v, w = state
    b_u, d_u, a_v, a_w, g_u, g_v, g_w, M = args
    s = 1+u**2+v**2
    return (M*(b_u*u**2/s + d_u*w**2/(1+w**2))-g_u*u,
            M*a_v/s - g_v*v,
            M*a_w/s-g_w*w)

@jax.jit
def det_solve(fun, t0, tf, state0, arg, i_species, solver=Tsit5, dt=0.1):
    save = SaveAt(ts=jnp.linspace(t0, tf, int((tf-t0)/dt)+1))
    term = ODETerm(fun)
    soln = diffeqsolve(term, solver, t0, tf, dt, state0, arg, saveat=save)
    if i_species == None:
        return soln.ts, soln.ys*100 #scaling in line with stochastic model
    else:            
        return soln.ts, soln.ys[i_species]*100
