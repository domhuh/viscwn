from env.boolean_network import BooleanNetwork
from env.sinr_network import SINRNetwork
from policy.mpc import MPC_Policy
env=BooleanNetwork(n_sbs=5, n_user=100)
env.view(MPC_Policy(env))
