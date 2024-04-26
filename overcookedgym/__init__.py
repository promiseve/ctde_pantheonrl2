from gym.envs.registration import register

register(
    id='OvercookedMultiEnv-v0',
    entry_point='overcookedgym.overcooked:OvercookedMultiEnv'
)

register(
    id='OvercookedMultiEnvWrapper-v0',
    entry_point='overcookedgym.overcooked:OvercookedMultiEnvWrapper',
    kwargs={'layout_name': 'simple', 'num_agents': 2}
)