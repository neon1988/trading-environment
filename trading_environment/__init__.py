from gym.envs.registration import register

register(
    id='tradingEnvironment-v0',
    entry_point='trading_environment.envs:tradingEnvironment',
    #timestep_limit=60,
)
