from gym.envs.registration import register

register(
    id='TradingEnvironment-v0',
    entry_point='trading_environment.envs:TradingEnvironment',
    #timestep_limit=60,
)
