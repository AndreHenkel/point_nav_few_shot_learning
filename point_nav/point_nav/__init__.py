from gym.envs.registration import register

register(
        id='PointNav-v0',
        entry_point='point_nav.envs:PointNavEnv',
        )
