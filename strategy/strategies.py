from strategy.random import random_sampling
from strategy.meta.random_meta import RandomMetaProcedure
from strategy.meta.thompson_meta import ThompsonMetaProcedure
from strategy.meta.albl_meta import ALBLMetaProcedure


def get_strategies(name, sub_names):
    if name == "random":
        return random_sampling, None
    elif name == "random_meta":
        strategy = RandomMetaProcedure(sub_names)
        return strategy.sample, strategy
    elif name == "thompson_pos":
        strategy = ThompsonMetaProcedure(sub_names, "pos")
        return strategy.sample, strategy
    elif name == "thompson_div":
        strategy = ThompsonMetaProcedure(sub_names, "div")
        return strategy.sample, strategy
    elif name == "albl_meta":
        strategy = ALBLMetaProcedure(sub_names)
        return strategy.sample, strategy
