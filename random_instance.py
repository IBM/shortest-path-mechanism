#
# Copyright contributors to the shortest-path-mechanism project
# SPDX-License-Identifier: Apache-2.0
#
# Author: Hirota Kinoshita
#


"""Random instance generators."""

from collections import namedtuple
import random

from valuation import Valuation


def random_environment(num_of_players: int, num_of_options: int, size_of_domain: int, value_min: int, value_max: int):
    """Return a random instance.
    """
    valuations = [
        [Valuation({x: random.randint(value_min, value_max) for x in range(num_of_options)})
        # [Valuation({x: random.uniform(-1, 1) for x in range(num_of_options)})
         for t in range(size_of_domain)]
        for i in range(num_of_players)
    ]

    weights = [1 for i in range(num_of_players)]

    def boost(option):
        return 0

    def maximizer(types):
        Result = namedtuple('MaximizerResult', ['option', 'value'])
        def social_welfare(x):
            return sum(0 if v_i is None else v_i(x) for v_i, weight in zip(types, weights))
        option = max(range(num_of_options), key=social_welfare)
        return Result(option, social_welfare(option))

    return (valuations, weights, boost, maximizer)


def random_weighted_environment(num_of_players: int, num_of_options: int, size_of_domain: int, value_min, value_max, weight_min, weight_max, lambda_min, lambda_max):
    """Return a random weighted instance.
    """
    valuations = [
        [Valuation({x: random.randint(value_min, value_max) for x in range(num_of_options)})
        # [Valuation({x: random.uniform(-1, 1) for x in range(num_of_options)})
         for t in range(size_of_domain)]
        for i in range(num_of_players)
    ]

    assert weight_min > 0
    weights = [random.uniform(weight_min, weight_max) for i in range(num_of_players)]
    boosts = [random.uniform(lambda_min, lambda_max) for x in range(num_of_options)]

    def boost(option):
        return boosts[option]

    def maximizer(types):
        Result = namedtuple('MaximizerResult', ['option', 'value'])
        def social_welfare(x):
            return sum(0 if v_i is None else v_i(x) for v_i, weight in zip(types, weights))
        option = max(range(num_of_options), key=social_welfare)
        return Result(option, social_welfare(option))

    return (valuations, weights, boost, maximizer)
