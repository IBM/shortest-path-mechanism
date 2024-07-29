#
# Copyright contributors to the shortest-path-mechanism project
# SPDX-License-Identifier: Apache-2.0
#
# Author: Hirota Kinoshita
#


"""Affine maximizer template."""

from collections import namedtuple


class AffineMaximizer:
    def __init__(self, options, weights, boost):
        self.options = options
        self.weights = weights
        self.boost = boost

    def __call__(self, types):
        def affine_welfare(option):
            return sum(0 if v_i is None else w_i * v_i(option) for v_i, w_i in zip(types, self.weights)) + self.boost(option)
        best_option = max(self.options, key=affine_welfare)
        
        Result = namedtuple('MaximizerResult', ['option', 'value'])
        return Result(best_option, affine_welfare(best_option))
