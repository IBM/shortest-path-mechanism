#
# Copyright contributors to the shortest-path-mechanism project
# SPDX-License-Identifier: Apache-2.0
#
# Author: Hirota Kinoshita
#


"""Valuation template."""

import random


class Valuation:
    def __init__(self, dic: dict):
        self.mp = dic

    def __call__(self, option):
        return self.mp[option]

    def add_value(self, option, value):
        self.mp[option] += value

    def set_value(self, option, value):
        self.mp[option] = value

    @staticmethod
    def random_valuation(options, value_min, value_max):
        return Valuation({option: random.randint(value_min, value_max) for option in options})
