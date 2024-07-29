#
# Copyright contributors to the shortest-path-mechanism project
# SPDX-License-Identifier: Apache-2.0
#
# Author: Hirota Kinoshita
#


"""Implementation of the VCG mechanism."""

from mechanism import SociallyEfficientMechanism


# h_i(v_{-i}) := 1/w_i min_{v_i} max_{\phi} sum_{j} v_j(\phi)
class VCGbudget(SociallyEfficientMechanism):
    def compute_payments(self, type_ids: list):
        social_best = self.maximizer(
            [self.valuations[i][type_ids[i]] for i in self.agents()])
        return [min(self.maximizer([self.valuations[j][type_ids[j]] if j != i
                else v_i for j in self.agents()]).value for v_i in self.valuations[i])
                - social_best.value
                + self.valuations[i][type_ids[i]](social_best.option)
                for i in self.agents()]


class AMAbudget(SociallyEfficientMechanism):
    def compute_payments(self, type_ids: list):
        social_best = self.maximizer(
            [self.valuations[i][type_ids[i]] for i in self.agents()])
        return [(min(self.maximizer([self.valuations[j][type_ids[j]] if j != i
                else v_i for j in self.agents()]).value for v_i in self.valuations[i])
                 - social_best.value) / self.weights[i]
                + self.valuations[i][type_ids[i]](social_best.option)
                for i in self.agents()]


# https://en.wikipedia.org/wiki/Vickrey%E2%80%93Clarke%E2%80%93Groves_mechanism#The_Clarke_pivot_rule
# h_i(v_{-i}) := 1/w_i max_{\phi} sum_{j\neq i} w_j v_j(\phi) + \lambda
class VCGClarke(SociallyEfficientMechanism):
    def compute_payments(self, type_ids: list):
        social_best = self.maximizer(
            [self.valuations[i][type_ids[i]] for i in self.agents()])
        return [self.maximizer([self.valuations[j][type_ids[j]] if j != i else None for j in self.agents()]).value
                - social_best.value
                + self.valuations[i][type_ids[i]](social_best.option)
                for i in self.agents()]


class AMAClarke(SociallyEfficientMechanism):
    def compute_payments(self, type_ids: list):
        social_best = self.maximizer(
            [self.valuations[i][type_ids[i]] for i in self.agents()])
        return [(self.maximizer([self.valuations[j][type_ids[j]] if j != i else None for j in self.agents()]).value
                - social_best.value) / self.weights[i]
                + self.valuations[i][type_ids[i]](social_best.option)
                for i in self.agents()]
