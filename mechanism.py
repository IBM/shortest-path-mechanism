#
# Copyright contributors to the shortest-path-mechanism project
# SPDX-License-Identifier: Apache-2.0
#


"""Mechanism template."""

class Mechanism:
    """A general mechanism class.
    """

    def __init__(self, valuations, weights, boost):
        """Initialize the mechanism.

        Args:
            valuations (2D-array-like): a collection of each agent's valuations. A valuation must be callable.
            weights (List-like): a list of positive weights for each agent.
            boost (Function-like): a function that maps each outcome to some value.
        """
        self.valuations = valuations
        self.weights = weights
        self.boost = boost

    def agents(self):
        """Return an iterable object that returns each agent.

        Yields:
            int: a agent.
        """
        yield from range(len(self.valuations))

    def run(self, type_ids: list) -> tuple:
        """Execute the mechanism.

        Args:
            type_ids (list): A list of types of each agent, represented as indices.

        Returns:
            tuple: a tuple consisting of an option, utilities, and payments.
        """
        return (self.compute_option(type_ids), self.compute_utilities(type_ids), self.compute_payments(type_ids))

    def compute_option(self, type_ids: list):
        raise NotImplementedError()

    def compute_utilities(self, type_ids: list):
        option = self.compute_option(type_ids)
        payments = self.compute_payments(type_ids)
        return [types[type_id](option) - payment for types, type_id, payment in zip(self.valuations, type_ids, payments)]

    def compute_payments(self, type_ids: list):
        raise NotImplementedError()


class SociallyEfficientMechanism(Mechanism):
    """A class of socially efficient mechanisms.
    """

    def __init__(self, valuations, weights, boost, maximizer):
        """Initialize the socially efficient mechanism.

        Args:
            valuations (2D-array-like): a collection of each agent's valuations. A valuation must be callable.
            weights (List-like): a list of positive weights for each agent.
            boost (Function-like): a function that maps each outcome to some value.
            maximizer (Function-like): a function that returns a pair of an optimal option and the maximum value.
        """
        super().__init__(valuations, weights, boost)
        self.maximizer = maximizer

    def compute_option(self, type_ids: list):
        """Determine an socially efficient option.

        Args:
            type_ids (list): A list of types of each agent, represented as indices.

        Returns:
            Any: an option that maximizes the total value.
        """
        return self.maximizer([types[type_id] for types, type_id in zip(self.valuations, type_ids)]).option
