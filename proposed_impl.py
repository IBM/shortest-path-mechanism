#
# Copyright contributors to the shortest-path-mechanism project
# SPDX-License-Identifier: Apache-2.0
#
# Author: Hirota Kinoshita
#


"""Implementation of the proposed mechanism."""

import networkx

from mechanism import SociallyEfficientMechanism


class Proposed(SociallyEfficientMechanism):
    """The proposed mechanism.
    """

    def compute_payments(self, type_ids: list):
        payments = []

        for i in self.agents():
            option_to_types = dict()
            for v_i in self.valuations[i]:
                option = self.maximizer(
                    [self.valuations[j][type_ids[j]] if j != i else v_i for j in self.agents()]).option
                if option not in option_to_types:
                    option_to_types[option] = []
                option_to_types[option].append(v_i)

            aux_node = 'AUX_NODE'
            vertices = list(option_to_types) + [aux_node]
            graph = networkx.DiGraph()
            graph.add_nodes_from(vertices)

            for option, types in option_to_types.items():
                graph.add_edge(aux_node, option, weight=min(
                    v_i(option) for v_i in types))
                for other_option in option_to_types:
                    graph.add_edge(other_option, option, weight=min(v_i(option) - v_i(other_option)
                                   for v_i in types))

            payments.append(networkx.shortest_path_length(graph, aux_node,
                                                          self.compute_option(type_ids), weight='weight', method="bellman-ford"))

        return payments
