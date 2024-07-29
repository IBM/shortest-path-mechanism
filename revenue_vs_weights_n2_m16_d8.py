#
# Copyright contributors to the shortest-path-mechanism project
# SPDX-License-Identifier: Apache-2.0
#
# Author: Hirota Kinoshita
#


"""Examine how revenue changes by weights for instances
with 2 agents, 16 options, 8 types per agent."""

import itertools
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from affine_maximizer import AffineMaximizer
from proposed_impl import Proposed
from vcg_impl import AMAbudget
from vcg_impl import AMAClarke
from valuation import Valuation


NUM_OF_SAMPLES = 10

NUM_OF_AGENTS = 2
agents = range(NUM_OF_AGENTS)

NUM_OF_OPTIONS = 16
options = range(NUM_OF_OPTIONS)

SIZE_OF_DOMAIN = 8

WEIGHT_SUM = 100
weight_range = np.column_stack(
    (np.arange(1, WEIGHT_SUM), WEIGHT_SUM - np.arange(1, WEIGHT_SUM)))


def plot_with_sample(plot_id: int, identical_domain: bool):
    """Plot expected revenue against weights, using a randomly sampled environment.

    Args:
        plot_id (int): an identifier for the present plot, which is used as a suffix of the figure.
        identical_domain (bool): whether two domains are identical.

    Returns:
        None
    """
    if identical_domain:
        # The same domain for two players.
        type_domain = [Valuation.random_valuation(
            options, -100, 100) for _ in range(SIZE_OF_DOMAIN)]
        type_domains = [type_domain for agent in agents]
    else:
        # Independently generated domains for two players.
        type_domains = [[Valuation.random_valuation(
            options, -100, 100) for _ in range(SIZE_OF_DOMAIN)] for agent in agents]

    def boost(option):
        del option
        return 0

    proposed_avgs = []
    amab_avgs = []
    amac_avgs = []

    for weights in tqdm(weight_range, leave=False):
        maximizer = AffineMaximizer(options, weights, boost)
        env = (type_domains, weights, boost, maximizer)
        proposed = Proposed(*env)
        amab = AMAbudget(*env)
        amac = AMAClarke(*env)
        proposed_revs = [sum(proposed.run(type_ids)[2]) for type_ids in itertools.product(
            range(SIZE_OF_DOMAIN), repeat=NUM_OF_AGENTS)]
        amab_revs = [sum(amab.run(type_ids)[2]) for type_ids in itertools.product(
            range(SIZE_OF_DOMAIN), repeat=NUM_OF_AGENTS)]
        amac_revs = [sum(amac.run(type_ids)[2]) for type_ids in itertools.product(
            range(SIZE_OF_DOMAIN), repeat=NUM_OF_AGENTS)]
        proposed_avgs.append(np.mean(proposed_revs))
        amab_avgs.append(np.mean(amab_revs))
        amac_avgs.append(np.mean(amac_revs))

    plt.plot(weight_range[:, 0], proposed_avgs, label='Proposed')
    plt.plot(weight_range[:, 0], amab_avgs, label='AMA-budget')
    plt.plot(weight_range[:, 0], amac_avgs, label='AMA-Clarke')
    plt.xlabel(fr'$w_0 = {WEIGHT_SUM} - w_1$')
    plt.ylabel('Expected revenue')
    plt.legend()
    plt.title(r'Revenue against weights ($|\mathcal{N}|=2,|\mathcal{X}|=16,|\mathcal{V}_i|=8,$'
              + f'{'identical' if identical_domain else 'distinct'} domain)')
    plt.savefig(Path(__file__).parent / 'figs' /
                f'{Path(__file__).stem}_{'idt' if identical_domain else 'dis'}_{plot_id:03}.pdf')
    plt.close()


if __name__ == '__main__':
    for sample_id in tqdm(range(NUM_OF_SAMPLES)):
        plot_with_sample(sample_id, identical_domain=True)
        plot_with_sample(sample_id, identical_domain=False)
