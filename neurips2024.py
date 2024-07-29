#
# Copyright contributors to the shortest-path-mechanism project
# SPDX-License-Identifier: Apache-2.0
#
# Author: Hirota Kinoshita, Takayuki Osogami
#


"""Experiments for NeurIPS 2024"""

import argparse
from pathlib import Path
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import random_instance
from proposed_impl import Proposed
from vcg_impl import VCGbudget
# from vcg_impl import VCGClarke


path_result = Path("results")
path_result.mkdir(exist_ok=True)


def set_rcParams():
    plt.rcParams["figure.figsize"] = (2.5, 1.8)
    plt.rcParams["lines.linewidth"] *= 0.3

    plt.rcParams["hatch.linewidth"] *= 0.5
    plt.rcParams["hatch.color"] = '0.8'

    plt.rcParams["axes.labelsize"] = 7
    plt.rcParams["legend.fontsize"] = 7
    plt.rcParams["xtick.labelsize"] = 5
    plt.rcParams["ytick.labelsize"] = 5

    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

    plt.rcParams["figure.subplot.left"] = 0.18
    plt.rcParams["figure.subplot.bottom"] = 0.18
    plt.rcParams["figure.subplot.right"] = 0.99
    plt.rcParams["figure.subplot.top"] = 0.99

    # plt.rcParams["figure.subplot.wspace"] = 0.18
    # plt.rcParams["figure.subplot.hspace"] = 0.23

    plt.rcParams['lines.markersize'] *= 0.5


def hist_diff(fix, n_max, m_max, d_max, value_min, value_max, reps: int, lower_bound=False):
    prefix = f"hist_diff_{fix}_{n_max}_{m_max}_{d_max}_{value_min}_{value_max}_{reps}_{lower_bound}"
    
    generator = random_instance.random_environment
    plt.figure()

    if fix == "n":
        params = list((n_max, random.randint(1, m_max), random.randint(1, d_max), value_min, value_max)
                      for i in range(reps))
    elif fix == "m":
        params = list((random.randint(1, n_max), m_max, random.randint(1, d_max), value_min, value_max)
                      for i in range(reps))
    elif fix == "d":
        params = list((random.randint(1, n_max), random.randint(1, m_max), d_max, value_min, value_max)
                      for i in range(reps))

    path_bud_diff = path_result / f"{prefix}_bud_diff.pkl"
    path_util_sum = path_result / f"{prefix}_util_sum.pkl"

    if path_bud_diff.exists() and path_util_sum.exists():
        bud_diff = pickle.load(open(path_bud_diff, "rb"))
        util_sum = pickle.load(open(path_util_sum, "rb"))

    else:
        bud_diff = []
        util_sum = []

        for param in tqdm(params):
            mechanism_param = generator(*param)
            type_ids = [random.randrange(len(types))
                        for types in mechanism_param[0]]

            proposed = Proposed(*mechanism_param)
            proposed_option, proposed_utilities, proposed_payments = proposed.run(
                type_ids)
            proposed_bud = -sum(proposed_payments)

            vcgb = VCGbudget(*mechanism_param)
            vcgb_option, vcgb_utilities, vcgb_payments = vcgb.run(type_ids)
            vcg_bud = -sum(vcgb_payments)

            # bud_diff.append(vcg_bud - proposed_bud)
            bud_diff.append(proposed_bud - vcg_bud)
            util_sum.append(sum(proposed_utilities))

        with open(path_bud_diff, "wb") as f:
            pickle.dump(bud_diff, f)

        with open(path_util_sum, "wb") as f:
            pickle.dump(util_sum, f)
        
    if lower_bound:
        bins = np.arange(min(bud_diff + util_sum),
                         max(bud_diff + util_sum) + 25, 25)
        # bins = np.concatenate([-np.arange(0, -min(bud_diff + util_sum), 1)[::-1], np.arange(1, max(bud_diff + util_sum) + 1, 1)])
        weights = np.ones_like(bud_diff) / len(bud_diff)
        freq1, _, _ = plt.hist(bud_diff, bins=bins, weights=weights,
                               label="from VCG-budget", alpha=0.7, color='black')
        freq2, _, _ = plt.hist(util_sum, bins=bins, weights=weights,
                               label="from Lower bound", alpha=0.3, color='grey')
        # plt.xticks(np.arange(min(bud_diff + util_sum), max(bud_diff + util_sum) + 2.5, 5))
        xticks = np.concatenate([-np.arange(0, -min(bud_diff + util_sum), 500)
                                [::-1], np.arange(500, max(bud_diff + util_sum), 500)])
        plt.xticks(xticks)
        plt.yticks(np.arange(0.0, max(freq1.max(), freq2.max()), 0.05))
        # plt.axvline(x=0, color='0.3', linewidth=0.2)
        # plt.hist(bud_diff, bins=np.arange(0, max(max(bud_diff), max(util_sum)) + 1, 1),
        # weights=np.ones_like(bud_diff) / len(bud_diff), label="vs. VCG-budget", alpha=0.7, color='black')
        # plt.hist(util_sum, bins=np.arange(0, max(max(bud_diff), max(util_sum)) + 1, 1),
        # weights=np.ones_like(util_sum) / len(util_sum), label="vs. Lower bound", alpha=0.25, hatch='//', color='grey')
        # plt.xticks(np.arange(0, max(max(bud_diff), max(util_sum) + 2.5), 5))
        # plt.yticks(np.arange(0.0, 0.35, 0.1))
        # plt.xlim(left=0)
        plt.xlabel("Difference")
        plt.ylabel("Frequency")
        plt.legend()
    else:
        bins = -np.arange(-25 * (value_max - value_min) // 200, -min(bud_diff) + 26 * (value_max - value_min) // 200, max([1, 25 * (value_max - value_min) // 200]))[::-1]
        freq, _, _ = plt.hist(bud_diff, bins=bins,
                              weights=np.ones_like(bud_diff) / len(bud_diff), alpha=0.7, color='black')
        xticks = -np.arange(0, -min(bud_diff) + 50 * (value_max - value_min) // 200, 100 * (value_max - value_min) // 200)[::-1]
        plt.xticks(xticks)
        plt.yticks(np.arange(0.0, freq.max(), 0.05))
        plt.xlabel("Difference")
        plt.ylabel("Frequency")

        print(freq)

    path_fig = path_result / f"{prefix}.pdf"    
    plt.savefig(path_fig, bbox_inches="tight")


def plot_diff_vs_n(n_max: int, m_max: int, d_max: int, value_min: int, value_max: int, reps: int, errfn):
    prefix = f"plot_diff_vs_n_{n_max}_{m_max}_{d_max}_{value_min}_{value_max}_{reps}_stddev"

    generator = random_instance.random_environment
    plt.figure()
    ax = plt.subplot()
    if n_max > 16:
        n_vals = [max(1, i * n_max // 16) for i in range(0, 17)]
    else:
        n_vals = range(1, n_max + 1)

    path_avgs = path_result / f"{prefix}_avgs.pkl"
    path_errs = path_result / f"{prefix}_errs.pkl"

    if path_avgs.exists() and path_errs.exists():
        diff_avgs = pickle.load(open(path_avgs, "rb"))
        diff_errs = pickle.load(open(path_errs, "rb"))
    else:
        diff_avgs = []
        diff_errs = []

        for n in tqdm(n_vals, leave=False):
            diffs = []

            for _ in tqdm(range(reps), leave=False):
                m = random.randint(1, m_max)
                d = random.randint(1, d_max)
                param = (n, m, d, value_min, value_max)

                mechanism_param = generator(*param)
                type_ids = [random.randrange(len(types))
                            for types in mechanism_param[0]]

                proposed = Proposed(*mechanism_param)
                proposed_option, proposed_utilities, proposed_payments = proposed.run(
                    type_ids)
                proposed_bud = -sum(proposed_payments)

                vcgb = VCGbudget(*mechanism_param)
                vcgb_option, vcgb_utilities, vcgb_payments = vcgb.run(type_ids)
                vcgb_bud = -sum(vcgb_payments)

                assert proposed_option == vcgb_option

                diffs.append(proposed_bud - vcgb_bud)

            diff_avgs.append(np.average(diffs))
            diff_errs.append(errfn(diffs))

        with open(path_avgs, "wb") as f:
            pickle.dump(diff_avgs, f)

        with open(path_errs, "wb") as f:
            pickle.dump(diff_errs, f)

    ax.errorbar(n_vals, diff_avgs, yerr=diff_errs,
                linestyle='solid', alpha=0.8, color='0')
    # # the horizontal line y = 0
    # ax_bud.hlines([0], 0, 1, transform=ax_bud.get_yaxis_transform(),
    #             colors='black', linewidth=1, linestyle='dashed')
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Average difference")
    ax.axhline(y=0, linestyle='dashed', alpha=0.3, color='grey')

    path_fig = path_result / f"{prefix}.pdf"    
    plt.savefig(path_fig, bbox_inches="tight")
    

def plot_diff_vs_m(n: int, m_max: int, d_max: int, value_min: int, value_max: int, reps: int, errfn):
    prefix = f"plot_diff_vs_m_{n}_{m_max}_{d_max}_{value_min}_{value_max}_{reps}_stddev"

    generator = random_instance.random_environment
    plt.figure()
    ax = plt.subplot()
    if m_max > 16:
        m_vals = [max(1, i * m_max // 16) for i in range(0, 17)]
    else:
        m_vals = range(1, m_max + 1)

    path_avgs = path_result / f"{prefix}_avgs.pkl"
    path_errs = path_result / f"{prefix}_errs.pkl"

    if path_avgs.exists() and path_errs.exists():
        diff_avgs = pickle.load(open(path_avgs, "rb"))
        diff_errs = pickle.load(open(path_errs, "rb"))
    else:
        diff_avgs = []
        diff_errs = []

        for m in tqdm(m_vals, leave=False):
            diffs = []

            for _ in tqdm(range(reps), leave=False):
                d = random.randint(1, d_max)
                param = (n, m, d, value_min, value_max)

                mechanism_param = generator(*param)
                type_ids = [random.randrange(len(types))
                            for types in mechanism_param[0]]

                proposed = Proposed(*mechanism_param)
                proposed_option, proposed_utilities, proposed_payments = proposed.run(
                    type_ids)
                proposed_bud = -sum(proposed_payments)

                vcgb = VCGbudget(*mechanism_param)
                vcgb_option, vcgb_utilities, vcgb_payments = vcgb.run(type_ids)
                vcgb_bud = -sum(vcgb_payments)

                assert proposed_option == vcgb_option

                diffs.append(proposed_bud - vcgb_bud)

            diff_avgs.append(np.average(diffs))
            diff_errs.append(errfn(diffs))

        with open(path_avgs, "wb") as f:
            pickle.dump(diff_avgs, f)

        with open(path_errs, "wb") as f:
            pickle.dump(diff_errs, f)
            
    ax.errorbar(m_vals, diff_avgs, yerr=diff_errs,
                linestyle='solid', alpha=0.8, color='0')
    # # the horizontal line y = 0
    # ax_bud.hlines([0], 0, 1, transform=ax_bud.get_yaxis_transform(),
    #             colors='black', linewidth=1, linestyle='dashed')
    ax.set_xlabel("Number of options")
    ax.set_ylabel("Average difference")
    ax.axhline(y=0, linestyle='dashed', alpha=0.3, color='grey')

    path_fig = path_result / f"{prefix}.pdf"    
    plt.savefig(path_fig, bbox_inches="tight")

    
def plot_diff_vs_d(n: int, m_max: int, d_max: int, value_min: int, value_max: int, reps: int, errfn):
    prefix = f"plot_diff_vs_d_{n}_{m_max}_{d_max}_{value_min}_{value_max}_{reps}_stddev"

    generator = random_instance.random_environment
    plt.figure()
    ax = plt.subplot()
    if d_max > 16:
        d_vals = [max(1, i * d_max // 16) for i in range(0, 17)]
    else:
        d_vals = range(1, d_max + 1)

    path_avgs = path_result / f"{prefix}_avgs.pkl"
    path_errs = path_result / f"{prefix}_errs.pkl"

    if path_avgs.exists() and path_errs.exists():
        diff_avgs = pickle.load(open(path_avgs, "rb"))
        diff_errs = pickle.load(open(path_errs, "rb"))
    else:
        diff_avgs = []
        diff_errs = []

        for d in tqdm(d_vals, leave=False):
            diffs = []

            for _ in tqdm(range(reps), leave=False):
                m = random.randint(1, m_max)
                param = (n, m, d, value_min, value_max)

                mechanism_param = generator(*param)
                type_ids = [random.randrange(len(types))
                            for types in mechanism_param[0]]

                proposed = Proposed(*mechanism_param)
                proposed_option, proposed_utilities, proposed_payments = proposed.run(
                    type_ids)
                proposed_bud = -sum(proposed_payments)

                vcgb = VCGbudget(*mechanism_param)
                vcgb_option, vcgb_utilities, vcgb_payments = vcgb.run(type_ids)
                vcgb_bud = -sum(vcgb_payments)

                assert proposed_option == vcgb_option

                diffs.append(proposed_bud - vcgb_bud)

            diff_avgs.append(np.average(diffs))
            diff_errs.append(errfn(diffs))

        with open(path_avgs, "wb") as f:
            pickle.dump(diff_avgs, f)

        with open(path_errs, "wb") as f:
            pickle.dump(diff_errs, f)
            
    ax.errorbar(d_vals, diff_avgs, yerr=diff_errs,
                linestyle='solid', alpha=0.8, color='0')
    # # the horizontal line y = 0
    # ax_bud.hlines([0], 0, 1, transform=ax_bud.get_yaxis_transform(),
    #             colors='black', linewidth=1, linestyle='dashed')
    ax.set_xlabel("Size of type domain")
    ax.set_ylabel("Average difference")
    ax.axhline(y=0, linestyle='dashed', alpha=0.3, color='grey')

    path_fig = path_result / f"{prefix}.pdf"    
    plt.savefig(path_fig, bbox_inches="tight")


def plot_budget_vs_n(n_max, m_max, d_max, value_min, value_max, reps, lower_bound=False):
    generator = random_instance.random_environment
    plt.figure()
    ax = plt.subplot()
    if n_max > 16:
        n_vals = [max(1, i * n_max // 16) for i in range(0, 17)]
    else:
        n_vals = range(1, n_max + 1)
    proposed_bud = []
    # vcgc_bud = []
    vcgb_bud = []
    lower_bnd = []

    for n in tqdm(n_vals, leave=False):
        tmp_proposed_bud = []
        # tmp_vcgc_bud = []
        tmp_vcgb_bud = []
        tmp_lower_bnd = []

        for _ in tqdm(range(reps), leave=False):
            m = random.randint(1, m_max)
            d = random.randint(1, d_max)
            param = (n, m, d, value_min, value_max)

            mechanism_param = generator(*param)
            type_ids = [random.randrange(len(types))
                        for types in mechanism_param[0]]

            proposed = Proposed(*mechanism_param)
            proposed_option, proposed_utilities, proposed_payments = proposed.run(
                type_ids)
            tmp_proposed_bud.append(-sum(proposed_payments))

            # vcgc = VCGClarke(*mechanism_param)
            # vcgc_option, vcgc_utilities, vcgc_payments = vcgc.run(type_ids)
            # tmp_vcgc_bud.append(-sum(vcgc_payments))

            vcgb = VCGbudget(*mechanism_param)
            vcgb_option, vcgb_utilities, vcgb_payments = vcgb.run(type_ids)
            tmp_vcgb_bud.append(-sum(vcgb_payments))

            assert proposed_option == vcgb_option

            tmp_lower_bnd.append(-sum(types[type_id](proposed_option)
                                 for types, type_id in zip(mechanism_param[0], type_ids)))

        lower_bnd.append(np.average(tmp_lower_bnd))
        proposed_bud.append(np.average(tmp_proposed_bud))
        # vcgc_bud.append(average(tmp_vcgc_bud))
        vcgb_bud.append(np.average(tmp_vcgb_bud))

    ax.plot(n_vals, proposed_bud, label="Proposed",
            linestyle='solid', alpha=0.8, color='0')
    # ax_bud.plot(x_vals, vcgc_bud, label="VCG-Clarke")
    ax.plot(n_vals, vcgb_bud, label="VCG-budget",
            linestyle='dotted', alpha=0.8, color='0.5')
    if lower_bound:
        ax.plot(n_vals, lower_bnd, label="Lower bound",
                linestyle='dashed', alpha=0.8, color='0.75')
    # # the horizontal line y = 0
    # ax_bud.hlines([0], 0, 1, transform=ax_bud.get_yaxis_transform(),
    #             colors='black', linewidth=1, linestyle='dashed')
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Average budget")
    ax.legend()


def plot_budget_vs_m(n, m_max, d_max, value_min, value_max, reps, lower_bound=False):
    generator = random_instance.random_environment
    plt.figure()
    ax = plt.subplot()
    if m_max > 16:
        m_vals = [max(1, i * m_max // 16) for i in range(0, 17)]
    else:
        m_vals = range(1, m_max + 1)
    proposed_bud = []
    # vcgc_bud = []
    vcgb_bud = []
    lower_bnd = []

    for m in tqdm(m_vals, leave=False):
        tmp_proposed_bud = []
        # tmp_vcgc_bud = []
        tmp_vcgb_bud = []
        tmp_lower_bnd = []

        for _ in tqdm(range(reps), leave=False):
            d = random.randint(1, d_max)
            param = (n, m, d, value_min, value_max)

            mechanism_param = generator(*param)
            type_ids = [random.randrange(len(types))
                        for types in mechanism_param[0]]

            proposed = Proposed(*mechanism_param)
            proposed_option, proposed_utilities, proposed_payments = proposed.run(
                type_ids)
            tmp_proposed_bud.append(-sum(proposed_payments))

            # vcgc = VCGClarke(*mechanism_param)
            # vcgc_option, vcgc_utilities, vcgc_payments = vcgc.run(type_ids)
            # tmp_vcgc_bud.append(-sum(vcgc_payments))

            vcgb = VCGbudget(*mechanism_param)
            vcgb_option, vcgb_utilities, vcgb_payments = vcgb.run(type_ids)
            tmp_vcgb_bud.append(-sum(vcgb_payments))

            assert proposed_option == vcgb_option

            tmp_lower_bnd.append(-sum(types[type_id](proposed_option)
                                 for types, type_id in zip(mechanism_param[0], type_ids)))

        lower_bnd.append(np.average(tmp_lower_bnd))
        proposed_bud.append(np.average(tmp_proposed_bud))
        # vcgc_bud.append(average(tmp_vcgc_bud))
        vcgb_bud.append(np.average(tmp_vcgb_bud))

    ax.plot(m_vals, proposed_bud, label="Proposed",
            linestyle='solid', alpha=0.8, color='0')
    # ax_bud.plot(x_vals, vcgc_bud, label="VCG-Clarke")
    ax.plot(m_vals, vcgb_bud, label="VCG-budget",
            linestyle='dotted', alpha=0.8, color='0.5')
    if lower_bound:
        ax.plot(m_vals, lower_bnd, label="Lower bound",
                linestyle='dashed', alpha=0.8, color='0.75')
    # # the horizontal line y = 0
    # ax_bud.hlines([0], 0, 1, transform=ax_bud.get_yaxis_transform(),
    #             colors='black', linewidth=1, linestyle='dashed')
    ax.set_xlabel("Number of options")
    ax.set_ylabel("Average budget")
    ax.legend()


def plot_budget_vs_d(n, m_max, d_max, value_min, value_max, reps, lower_bound=False):
    generator = random_instance.random_environment
    plt.figure()
    ax = plt.subplot()
    if d_max > 16:
        d_vals = [max(1, i * d_max // 16) for i in range(0, 17)]
    else:
        d_vals = range(1, d_max + 1)
    proposed_bud = []
    proposed_err = []
    # vcgc_bud = []
    vcgb_bud = []
    vcgb_err = []
    lower_bnd = []

    for d in tqdm(d_vals, leave=False):
        tmp_proposed_bud = []
        # tmp_vcgc_bud = []
        tmp_vcgb_bud = []
        tmp_lower_bnd = []

        for _ in tqdm(range(reps), leave=False):
            m = random.randint(1, m_max)
            param = (n, m, d, value_min, value_max)

            mechanism_param = generator(*param)
            type_ids = [random.randrange(len(types))
                        for types in mechanism_param[0]]

            proposed = Proposed(*mechanism_param)
            proposed_option, proposed_utilities, proposed_payments = proposed.run(
                type_ids)
            tmp_proposed_bud.append(-sum(proposed_payments))

            # vcgc = VCGClarke(*mechanism_param)
            # vcgc_option, vcgc_utilities, vcgc_payments = vcgc.run(type_ids)
            # tmp_vcgc_bud.append(-sum(vcgc_payments))

            vcgb = VCGbudget(*mechanism_param)
            vcgb_option, vcgb_utilities, vcgb_payments = vcgb.run(type_ids)
            tmp_vcgb_bud.append(-sum(vcgb_payments))

            assert proposed_option == vcgb_option

            tmp_lower_bnd.append(-sum(types[type_id](proposed_option)
                                 for types, type_id in zip(mechanism_param[0], type_ids)))

        lower_bnd.append(np.average(tmp_lower_bnd))
        proposed_bud.append(np.average(tmp_proposed_bud))
        # vcgc_bud.append(average(tmp_vcgc_bud))
        vcgb_bud.append(np.average(tmp_vcgb_bud))

        proposed_err.append(np.std(tmp_proposed_bud, ddof=1))
        vcgb_err.append(np.std(tmp_vcgb_bud, ddof=1))

    ax.plot(d_vals, proposed_bud, label="Proposed",
            linestyle='solid', alpha=0.8, color='0')
    # ax_bud.plot(x_vals, vcgc_bud, label="VCG-Clarke")
    ax.plot(d_vals, vcgb_bud, label="VCG-budget",
            linestyle='dotted', alpha=0.8, color='0.5')
    if lower_bound:
        ax.plot(d_vals, lower_bnd, label="Lower bound",
                linestyle='dashed', alpha=0.8, color='0.75')
    # # the horizontal line y = 0
    # ax_bud.hlines([0], 0, 1, transform=ax_bud.get_yaxis_transform(),
    #             colors='black', linewidth=1, linestyle='dashed')
    ax.set_xlabel("Size of type domain")
    ax.set_ylabel("Average budget")
    ax.legend()

    plt.figure()
    ax = plt.subplot()
    ax.errorbar(d_vals, proposed_bud, yerr=proposed_err, label="Proposed",
                linestyle='solid', alpha=0.8, color='0')
    # ax_bud.plot(x_vals, vcgc_bud, label="VCG-Clarke")
    ax.errorbar(d_vals, vcgb_bud, yerr=vcgb_err, label="VCG-budget",
                linestyle='dotted', alpha=0.8, color='0.5')
    if lower_bound:
        ax.plot(d_vals, lower_bnd, label="Lower bound",
                linestyle='dashed', alpha=0.8, color='0.75')
    # # the horizontal line y = 0
    # ax_bud.hlines([0], 0, 1, transform=ax_bud.get_yaxis_transform(),
    #             colors='black', linewidth=1, linestyle='dashed')
    ax.set_xlabel("Size of type domain")
    ax.set_ylabel("Average budget")
    ax.legend()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--hist", action="store_true")
    parser.add_argument("--vs_n", action="store_true")
    parser.add_argument("--vs_m", action="store_true")
    parser.add_argument("--vs_d", action="store_true")
    parser.add_argument("--fix", type=str, default="n")
    parser.add_argument("--n_max", type=int, default=32)
    parser.add_argument("--m_max", type=int, default=256)
    parser.add_argument("--d_max", type=int, default=16)
    parser.add_argument("--value_min", type=int, default=-100)
    parser.add_argument("--value_max", type=int, default=100)
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()

    set_rcParams()

    if args.hist:
        print("running hist_diff")
        hist_diff(
            fix=args.fix,
            n_max=args.n_max,
            m_max=args.m_max,
            d_max=args.d_max,
            value_min=args.value_min,
            value_max=args.value_max,
            reps=args.reps,
            lower_bound=False
            #n=16, m_max=256, d_max=16, value_min=-100, value_max=100, reps=1000, lower_bound=False)        
        )
        
    def stddev(arr):
        return np.std(arr, ddof=1)

    if args.vs_n:
        plot_diff_vs_n(
            n_max=args.n_max,
            m_max=args.m_max,
            d_max=args.d_max,
            value_min=args.value_min,
            value_max=args.value_max,
            reps=args.reps,
            errfn=stddev
            # n_max=32, m_max=256, d_max=16,
            # value_min=-100, value_max=100, reps=100, errfn=stddev)
        )

    if args.vs_m:
        plot_diff_vs_m(
            n=args.n_max,
            m_max=args.m_max,
            d_max=args.d_max,
            value_min=args.value_min,
            value_max=args.value_max,
            reps=args.reps,
            errfn=stddev
            # n=16, m_max=256, d_max=16,
            # value_min=-100, value_max=100, reps=100, errfn=stddev)
        )
        
    if args.vs_d:
        plot_diff_vs_d(
            n=args.n_max,
            m_max=args.m_max,
            d_max=args.d_max,
            value_min=args.value_min,
            value_max=args.value_max,
            reps=args.reps,
            errfn=stddev
            # n=16, m_max=256, d_max=16,
            # value_min=-100, value_max=100, reps=100, errfn=stddev)
        )

    # plot_budget_vs_n(
    #     n_max=32, m_max=256, d_max=16, value_min=-100, value_max=100, reps=100, lower_bound=False)
    # plot_budget_vs_m(
    #     n=16, m_max=256, d_max=16, value_min=-100, value_max=100, reps=100, lower_bound=False)
    # plot_budget_vs_d(
    #     n=16, m_max=256, d_max=16, value_min=-100, value_max=100, reps=100, lower_bound=False)
    

    # plt.tight_layout()
    plt.show()
