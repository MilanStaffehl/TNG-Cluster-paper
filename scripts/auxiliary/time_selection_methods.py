import sys
import timeit
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root_dir = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root_dir / "src"))

from library.processing import selection  # noqa: F401

a = np.zeros((1, ))
s = np.zeros((1, ))
is_subset = True


def main():
    # global is_subset
    sample_sizes = np.arange(2, 10, 1, dtype=int)
    selection_sizes = [0.01, 0.1, 0.3, 0.5, 0.9]
    # axes: sample size (0), selection size (1)
    results_shape = len(sample_sizes), len(selection_sizes)
    searchsort_times = np.zeros(results_shape)
    intersect_times = np.zeros(results_shape)
    iterate_times = np.zeros(results_shape)

    for i, sample_size_exp in enumerate(sample_sizes):
        for j, selection_size in enumerate(selection_sizes):
            iterations = min(max(10**(8 - int(sample_size_exp)), 1), 10000)
            global a
            global s
            a = np.random.randint(
                1, 10**(sample_size_exp + 2), 10**sample_size_exp
            )
            a = np.unique(a)
            np.random.shuffle(a)
            s = np.random.choice(a, int(selection_size * len(a)))
            if not is_subset:
                # contaminate s with some values not in a
                mask = np.random.choice([0, 1], size=s.shape,
                                        p=[0.8, 0.2]).astype(bool)
                s[mask] = 0  # zero is not in a

            # test iterate
            iterate_times[i][j] = timeit.timeit(
                "selection.select_if_in(a, s, mode='iterate', assume_subset=is_subset)",
                number=iterations,
                globals=globals(),
            ) / iterations
            print(
                f"Tested iterate for sample size 1e{sample_size_exp} and "
                f"selection size {selection_size} at {iterations} iterations: "
                f"{iterate_times[i][j]} s"
            )
            # test intersect
            intersect_times[i][j] = timeit.timeit(
                "selection.select_if_in(a, s, mode='intersect', assume_subset=is_subset)",
                number=iterations,
                globals=globals(),
            ) / iterations
            print(
                f"Tested intersect for sample size 1e{sample_size_exp} and "
                f"selection size {selection_size} at {iterations} iterations: "
                f"{iterate_times[i][j]} s"
            )
            # test searchsorted
            searchsort_times[i][j] = timeit.timeit(
                "selection.select_if_in(a, s, mode='searchsort', assume_subset=is_subset)",
                number=iterations,
                globals=globals(),
            ) / iterations
            print(
                f"Tested searchsorted for sample size 1e{sample_size_exp} and "
                f"selection size {selection_size} at {iterations} iterations: "
                f"{searchsort_times[i][j]} s"
            )

    print("Iterate times:")
    print(iterate_times)
    print("Searchsorted times:")
    print(searchsort_times)

    fig, axes = plt.subplots(figsize=(4, 4))
    fig.set_tight_layout(True)
    axes.set_xlabel("Sample size [log10]")
    axes.set_ylabel("Mean runtime [log10 s]")
    axes.set_yscale("log")
    if is_subset:
        axes.set_title("s is subset of a")
    else:
        axes.set_title("s is NOT a subset of a")

    n_selections = results_shape[-1]
    for k in range(n_selections):
        # iterate
        color = np.array([0, 0, 1]) * (k + 1) / n_selections
        axes.plot(
            sample_sizes,
            iterate_times[:, k],
            marker="^",
            linestyle="solid",
            color=color
        )
        # intersect
        color = np.array([0, 1, 0]) * (k + 1) / n_selections
        axes.plot(
            sample_sizes,
            iterate_times[:, k],
            marker="v",
            linestyle="solid",
            color=color
        )
        # searchsorted
        color = np.array([1, 0, 0]) * (k + 1) / n_selections
        axes.plot(
            sample_sizes,
            searchsort_times[:, k],
            marker="o",
            linestyle="solid",
            color=color
        )

    if is_subset:
        filename = "timeit_subset"
    else:
        filename = "timeit_not_subset"
    fig.savefig(f"tmp/{filename}.pdf", bbox_inches="tight")

    # save data
    np.savez(
        f"tmp/{filename}.npz",
        iterate_times=iterate_times,
        intersect_times=intersect_times,
        searchsort_times=searchsort_times,
    )


if __name__ == '__main__':
    main()
