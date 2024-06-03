import timeit

import matplotlib.pyplot as plt
import numpy as np

a = np.zeros((1, ))
s = np.zeros((1, ))


def iterate(a, s):
    return np.nonzero(np.isin(a, s))[0]


def searchsorted(a, s):
    a_sorted_indices = np.argsort(a)
    indices = np.searchsorted(a[a_sorted_indices], s)
    return a_sorted_indices[indices]


def main():
    # axes: sample size (0), selection size (1)
    searchsorted_times = np.zeros((7, 9))
    iterate_times = np.zeros((7, 9))

    for i, sample_size_exp in enumerate(range(2, 9, 1)):
        for j, selection_size in enumerate(range(10, 100, 10)):
            iterations = min(10**(8 - sample_size_exp), 10000)
            global a
            global s
            a = np.random.randint(
                1, 10**(sample_size_exp + 2), 10**sample_size_exp
            )
            a = np.unique(a)
            np.random.shuffle(a)
            s = np.random.choice(a, int(selection_size / 100 * len(a)))
            # test iterate
            iterate_times[i][j] = timeit.timeit(
                "iterate(a, s)",
                number=iterations,
                globals=globals(),
            ) / iterations
            print(
                f"Tested iterate for sample size 1e{sample_size_exp} and "
                f"selection size {selection_size}% at {iterations} iterations: "
                f"{iterate_times[i][j]} s"
            )
            # test searchsorted
            searchsorted_times[i][j] = timeit.timeit(
                "searchsorted(a, s)",
                number=iterations,
                globals=globals(),
            ) / iterations
            print(
                f"Tested searchsorted for sample size 1e{sample_size_exp} and "
                f"selection size {selection_size}% at {iterations} iterations: "
                f"{searchsorted_times[i][j]} s"
            )

    print("Iterate times:")
    print(iterate_times)
    print("Searchsorted times:")
    print(searchsorted_times)

    fig, axes = plt.subplots(figsize=(4, 4))
    fig.set_tight_layout(True)
    axes.set_xlabel("Sample size [log10]")
    axes.set_ylabel("Mean runtime [log10 s]")
    axes.set_yscale("log")
    xs = np.linspace(2, 10, 8)

    for k in range(9):
        # iterate
        color = np.array([0, 0, 1]) * (k + 1) / 9
        axes.plot(
            xs,
            iterate_times[:, k],
            marker="^",
            linestyle="solid",
            color=color
        )
        # searchsorted
        color = np.array([1, 0, 0]) * (k + 1) / 9
        axes.plot(
            xs,
            searchsorted_times[:, k],
            marker="o",
            linestyle="solid",
            color=color
        )

    fig.savefig("timeit.png", bbox_inches="tight")


if __name__ == '__main__':
    main()
