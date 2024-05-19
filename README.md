# Maximum random baseline

This package implements a simple random baseline that accounts for dataset size
and evaluation set reuse.

The baseline is the expected maximum accuracy across multiple random classifiers,
as described in [Stronger Random Baselines for In-Context Learning](https://arxiv.org/pdf/2404.13020).

## Installation

You can install this package with pip:

```
pip install max-random-baseline
```

This package requires `numpy` and the
[`PoiBin` Python implementation](https://github.com/tsakim/poibin)
of Poisson binomial random variables. 
For ease of installation, `PoiBin` is included in this package.
`PoiBin` is copyright (c) 2016 Mika J. Straka under an MIT License.

## Quick start

The simplest way to get baseline numbers is to import and call the function
`max_random_baseline`:

```
from max_random_baseline import max_random_baseline
```

The syntax is ```max_random_baseline(n, p, t)```:

- `n`: number of examples in the dataset
- `p`: probability of guessing correctly on a single example.
  In the binary classification setting, `p = 0.5`.
- `t`: number of random classifiers to take the maximum over.
  For example, if you are comparing the baseline to the best validation accuracy
  across 10 different classifiers, then set `t = 10`.

For example, to get the expected maximum accuracy among 10 classifiers that
guess uniformly at random on a binary classification task with `n = 100`
examples: `max_random_baseline(100, 0.5, 10)`.


## Datasets with different numbers of labels for each example

There are additional formats for specifiying the probability of guessing correctly on each example.
Say you have a dataset with `n = 100` examples, where 50 of the examples have 2
possible labels (0.5 probability of guessing correctly) and the other 50 have 5
possible labels (0.2 probability of guessing correctly). The two equivalent syntaxes
for getting the maximum random baseline for this dataset with `t = 10` are:


- `max_random_baseline(100, {2: 50, 5: 50}, 10)`
    
   The second argument maps a number of possible labels to the number of
   examples with that many labels.

- `max_random_baseline(100, [0.5] * 50 + [0.2] * 50, 10)`
    
    The second argument is a list of `n` probabilities, one for each of the examples.

## Additional features

- `max_random_p_value(acc, n, p, t)`

    Returns the $p$-value of a given accuracy `acc` with respect to the maximum
    random baseline with parameters `n`, `p`, `t`.

- `max_random_pmf(num_correct, n, p, t)`

    Evaluates the probability mass function of the maximum order statistic of
    `t` draws from the Poisson binomial random variable with parameters `n` and `p`
    at the value `num_correct`. Note that `num_correct` is an integer between 0
    and `n` (inclusive) rather than an accuracy.

- `max_random_F(num_correct, n, p, t)`

    Evaluates the distribution function of the maximum order statistic of
    `t` draws from the Poisson binomial random variable with parameters `n` and `p`
    at the value `num_correct`. Note that `num_correct` is an integer between 0
    and `n` (inclusive) rather than an accuracy.

In all cases, the parameters `n`, `p`, and `t` are as described above.
For all of these functions, the `p` argument can also take the additional formats
described above.

We also expose the class `MaxOrderStatisticPoissonBinomial`, allowing for faster
calculations of the baseline when the values of `n` and `p` are fixed (`t` can
still take different values).

For example, to print the maximum random baseline for fixed `n=100` and `p=0.5`
with two values of `t`:

```
from max_random_baseline import MaxOrderStatisticPoissonBinomial
max_order_statistic = MaxOrderStatisticPoissonBinomial(100, 0.5)
print(max_order_statistic.max_random_baseline(t=10))
print(max_order_statistic.max_random_baseline(t=100))
```

Parameter ranges used in the paper have been tested.
Calculating values for large `n` will take a while.
If you find numerical instability outside these ranges, please report an issue.