[Temporal Difference Learning in Python](https://harderchoices.com/2018/06/07/temporal-difference-learning-in-python/)

By: Jeremi (June 7, 2018)

Key Points:

- "DP + MC = TD"

Basically ideas from Monte Carlo and Dynamic Programming added together is "Temporal Difference Learning".

__Monte Carlo__: because TD learns from experience, without a model of any kind

__Dynamic Programming__: as TD doesn't wait for episode completion. TD bootstraps.

*Bootstrap: using one or more estimated values in the update step for the same kind of estimated value.

### SARSA

State -> Action -> Reward -> State' -> Action'


