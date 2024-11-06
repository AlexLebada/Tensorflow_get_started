# HMM elements: States, observations and transitions
import tensorflow_probability as tfp
import tensorflow as tf


tfd = tfp.distributions
# 2-el vector: 0 - cold day, 1 - hot day
# Initial day: 80% to be cold day
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])
# Cold day has 30% of being followed by hot day, while hot day has 20% being followed by cold day
transition_distribution = tfd.Categorical(probs=[[0.3, 0.7],
                                                 [0.2,0.8]])
# cold day temp distrib: mean=0, std=5;  hot day temp distrib: mean=15, std=10
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

# .mean() is the computation graph for calculating the probability of day temperature
mean = model.mean()

# using this session to evaluate part of the graph
with tf.compat.v1.Session() as sess:
  print(mean.numpy())