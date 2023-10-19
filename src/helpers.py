# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(686)
DIG_A, DIG_B = 4, 9
SIDE = 28
MAX_PIX_VAL = 255

#############################################################
# Prepare dataset
#############################################################

from keras.datasets import mnist
# Load the MNIST dataset keeping test set unnamed
(x_train, y_train), _ = mnist.load_data()
# x_train = N x SIDE x SIDE array of integers [0, MAX_PIX_VAL] that reveal the pixel intensity
# y_train = N array of integers [1, 10]
# N = 60,000

#############################################################
# Keep only DIG_As and DIG_Bs for this binary classification
#############################################################
 
# Return boolean array the same shape as y_train,
# with True for elements equal to DIG_A or DIG_B, False otherwise
indices = np.logical_or(
    np.equal(y_train, DIG_A), 
    np.equal(y_train, DIG_B)
    )
# Update x_train and y_train to remove all elements marked False in indices
x_train = x_train[indices]
y_train = y_train[indices]
# x_train = N x SIDE x SIDE array of integers [0, MAX_PIX_VAL] that reveal the pixel intensity
# y_train = N array of integers {DIG_A, DIG_B}
# N ~ 12,000


#############################################################
# Shape
#############################################################

assert len(x_train) == len(y_train)
N = len(x_train)

#############################################################
# Normalise pixel intensities
#############################################################

x_train = x_train / float(MAX_PIX_VAL)
# x_train = N x SIDE x SIDE array of floats [0., 1.] that reveal the pixel intensity
# y_train = N array of integers {DIG_A, DIG_B}
# N ~ 12,000

#############################################################
# Shuffle data
#############################################################

# Shuffle x_train and y_train in the same way,
# so that corresponding pairs remain intact
indices = np.arange(N)
np.random.shuffle(indices)  # mutating shuffle
# Apply the shuffled indices to x_train and y_train
x_train = x_train[indices]  # indices now an int array, not a boolean array
y_train = y_train[indices]
# For the new x_train and y_train, 
# the (e.g.) 42nd element is the kth element of the former x_train and y_train
# where k == the 42nd element of indices

#############################################################
# Sanity checks
#############################################################

# Floating point equality is problematic, so use epsilon
close_enough = lambda a, b : abs(b - a) < 1e-6

assert x_train.shape == (N, SIDE, SIDE)
assert y_train.shape == (N,)
assert set(y_train) == {DIG_A, DIG_B}
assert close_enough(np.min(x_train), 0.)
assert close_enough(np.max(x_train), 1.)
assert abs(N - 12000) < 500

print(f"Prepared {N:,} training examples")

#############################################################
# Success metrics
#############################################################
"""
acc, loss

prob p represents model's prob mass for DIG_B

predictor a function that takes an image and returns a probability
"""

def accuracy(predicted_labels, true_ys):
    # Could also use np.equal
    return np.mean([1. if l == y else 0. 
             for l, y in zip(predicted_labels, true_ys)])


def cross_entropy_loss(predicted_probs, true_ys):
    """
    Cross entropy loss - average over prediction truth pairs
    of minus the log of the probability mass the model
    put on the outcome being True
    
    E.g., if the true_y == 9, entropy loss is the probability
    the model put on 9 being the label
    """
    # Could also use np.equal
    return np.mean([ - np.log(p if y == DIG_B else 1 - p) 
             for p, y in zip(predicted_probs, true_ys)])


def success_metrics(predictor, xs, ys):
    probs = [predictor(x) for x in xs]
    labels = [DIG_B if p > 0.5 else DIG_A for p in probs]
    acc = accuracy(labels, ys)
    loss = cross_entropy_loss(probs, ys)
    
    return {"acc": acc, "loss": loss}
    

#############################################################
# Sanity checks using placeholder predictors
#############################################################

# 1% chance of DIG_B being the case, so appropriate to call it DIG_A
# sure_A will be correct for roughly half the training examples,
# with a low loss around 0
# on the other half of the training inputs, it will make a prediction
# that is wrong, with high confidence
# cross entropy loss will heavily penalise this
sure_A = lambda x : 0.01
# 99% chance of DIG_B being the case, so appropriate to call it DIG_B
sure_B = lambda x : 0.99
# maybe_A will have a moderate loss when correct, and moderate loss when wrong
maybe_A = lambda x : 0.4
maybe_B = lambda x : 0.6
fifty_fifty = lambda x : 0.5

sA = success_metrics(sure_A, x_train, y_train)["acc"]
sB = success_metrics(sure_B, x_train, y_train)["acc"]
# Probabilities should sum to 1
assert close_enough(sA + sB, 1.)

# Check single case
sA = success_metrics(sure_A, x_train[:1], [DIG_A])["acc"]
sB = success_metrics(sure_A, x_train[:1], [DIG_B])["acc"]
# Single case, so each can only be 100% or 0%
assert close_enough(sA, 1.)
assert close_enough(sB, 0.)

sA = success_metrics(sure_A, x_train, y_train)["loss"]
sB = success_metrics(sure_B, x_train, y_train)["loss"]
mA = success_metrics(maybe_A, x_train, y_train)["loss"]
mB = success_metrics(maybe_B, x_train, y_train)["loss"]
f5 = success_metrics(fifty_fifty, x_train, y_train)["loss"]

# The closer to the baseline rate, the less the loss should be
assert f5 < mA < sA
assert f5 < mB < sB
# The fifty_fifty loss is always log(2)
# because its predictor always says p == 0.5
# - log(0.5) == log(2)
assert close_enough(f5, np.log(2))

print("Sanity checks succeeded")