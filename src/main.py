# -*- coding: utf-8 -*-
import numpy as np
from keras.datasets import mnist
np.random.seed(686)
DIG_A, DIG_B = 4, 9
SIDE = 28
MAX_PIX_VAL = 255

#############################################################
# Prepare dataset
#############################################################

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
# close_enough = lambda a, b : abs(b - a) < 1e-6  # NOTE: for scalars only
close_enough = lambda a, b : np.linalg.norm(np.array(b - a).flatten()) < 1e-6

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

print("Success metric checks passed")

#############################################################
# Linear model - forward model
#############################################################

# ~~~~~~ Manipulating weights ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Make array of independent standard Gaussian elements
# Scale weights by SIDE * SIDE: w.dot(x) will be at most on order of +/- sqrt(SIDE * SIDE)
# scaling this way means decision function values will be at most on order of +/- 1
linear_init = lambda : np.random.randn(SIDE * SIDE) / np.sqrt(SIDE * SIDE)

def linear_displace(w, coef, g):
    return w + coef * g

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Limit value range of z
clip = lambda z : np.maximum(-15., np.minimum(+15., z))

# Sigmoid function for hidden unit
sigmoid = lambda z : 1. / (1. + np.exp(-clip(z)))


def linear_predict(weights, input):
    """
    Applies sigmoid function to the dot product of
    node weights and input
    """
    return sigmoid(weights.dot(input.flatten()))


#############################################################
# Sanity checks
#############################################################

# Initialise weights
w = linear_init()

sA = success_metrics(lambda x : linear_predict(+w, x), x_train, y_train)["acc"]
sB = success_metrics(lambda x : linear_predict(-w, x), x_train, y_train)["acc"]
# Probabilities should sum to 1
assert close_enough(sA + sB, 1.)

f5 = success_metrics(lambda x : linear_predict(0*w, x), x_train, y_train)["loss"]
assert close_enough(f5, np.log(2))

x = w.reshape(SIDE, SIDE)
sA = success_metrics(lambda x : linear_predict(+w, x), [x], [DIG_A])["acc"]
sB = success_metrics(lambda x : linear_predict(+w, x), [x], [DIG_B])["acc"]
assert close_enough(sA, 0.)
assert close_enough(sB, 1.)

print("Forward model checks passed")

#############################################################
# Linear model - backward pass
#############################################################
"""
For given x, y, get derivative (with respect to w) of
    l(w) = loss(sigmoid(w.dot(x)), y)
    rewritten to emphasise dependance on w, 
                de-emphasise dependance on fixed x, y:
    l(w) = loss_at_y(sigmoid(dot_with_x(w)))
    where loss_at_y(p) = - log(p if y == DIG_B else 1-p)
    where sigmoid(z) = 1/(1+exp(-z))
    where dot_with_x(w) = w.dot(x)

By __CHAIN_RULE__:
    l'(w) = (
          loss_at_y'(sigmoid(dot_with_x(w)))
        * sigmoid'(dot_with_x(w))
        * dot_with_x'(w)
    ) = (
          loss_at_y'(p)
        * sigmoid'(z)
        * dot_with_x'(w)
    )
    where z = dot_with_x(w)
    where p = sigmoid(z)
    NOTE: appearance of terms from forward pass!
"""

# NOTE: both linear_backprop_unsimp() and linear_backprop() are correct
# the latter is purely a simplification, but both may be used

def linear_backprop_unsimp(w, x, y):
    z = w.dot(x.flatten())
    p = sigmoid(z)
    # Deriv. of log of X is deriv. of X, divided by X
    dl_dp = - (+1 if y == DIG_B else -1)/(p if y == DIG_B else 1-p)
    # p = sigmoid(z)
    # Deriv. of sigmoid is (sigmoid * (1 - sigmoid))
    dp_dz = p * (1 - p)
    # Deriv. of w*x w.r.t. w = x
    dz_dw = x.flatten()
    
    dl_dw = dl_dp * dp_dz * dz_dw
    
    return dl_dw


def linear_backprop(w, x, y):
    """
    Simplifies linear_backprop_unsimplified()
    """
    z = w.dot(x.flatten())
    p = sigmoid(z)
    """
        dl_dp = -1/p if y == DIG_B else +1/(1-p)
        dp_dz = p * (1 - p)
        dl_dz = dl_dp * dp_dz = - (1 - p) if y == DIG_B else +p
    """
    # Interpret dl_dz as residual error of p as estimator of one-hot version of y
    dl_dz = p - (1 if y == DIG_B else 0)
    # Deriv. of w*x w.r.t. w = x
    dz_dw = x.flatten()
    
    dl_dw = dl_dz * dz_dw
    
    return dl_dw


#############################################################
# Sanity checks
#############################################################

for _ in range(10):
    # Create test variables
    w = linear_init()
    idx = np.random.randint(N)
    x = x_train[idx]
    y = y_train[idx]
    
    # Check that simplification preserved answer
    g_unsimp    = linear_backprop_unsimp(w, x, y)
    g           = linear_backprop       (w, x, y)
    assert close_enough(g_unsimp, g)
    
    # Do step of gradient descent, check loss decreased
    before = success_metrics(lambda xx: linear_predict(w, xx), [x], [y])["loss"]
    w = w - 0.01 * g
    after = success_metrics(lambda xx: linear_predict(w, xx), [x], [y])["loss"]
    assert after < before

print("Back propagation checks passed")

#############################################################
# Vanilla neural network model
#############################################################
"""
Any architecture can be used, as long as it is differentiable.

We want a parameterised feature transformation function,
that we can eventually linearly classify.

A good method is to alternate learned linearities
with fixed non-linearities.

Leaky ReLU - instead of a slope between 0,1, we have
a slope between 1/10th and 1 or 1/5th and 1.
lrelu(z) = max(z/10, z)
It helps to address vanishing gradient problems.

    x
    h0 -------> z1 -----> h1 ----> z2 ------> h2 -----> z3 -----> p
    |                                                          
    |           |        |                                        
    |           |        |         |          |                   
    |   C*      | lrelu  |   B*    |  lrelu   |   A*    | sigmoid |
    |           |        |         |          |                   
    |           |        |                    1                                       
    |                    1 
    1                                                     
    D0          D1      D1        D2         D2        D3         1
    SIDE*SIDE   32                32                   1

In general, we want the dimensions for the first hidden
layer to be substantially larger, to provide elbow-room to
learn good high-level features.
"""

# ~~~~~~ Weight helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

D0 = SIDE * SIDE
D1 = 32
D2 = 32
D3 = 1
def vanilla_init():
    A = np.random.randn(    D2) / np.sqrt( 1 + D2)
    B = np.random.randn(D2, D1) / np.sqrt(D2 + D1)
    C = np.random.randn(D1, D0) / np.sqrt(D1 + D0)
    return (A, B, C)



def vanilla_displace(abc, coef, g):
    A, B, C     = abc
    gA, gB, gC  = g
    return (
        A + coef * gA,
        B + coef * gB,
        C + coef * gC
    )


# ~~~~~~ Forward pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

lrelu = lambda z : np.maximum(z/10, z)
step = lambda z : np.heaviside(z, 0.5)
dlrelu_dz = lambda z : 0.1 + (1. - 0.1)*step(z)

def vanilla_predict(abc, x):
    A, B, C = abc
    #
    h0 = x.flatten()
    #
    z1 = C.dot(h0)
    h1 = lrelu(z1)
    #
    z2 = B.dot(h1)
    h2 = lrelu(z2)  # learned featurisation
    #
    z3 = A.dot(h2)  # linear classifier
    p = sigmoid(z3)
    #
    return p


# ~~~~~~ Checks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Initialise weights
A, B, C = vanilla_init()

# Check that linear layer makes sense
# Compare same featurisations with +ve and -ve A, holding B and C constant
sA = success_metrics(lambda x : vanilla_predict((+A, B, C), x), x_train, y_train)["acc"]
sB = success_metrics(lambda x : vanilla_predict((-A, B, C), x), x_train, y_train)["acc"]
# Probabilities should sum to 1
assert close_enough(sA + sB, 1.)

f5 = success_metrics(lambda x : vanilla_predict((0*A, B, C), x), x_train, y_train)["loss"]
assert close_enough(f5, np.log(2))

# If A, B, C all > 0, that will mean all x's are non-negative
# so we know the overall output from the network will
# be positive, since it's very unlikely to be 0.

# Check end-to-end positivity
x = x_train[0]
y = y_train[0]
A = np.abs(A)
B = np.abs(B)
C = np.abs(C)
acc_ppp = success_metrics(lambda x : vanilla_predict((A,  B,  C), x), [x], [DIG_B])["acc"]
acc_ppn = success_metrics(lambda x : vanilla_predict((A,  B, -C), x), [x], [DIG_B])["acc"]
acc_pnp = success_metrics(lambda x : vanilla_predict((A, -B,  C), x), [x], [DIG_B])["acc"]
acc_pnn = success_metrics(lambda x : vanilla_predict((A, -B, -C), x), [x], [DIG_B])["acc"]
assert close_enough(acc_ppp, 1.)
assert close_enough(acc_ppn, 0.)
assert close_enough(acc_pnp, 0.)
assert close_enough(acc_pnn, 1.)

print("Vanilla neural net checks passed")

# ~~~~~~ Backward pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def vanilla_backprop(abc, x, y):
    A, B, C = abc
    #
    h0 = x.flatten()
    #
    z1 = C.dot(h0)
    h1 = lrelu(z1)
    #
    z2 = B.dot(h1)
    h2 = lrelu(z2)  # learned featurisation
    #
    z3 = A.dot(h2)  # linear classifier
    p = sigmoid(z3)
    #
    
    dl_dz3 = p - (1 if y == DIG_B else 0)
    dl_dh2 = dl_dz3 * A  # A = dz3_dh2  # scalar * scalar
    dl_dz2 = dl_dh2 * dlrelu_dz(z2)  # vector * vector
    dl_dh1 = dl_dz2.dot(B)  # vector * matrix, so dot
    dl_dz1 = dl_dh1 * dlrelu_dz(z1)  # vector * vector
    
    dl_dA = dl_dz3 * h2  # scalar * vector
    dl_dB = np.outer(dl_dz2, h1)  # to get shapes to match
    dl_dC = np.outer(dl_dz1, h0)
    
    return (dl_dA, dl_dB, dl_dC)


#############################################################
# Sanity checks
#############################################################

for _ in range(10):
    # Create test variables
    abc = vanilla_init()
    idx = np.random.randint(N)
    x = x_train[idx]
    y = y_train[idx]
    
    # Do step of gradient descent, check loss decreased
    before = success_metrics(lambda xx: vanilla_predict(abc, xx), [x], [y])["loss"]
    g = vanilla_backprop(abc, x, y)
    abc = vanilla_displace(abc, - 0.01, g)
    after = success_metrics(lambda xx: vanilla_predict(abc, xx), [x], [y])["loss"]
    assert after < before

print("Vanilla back propagation checks passed")

#############################################################
# Training loop
#############################################################

# ~~~~~~ Training parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Original values did not converge, so dec LR, 
T = 15001  # 10001
DT = 1000
LEARNING_RATE = 0.01  # 0.1
ANNEAL_T = 4000  # 1000

idx = 0

def next_training_example():
    global idx
    xy = x_train[idx], y_train[idx]
    idx += 1
    idx %= N  # if idx == N, set to 0 so as to loop
    return xy   


# ~~~~~~ Interface with model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# INIT, BACK, DISP, PRED = (
#     (linear_init, linear_backprop, linear_displace, linear_predict)
# )

INIT, BACK, DISP, PRED = (
    (vanilla_init, vanilla_backprop, vanilla_displace, vanilla_predict)
)

# ~~~~~~ SGD - ENGINE of machine learning ~~~~~~~~~~~~~~~~~~~

# Initialise w
w = INIT()
# Add momentum for physics simulation, so that instead of displacing
# by the gradient, we displace by momentum
# the gradient will then affect things by changing momentum
m = DISP(w, -1., w)  # hacky way to set m = 0 of same shape as w
# the momentum will accumulate gradients, so each update made
# is the "average wisdom" of a group of previous gradients
# momentum makes SGD less prone to becoming stuck in a local
# minima due to the presence of a minor bump in the road

for t in range(T):
    x, y = next_training_example()
    g = BACK(w, x, y)
    LR = LEARNING_RATE * float(ANNEAL_T) / (ANNEAL_T + t)
    m = DISP(m, -0.1, m)  # m forgets a bit of its past ... .1 is drag coef
    m = DISP(m, +1., g)  # add gradient to momentum
    w = DISP(w, -LR, m)  # update based on momentum
    
    if t % DT : continue
    
    ms = success_metrics(lambda x: PRED(w, x), x_train, y_train)
    print(f"at step {t:6d}; tr acc {ms['acc']:4.2f}; tr loss {ms['loss']:5.3f}")
    