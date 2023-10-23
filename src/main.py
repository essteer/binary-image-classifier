# -*- coding: utf-8 -*-
import numpy as np
from keras.datasets import mnist
import tqdm
np.random.seed(686)
DIG_A, DIG_B = 4, 9
SIDE = 28
MAX_PIX_VAL = 255
NB_TRAIN = 1000

#~~~~~ Notes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
This model has:
-- no bias trick
-- no model selection
-- no feature engineering
-- no pooling

It does have:
-- stride
-- momentum
"""
#############################################################
# Prepare dataset
#############################################################

# Load the MNIST dataset keeping test set unnamed
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = N x SIDE x SIDE array of integers [0, MAX_PIX_VAL] that reveal the pixel intensity
# y_train = N array of integers [1, 10]
# N = 60,000

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Keep only DIG_As and DIG_Bs for this binary classification
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
def filter_digits(xs, ys):
    """
    Return boolean array the same shape as y_train,
    with True for elements equal to DIG_A or DIG_B, 
    False otherwise
    """
    indices = np.logical_or(
        np.equal(ys, DIG_A), 
        np.equal(ys, DIG_B)
        )
    # Update x_train and y_train 
    # to remove all elements marked False in indices
    xs = xs[indices]
    ys = ys[indices]
    
    return xs, ys


x_train, y_train = filter_digits(x_train, y_train)
x_test, y_test = filter_digits(x_test, y_test)
# x_train = N x SIDE x SIDE array of integers [0, MAX_PIX_VAL] 
# that reveal the pixel intensity
# y_train = N array of integers {DIG_A, DIG_B}
# N ~ 12,000

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Normalise pixel intensities
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

x_train = x_train / float(MAX_PIX_VAL)
x_test = x_test / float(MAX_PIX_VAL)
# x_train = N x SIDE x SIDE array of floats [0., 1.] that reveal the pixel intensity
# y_train = N array of integers {DIG_A, DIG_B}
# N ~ 12,000

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Shuffle data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def shuffle(xs, ys):
    """
    Shuffle x_train (x_test) and y_train (y_test) in the same way,
    so that corresponding pairs remain intact
    For the new xs and ys, the (e.g.) 42nd element is the kth 
    element of the former xs and ys where k == the 42nd element of indices
    """
    indices = np.arange(len(xs))
    np.random.shuffle(indices)  # mutating shuffle
    # Apply the shuffled indices to x_train and y_train
    xs = xs[indices]  # indices now an int array, not a boolean array
    ys = ys[indices]
    
    return xs, ys


x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)

#~~~~~~ Pare down training set ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x_train = x_train[:NB_TRAIN]
y_train = y_train[:NB_TRAIN]

#~~~~~~ Add intensity noise ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x_train = x_train + np.random.randn(*x_train.shape)
x_train = np.maximum(0., np.minimum(1., x_train))

#~~~~~~ Shape ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
assert len(x_train) == len(y_train)
N = len(x_train)

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
assert abs(N - min(NB_TRAIN, 12000)) < 500

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


def success_metrics(predictor, xs, ys, verbose=False):
    # Progress bar
    xs = tqdm.tqdm(xs) if verbose else xs
    
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


# ~~~~~~ Checks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
# Convoluted neural network model
#############################################################
"""
Any architecture can be used, as long as it is differentiable.

In this model we de-emphasise pooling.

In the chart below, we transform inputs (top) to outputs (bottom).

                    height x width x channels
    x                   28 x 28 x 1
        avgpool                                     2x2
    h0                  14 x 14 x 1
        conv                            weight C    5x5x8x1 stride 2x2
    z1                   5 x  5 x 8
        lrelu
    h1                   5 x  5 x 8   
        conv                            weight B    5x5x8x1 stride 1x1
    z2                   5 x  5 x 4
        lrelu
    h2                   5 x  5 x 4
        dense                           weight A    1x(5*5*4)
    z3                            1
        sigmoid
    p                             1          

If we wanted to add bias to the h1 layer, for example,
we could set the dimensions to 5x5x9, since this would
add one further 5x5 layer.

1x1 convolution has become more popular in recent years.
These still have an effect, but do so without 
reference to their neighbour; each location only relates
to its corresponding location.
"""

# ~~~~~~ Weights helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def conv_init():
    A = np.random.randn(5 * 5 * 4) / np.sqrt(1 + 5 * 5 * 4)
    B = np.random.randn(1, 1, 4, 8) / np.sqrt(4 + 1 * 1 * 8)
    C = np.random.randn(5, 5, 8, 1) / np.sqrt(8 + 5 * 5 * 1)
    return (A, B, C)


def conv_displace(abc, coef, g):
    A, B, C     = abc
    gA, gB, gC  = g
    return (
        A + coef * gA,
        B + coef * gB,
        C + coef * gC
    )

# ~~~~~~ Building blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def avgpool2x2(x):
    H, W, C = x.shape  # height, width, channel
    # return an array of shape (H/2 x W/2 x C)
    # if doing this for more complex pools we can
    # use np.transpose instead of this manual version
    return (  x[0:H:2, 0:W:2]
            + x[0:H:2, 1:W:2]
            + x[1:H:2, 0:W:2]
            + x[1:H:2, 1:W:2])/4.
    

def conv(x, weights, stride=1):
    H, W, C = x.shape  # height, width, channel
    KH, KW, OD, ID = weights.shape  # kernel / output / input
    assert C == ID
    HH, WW = int((H - KH + 1)/stride), int((W - KW + 1)/stride)
    # return an array of shape HH x WW x OD
    return np.array(
        [[
            np.tensordot(
                weights,                     # KH x WH x OD x ID
                x[h:h+KH, w:w+KW],           # KH x KW      x ID
                ((0, 1, 3), (0, 1, 2))
        )
             for w in range(0, WW*stride, stride)]
         for h in range(0, HH*stride, stride)]
        )
    

# ~~~~~~ Checks for forward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Scaling and shape tests
aa = np.ones((8, 12, 7))  # test with factors of diff. primes (2, 3, 7)
pp = np.ones((4, 6, 7))
assert close_enough(avgpool2x2(aa), pp)

ww = np.ones((3, 3, 5, 7))
cc = (3*3*7)*np.ones((6, 10, 5))
assert close_enough(conv(aa, ww, stride=1), cc)

# Orientation test
bb = np.array([1*np.eye(4), 3*np.eye(4)])  # 2 x 4 x 4
"""
bb == [
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], 
    [[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]]
    ]
"""
pp = np.array([[[1, 1, 0, 0], [0, 0, 1, 1]]])
assert close_enough(avgpool2x2(bb), pp)

ww = np.zeros((2, 2, 1, 4))
ww[0, 0, :, :] = 1 + np.arange(4)
cc = np.array([1, 2, 3])[np.newaxis, :, np.newaxis]  # shape 1 x 3 x 1
assert close_enough(conv(bb, ww, stride=1), cc)

print("Conv neural net checks passed")

# ~~~~~~ Derivatives ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
Why do we want dconv(x, w)/dw? 
So that we can compute dl/dw from dl/dconv(x, w).
This can be simplified by writing a function that
directly gives dl/dw from dl/dconv(x, w).
"""
def Dw_conv(x, weights_shape, dl_dconv, stride=1):
    H, W, C = x.shape  # height, width, channel
    KH, KW, OD, ID = weights_shape  # kernel / output / input
    assert C == ID
   
    HH, WW = int((H - KH + 1)/stride), int((W - KW + 1)/stride)
    assert dl_dconv.shape == (HH, WW, OD)
    
    # return an array of shape KH x KW x OD x ID
    HS, WS = HH*stride, WW*stride
    dl_dw = np.array(
        [[np.tensordot(
            dl_dconv,                                # HH x WW x OD
            x[dh:dh+HS:stride, dw:dw+WS:stride],     # HH x WW x ID
            ((0, 1), (0, 1))
        )
            for dw in range(KW)] 
         for dh in range(KH)]
    )
    
    return dl_dw


"""
Why do we want dconv(x, w)/dx? 
So that we can compute dl/dx from dl/dconv(x, w).
This can be simplified by writing a function that
directly gives dl/dx from dl/dconv(x, w).
"""
def Dx_conv(x_shape, weights, dl_dconv, stride):
    H, W, C = x_shape  # height, width, channel
    KH, KW, OD, ID = weights.shape  # kernel / output / input
    assert C == ID
   
    HH, WW = int((H - KH + 1)/stride), int((W - KW + 1)/stride)
    assert dl_dconv.shape == (HH, WW, OD)
    
    # return H x W x ID
    dl_dx = np.zeros((H, W, ID), dtype=np.float32)
    for h in range(KH):
        for w in range(KW):
            dl_dx[h:h+HH*stride:stride, w:w+WW*stride:stride] += (
                np.tensordot(dl_dconv,          # HH x WW x OD
                             weights[h, w],      # OD x ID
                             ((2,), (0,))
                              )
            )
    
    return dl_dx


# ~~~~~~ Forward pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
                    height x width x channels
    x                   28 x 28 x 1
        avgpool                                     2x2
    h0                  14 x 14 x 1
        conv                            weight C    5x5x8x1 stride 2x2
    z1                   5 x  5 x 8
        lrelu
    h1                   5 x  5 x 8   
        conv                            weight B    5x5x8x1 stride 1x1
    z2                   5 x  5 x 4
        lrelu
    h2                   5 x  5 x 4
        dense                           weight A    1x(5*5*4)
    z3                            1
        sigmoid
    p                             1          

"""
def conv_predict(abc, x):
    A, B, C = abc
    
    h0 = avgpool2x2(x[:, :, np.newaxis])
    
    z1 = conv(h0, C, stride=2)
    h1 = lrelu(z1)
    
    z2 = conv(h1, B, stride=1)
    h2 = lrelu(z2)
    
    z3 = A.dot(h2.flatten())
    p = sigmoid(z3)
    
    return p


# ~~~~~~ Backward pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def conv_backprop(abc, x, y):
    A, B, C = abc
    
    h0 = avgpool2x2(x[:, :, np.newaxis])
    
    z1 = conv(h0, C, stride=2)
    h1 = lrelu(z1)
    
    z2 = conv(h1, B, stride=1)
    h2 = lrelu(z2)
    
    z3 = A.dot(h2.flatten())
    p = sigmoid(z3)
    
    dl_dz3 = p - (1 if y == DIG_B else 0)
    dl_dh2 = dl_dz3 * A.reshape(h2.shape)
    dl_dz2 = dl_dh2 * dlrelu_dz(z2)
    dl_dh1 = Dx_conv(h1.shape, B, dl_dz2, stride=1)
    dl_dz1 = dl_dh1 * dlrelu_dz(z1)
    
    dl_dA = dl_dz3 * h2.flatten()
    dl_dB = Dw_conv(h1, B.shape, dl_dz2, stride=1)
    dl_dC = Dw_conv(h0, C.shape, dl_dz1, stride=2)
    
    return (dl_dA, dl_dB, dl_dC)

#############################################################
# Training loop
#############################################################

# ~~~~~~ Training parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Original values did not converge, so dec LR, 
T = 15001  # 10001
DT = 1000
LEARNING_RATE = 0.01  # 0.1
ANNEAL_T = 4000  # 1000
DRAG_COEF = 0.1

idx = 0

def next_training_example():
    global idx, x_train, y_train
    xy = x_train[idx], y_train[idx]
    idx += 1
    
    if idx == N:
        idx = 0
        x_train, y_train = shuffle(x_train, y_train)
    return xy   


# ~~~~~~ Interface with model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FUNCS_BY_MODEL = {
    "linear": (linear_init, linear_backprop, linear_displace, linear_predict),
    "vanilla": (vanilla_init, vanilla_backprop, vanilla_displace, vanilla_predict),
    "conv": (conv_init, conv_backprop, conv_displace, conv_predict)
}#

# ~~~~~~ SGD - ENGINE of machine learning ~~~~~~~~~~~~~~~~~~~

for MODEL in ("linear", "vanilla", "conv"):
    
    print("\n"*4)
    print(MODEL)
    print("\n"*2)
    INIT, BACK, DISP, PRED = FUNCS_BY_MODEL[MODEL]
    
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
        m = DISP(m, -DRAG_COEF, m)  # m forgets a bit of its past
        m = DISP(m, +1., g)  # add gradient to momentum
        w = DISP(w, -LR, m)  # update based on momentum
        
        if t % DT : continue
        
        xs = x_train[-1000:]
        ys = y_train[-1000:]
        ms_train = success_metrics(lambda x: PRED(w, x), xs, ys)
        
        xs = x_test[-1000:]
        ys = y_test[-1000:]
        ms_test = success_metrics(lambda x: PRED(w, x), xs, ys)
        
        print(f"at step {t:6d}; tr acc {ms_train['acc']:4.2f}; tr loss {ms_train['loss']:5.3f}; te acc {ms_test['acc']:4.2f}; te loss {ms_test['loss']:5.3f}")
        
    
    xs = x_train[:]
    ys = y_train[:]
    ms_train = success_metrics(lambda x: PRED(w, x), xs, ys, verbose=True)
    
    xs = x_test[:]
    ys = y_test[:]
    ms_test = success_metrics(lambda x: PRED(w, x), xs, ys, verbose=True)
    
    print("After all training")
    print(f"at step {t:6d}; tr acc {ms_train['acc']:4.2f}; tr loss {ms_train['loss']:5.3f}; te acc {ms_test['acc']:4.2f}; te loss {ms_test['loss']:5.3f}")
        