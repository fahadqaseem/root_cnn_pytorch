"""
THE SIMPLEST POSSIBLE ML CODE  (linear regression in PyTorch)
=============================================================

Goal: fit a straight line   y = w * x + b   to some points.

There is only ONE weight (w) and ONE bias (b). No images, no CNN.
But the training loop is the EXACT same five stages as the CNN file:

    1. forward   -> prediction
    2. loss      -> how wrong we are
    3. zero grad -> clear old gradients
    4. backward  -> backprop computes gradients
    5. step      -> optimizer updates w and b

If you understand these five stages here, you understand them everywhere.
"""

import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
# We MAKE fake data on purpose so we know the right answer.
# The true line is  y = 2*x + 1 .  We add a little noise to make it realistic.
# The model's job is to DISCOVER that w should be ~2 and b should be ~1.
#
# Shapes matter: nn.Linear expects a 2D tensor of shape (num_samples, num_features).
# Here we have 100 samples, each with 1 feature, so shape is (100, 1).
torch.manual_seed(0)  # makes the random noise reproducible

x = torch.linspace(-3, 3, 100).unsqueeze(1)   # shape (100, 1)
true_w, true_b = 2.0, 1.0
noise = 0.5 * torch.randn(x.shape)            # random wiggle, shape (100, 1)
y = true_w * x + true_b + noise               # shape (100, 1)


# ----------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------
# nn.Linear(1, 1) is a single layer with exactly one weight and one bias.
# It computes  output = w * input + b .  That IS linear regression.
# The w and b start as random numbers; training will fix them.
model = nn.Linear(in_features=1, out_features=1)


# ----------------------------------------------------------------------
# LOSS + OPTIMIZER
# ----------------------------------------------------------------------
# Loss: Mean Squared Error. It measures the average squared gap between the
# predicted y and the true y. Smaller loss = the line fits better.
loss_fn = nn.MSELoss()

# Optimizer: plain gradient descent (SGD). It owns w and b (model.parameters())
# and updates them using their gradients. lr is the step size.
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)


# ----------------------------------------------------------------------
# TRAINING LOOP  (the five stages, repeated)
# ----------------------------------------------------------------------
epochs = 100
for epoch in range(1, epochs + 1):

    # 1. FORWARD: run the inputs through the model to get predictions
    y_pred = model(x)                 # shape (100, 1)

    # 2. LOSS: compare predictions to the truth
    loss = loss_fn(y_pred, y)         # a single scalar tensor

    # 3. ZERO GRAD: clear gradients left over from the previous step
    #    (PyTorch adds gradients up by default, so we must reset them)
    optimizer.zero_grad()

    # 4. BACKWARD: backprop fills in w.grad and b.grad via the chain rule
    loss.backward()

    # 5. STEP: the optimizer nudges w and b downhill to lower the loss
    optimizer.step()

    # (just for us to watch) print progress every 20 epochs
    if epoch % 20 == 0:
        print(f"epoch {epoch:>3} | loss {loss.item():.4f}")


# ----------------------------------------------------------------------
# RESULT
# ----------------------------------------------------------------------
# After training, the learned weight and bias should be close to 2 and 1.
# .item() pulls the single number out of the tensor.
learned_w = model.weight.item()
learned_b = model.bias.item()
print(f"\nlearned:  y = {learned_w:.3f} * x + {learned_b:.3f}")
print(f"true:     y = {true_w} * x + {true_b}")