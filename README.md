# Norch
Norch is a PyTorch-like deep learning framework built using just NumPy. It provides a simple and intuitive interface for building and training deep neural networks.
This document provides an overview of Norch's features and instructions for how to use Norch in your Python projects.
(For inner working look into the code documentation itself, the code is explained.)

## Features
### Norch provides the following layers:
- Linear  
```python
from norch.layer import Linear

linear = Linear(in_features, out_features, bias=True, init_method='normal')
# Forward propagation
out = linear(input)
# Backward propagation
grad = linear.backward(<grad_from_next_layer>)
```
- ReLU
```python
from norch.layer import ReLU

relu = ReLU()
# Forward propagation
out = relu(input)
# Backward propagation
grad = relu.backward(<grad_from_next_layer>)
```
- Sigmoid
```python
from norch.layer import Sigmoid

sigmoid = Sigmoid()
# Forward propagation
out = sigmoid(input)
# Backward propagation
grad = sigmoid.backward(<grad_from_next_layer>)
```

- Sequential
```python
from norch.layer import Sequential

seq = Sequential(
    [
        linear,
        sigmoid
        ]
)
# Forward propagation
out = seq(input)
# Backward propagation
grad = seq.backward(<grad_from_next_layer>)
```

### Norch provides the following loss functions:
- Mean Squared Error (MSE)
```python
from norch.loss import MSE

criterion = MSE()
# Forward propagation
loss = criterion(y_pred, y_true)
# Backward propagation
dl_wrt_dpred = loss.backward()
```
- Hinge 
```python
from norch.loss import Hinge

criterion = Hinge(margin=1)
# Forward propagation
loss = criterion(y_pred, y_true)
# Backward propagation
dl_wrt_dpred = loss.backward()
```
- Cross-Entropy With Logits
```python
from norch.loss import CrossEntropyWithLogitsLoss

criterion = CrossEntropyWithLogitsLoss()
# Forward propagation
loss = criterion(y_pred, y_true)
# Backward propagation
dl_wrt_dpred = loss.backward()
```

### Norch provides the following optimizer:
- Stochastic Gradient Descent (SGD)
```python
from norch.optim import SGD

params = model.param
optimizer = SGD(params=params,learning_rate=0.1)
# Set gradient to zero befor forward propagation
# Forward propagation
# Backward propagation
# Update weights by calling following method
optimizer.step()
```

### Norch provides the following Neural Networks:
- Multi Layer Perceptron (MLP)
```python
from norch.nn import MLP

# Create a model
model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, init_method='normal', activation='sigmoid', verbose=True)

# Fit the model
train_loss, val_loss = model.fit(X_train=X_train, y_train=y_train, loss_fn=loss_fn, optimizer_fn=optimizer, X_val=X_test, y_val=y_test, batch_size=batch_size, epochs=epochs, patience=patience)

# Prediction
test_pred = model.predict(X_test)
```

## Conclusion
Norch provides a simple and intuitive interface for building and training deep neural networks using just NumPy. With its linear and sequential layers, activation functions, and optimizers, Norch can be used for a variety of deep learning tasks. The provided loss functions also make it easy to choose the appropriate objective for your problem.

