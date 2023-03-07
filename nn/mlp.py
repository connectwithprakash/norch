import numpy as np
from norch.layer import Linear, Sequential
from norch.layer.activation import ReLU, Sigmoid


class MLP(Sequential):
    """
    Multi-layer Perceptron class inheriting from Sequential.

    Attributes:
    -----------
    input_dim : int
        The number of dimensions in the input layer.
    hidden_dims : list
        A list of integers representing the number of nodes in each hidden layer.
    output_dim : int
        The number of dimensions in the output layer.
    init_method : str, optional, default: 'normal'
        The method used to initialize the weights of the layers.
    activation : str, optional, default: 'relu'
        The activation function to be used in each hidden layer. Valid options are 'relu' or 'sigmoid'.
    verbose : bool, optional, default: True
        Whether to print updates during training.

    Methods:
    --------
    load_param(params)
        Load parameters for each layer from a given list of parameter values.
    train(X, y, loss_fn, optimizer_fn, batch_size=32)
        Train the model on the given data and loss function with a given optimizer.
    validate(X, y, loss, batch_size=32)
        Evaluate the model on the given data and loss function.
    fit(X_train, y_train, loss_fn, optimizer_fn, X_val=None, y_val=None, batch_size=32, epochs=10, patience=None)
        Train the model on the given training data, loss function, and optimizer, optionally using validation data and early stopping.
    predict(X)
        Predict the output of the model for the given input.
    """

    def __init__(self, input_dim, hidden_dims, output_dim, init_method='normal', activation='relu', verbose=True):
        """
        Constructor for the MLP class.

        Parameters:
        -----------
        input_dim : int
            The number of dimensions in the input layer.
        hidden_dims : list
            A list of integers representing the number of nodes in each hidden layer.
        output_dim : int
            The number of dimensions in the output layer.
        init_method : str, optional, default: 'normal'
            The method used to initialize the weights of the layers.
        activation : str, optional, default: 'relu'
            The activation function to be used in each hidden layer. Valid options are 'relu' or 'sigmoid'.
        verbose : bool, optional, default: True
            Whether to print updates during training.
        """
        self.verbose = verbose
        modules = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            modules.append(
                Linear(dims[i], dims[i + 1], init_method=init_method, bias=True))
            if i < len(dims) - 2:
                if activation == 'relu':
                    modules.append(ReLU())
                elif activation == 'sigmoid':
                    modules.append(Sigmoid())
                else:
                    raise ValueError("Invalid activation function")
        super().__init__(*modules)

    def load_param(self, params):
        """
        Load parameters for each layer from a given list of parameter values.

        Parameters:
        -----------
        params : list
            A list of tuples representing the parameter values for each layer.
        """
        for i, layer in enumerate(self.layers):
            layer.load_param(*params[i])

    def train(self, X, y, loss_fn, optimizer_fn, batch_size=32):
        """
        Trains the MLP on the input data `X` with corresponding targets `y` using the specified `loss_fn`
        and `optimizer_fn`.

        Args:
            X (numpy.ndarray): Input data of shape `(N, D)` where `N` is the number of samples and `D` is the
                input dimension.
            y (numpy.ndarray): Target values of shape `(N, C)` where `C` is the number of classes in the output.
            loss_fn (callable): A function that takes in the predicted values and target values and computes
                the loss.
            optimizer_fn (callable): A function that takes in the model parameters and the computed gradients
                and updates the parameters.
            batch_size (int, optional): The number of samples per gradient update. Defaults to 32.

        Returns:
            float: The average loss over the entire input dataset.
        """
        epoch_loss = 0
        iters = range(0, X.shape[0], batch_size)
        n_iters = len(iters)
        for i in iters:
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            # Forward pass
            optimizer_fn.zero_grad()
            y_pred = self(X_batch)
            # Compute loss
            loss_value = loss_fn(y_pred, y_batch)
            epoch_loss += loss_value/n_iters
            # Backward pass
            gradwrtpred = loss_fn.backward()
            # Backpropagate
            super().backward(gradwrtoutput=gradwrtpred)
            # Update parameters
            optimizer_fn.step()

        return epoch_loss

    def validate(self, X, y, loss, batch_size=32):
        """
        Validate the model on the given input and output data.

        Parameters:
        -----------
        X : numpy array
            Input data of shape (n_samples, input_dim).
        y : numpy array
            Output data of shape (n_samples, output_dim).
        loss : callable
            Loss function to compute the validation loss.
        batch_size : int, optional (default=32)
            Batch size to use during validation.

        Returns:
        --------
        epoch_loss : float
            Average loss over all the batches during the validation.
        """
        epoch_loss = 0
        iters = range(0, X.shape[0], batch_size)
        n_iters = len(iters)
        for i in iters:
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            y_pred = self(X_batch)
            loss_value = loss(y_pred, y_batch)
            epoch_loss += loss_value/n_iters

        return epoch_loss

    def fit(self, X_train, y_train, loss_fn, optimizer_fn, X_val=None, y_val=None, batch_size=32, epochs=10, patience=None):
        """    Parameters:
        -----------
        X_train : numpy.ndarray
            The training data of shape (n_samples, n_features)
        y_train : numpy.ndarray
            The training labels of shape (n_samples, n_classes)
        loss_fn : function
            The loss function used to compute the loss between predictions and ground truth labels
        optimizer_fn : function
            The optimizer function used to update the network's weights during training
        X_val : numpy.ndarray, optional
            The validation data of shape (n_samples_val, n_features), default=None
        y_val : numpy.ndarray, optional
            The validation labels of shape (n_samples_val, n_classes), default=None
        batch_size : int, optional
            The batch size used during training, default=32
        epochs : int, optional
            The number of epochs to train the network, default=10
        patience : int, optional
            The number of epochs to wait before early stopping if the validation loss doesn't improve, default=None

        Returns:
        --------
        train_loss : list
            The list of training losses for each epoch
        val_loss : list
            The list of validation losses for each epoch, or an empty list if validation data isn't provided
        """

        train_loss = []
        val_loss = []
        best_val_loss = np.inf
        best_param = self.param
        best_epoch = 0

        for epoch in range(epochs):
            # Shuffle training data
            idx = np.random.permutation(X_train.shape[0])
            X_train = X_train[idx]
            y_train = y_train[idx]

            epoch_loss = self.train(
                X_train, y_train, loss_fn, optimizer_fn, batch_size)
            train_loss.append(epoch_loss)

            if (X_val is not None) and (y_val is not None):
                epoch_val_loss = self.validate(
                    X_val, y_val, loss_fn, batch_size)
                val_loss.append(epoch_val_loss)

                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_param = self.param
                    best_epoch = epoch
                elif (patience is not None) and ((epoch - best_epoch) >= patience):
                    if self.verbose:
                        print("Early stopping")
                    break

                if self.verbose:
                    print(
                        f"Epoch [{epoch + 1}/{epochs}] - train_loss: {train_loss[-1]:.4f} - val_loss: {val_loss[-1]:.4f}")
            else:
                if self.verbose:
                    print(
                        f"Epoch [{epoch + 1}/{epochs}] - train_loss: {train_loss[-1]:.4f}")

        if best_param is not None:
            self.load_param(best_param)
        return train_loss, val_loss

    def predict(self, X):
        """
        Generate predictions for the given data using the trained network.

        Parameters:
        -----------
        X : numpy.ndarray
            The data to generate predictions for, of shape (n_samples, n_features)

        Returns:
        --------
        numpy.ndarray
            The predictions generated by the network, of shape (n_samples, n_classes)
        """
        return self(X)
