from typing import Literal, Union

import numpy as np
import pandas as pd
import torch


class SynthControlModel(torch.nn.Module):
    """Base model for prodessing synth control.
    Hack from Lev Hojoyan for linear regression with condition.

    Need: math:`\sum_i w_i = 1`, :math:`w_i > 0`, No in sklearn.
    Set problem in pytorch :mod:`pytorch` as:

    .. math::
        y = \sum_{i=1}^K \mathrm{sigmoid} (w_i) \cdot x_i

    optimizing :math:`w_i`. Set final regression coefs as :math:`\mathrm{sigmoid} (\hat{w}_i)`
    from found optimum -- this gives both conditions at once.

    Args:
        seed (int): Random seed. Default to 42.
        loss_func (Literal["mse", "mae"], optional): Loss function of optimizer. Default to "mse".
        epochs (int): Number of learning epochs. Default to 2000.
        lr (float): Learning rate for torch optimizer. Defaults to 0.1.

    Returns:
        None
    """

    def __init__(
        self,
        seed: int = 42,
        loss_func: Literal["mse", "mae"] = "mse",
        device: Literal["cpu", "cuda", "cuda:0"] = "cpu",
        epochs: int = 2000,
        lr: float = 0.1,
    ) -> None:
        """
        Initialize the SynthControlModel.

        Args:
            seed (int): Random seed for reproducibility. Defaults to 42.
            loss_func (Literal["mse", "mae"]): Loss function for the optimizer. Defaults to "mse".
            epochs (int): Number of learning epochs. Defaults to 2000.
            lr (float): Learning rate for the torch optimizer. Defaults to 0.1.
            device (Literal["cpu", "cuda", "cuda:0"]): For gpu-rich millionaires

        Returns:
            None
        """

        super().__init__()
        self.seed = seed
        self.lr = lr
        self.epochs = epochs
        self.__weights = None
        self.__loss_value = None
        possible_loss_functions = ["mse", "mae"]
        possible_devices = ["cpu", "cuda", "cuda:0"]
        self.loss_func = loss_func
        assert device in possible_devices, f"Device should be from {possible_devices}"
        self.device = device
        assert (
            loss_func in possible_loss_functions
        ), f"Loss function should be from {possible_loss_functions}"
        if loss_func == "mse":
            self.crit = torch.nn.MSELoss(reduction="sum")
        elif loss_func == "mae":
            self.crit = torch.nn.L1Loss(reduction="sum")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, K) where N is the number of samples and K is the number of features.

        Returns:
            torch.Tensor: Output tensor of shape (N,) representing the predicted values.
        """
        # return torch.matmul(x, self.w)
        return torch.matmul(x, torch.softmax(self.w, dim=-1))

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]
    ) -> None:
        """
        Fit the model to the given data.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Input features of shape (N, K) where N is the number of samples and K is the number of features.
            y (Union[np.ndarray, pd.DataFrame]): Target values of shape (N,).

        Returns:
            None
        """
        self.x = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).reshape(y.shape[0])
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        # initiate start weights
        self.w = torch.nn.Parameter(
            torch.normal(
                0,
                0.1,
                (X.shape[1],),
                device=self.device,
                generator=self.generator,
                dtype=torch.float32,
                requires_grad=True,
            )
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        for t in range(self.epochs):
            y_pred = self.forward(self.x)
            loss = self.crit(y_pred, self.y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.__weights = torch.softmax(
            list(self.parameters())[0].T.detach(), dim=-1
        ).numpy()
        self.__loss_value = loss.detach().numpy()

    def get_weights(self) -> np.ndarray:
        """
        Get the weights of the model.

        Returns:
            np.ndarray: Array of weights of shape (K,) where K is the number of features.
        """
        return self.__weights

    def get_loss_value(self) -> float:
        """
        Get the loss value after fitting the model.

        Returns:
            float: Loss value of the model.
        """
        return self.__loss_value
