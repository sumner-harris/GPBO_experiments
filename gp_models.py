from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from botorch.models import SingleTaskGP
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means import ConstantMean
from botorch.models.transforms import Normalize, Standardize
import torch
from typing import Optional

def create_single_task_gp(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    train_Yvar: Optional[torch.Tensor] = None,
    ard: bool = True,
) -> SingleTaskGP:
    d = train_X.shape[-1]
    covar = ScaleKernel(RBFKernel(ard_num_dims=d if ard else 1))
    mean  = ConstantMean()

    # 1) set up your likelihood as before…
    if train_Yvar is not None:
        likelihood = FixedNoiseGaussianLikelihood(
            noise=train_Yvar, learn_additional_noise=True
        )
        gp_args = dict(
            train_Yvar=train_Yvar,
            likelihood=likelihood,
        )
    else:
        gp_args = {}

    # 2) add BoTorch input & outcome transforms
    input_tf = Normalize(d=d)         # scales each column of X to [0,1]
    outcome_tf = Standardize(m=1)     # zero-centers & scales Y to unit std

    model = SingleTaskGP(
        train_X,
        train_Y,
        covar_module=covar,
        mean_module=mean,
        input_transform=input_tf,
        outcome_transform=outcome_tf,
        **gp_args,
    )
    return model
