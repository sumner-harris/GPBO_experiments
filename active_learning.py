import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional,Tuple
import os
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import (
    qUpperConfidenceBound,
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qSimpleRegret,
    qNegIntegratedPosteriorVariance
)
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from gpytorch.mlls import ExactMarginalLogLikelihood
from gp_models import create_single_task_gp
from gpytorch.settings import cholesky_jitter

def step_GP(
    X: torch.Tensor,
    Y: torch.Tensor,
    X_test: torch.Tensor,
    bounds: Optional[torch.Tensor] = None,
    Y_var: Optional[torch.Tensor] = None,
    acq_func: str = "UCB",
    beta: float = 1.0,
    num_restarts: int = 5,
    raw_samples: int = 20,
    batch: int = 1,
    maximize: bool = True,
    discrete: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform one GP-based acquisition step:

      1. Fit a GP model to (X, Y) (with optional per-point variances Y_var)
      2. Compute posterior mean/variance on X_test
      3. Propose `batch` new points via the specified acquisition function
         * If discrete=True, picks from X_test via optimize_acqf_discrete
         * Else, uses optimize_acqf over bounds
      4. Return (candidates, mean, var, acq_vals_test)

    Args:
        X: torch.Tensor of shape (n, d) training inputs
        Y: torch.Tensor of shape (n, 1) training targets
        X_test: torch.Tensor of shape (m, d) for evaluation / discrete choices
        bounds: Optional[torch.Tensor of shape (2, d)] domain bounds.
                Required if discrete=False.
        Y_var: Optional[torch.Tensor of shape (n, 1)] per-point noise variances
        acq_func: one of "UCB", "EI", "QNEI", "REGRET", "UNC"
        beta: float, UCB exploration parameter
        num_restarts: int, acquisition‐opt restarts
        raw_samples: int, init samples for acquisition
        batch: int, number of candidates to generate
        maximize: bool, if True maximize objective, if False minimize
        discrete: bool, if True pick only from X_test via optimize_acqf_discrete

    Returns:
        candidates: Tensor(batch, d)
        mean:       Tensor(m, 1)
        var:        Tensor(m, 1)
        acq_vals_test:   Tensor(m,)  acquisition values on X_test
    """
    # 1) Possibly flip sign for minimization
    Y_train = Y if maximize else -Y

    # 2) Build & fit the GP (with optional heteroskedastic noise)
    with cholesky_jitter(1e-6):
        gp = create_single_task_gp(
            train_X=X,
            train_Y=Y_train,
            train_Yvar=Y_var,     # None → default homoskedastic
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

    # 3) Compute posterior on X_test
    posterior = gp.posterior(X_test)
    mean = posterior.mean.detach()
    var  = posterior.variance.detach()
    if not maximize:
        mean = -mean

    # 4) Build acquisition function
    mode = acq_func.strip().upper()
    if mode == "UCB":
        acqf = qUpperConfidenceBound(gp, beta=beta)
    elif mode in ("EI", "EXPECTED_IMPROVEMENT"):
        best_f = Y_train.max()
        acqf = qLogExpectedImprovement(model=gp, best_f=best_f)
    elif mode in ("QNEI", "NOISY_EI"):
        acqf = qLogNoisyExpectedImprovement(model=gp, X_baseline=X)
    elif mode == "REGRET":
        acqf = qSimpleRegret(model=gp)
    elif mode in ("UNC", "MAXVAR", "INTEGRATED_VAR"):
        mc_points = X_test.unsqueeze(0)   # shape (1, m, d)
        acqf = qNegIntegratedPosteriorVariance(model=gp, mc_points=mc_points)
    else:
        raise ValueError(f"Unsupported acquisition function: {acq_func}")

    # filter out previously sampled points
    acqf.X_pending = X

    # 5) Optimize acquisition
    if discrete:
        mask = ~((X_test.unsqueeze(1) == X.unsqueeze(0)).all(-1).any(-1))
        masked_grid = X_test[mask]
        if masked_grid.shape[0] == 0:
            raise ValueError("All available points have been sampled. Increase grid or use discrete=False.")
        # pick from X_test only; bounds not needed
        print('Calculating Acquisition...')
        candidates, acq_vals = optimize_acqf_discrete(
            acq_function=acqf,
            choices =masked_grid,
            q=batch,
        )
        print('Finished calculating Acquisition.')
    else:
        # continuous domain search over bounds
        if bounds is None:
            raise ValueError("bounds must be provided when discrete=False")
        print('Calculating Acquisition...')
        candidates, _ = optimize_acqf(
            acqf,
            bounds=bounds,
            q=batch,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        print('Finished calculating Acquisition.')
        # evaluate acquisition values at X_test for reporting
        
    gp.eval()
    with torch.no_grad():
        #acq_vals = acqf(X_test.unsqueeze(1)).squeeze(-1)
        acq_vals = evaluate_in_batches(acqf, X_test, batch_size=10_000)

    # 6) Return
    return candidates, mean, var, acq_vals

def evaluate_in_batches(acqf, X, batch_size=10_000):
    all_vals = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            print(i, 'of', X.size(0))
            Xi = X[i : i + batch_size]
            vals = acqf(Xi.unsqueeze(1)).squeeze(-1)
            all_vals.append(vals)
    return torch.cat(all_vals)

def predict_posterior(
    X: torch.Tensor,
    Y: torch.Tensor,
    test_X: torch.Tensor,
    maximize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a SingleTaskGP on (X, Y) and compute posterior mean and variance on test_X.

    Args:
        X: torch.Tensor of shape (n, d) training inputs
        Y: torch.Tensor of shape (n, 1) training targets
        test_X: torch.Tensor of shape (m, d) test inputs
        maximize: bool, if True maximize objective, if False minimize

    Returns:
        mean: torch.Tensor of shape (m, 1) posterior mean
        var:  torch.Tensor of shape (m, 1) posterior variance
    """
    # 1. Prepare targets: flip sign if minimizing
    Y_train = Y if maximize else (-Y)

    # 2. Fit GP
    gp = create_single_task_gp(X, Y_train)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    # 3. Compute posterior
    posterior = gp.posterior(test_X)
    mean = posterior.mean.detach()
    var  = posterior.variance.detach()

    # 4. Flip mean back if minimizing
    if not maximize:
        mean = -mean

    return mean, var

def eval_acq_grid(
    acqf,
    X_grid: torch.Tensor,
    q: int = 1,
) -> torch.Tensor:
    """
    Evaluate a q-point acquisition function `acqf` on every point in
    `X_grid`, where X_grid.shape = (*grid_shape, d).

    For each of the G = prod(grid_shape) points, we build a batch of size q
    by repeating the same point q times, feed it into acqf, and get one value
    per point. Returns a tensor of shape grid_shape.

    Args:
        acqf: a BoTorch q-acquisition object (expects input of shape (N, q, d))
        X_grid: torch.Tensor of shape (*grid_shape, d)
        q: batch size for the acquisition

    Returns:
        acq_vals: torch.Tensor of shape grid_shape
    """
    # 1) remember the grid shape and input dim
    *grid_shape, d = X_grid.shape
    # 2) flatten to (N, d)
    X_flat = X_grid.reshape(-1, d)          # N = G

    # 3) build (N, q, d) by repeating each point q times
    X_input = X_flat.unsqueeze(1).repeat(1, q, 1)

    # 4) evaluate acquisition
    with torch.no_grad():
        vals = acqf(X_input)                # → (N, 1)
        vals = vals.squeeze(-1)             # → (N,)

    # 5) reshape back to grid_shape
    return vals.view(*grid_shape)

def append_data(
    X: torch.Tensor,
    Y: torch.Tensor,
    Y_var: torch.Tensor,
    X_new: torch.Tensor,
    Y_new: torch.Tensor,
    Y_var_new: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    acq: torch.Tensor,
    save_path: str = "",           # directory in which to save
    base_filename: str = "data",   # prefix for the file
    step: Optional[int] = None,    # if given, appended to filename
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Appends new data to (X,Y) and new predictions to (mean,var,acq),
    then saves everything as a torch file in `save_path` with name
    `{base_filename}_{step}.pt` (or `{base_filename}.pt` if step is None).

    If save_path == "", defaults to "./test_results".
    """
    # 1) Concatenate
    X_all = torch.cat([X, X_new], dim=0)
    Y_all = torch.cat([Y, Y_new], dim=0)
    Y_var_all = torch.cat([Y_var, Y_var_new], dim=0)

    # 2) Determine directory
    save_dir = save_path or "test_results"
    os.makedirs(save_dir, exist_ok=True)

    # 3) Build filename
    if step is None:
        filename = f"{base_filename}.pt"
    else:
        filename = f"{base_filename}_{step}.pt"
    full_path = os.path.join(save_dir, filename)

    # 4) Save as torch file
    torch.save({
        "X":    X_all.cpu(),
        "Y":    Y_all.cpu(),
        "Y_var":Y_var_all.cpu(),
        "mean": mean.cpu(),
        "var":  var.cpu(),
        "acq":  acq.cpu()
    }, full_path)

    return X_all, Y_all

def save_diagnostics(
    X_test: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    mean: torch.Tensor,
    acq: torch.Tensor,
    maximize: bool=True,
    save_path: str = "",           # directory in which to save
    base_filename: str = "data",   # prefix for the file
    step: Optional[int] = None,    # if given, appended to filename
 )-> Tuple[torch.Tensor, torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    # ◾ Report the optimum measured
    Y_flat = Y.squeeze(-1)
    if maximize:
        best_idx = Y_flat.argmax()
    else:
        best_idx = Y_flat.argmin()
    
    X_best = X[best_idx]
    Y_best = Y_flat[best_idx]

    # ◾ Report the optimum predicted
    mean_flat = mean.squeeze(-1)
    if maximize:
        best_idx = mean_flat.argmax()
    else:
        best_idx = mean_flat.argmin()
    
    X_best_pred = X_test[best_idx]
    Y_best_pred = mean_flat[best_idx]

    acq_max = acq.max()
    
    print("Best measured:")
    print(f"   X* = {X_best.numpy()}")
    print(f"   Y* = {Y_best.item():.4f}")
    print("Best predicted:")
    print(f"   X = {X_best_pred.numpy()}")
    print(f"   Y = {Y_best_pred.item():.4f}")
    print("Acq. Max:")
    print(acq_max.item())

    # Build filename
    if step is None:
        filename = f"{base_filename+'_diagnostics'}.pt"
    else:
        filename = f"{base_filename+'_diagnostics'}_{step}.pt"
    full_path = os.path.join(save_path, filename)

    torch.save({
        "X_best":    X_best.cpu(),
        "Y_best":    Y_best.item(),
        "X_best_pred": X_best_pred.cpu(),
        "Y_best_pred":  Y_best_pred.item(),
        "acq_max":  acq_max
    }, full_path)

    return X_best, Y_best, X_best_pred, Y_best_pred, acq_max

def plot_2D(X,Y,x1,x2,mean,var,acq):
    x1_shape=x1.shape[-1]
    x2_shape=x2.shape[-1]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    # Posterior mean
    pcm = ax[0].pcolormesh(x1, x2,
                           mean.numpy().reshape(x1_shape,x2_shape),
                           shading="auto")
    ax[0].scatter(X[:,0], X[:,1], c=Y, edgecolor="k")
    ax[0].set_title("GP Posterior Mean")
    fig.colorbar(pcm, ax=ax[0])
    
    # Posterior variance
    pcm2 = ax[1].pcolormesh(x1, x2,
                           var.numpy().reshape(x1_shape,x2_shape),
                           shading="auto")
    ax[1].scatter(X[:,0], X[:,1], c=Y, edgecolor="k")
    ax[1].set_title("GP Posterior Variance")
    fig.colorbar(pcm2, ax=ax[1])
    # Acquisition
    pcm3 = ax[2].pcolormesh(x1, x2,
                            acq.exp().numpy().reshape(x1_shape,x2_shape),
                            shading="auto")
    ax[2].scatter(X[:,0], X[:,1], c=Y, edgecolor="k")
    ax[2].set_title("Acquisition Surface")
    fig.colorbar(pcm3, ax=ax[2])
    
    plt.tight_layout()
    plt.show()