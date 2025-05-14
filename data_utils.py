import pandas as pd
import torch
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt

def read_print_data(file_path: str,
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reads the Excel at `file_path` (and optional `sheet_name`), 
    extracts the columns:
       X = [Speed, Sintering Power (const), Printing Passes, Sintering Passes]
       Y = Average Resistance
       Y_var = Std Dev
    Converts them into torch tensors of shape (N,4), (N,1), (N,1),
    prints their shapes, and returns (X, Y, Y_var).
    """
    # 1) Load into a DataFrame
    df = pd.read_excel(file_path)
    
    # 2) Define which columns go where
    X_cols    = ['Speed',
                 'Sintering Power (const)',
                 'Printing Passes',
                 'Sintering Passes']
    Y_col     = 'Average Resistance'
    Yvar_col  = 'Std Dev'
    
    # 3) Extract numpy arrays
    X_np    = df[X_cols].to_numpy(dtype=float)     # (N,4)
    Y_np    = df[[Y_col]].to_numpy(dtype=float)    # (N,1)
    Yvar_np = df[[Yvar_col]].to_numpy(dtype=float) # (N,1)
    
    # 4) Convert to torch tensors
    X     = torch.from_numpy(X_np).double()     # torch.Size([N,4])
    Y     = torch.from_numpy(Y_np).double()     # torch.Size([N,1])
    Y_var = torch.from_numpy(Yvar_np).double()  # torch.Size([N,1])
    
    # 5) Print to verify
    print(f"X: {X.shape}, Y: {Y.shape}, Y_var: {Y_var.shape}")
    
    return X, Y, Y_var
    
def plot_gp_mean_projections(
    mean: torch.Tensor,
    grids: list,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    param_names: list[str] = None,
    figsize=(15, 5),
):
    """
    Plot all pairwise 2D projections of a GP posterior mean for N-dimensional inputs.

    Args:
        mean:       torch.Tensor of shape (prod(n_i),) containing the posterior mean 
                    values on the full grid (flattened).
        grids:      list of N one-dimensional torch.Tensor or array, each of length n_i.
        X_train:    torch.Tensor of shape (M, N) with the training inputs.
        Y_train:    torch.Tensor of shape (M, 1) or (M,) with the training outputs.
        param_names: optional list of N strings for axis labels; defaults to ["x0", "x1", …].
        figsize:    size of the full figure (width, height).

    Notes:
        • The function arranges subplots in 3 columns and as many rows as needed.
        • Each subplot shows the posterior mean averaged over the other N−2 dimensions.
    """
    D = len(grids)
    # default names
    if param_names is None:
        param_names = [f"x{i}" for i in range(D)]

    # reshape the flat mean into an N-dim array
    shapes = [int(g.shape[0]) for g in grids]
    mean_np = mean.detach().cpu().numpy().reshape(*shapes)

    # prepare numpy versions of the grids
    grid_np = [
        g.detach().cpu().numpy() if torch.is_tensor(g) else np.asarray(g)
        for g in grids
    ]

    # all combinations of two distinct dims
    combos = list(itertools.combinations(range(D), 2))
    n_plots = len(combos)
    n_cols = 3
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for idx, (i, j) in enumerate(combos):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        # average out all other axes
        other_axes = [k for k in range(D) if k not in (i, j)]
        proj = mean_np.mean(axis=tuple(other_axes))  # shape (n_i, n_j)

        # create mesh for dims i,j
        Gi, Gj = np.meshgrid(grid_np[i], grid_np[j], indexing="ij")

        pcm = ax.pcolormesh(Gi, Gj, proj, shading="auto")
        ax.scatter(
            X_train[:, i].cpu().numpy(),
            X_train[:, j].cpu().numpy(),
            c=Y_train.squeeze(-1).cpu().numpy(),
            edgecolor="k",
            cmap="viridis",
        )
        ax.set_xlabel(param_names[i])
        ax.set_ylabel(param_names[j])
        ax.set_title(f"{param_names[i]} vs {param_names[j]}")
        fig.colorbar(pcm, ax=ax)

    # hide any unused subplots
    for idx in range(n_plots, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        fig.delaxes(axes[r][c])

    plt.tight_layout()
    plt.show()