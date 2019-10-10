import numpy as np
import scipy.sparse.linalg as splinalg
import seaborn as sns

def _process_default_kwargs(kwargs, default_kwargs):
    if kwargs is None:
        kwargs = {}
    for kwarg, value in default_kwargs.items():
        if kwarg not in kwargs:
            kwargs[kwarg] = value
    return kwargs

def get_spectra_and_bases(ops):
    spectra, bases = zip(*[np.linalg.eigh(op) for op in ops])
    return np.stack(spectra), np.stack(bases)

def get_spectra_and_bases_sparse(sparse_ops, eigsh_kwargs=None):
    default_eigsh_kwargs = {'k': 6,
                            'which': 'LA',
                            'return_eigenvectors': True}
    eigsh_kwargs = _process_default_kwargs(eigsh_kwargs, default_eigsh_kwargs)
    spectra, bases = zip(*[splinalg.eigsh(op, **eigsh_kwargs)
                           for op in sparse_ops])
    sorts = [np.argsort(spectrum) for spectrum in spectra]
    spectra = [spectrum[sort] for spectrum, sort in zip(spectra, sorts)]
    bases = [basis[:,sort] for basis, sort in zip(bases, sorts)]
    return np.stack(spectra), np.stack(bases)

def plot_spectra(spectra, ts, ax):
    for n, (eigval_trace, color) in enumerate(zip(spectra.T,
            sns.color_palette('cubehelix', spectra.shape[1]))):
        ax.plot(ts, eigval_trace, color=color, label=n)
    ax.legend(fontsize=6)
