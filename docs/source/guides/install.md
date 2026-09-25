Install
-------
Ideally, if we had time and more experience with Python and building packages in general, the requirements would be much lighter. As is, *treillis* requires:

- [Python](http://python.org/) >= 3.11
- [numba](https://numba.readthedocs.io/en/stable/#) <0.66
- [numpy](https://numpy.org) < 2.5
- [scipy](https://scipy.org)
- [dill](https://github.com/uqfoundation/dill )
- [tqdm](https://tqdm.github.io/)
- [ipyparallel](https://ipyparallel.readthedocs.io/)
- [meshio](https://github.com/nschloe/meshio)
- [matplotlib](https://matplotlib.org/)
- [datoviz](https://datoviz.org/)
- [pytables](https://www.pytables.org/)
- [scikit sparse](https://scikit-sparse.readthedocs.io/)

### Install from the source as editable
Clone the repository.

#### Using Pixi
Install the latest version of [Pixi](https://pixi.prefix.dev/latest/), open a command terminal in the folder root and input `pixi shell`.

#### Using conda
Because *treillis* uses [Datoviz](https://datoviz.org/) for 3D interactive plotting, which does not yet have a conda release, you need to create the environment before being able to install

- Open a conda command terminal in the folder root.
- Create the treillis environment from environment.yml with

```conda env create --name <YOUR ENVT NAME> --file=environment.yml```

- activate the environment `conda activate <YOUR ENVT NAME>`
- install *treillis* in the environment as editable with 

```
conda install conda-build
conda develop .
```

OR install *treillis* with `conda install treillis`

### Pypi
Because the mechanics module of *treillis* is based on [scikit-sparse](https://github.com/scikit-sparse/scikit-sparse), it has the same issues with pip install: SuiteSparse needs to be setup first before building the package. Hence, the next paragraph is copied from the scikit-sparse documentation:

> To install `scikit-sparse`, you need to have the [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html) library installed on your system.
>
> It is recommended that you install SuiteSparse and the scikit-sparse dependencies in a virtual environment, to avoid conflicts with other packages. We recommend using Anaconda:
>
>  ```
>    $ conda create -n scikit-sparse python>=3.10 suitesparse
>    $ conda activate scikit-sparse
>  ```
>
>If you are not using Anaconda, you can install SuiteSparse using your preferred package manager.
>
>On MacOS, you can use [Homebrew](http://brew.sh):
>
>  ```
>    $ brew install suite-sparse
>  ```
>
>On Debian/Ubuntu systems, use the following command:
>
>  ```
>    $ sudo apt-get install python-scipy libsuitesparse-dev
>  ```
>
>On Arch Linux, run:
>
>  ```
>    $ sudo pacman -S suitesparse
>  ```
