Install
=======
## Install from the source as editable
Clone the repository.

### Using Pixi
Install the latest version of [Pixi](https://pixi.prefix.dev/latest/), open a command terminal in the folder root and input `pixi shell`.

### Using pip or conda
- Open a conda command terminal in the folder root.
- Create the treillis environment from environment.yml with
```conda env create --name treillis --file=environment.yml```
- install treillis in the environment as editable
   - with pip: `pip install -e .`
   - with conda: `conda install conda-build; conda develop .`


## Pypi
In your environment, write `pip install treillis==1.4`