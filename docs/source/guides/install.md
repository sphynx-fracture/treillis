# Installation

*treillis* is still under active development, making it not so easy to install. For development, Pixi is used, making it the easiest way to install a working environment. However, Spyder has not yet updated spyder-kernels to load Pixi environments, meaning that if your IDE is Spyder, like it is for us, you might need to fight a bit to get everything working.

Whatever your installation method and IDE, we heavily recommend creating a dedicated environment for the use of *treillis*.

## Using Pixi
### Installing treillis

This is the fastest and easiest way to get *treillis* running, but you need to have [Pixi](https://pixi.prefix.dev/latest/installation/) installed first.

Clone *treillis* locally and once Pixi is installed, go to the root of the folder in which you have loaded *treillis*, where you have the files `pyproject.toml` and `pixi.lock`, and open a command line here. In the command run either

```
pixi shell
```

or 

```
pixi install
```

And you now have installed *treillis*!

### Running on Spyder
The real difficulty with this method is that Pixi and Spyder do not yet have easily compatible kernels, meaning Spyder does not see the environment you have just created.

To get Spyder to see your environment, go in Settings > Python Interpreter > Selected interpreter, and you will now have to find the python.exe file in the Pixi default environement. Usually you find it in the `.pixi/envs/default` folder.