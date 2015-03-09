# nelder-mead

Pure Python/Numpy implementation of the Nelder-Mead optimization algorithm.

## Why?

For inclusion in projects with limited support for 3rd party libraries, such as PyPy projects, Google App Engine projects, etc. 

To the best of my knowledge the only open-source implementation of Nelder-Mead is the one packaged with SciPy. However SciPy is not available in PyPy (yet), or on Google App Engine. But both PyPy and GAE support Numpy, and can run the present Nelder-Mead implementation.

## Reference

See the description of the Nelder-Mead algorithm on Wikipedia: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method