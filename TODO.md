Things which still need to be done:

- [X] Implement R<sub>F</sub>
- [X] Implement R<sub>C</sub>
- [X] Implement R<sub>J</sub>
- [ ] Implement R<sub>D</sub>
- [ ] Type checking
- [ ] Add NumPy style docstrings
- [X] Add tests using the Boost datasets: scipy/special/tests/data/boost
- [ ] Introduce test generators instead of a bunch of `asserts`.  Each check
      should be independent.
- [ ] Module for calculating Legendre forms using Carlson forms

Once these are complete, I would like to try to introduce these into the SciPy
stack.

R<sub>J</sub> needs to be modified to compute Cauchy principal values in order
to deal with negative values for `p`, which the Boost dataset uses.

