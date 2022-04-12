# jax_single_use_rng
A simple wrapper I made after the 73rd time I re-used an RNG key on accident.

# Installation
`pip install -e .`

# Usage
```python
from jax_safe_prng import SafePRNGKey

rng = SafePRNGKey(12345)
jax.random.Uniform(rng.key) # Works
jax.random.Uniform(rng.key) # Error

rng1, rng2 = rng.split() # New SafePRNGKey instances
```
