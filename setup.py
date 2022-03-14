import setuptools

setuptools.setup(
    name='jax-safe-prng-key',
    version='0.0.1',
    author='Davis Yoshida',
    author_email='davis.yoshida@gmail.com',
    description='Single use PRNG keys for JAX',
    long_description='A simple wrapper around JAX\'s PRNG keys to minimize RNG reuse bugs.',
    long_description_content_type='text',
    url=None,
    packages=['jax_safe_prng'],
)
