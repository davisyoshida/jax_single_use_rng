import jax

@jax.tree_util.register_pytree_node_class
class SafePRNGKey:
    def __init__(self, seed=None, *, key=None):
        if not ((seed is not None) ^ (key is not None)):
            raise ValueError('Must specify exactly one of seed or key')
        self._key = jax.random.PRNGKey(seed) if seed is not None else key
        self._has_used = False
        self._has_split = False

    @property
    def has_used(self):
        return self._has_used

    @property
    def has_split(self):
        return self._has_split

    @property
    def key(self):
        if self._has_used:
            raise ValueError('Attempted re-use of RNG key.')
        self._has_used = True
        return self._key

    def peek_key(self):
        return self._key

    def split(self, n=2):
        if self._has_split:
            raise ValueError('Attempted to split RNG key multiple times.')
        self._has_split = True
        key = self._key
        return SafePRNGKey(key=jax.random.split(self._key, n))

    def __iter__(self):
        for k in self.key:
            yield SafePRNGKey(key=k)

    def tree_flatten(self):
        return ([self._key], (self.has_used, self.has_split))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        result = cls(key=children[0])
        result._has_used, result._has_split = aux_data
        return result
