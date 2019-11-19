class Freezable(object):
    """
    Class whose attributes are not modifiable after method `freeze()` is called.
    Moreover, once `close()` method has been called, no further attribute can be
    added.
    """
    _isfrozen = False
    _isclosed = False
    # set of exceptional attributes which won't follow the restrictions.
    _whitelist = set()

    def __setattr__(self, key, value):
        if key in self._whitelist:
            pass
        elif self._isfrozen:
            raise KeyError("Frozen instance: instance = {}".format(self))
        elif self._isclosed and not hasattr(self, key):
            raise KeyError("Attribute <{}> not present in closed instance = {})".format(key, self))
        object.__setattr__(self, key, value)

    def freeze(self):
        self._isfrozen = True

    def close(self):
        self._isclosed = True

    def addtowhitelist(self, key):
        self._whitelist.add(key)


