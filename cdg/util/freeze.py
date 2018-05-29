# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Class with freezable attributes (`Freezable`) and a direct use of it
# (`Parameters`).
#
#
# --------------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# --------------------------------------------------------------------------------
# Author: Daniele Zambon
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 10/04/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import cdg.util.logger
import cdg.util.serialise
import datetime


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
            raise KeyError("Attribute {} not present in closed instance = {})".format(key, self))
        object.__setattr__(self, key, value)

    def freeze(self):
        self._isfrozen = True

    def close(self):
        self._isclosed = True

    def addtowhitelist(self, key):
        self._whitelist.add(key)


class Parameters(Freezable, cdg.util.logger.Loggable, cdg.util.serialise.Pickable):
    """
    Generic class for parameters.
    """

    def __init__(self):
        cdg.util.logger.Loggable.__init__(self)
        cdg.util.freeze.Freezable.__init__(self)
        self.addtowhitelist('_log')
        self.set_default()
        self.close()

    def __setattr__(self, key, value):
        try:
            cdg.util.freeze.Freezable.__setattr__(self, key, value)
            self.log.info("setting parameter %s: %s" % (key, value))
        except KeyError as e:
            raise cdg.util.errors.CDGForbidden(str(e))

    def set_default(self):
        self.creation_time = datetime.datetime.now()
