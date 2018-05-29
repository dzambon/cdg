# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Logging tool for cdg
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
# Last Update: 12/04/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import logging
import logging.handlers
import cdg.util.serialise
import cdg.util.errors

# Defines the levels
# ------------------

DISABLED = logging.CRITICAL + 1
CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
ALL = logging.NOTSET + 1
NOTSET = logging.NOTSET

# Defaults
# --------

_logging_session = 'cdg_default.log'
_stdout_level = WARNING
_filelog_level = DISABLED
_runs_enabled = False


# Manage the logger
# -----------------

def set_stdout_level(level):
    global _stdout_level
    _stdout_level = level


def set_filelog_level(level, name=None):
    global _logging_session
    global _filelog_level
    if name is not None:
        _logging_session = 'cdg_' + name + '.log'
    _filelog_level = level
    return _logging_session


def enable_logrun(level=True):
    global _runs_enabled
    _runs_enabled = level


# Create the logger
# -----------------

_logger_instance = None


def _call_logger():
    global _logger_instance

    if _logger_instance is None:
        _logger_instance = logging.getLogger('CDG')
        _logger_instance.setLevel(ALL)

        formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(name)s:%(message)s')

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        sh.setLevel(_stdout_level)
        _logger_instance.addHandler(sh)

        if _filelog_level is not DISABLED:
            fh = logging.FileHandler(_logging_session, mode='a', encoding=None)
            fh.setFormatter(formatter)
            fh.setLevel(_filelog_level)
            _logger_instance.addHandler(fh)


    if _logger_instance is None:
        raise cdg.util.errros.CDGImpossible("this should never happen")

    return _logger_instance


# Thanks to https://stackoverflow.com/a/15835863
class Loggable(cdg.util.serialise.Pickable):
    def __init__(self):
        # cdg.util.freeze.Freezable.__init__(self)
        self.skip_from_serialising(['_log'])

    @property
    def log(self):
        return _call_logger()

    # def log_flush(self):
    #     for h in _call_logger().handlers:
    #         h.flush()
    # def setLevel(self, level):
    #     self._log.basicConfig(level=level)

    def logrun(self, fun):
        if _runs_enabled:
            fun()

    def logplot(self, datalist, stylelist=None):
        if not isinstance(datalist, list):
            self.log.warning("datalist is not a list. No figure is produces")
        n = len(datalist)
        if stylelist is None:
            stylelist = n * ['x']

        def tmp():
            import matplotlib.pyplot as plt
            for i in range(n):
                if datalist[i].ndim == 1:
                    plt.plot(datalist[i], stylelist[i])
                elif datalist[i].shape[1] == 1:
                    plt.plot(datalist[i][:, 0], stylelist[i])
                elif datalist[i].shape[0] == 1:
                    plt.plot(datalist[i][0, :], stylelist[i])
                else:
                    plt.plot(datalist[i][:, 0], datalist[i][:, 1], stylelist[i])
            plt.show()

        self.logrun(tmp)


genlog = Loggable()


def glog():
    return genlog.log
