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
from . import *
import logging
import logging.handlers
from datetime import datetime
# import cdg.util.serialise
# import cdg.util.errors

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

def create_new_logger(name, stdout_level=None, filelog_level=None, date_in_fname=True, log_session=None):
    new_logger = logging.getLogger(name)
    new_logger.setLevel(ALL)
    formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(name)s:%(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    if stdout_level is None:
        stdout_level = _stdout_level
    sh.setLevel(stdout_level)
    new_logger.addHandler(sh)
    
    if filelog_level is None:
        filelog_level = _filelog_level

    if log_session is None:
        log_session, idcode = get_filelog_fname(name=name, date_in_fname=date_in_fname)
    else:
        log_session = _logging_session
        idcode = 100000

    if filelog_level is not DISABLED:
        fh = logging.FileHandler(log_session, mode='a', encoding=None)
        fh.setFormatter(formatter)
        fh.setLevel(filelog_level)
        new_logger.addHandler(fh)
    
    return new_logger, idcode

def set_stdout_level(level):
    assert _logger_instance is None, 'The logging levels are global, and can only be set once.' + \
                                     'Try to move the call to `set_stdout_level` earlier in the code.'
    global _stdout_level
    _stdout_level = level


def get_filelog_fname(name, date_in_fname):
    log_session = 'cdg'
    idcode = 'nocode'
    if date_in_fname:
        idcode = datetime.now().strftime('%f')
        log_session += datetime.now().strftime('_%G%m%d_%H%M')
    if name is not None:
        log_session += '_' + name
    log_session += '_' + idcode + '.log'
    return log_session, idcode

def set_filelog_level(level, name=None, date_in_fname=True):
    assert _logger_instance is None, 'The logging levels are global, and can only be set once.' + \
                                     'Try to move the call to `set_filelog_level` earlier in the code.'
    global _filelog_level
    _filelog_level = level
    global _logging_session
    _logging_session, idcode = get_filelog_fname(name=name, date_in_fname=date_in_fname)
    return _logging_session, idcode

def enable_logrun(level=True):
    assert _logger_instance is None, 'The logging levels are global, and can only be set once.' + \
                                     'Try to move the call to `enable_logrun` earlier in the code.'
    global _runs_enabled
    _runs_enabled = level


# Create the logger
# -----------------

_logger_instance = None

def _call_logger():
    global _logger_instance

    if _logger_instance is None:
        _logger_instance, _ = create_new_logger(name='CDG', log_session=_logging_session,
                                                stdout_level=_stdout_level, filelog_level=_filelog_level)

    assert _logger_instance is not None
    return _logger_instance


# Thanks to https://stackoverflow.com/a/15835863
# class Loggable(cdg.util.serialise.Pickable):
class Loggable(object):
    def __init__(self):
        if issubclass(type(self), Pickable):
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


# This is intended as a generic logger. Tecnically is an instance of Loggable,
# thus an object which can produce logs. Notice that the creation of the actual
# logger inside genlog is not created until the first call to log: this trick
# allows to set the levels once for all.
genlog = Loggable()

# These methods expose some common log methods
def debug(msg):
    return genlog.log.debug(msg)
def info(msg):
    return genlog.log.info(msg)
def warning(msg):
    return genlog.log.warning(msg)
def error(msg):
    return genlog.log.error(msg)


