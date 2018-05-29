# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Errors.
#
# --------------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# --------------------------------------------------------------------------------
# Author: Daniele Zambon
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 16/11/2017
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class CDGError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return '[cdg] ' + repr(self.value)


# class CDGNotImplemented(CDGError):
#     def __init__(self):
#         CDGError.__init__(self, 'Not implemented yet.')

class CDGAbstractMethod(CDGError):
    def __init__(self, message=""):
        CDGError.__init__(self, "Pretending to be an abstract method... " + message)


class CDGImpossible(CDGError):
    def __init__(self, message=""):
        CDGError.__init__(self, "This should never happen. " + message)


class CDGForbidden(CDGError):
    def __init__(self, message=""):
        CDGError.__init__(self, "You are not allowed to do this. " + message)

# def fake_abstract_class(instance):
#     raise CDGAbstractMethod("Pretending to be an abstract method " + type(instance).__name__)
