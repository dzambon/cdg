
'''
Already available errors:
 - ValueError
 - NotImplementedError
'''

class CDGError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return '[cdg] ' + repr(self.value)

class AbstractMethodError(CDGError):
    def __init__(self, message=""):
        CDGError.__init__(self, "Pretending to be an abstract method... " + message)


class ImpossibleError(CDGError):
    def __init__(self, message=""):
        CDGError.__init__(self, "This should never happen. " + message)

class ForbiddenError(CDGError):
    def __init__(self, message=""):
        CDGError.__init__(self, "You are not allowed to do this. " + message)

class EDivisiveRFileNotFoundError(CDGError):
    def __init__(self, message=""):
        CDGError.__init__(self, "EDivisive_R CPM raise a FileNotFoundError. " + message)
# def fake_abstract_class(instance):
#     raise CDGAbstractMethod("Pretending to be an abstract method " + type(instance).__name__)
