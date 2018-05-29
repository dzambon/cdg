# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Serialization with pickle.
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
import pickle


class Pickable(object):
    _pickleFile = 'cdgpickle.pkl'
    _skip = set(['_pickleFile '])

    def skip_from_serialising(self, skipthese):
        for s in skipthese:
            self._skip.add(s)

    def keep_in_serialising(self, keepthese):
        for k in keepthese:
            for key in self.__dict__:
                if not (key in keepthese):
                    self._skip.append(k)

    def getPickleFile(self):
        return self._pickleFile

    def serialise(self, pickleFile):

        if not (pickleFile is None):
            self._pickleFile = pickleFile
        else:
            pass

        pickle.dump(self.getPickleDictionary(), open(self.getPickleFile(), "wb"))

    def getPickleDictionary(self):
        # Save as pickle file.
        pickleDictionary = {}
        pickleDictionary['current_classes'] = {}
        for key in self.__dict__:
            if not (key in self._skip):
                if isinstance(self.__dict__[key], Pickable):
                    pickleDictionary['current_classes'][key] = type(self.__dict__[key])
                    pickleDictionary[key] = self.__dict__[key].getPickleDictionary()
                else:
                    pickleDictionary[key] = self.__dict__[key]
        return pickleDictionary

    def deserialise(self, pickleFile):

        if not (pickleFile is None):
            self._pickleFile = pickleFile

        pickledDict = pickle.load(open(self._pickleFile, "rb"))

        self.deserialise_from_dict(pickledDict=pickledDict)

    def deserialise_from_dict(self, pickledDict):

        current_classes = pickledDict.pop('current_classes', None)
        for key in pickledDict:
            if key in current_classes:
                self.__dict__[key] = current_classes[key]()
                self.__dict__[key].deserialise_from_dict(pickledDict=pickledDict[key])
            else:
                self.__dict__[key] = pickledDict[key]
