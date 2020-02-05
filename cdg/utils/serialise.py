import pickle


class Pickable(object):
    
    def __init__(self):
        self._pickleFile = 'cdgpickle.pkl'
        self._to_skip = set(['_pickleFile'])
        self._to_keep = None  # when None means keep everything
        # self._my_class = None

    def skip_from_serialising(self, skipthese):
        '''Force to skip these keys.'''
        for s in skipthese:
            self._to_skip.add(s)
            if self._to_keep is not None:
                assert not s in self._to_keep, 'key {} cannot be kept and skipped.'.format(s)

    def keep_in_serialising(self, keepthese):
        '''Force to keep these keys.'''
        for k in keepthese:
            if self._to_keep is None:
                self._to_keep = set([])
            self._to_keep.add(k)
            assert not k in self._to_skip, 'key {} cannot be kept and skipped.'.format(k)
            # for key in self.__dict__:
            #     if not (key in keepthese):
            #         self._skip.add(k)

    def get_pickle_file(self):
        return self._pickleFile

    def serialise(self, pickleFile):

        if not (pickleFile is None):
            self._pickleFile = pickleFile
        else:
            pass

        pickle.dump(self.get_dict_to_pickle(), open(self.get_pickle_file(), "wb"))

    def get_dict_to_pickle(self):
        # Save as pickle file.
        dict_to_pickle = {}
        dict_to_pickle['_my_class'] = type(self)
        dict_to_pickle['current_classes'] = {}

        attr_to_keep = set()
        for key in self.__dict__:
            if key in self._to_skip:
                if self._to_keep is None:
                    pass
                elif key in self._to_keep:
                    # give priority to keep
                    attr_to_keep.add(key)
            else:
                attr_to_keep.add(key)

        for key in attr_to_keep:
            if isinstance(self.__dict__[key], Pickable):
                dict_to_pickle['current_classes'][key] = type(self.__dict__[key])
                dict_to_pickle[key] = self.__dict__[key].get_dict_to_pickle()
            else:
                dict_to_pickle[key] = self.__dict__[key]
        #
        # for key in self.__dict__:
        #     if not (key in to_be_skipped):
        #         if isinstance(self.__dict__[key], Pickable):
        #             pickleDictionary['current_classes'][key] = type(self.__dict__[key])
        #             pickleDictionary[key] = self.__dict__[key].getPickleDictionary()
        #         else:
        #             pickleDictionary[key] = self.__dict__[key]
        return dict_to_pickle

    @classmethod
    def deserialise(cls, pickleFile):

        file_to_deser = cls._pickleFile if pickleFile is None else pickleFile
        pickled_dict = pickle.load(open(file_to_deser, "rb"))

        instance = pickled_dict['_my_class']()
        instance._pickleFile = file_to_deser
        instance.deserialise_from_dict(pickledDict=pickled_dict)
        return instance

    def deserialise_from_dict(self, pickledDict):
        current_classes = pickledDict.pop('current_classes', None)
        for key in pickledDict:
            if key in current_classes:
                try:
                    self.__dict__[key] = current_classes[key]()
                except TypeError as err:
                    # print("self: {}, key: {}".format(type(self), key))
                    err.args += ("self: {}, current_classes[{}]: {}".format(type(self), key, current_classes[key]),)
                    raise
                for k in self.__dict__[key].__dict__:
                    self.__dict__[key].__dict__[k] = None
                self.__dict__[key].deserialise_from_dict(pickledDict=pickledDict[key])
            else:
                self.__dict__[key] = pickledDict[key]



