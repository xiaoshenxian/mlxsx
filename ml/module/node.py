# -*- coding: utf-8 -*-

class DataPackage:
    pass

class TfNode:
    def __init__(self, the_input, running_type):
        self.input=the_input
        self.output=None
        self.running_type=running_type

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output
