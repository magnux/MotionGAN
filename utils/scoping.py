from __future__ import absolute_import, division, print_function
from contextlib import contextmanager


class Scoping:
    def __init__(self):
        self._name_stack = ''

    @staticmethod
    def get_global_scope():
        global_scope = None
        for val in globals().values():
            if isinstance(val, Scoping):
                global_scope = val

        if global_scope is not None:
            return global_scope
        else:
            global scoping
            scoping = Scoping()
            return scoping

    @contextmanager
    def name_scope(self, scope):
        try:
            old_stack = self._name_stack
            if self._name_stack == '':
                self._name_stack = scope
            else:
                self._name_stack += '/' + scope
            yield self._name_stack
        finally:
            self._name_stack = old_stack

    def __str__(self):
        return self._name_stack

    def __add__(self, other):
        return str(self) + '/' + str(other)


if __name__ == "__main__":
    my_scope = Scoping.get_global_scope()

    def _test_global():
        my_scopex = Scoping.get_global_scope()
        print('global test:', my_scopex)

    with my_scope.name_scope('foo') as scope:
        print(scope)
        print(my_scope)
        with my_scope.name_scope('bar'):
            print(scope)
            print(my_scope)
            with my_scope.name_scope('baz'):
                print(scope)
                print(my_scope)
                _test_global()
                print(my_scope + 123)
