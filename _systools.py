#!/usr/bin/python
# -*- coding: utf-8 -*-

#    _systools.py
#    Copyright (C) 2025 Xu Yuyang

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# 

from numbers import Real

def printPercent(progress:Real, total:Real, decimals:int = 2, prefix:str = '', suffix:str = ''):
    '''Print the progress in the form of a percentage.
    
    ### Parameters:
    * `progress`: a real number describes the progress.
    * `total`: a real number that describes the total work that need to be done. 
    The persentage will have the value of `progress`/`total`.
    * `decimals`: (optional) an integer that controls the number of decimal places to be retained. Default is `2`.
    * `prefix`: (optional) a string to be printed before the percentage.
    * `suffix`: (optional) a string to be printed after the percentage.
    '''
    
    percent = (f"{{0:{decimals+4}.{decimals}f}}").format(100 * (progress / float(total)))
    print(f'{prefix}{percent}%', end=suffix)

import numpy as np

errors = [
    'divide', # ZeroDivision
    'over',   # Overflow
    'under',  # Underflow
    'invalid' # Invalid floating-point operation
]
behaviors = [
    'ignore', 
    'warn', 
    'raise', 
    'call', 
    'print', 
    'log'
]

class error_behavior:
    '''A context class for setting the error handling behavior of NumPy.
    
    ### Usage
    
    #### Set how all floating-point errors are handled (for example, `'ignore'`)
    >>> with error_behavior('ignore'):
    >>>     a = np.array([1, 2**-1000, 2**1000])
    >>>     b = np.array([0, 2**1000, 2**-1000])
    >>>     a /= b

    #### Set how to handle all floating-point errors and set separately for specific error(s).
    >>> with error_behavior('ignore', under = 'warn'):
    >>>     a = np.array([1, 2**-1000, 2**1000])
    >>>     b = np.array([0, 2**1000, 2**-1000])
    >>>     a /= b

    #### Set with an error dictionary
    >>> with error_behavior({'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}):
    >>>     a = np.array([1, 2**-1000, 2**1000])
    >>>     b = np.array([0, 2**1000, 2**-1000])
    >>>     a /= b
    
    ### Parameters:
    * `arg`: `dict`|`str`, see Usage. The priority of this parameter is lower than that of `**kwargs`.
    * `**kwargs`: `str`, see Usage.
    '''
    def __init__(self, arg:dict|str|None = None, **kwargs:str):
        self.old_setting = np.geterr()
        
        # Dealing with new_setting
        if arg is None:
            self.new_setting = kwargs
        elif isinstance(arg, dict):
            self.new_setting = arg
            arg.update(kwargs)
        elif isinstance(arg, str):
            if kwargs:
                for error in errors:
                    if error not in kwargs.keys():
                        kwargs[error] = arg
                self.new_setting = kwargs
            else:
                self.new_setting = {'all': arg}
        return None
    
    def __enter__(self):
        np.seterr(**self.new_setting)
        return self
    
    def __exit__(self, *args):
        np.seterr(**self.old_setting)
        if args:
            return False
        return True

from functools import cached_property as _cached_property
from typing import Callable, Any
type Value = Any
type Instance = Any

class cached_property(_cached_property):
    '''Modified cached_property that allows you to register functions that will
    be called before or after the value of the property has been changed.
    
    ### Usage:
    For example
    ```
    def alert_before(instance, value):
        print(f'The `baz` attribute of instance {instance} will be modified to {value}.')
    
    def alert_after(instance, value):
        print(f'The `baz` attribute of instance {instance} has been modified to {value}.')
    
    class foo:
        @cached_property
        def baz(self):
            return 0

        # Give an alert before baz is modified.
        baz.run_before_set(alert_before)

        # Give an alert after baz is modified.
        baz.run_after_set(alert_after)
    
    bar = foo()
    bar.baz = 1
    bar.baz = 2
    ```
    '''
    def __init__(self, func):
        super().__init__(func)
        self.funcs_before_set = []
        self.funcs_after_set = []

    def run_before_set(self, func: Callable[[Instance, Value], None]):
        self.funcs_before_set.append(func)
        return self
    
    def run_after_set(self, func: Callable[[Instance, Value], None]):
        self.funcs_after_set.append(func)
        return self
    
    def __set__(self, instance, value):
        for func in self.funcs_before_set:
            func(instance, value)
        instance.__dict__[self.attrname] = value
        for func in self.funcs_after_set:
            func(instance, value)
        return None