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

# 命令行进度条
def printPercent(progress, total, decimals = 2, prefix:str = '', suffix:str = ''):
    '''打印命令行进度条'''
    percent = (f"{{0:{decimals+4}.{decimals}f}}").format(100 * (progress / float(total)))
    print(f'{prefix}{percent}%', end=suffix)

# 简易错误处理
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
    '''用于简单设置 numpy 的错误处理方式'''
    def __init__(self, arg:dict|str|None = None, **kwargs):
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

# 修改版 cached_property
from functools import cached_property as _cached_property
from typing import Callable, Any
type Value = Any
type Instance = Any

class cached_property(_cached_property):
    '''Modified cached_property / 修改版 cached_property'''
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

if __name__ == "__main__" and False:
    class a:
        @cached_property
        def b(self):
            return 0
        @b.run_before_set
        def b(self, value):
            print(f'Now setting b from {self.b} to {value}')
        @b.run_after_set
        def b(self, value):
            print(f'Setted, b is now {self.b}')
    test = a()
    test.b = 1