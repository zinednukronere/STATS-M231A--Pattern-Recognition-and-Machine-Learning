#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'fizzBuzz' function below.
#
# The function accepts INTEGER n as parameter.
#

def fizzBuzz(n):
    for k in range(1,n+1):
        if k%3==0 and k%5==0:
            print("FizzBuzz")
        elif k%3==0 and k%5!=0:
            print("Fizz")
        elif k%5==0 and k%3!=0:
            print("Buzz")
        else:
            print(k)        
 

if __name__ == '__main__':
    n = int(input().strip())

    fizzBuzz(n)
