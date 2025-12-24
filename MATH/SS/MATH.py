import numpy

FILE_MATH = 'Coefficients-B10.txt'
FILE_PATH = 'pscratch/sd/y/yhzhang/Tensor/MATH/SS/'
FILE = open(FILE_PATH + FILE_MATH,'r').readlines()
MATH = ''
for LINE in FILE:

    LINE = LINE.replace('Log','numpy.log')
    LINE = LINE.replace('[','(')
    LINE = LINE.replace(']',')')
    LINE = LINE.replace(' ','')
    MATH = MATH + LINE

MATH = MATH.replace('=', ' = ')
MATH = MATH.replace('+', ' + ')
MATH = MATH.replace('-', ' - ')
MATH = MATH.replace('*', ' * ')
MATH = MATH.replace('/', ' / ')
MATH = MATH.replace('^','**')

open(FILE_PATH + FILE_MATH,'w').writelines(MATH)