import numpy

FILE_MATH = 'Coefficients-B1.txt'
FILE_PATH = '/pscratch/sd/y/yhzhang/LimberCloud/MATH/SS/'
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
MATH = MATH.replace('^',' ** ')

open(FILE_PATH + FILE_MATH,'w').writelines(MATH)