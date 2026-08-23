from pathlib import Path

FILE_PATH = Path(__file__).resolve().parent / 'Coefficient_B01.txt'
FILE = FILE_PATH.read_text().splitlines()

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

FILE_PATH.write_text(MATH)
