import numpy as np
import struct

bits = {'0':'0000', '1':'0001', '2':'0010', '3':'0011', 
        '4':'0100', '5':'0101', '6':'0110', '7':'0111', 
        '8':'1000', '9':'1001', 'a':'1010', 'b':'1011', 
        'c':'1100', 'd':'1101', 'e':'1110', 'f':'1111'}

drop = {'0':'0', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', 
        '8':'0', '9':'1', 'a':'2', 'b':'3', 'c':'4', 'd':'5', 'e':'6', 'f':'7'}

def int64_to_hex(f):
    if f >= 0:
        pad = '0'
    else:
        pad = 'F'
    s = hex((f + (1 << 64)) % (1 << 64))
    s = s[2:]           # remove the 0x prefix
    while len(s) < 16:  # pad with 0 or F depending on sign
        s = pad + s
    return s
#end of int64_to_hex

def double_to_hex(f):
    s = hex(struct.unpack('<Q', struct.pack('<d', f))[0])
    s = s[2:]           # remove the 0x prefix
    while len(s) < 16:  # pad with zeros
        s = '0' + s
    return s
#end of double_to_hex

def print_float64(x):
    """Print a 64-bit floating-point number in various formats.
    """
    print('input     :', x)
    # Cast the input to a 64-bit float
    x = np.float64(x)
    xhex = double_to_hex(x)
    print('as float64: {:.16e}'.format(x))
    print('as hex    : ' + xhex)
    if bits[xhex[0]][0] == '0':
        sign = '0 means +'
    else:
        sign = '1 means -'
    print('sign      :', sign)
    expostr = drop[xhex[0]] + xhex[1:3]
    expo = int(expostr, 16)
    if expo == 0:
        print('exponent  :', expostr, 'means zero or denormal')
    elif expo == 2047:
        print('exponent  :', expostr, 'means inf or nan')
    else:
        print('exponent  :', expostr, 'means', expo, '- 1023 =', expo - 1023)
        mantissa = '1.'
        for i in range(3,16):
            mantissa = mantissa + bits[xhex[i]]
        print('mantissa  :', mantissa)
    print()
#end of print_float64
