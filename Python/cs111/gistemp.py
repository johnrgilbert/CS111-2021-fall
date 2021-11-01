import numpy as np
import numpy.linalg as npla
import json
import scipy

#######################################################################
# Read GISTEMP data from the JSON file downloaded from a NASA website #
#                                                                     #
# Reference: https://data.giss.nasa.gov/gistemp/                      #
#######################################################################

def get_gistemp(filepath = 'annual_temps.json'):
    """Read the NASA GISTEMP data set from a JSON file.
    input: filepath (default 'annual_temps.json')
    output: two numpy 1D arrays: year, temp
    Outputs are in increasing order by year.
    """
    with open(filepath) as f:
        D = json.load(f)
    year = []
    temp = []
    for r in D:
        if r['Source'] == 'GISTEMP':
            year.append(r['Year'])
            temp.append(r['Mean'])
    perm = np.argsort(year)
    year = np.array(year)[perm]
    temp = np.array(temp)[perm]
    return (year, temp)
# end of get_gistemp()

