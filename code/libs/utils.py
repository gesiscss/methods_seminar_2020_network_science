__author__ = 'lisette.espin'

######################################################################################################################
# SYSTEM DEPENDENCES
######################################################################################################################
import sys, os
import operator
from datetime import datetime

######################################################################################################################
# FUNCTIONS
######################################################################################################################
def printf(msg, logfile=None):
    strtowrite = "[{}] {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(strtowrite)
    if logfile is not None:
        with open(logfile, 'a') as f:
            f.write('{}\n'.format(strtowrite))

def sortDictByValue(x,desc):
    sorted_x = sorted(x.items(), key=operator.itemgetter(1),reverse=desc)
    return sorted_x

def sortDictByKey(x,desc):
    sorted_x = sorted(x.items(), key=operator.itemgetter(0),reverse=desc)
    return sorted_x

def _swap_cols(arr, frm, to):
    arr[:,[frm, to]] = arr[:,[to, frm]]

def _swap_rows(arr, frm, to):
    arr[[frm, to],:] = arr[[to, frm],:]

def appendToFile(str, path):
    with open(path,'a') as f:
        f.write(str)