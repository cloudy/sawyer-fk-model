#/usr/bin/env python3

import subprocess
import numpy as np 

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)    
    return int(result.strip().split()[0])

# dropcols
def data_generator(fname, batchsize = 256, dof = 7, posor = 6, dofoffset = 7, dropcol = None):
    f = open(fname)
    flen = file_len(fname)
    i = 0
    while True:
        chunk = np.array([np.fromstring(f.readline(), sep=',') for _ in range(batchsize)])
        if i >= flen:
            break
        i += batchsize
        res = (chunk[:,:dof], chunk[:,dofoffset: dofoffset + posor])  
        #print(res[0], end='\n\n')
        #print("Shape: ", res[0].shape) 

        yield res 



# Tester
if __name__ == '__main__':
    tfile = '/data/sawyer_fk_data/7DOF/UFMD7_10M.txt'

    print(file_len(tfile))
    
    val = 0
    for p in data_generator(tfile): #dgen():
        print("LOL:", end = " ")
        print(p[0])
        print(p[0].shape)
        #val += p.shape[0]

    print(val)
