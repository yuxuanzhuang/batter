import numpy as np
from loguru import logger

# Copy from fe-toolkit to handle outliers in the MMBAR energy data.
def SizedChunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def MakeChunksWithSize( istart, istop, size ):
    # return a list of list of indexes from istart to istop
    # each sublist contains approximately "size" elements
    chunks = [ [ i for i in gen ]
               for gen in SizedChunks(range(istart,istop),size) ]
    
    if len(chunks) > 1:
        if len(chunks[-1]) < size*2/3:
            chunks[-2].extend(chunks[-1])
            del chunks[-1]
    return chunks

def MakeGroupedChunks( ene, size ):
    # return a list of list of indexes. This extends MakeChunksWithSize
    # by grouping adjacent chunks together if their means pass the t-test
    from scipy.stats import ttest_ind
    cidxs = MakeChunksWithSize(0,ene.shape[0],size)
    nchk = len(cidxs)
    ichk = 0
    #print("input chunk len ",nchk)
    while ichk < nchk-1:
        t,p = ttest_ind(ene[cidxs[ichk]],ene[cidxs[ichk+1]],
                        equal_var=False)
        if p > 0.05:
            cidxs[ichk].extend(cidxs[ichk+1])
            del cidxs[ichk+1]
            nchk -= 1
        else:
            ichk += 1
    return cidxs


def exclude_outliers(df, iclam):
    df = df.replace([np.inf, -np.inf], 1e9)
    efeps = df.values
    skips = [False] * efeps.shape[0]
    lams = df.columns.values

    avgs = np.mean(efeps,axis=0)
    meds = np.median(efeps,axis=0)
    meds = meds - meds[iclam]
    
    cidxs = MakeGroupedChunks( efeps[:,iclam], 200 )
    nchunks = len(cidxs)
    for ichunk in range(nchunks):
        idxs = cidxs[ichunk]
        ref = [ x for x in efeps[idxs,iclam] ]
        ref.sort()
        n = len(idxs)
        nmin = int(0.05*n)
        nmax = int(0.95*n)
        ref = ref[nmin:nmax]
        refm = np.median(ref)
        refs = np.std(ref)

        for i in idxs:
            for ilam,plam in enumerate(lams):
                if meds[ilam] > 10000:
                    if efeps[i,ilam] < refm-3*refs - 1000:
                        skips[i] = True
                        #refa = np.mean(ref)
                        #print("skip %6i e=%12.3e avg=%12.3e median=%12.3e std=%12.3e"%(i, efeps[i,ilam], refa, refm, refs))
                        break
    logger.debug(f"skips: {np.sum(skips)} for {iclam} {lams[iclam]}")
    return df[~np.array(skips)]