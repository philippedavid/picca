import scipy as sp
from .data import forest
from scipy.interpolate import interp1d
from scipy.linalg import svd

def tophat_resampler(pix_data, llmin, llmax, llmin_rest, llmax_rest, dll):

    print('DLL: {}'.format(dll))
    ll_new = sp.arange(llmin, llmax, dll)

    for t,d_list in pix_data.items():
        for d in d_list:
            ## resample matrix:
            A = abs(d.ll-ll_new[:,None])
            w = A>dll/2
            A[w] = 0.
            A[~w] = 1.
            fl_new = A.dot(d.fl*d.iv)
            iv_new = A.dot(d.iv)

            ll_new_rest = ll_new - sp.log10(1+d.zqso)
            w = iv_new > 0
            w &= (ll_new_rest>llmin_rest) & (ll_new_rest<llmax_rest)

            fl_new = fl_new[w]
            iv_new = iv_new[w]
            fl_new /= iv_new

            d.fl = fl_new
            d.iv = iv_new
            d.ll = ll_new[w]

        pix_data[t] = coadd(d_list)

    return pix_data

def gaussian_resampler(pix_data, llmin, llmax, llmin_rest, llmax_rest, dll):

    print('DLL: {}'.format(dll))
    ll_new = sp.arange(llmin, llmax, dll)

    count=0
    for t,d_list in pix_data.items():
        print('\rresampling {}%'.format(round(count/len(pix_data)*100,3)), end='')
        count+=1
        for d in d_list:
            ## resample matrix:
            A = abs(d.ll-ll_new[:,None])/dll
            ## set to zero above 5 sigma
            w = A>5
            A[w]=0
            A[~w] = sp.exp(-A[~w]**2/2)
            fl_new = A.dot(d.fl*d.iv)
            iv_new = A.dot(d.iv)

            ll_new_rest = ll_new - sp.log10(1+d.zqso)
            w = iv_new > 0
            w &= (ll_new_rest>llmin_rest) & (ll_new_rest<llmax_rest)

            fl_new = fl_new[w]
            iv_new = iv_new[w]
            fl_new /= iv_new

            d.fl = fl_new
            d.iv = iv_new
            d.ll = ll_new[w]

        pix_data[t] = coadd(d_list)

    return pix_data

def spectro_perf_resampler(pix_data, llmin, llmax, llmin_rest, llmax_rest, dll):
    ll_new = sp.arange(llmin, llmax, dll)
    nbins = len(ll_new)
    count=0
    for t,d_list in pix_data.items():
        if count%1==0:
            print("\rcoadding {}%".format(round(count*100/len(pix_data),3)),end='')
        count+=1
        F = sp.zeros(nbins)
        C = sp.zeros((nbins, nbins))

        for d in d_list:
            wd = interp1d(d.ll, d.reso*1e-4, bounds_error = False, 
                    fill_value=1e-4)

            R = (d.ll[:,None]-ll_new)/wd(ll_new)
            R = sp.exp(-R**2/2)
            R /= R.sum(axis=1)[:,None]
            R = sp.sqrt(d.iv[:,None])*R

            F += (sp.sqrt(d.iv)*d.fl).dot(R)
            C += R.T.dot(R)

        try:
            u,s,vt = svd(C)
        except:
            pix_data[t] = None
            continue
            
        fl_new = vt.T.dot(u.T.dot(F))
        s = sp.diag(s)
        Q = vt.T.dot(s.dot(vt))
        norm = Q.sum(axis=1)
        w = norm>0
        Q[w,:] /= norm[w,None]

        d0 = d_list[0]
        ll_new_rest = ll_new - sp.log10(1+d0.zqso)
        w &= (ll_new_rest>llmin_rest) & (ll_new_rest<llmax_rest)

        fl_new = fl_new[w]/norm[w]
        iv_new = norm[w]
        reso = Q.std(axis=1)[w]

        assert w.sum() == len(iv_new)

        pix_data[t] = forest(ll_new[w], fl_new, iv_new, d0.thid,
                d0.ra, d0.dec, d0.zqso, d0.plate, d0.mjd, d0.fid,
                d0.order, reso=reso)

    return {t:v for t,v in pix_data.items() if v is not None}

def coadd(data_list):
    '''
    Takes a list of forest objects and returns the 
    coaddition. It assumes that all objects are on a 
    common wavelength grid

    Usage:
        f = coadd(data_list)

        data_list: iterable of forest objects

        Returns:
            a forest object with the coadded flux and ivar


    '''

    d0 = data_list[0]
    for d1 in data_list[1:]:
        d0 += d1

    return d0

