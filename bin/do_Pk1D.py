#!/usr/bin/env python


import fitsio
import argparse
import glob
import sys
import numpy as np

from picca import constants
from picca.Pk1D import Pk1D, compute_Pk_raw, compute_Pk_noise, compute_cor_reso, fill_masked_pixels, split_forest
from picca.data import delta

from array import array

def make_tree(tree,nb_bin_max):

    zqso = array( 'f', [ 0. ] )
    mean_z = array( 'f', [ 0. ] )
    mean_reso = array( 'f', [ 0. ] )
    mean_SNR = array( 'f', [ 0. ] )
    nb_mask_pix = array( 'f', [ 0. ] )
    
    plate = array( 'i', [ 0 ] )
    mjd = array( 'i', [ 0 ] )
    fiber = array( 'i', [ 0 ] )
    
    nb_r = array( 'i', [ 0 ] )
    k_r = array( 'f', nb_bin_max*[ 0. ] )
    Pk_r = array( 'f', nb_bin_max*[ 0. ] )
    Pk_raw_r = array( 'f', nb_bin_max*[ 0. ] )
    Pk_noise_r = array( 'f', nb_bin_max*[ 0. ] )
    Pk_diff_r = array( 'f', nb_bin_max*[ 0. ] )
    cor_reso_r = array( 'f', nb_bin_max*[ 0. ] )

    tree.Branch("zqso",zqso,"zqso/F")
    tree.Branch("mean_z",mean_z,"mean_z/F")
    tree.Branch("mean_reso",mean_reso,"mean_reso/F")
    tree.Branch("mean_SNR",mean_SNR,"mean_SNR/F")
    tree.Branch("nb_masked_pixel",nb_mask_pix,"nb_mask_pixel/F")

    tree.Branch("plate",plate,"plate/I")
    tree.Branch("mjd",mjd,"mjd/I")
    tree.Branch("fiber",fiber,"fiber/I")

    tree.Branch( 'NbBin', nb_r, 'NbBin/I' )
    tree.Branch( 'k', k_r, 'k[NbBin]/F' )
    tree.Branch( 'Pk_raw', Pk_raw_r, 'Pk_raw[NbBin]/F' )
    tree.Branch( 'Pk_noise', Pk_noise_r, 'Pk_noise[NbBin]/F' )
    tree.Branch( 'Pk_diff', Pk_diff_r, 'Pk_diff[NbBin]/F' )
    tree.Branch( 'cor_reso', cor_reso_r, 'cor_reso[NbBin]/F' )
    tree.Branch( 'Pk', Pk_r, 'Pk[NbBin]/F' )
    
    return zqso,mean_z,mean_reso,mean_SNR,plate,mjd,fiber,\
    nb_mask_pix,nb_r,k_r,Pk_r,Pk_raw_r,Pk_noise_r,cor_reso_r,Pk_diff_r

def compute_mean_delta(ll,delta,zqso):

    for i in range (len(ll)):
        ll_obs= np.power(10.,ll[i])
        ll_rf = ll_obs/(1.+zqso)
        hdelta.Fill(ll_obs,ll_rf,delta[i])
        hdelta_RF.Fill(ll_rf,delta[i])
        hdelta_OBS.Fill(ll_obs,delta[i])

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--out-dir', type = str, default = None, required=True,
                        help = 'output file name')

    parser.add_argument('--in-dir', type = str, default = None, required=True,
                        help = 'data directory')

    parser.add_argument('--mode', type = str, default = None, required=False,
                        help = ' root or fits, if root call PyRoot')

    parser.add_argument('--SNR-min',type = float,default=2.,required=False,
                        help = 'minimal mean SNR per pixel ')
    
    parser.add_argument('--reso-max',type = float,default=85.,required=False,
                        help = 'maximal resolution in km/s ')

    parser.add_argument('--nb-part',type = int,default=3,required=False,
                        help = 'Number of parts in forest')

    parser.add_argument('--nb-pixel-min',type = int,default=75,required=False,
                        help = 'Minimal number of pixels in a part of forest')

    parser.add_argument('--nb-pixel-masked-max',type = int,default=30,required=False,
                        help = 'Maximal number of masked pixels in a part of forest')

    parser.add_argument('--noise-estimate', type = str, default = 'mean_diff', required=False,
                        help = ' Estimate of Pk_noise  pipeline/diff/mean_diff')
    

    args = parser.parse_args()

#   Create root file
    if (args.mode=='root') :
        from ROOT import TCanvas, TH1F, TFile, TTree, TProfile2D, TProfile
        storeFile = TFile("Testpicca.root","RECREATE","PK 1D studies studies");
        nb_bin_max = 700
        tree = TTree("Pk1D","SDSS 1D Power spectrum Ly-a");
        zqso,mean_z,mean_reso,mean_SNR,plate,mjd,fiber,\
        nb_mask_pix,nb_r,k_r,Pk_r,Pk_raw_r,Pk_noise_r,cor_reso_r,Pk_diff_r = make_tree(tree,nb_bin_max)
        hdelta  = TProfile2D( 'hdelta', 'delta mean as a function of lambda-lambdaRF', 34, 3600., 7000., 16, 1040., 1200., -5.0, 5.0)
        hdelta_RF  = TProfile( 'hdelta_RF', 'delta mean as a function of lambdaRF', 320, 1040., 1200., -5.0, 5.0)
        hdelta_OBS  = TProfile( 'hdelta_OBS', 'delta mean as a function of lambdaOBS', 1700, 3600., 7000., -5.0, 5.0)

        
# Read Deltas
    fi = glob.glob(args.in_dir+"/*.fits.gz")
    data = {}
    ndata = 0

    for i,f in enumerate(fi):
        if i%1==0:
            sys.stderr.write("\rread {} of {} {}".format(i,len(fi),ndata))
        hdus = fitsio.FITS(f)
        dels = [delta.from_fitsio(h,Pk1D_type=True) for h in hdus[1:]]
        ndata+=len(dels)
        if (args.mode=='fits') :
            out = fitsio.FITS(args.out_dir+'/Pk1D-'+str(i)+'.fits.gz','rw',clobber=True)
        print ' ndata = ',ndata
        for d in dels:

            # Selection over the SNR and the resolution
            if (d.mean_SNR<=args.SNR_min or d.mean_reso>=args.reso_max) : continue

            # minimum number of pixel in forest
            nb_pixel_min = args.nb_pixel_min
            if (len(d.ll)<nb_pixel_min) : continue

            # Split in n parts the forest
            nb_part_max = len(d.ll)/nb_pixel_min
            nb_part = min(args.nb_part,nb_part_max)
            m_z_arr,ll_arr,de_arr,diff_arr,iv_arr = split_forest(nb_part,d.dll,d.ll,d.de,d.diff,d.iv)
            for f in range(nb_part): 
            
                # Fill masked pixels with 0.
                ll_new,delta_new,diff_new,iv_new,nb_masked_pixel = fill_masked_pixels(d.dll,ll_arr[f],de_arr[f],diff_arr[f],iv_arr[f])
                if (nb_masked_pixel> args.nb_pixel_masked_max) : continue
                if (args.mode=='root'): compute_mean_delta(ll_new,delta_new,d.zqso)
            
                # Compute Pk_raw
                k,Pk_raw = compute_Pk_raw(delta_new,ll_new)

                # Compute Pk_noise
                run_noise = False
                if (args.noise_estimate=='pipeline'): run_noise=True
                Pk_noise,Pk_diff = compute_Pk_noise(iv_new,diff_new,ll_new,run_noise)               

                # Compute resolution correction
                delta_pixel = d.dll*np.log(10.)*constants.speed_light/1000.
                cor_reso = compute_cor_reso(delta_pixel,d.mean_reso,k)

                # Compute 1D Pk
                if (args.noise_estimate=='pipeline'):
                    Pk = (Pk_raw - Pk_noise)/cor_reso
                elif (args.noise_estimate=='diff'):
                    Pk = (Pk_raw - Pk_diff)/cor_reso
                elif (args.noise_estimate=='mean_diff'):
                    selection = (k>0) & (k<0.02)
                    Pk_mean_diff = sum(Pk_diff[selection])/float(len(Pk_diff[selection]))
                    Pk = (Pk_raw - Pk_mean_diff)/cor_reso

                # Build   Pk1D
                Pk1D_final = Pk1D(d.ra,d.dec,d.zqso,d.mean_z,d.plate,d.mjd,d.fid,k,Pk_raw,Pk_noise,cor_reso,Pk)

                # save in root format
                if (args.mode=='root'):
                    zqso[0] = d.zqso
                    mean_z[0] = m_z_arr[f]
                    mean_reso[0] = d.mean_reso
                    mean_SNR[0] = d.mean_SNR
                    nb_mask_pix[0] = nb_masked_pixel

                    plate[0] = d.plate
                    mjd[0] = d.mjd
                    fiber[0] = d.fid

                    nb_r[0] = min(len(k),nb_bin_max)
                    for i in range(nb_r[0]) :
                        k_r[i] = k[i]
                        Pk_raw_r[i] = Pk_raw[i]
                        Pk_noise_r[i] = Pk_noise[i]
                        Pk_diff_r[i] = Pk_diff[i]
                        Pk_r[i] = Pk[i]
                        cor_reso_r[i] = cor_reso[i]
                               
                    tree.Fill()

                # save in fits format

                if (args.mode=='fits'):
                    hd={}
                    hd["RA"]=d.ra
                    hd["DEC"]=d.dec
                    hd["Z"]=d.zqso
                    hd["MEANZ"]=d.mean_z
                    hd["MEANRESO"]=d.mean_reso
                    hd["MEANSNR"]=d.mean_SNR

                    cols=[k,Pk]
                    names=['k','Pk']
                
                    out.write(cols,names=names,header=hd)
        if (args.mode=='fits'):
            out.close()
        
# Store root file results           
    if (args.mode=='root'):
         storeFile.Write()

   
    print "all done"


