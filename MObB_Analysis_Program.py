# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 09:32:32 2022

@author: jbett
"""
##############################################################################
############ Import modules, if any modules are missing from #################
############ your pyhton installation, you must download and #################
############ install them according to publishers instructions ###############
###############################################################################

import pandas as pd
from pyteomics import mzxml, auxiliary
import numpy as np
import math as math
import matplotlib as mp
import mpl_toolkits.mplot3d.art3d as art3d
import scipy as sp
from sklearn import metrics
import statistics as st
import matplotlib.patches as mpl_patches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import pearsonr
from scipy import stats
from numpy import trapz
import pickle

##############################################################################
############ MObB defined functions, it is recommended that only #############
################### advanced users makeedits to lines 34-607 ####################
##############################################################################


# Calculate the dot product of two vectors
############## Input #################################
# v1, v2 = numeric vectors of the same length
############## Output #################################
# A numeric value that is the dot product between v1 and v2

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

# Calculate the length of a vector
############## Input #################################
# v = a 1D vector of any length
############## Output #################################
# A numeric value that is the legth of vector v

def length(v):
  return math.sqrt(dotproduct(v, v))


######## Create amino acid to atomic composition map #########################
############## each amino acid is converted into a ###########################
#################### vector [C,H,N,O,S] of counts ############################

A= [3,7,1,2,0]
R= [6,14,4,2,0]
N= [4,8,2,3,0]
D= [4,7,1,4,0]
C= [3,7,1,2,1]
Q= [5,10,2,3,0]
E= [5,9,1,4,0]
G= [2,5,1,2,0]
H= [6,9,3,2,0]
I= [6,13,1,2,0]
L= [6,13,1,2,0]
K= [6,14,2,2,0]
M= [5,11,1,2,1]
F =[9,11,1,2,0]
P= [5,9,1,2,0]
S= [3,7,1,3,0]
T= [4,9,1,3,0]
W= [11,12,2,2,0]
Y= [9,11,1,3,0]
V= [5,11,1,2,0]
aakeys=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
aaempiricalvalues =[A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V]
AAcompmap=dict(zip(aakeys,aaempiricalvalues))

# Create a theoretical isotope distribution for 16O/18O labeled pairs, given
# a known ratio between 16O intensity and total intensity
############## Input #################################
# fn = a known or hypothesized ratio between 16O intensity and total intensity
#      for a 16O/18O labeled pair

# theodist = output of MObB command gettheodist, ran with an offset = 0
############## Output #################################
# z = an Nx2 array where each row (N) is an isotopic mass,
#     Column 0 = m/z values of isotopologues
#     Column 0 = isotopologue probabilities, given natural isotopes

def Createtheoretical(fn,theodist):
    dist1=np.pad(theodist, (0, 2), 'constant')
    dist2=np.pad(theodist, (2, 0), 'constant')
    dist3=(dist1*fn) + (dist2*(1 - fn));
    z=dist3[:,0:2]
    return z

# Analyze the similiarity in elution profiles (XICs) of light and heavy
# labeled isotopologes in 16O/18O labeled pairs
############## Input #################################
# modelin = the output of MObB command modelpeaks()

# timesarray = an array of Retention Times corresponding to modelin
############## Output #################################
# RTcos = the cosine similiarity between XICs light and heavy labeled peptides

# RTdiff = the distance (second,minutes) between the intensity weighted centers
#          of light and heavy labeled XICs

def RTanalysis(modelin,timesarray):
    RTlength=max(timesarray)-min(timesarray)
    XIClight=modelin[0]+modelin[1]
    for n in range(2,len(modelin)):
        if n == 2:
            XICheavy=modelin[n]
        else:
            XICheavy=XICheavy+modelin[n]
            
    RTcos = 1-sp.spatial.distance.cosine(XIClight,XICheavy)
    weightedscansl = sum((timesarray * XIClight))/sum(XIClight)
    weightedscansh = sum((timesarray * XICheavy))/sum(XICheavy)
    RTdiff = abs(weightedscansl-weightedscansh)/RTlength
    return RTcos,RTdiff

##############################################################################
# Lines 133-322 define the protocols for fitting an elution model to 16O/18O
# labeled pairs, the general strategy is to model elution profiles as the sum
# no more than two assymetric gaussians. A complete description of the
# strategy for creating elution models can be found in ref XXXXXXXXXXXXX
##############################################################################
def two_peaks(t, *pars):    
    'function of two overlapping peaks'
    a10 = pars[0]  # peak area
    a11 = pars[1]  # elution time
    a12 = pars[2]  # width of gaussian
    a13 = pars[3]  # exponential damping term
    a20 = pars[4]  # peak area
    a21 = pars[5]  # elution time
    a22 = pars[6]  # width of gaussian
    a23 = pars[7]  # exponential damping term   
    p1 = asym_peak(t, [a10, a11, a12, a13])
    p2 = asym_peak(t, [a20, a21, a22, a23])
    return p1 + p2



def asym_peakx(t, *pars):
    'from Anal. Chem. 1994, 66, 1294-1301'
    a0 = pars[0]  # peak area
    a1 = pars[1]  # elution time
    a2 = pars[2]  # width of gaussian
    a3 = pars[3]  # exponential damping term
    f = (a0/2/a3*np.exp(a2**2/2.0/a3**2 + (a1 - t)/a3)
         *(erf((t-a1)/(np.sqrt(2.0)*a2) - a2/np.sqrt(2.0)/a3) + 1.0))
    return f



def modelpeaks(feat):
    intsfitlist=[]
    intslist=[]
    areas=[]
    maxpeaks=[]
    intsarray=[]
    cosarray=[]
    feat=feat
    areas=[]
    maxiso=feat[:,4].max()
    maxiso=maxiso.astype(np.int64)+1
    r2list=[]
    r22list=[]
    featmodel = {}
    for n in range(-2,3):
        featsub=feat[:,4]==n
        featsubs=feat[featsub,:]
        times=featsubs[:,3]
        ints=featsubs[:,1]
        peaks,_=find_peaks(ints)
        prominences = peak_prominences(ints, peaks)[0]
        centers=pd.DataFrame({'peaks':peaks,'prominences':prominences})
        centers=centers.sort_values(by='prominences')
        centers=centers.reset_index()
        if len(centers)>0:
            RT1=centers['peaks'][len(centers)-1]
        else:
            break
        if len(centers) > 1:
            RT2=centers['peaks'][len(centers)-2]
        prominences.sort()
        

        
        parguess = (np.trapz(featsubs[:,1],times), featsubs[RT1,3], 0.05, 0.05)
        try:
            popt, pcov = curve_fit(asym_peakx, times, ints, parguess)
        except RuntimeError:
            popt=[]
            pcov=[]
        
        
        if len(popt)==0:
            intsfit = np.nan
            cor = 0
        if len(popt)>0:
            intsfit=asym_peakx(times,*popt)
            cor,_=pearsonr(ints,intsfit)
            if cor >= 0.9 or len(centers) <= 1:
                peak1 = asym_peakx(times, *popt)
                featmodel["{}".format(n)] = {}
                featmodel["{}".format(n)]['peak1'] = peak1
                featmodel["{}".format(n)]['peak2'] = np.zeros(len(times))
                
                    
        if cor < 0.9 and len(centers) > 1:
            parguess2 = (np.trapz(featsubs[:,1],times)/2, featsubs[RT1,3], 0.05, 0.05,
                    np.trapz(featsubs[:,1],times)/2, featsubs[RT2,3], 0.05, 0.05)
            
            try:
                popt2, pcov = curve_fit(two_peaks, times, ints, parguess2)
            except:
                print("No model could be fit")
                popt2=[]
                pcov=[]
                pass
                
            intsfit2=two_peaks(times,*popt2)
            cor2,_=pearsonr(ints,intsfit2)
            if cor2 > cor:
                intsfit=intsfit2
                popt=popt2
                
                peak1pars = popt[0:4]
                peak1 = asym_peakx(times,*peak1pars)
                
                peak2pars = popt[4:8]
                peak2 = asym_peakx(times,*peak2pars)
                
                
                featmodel["{}".format(n)] = {}
                featmodel["{}".format(n)]['peak1'] = peak1
                featmodel["{}".format(n)]['peak2'] = peak2
    

    path1=[]
    path2=[]
    xx=-1
    for z in range(-2,3):
        isoin = featmodel['{}'.format(z)]
        in1=isoin['peak1']
        in2=isoin['peak2']
        if z == -2:
            path1.append(in1)
            path2.append(in2)
        else:
            adj1=path1[xx]
            adj2=path2[xx]
            if 1-sp.spatial.distance.cosine(in1,adj1)>=0.6 or 1-sp.spatial.distance.cosine(in2,adj1)>=0.6:
                if 1-sp.spatial.distance.cosine(in1,adj1)>=0.6 and 1-sp.spatial.distance.cosine(in2,adj1)>=0.6:
                    path1.append(in1+in2)
                else:
                    if 1-sp.spatial.distance.cosine(in1,adj1)>=0.6:
                        path1.append(in1)
                    if 1-sp.spatial.distance.cosine(in2,adj1)>=0.6:
                        path1.append(in2)
            else:
                path1.append(np.zeros(len(times)))
                
            if 1-sp.spatial.distance.cosine(in1,adj2)>=0.6 or 1-sp.spatial.distance.cosine(in2,adj2)>=0.6:
                if 1-sp.spatial.distance.cosine(in1,adj1)>=0.6 and 1-sp.spatial.distance.cosine(in2,adj1)>=0.6:
                    path1.append(in1+in2)
                else:               
                    if 1-sp.spatial.distance.cosine(in1,adj2)>=0.6:
                        path2.append(in1)
                    if 1-sp.spatial.distance.cosine(in2,adj2)>=0.6:
                        path2.append(in2)
            else:
                path2.append(np.zeros(len(times)))
        xx=xx+1



    graph=[]
    graphsub=[]
    for g in range(0,len(path1)):
        if sum(path1[g])>0:
            graphsub.append(path1[g])
    if len(graphsub)>4:
        graph.append(graphsub)
    
    graphsub=[]
    for g in range(0,len(path2)):
        if sum(path2[g])>0:
            graphsub.append(path2[g])
    if len(graphsub)>4:
        graph.append(graphsub)
    
    finalmodel=[]
    peakmaxs=[]
    if len(graph) > 1:
        for x in range(0,len(graph[0])):
                finalmodelx = graph[0][x]+graph[1][x]
                finalmodel.append(finalmodelx)
                peakmaxs.append(max(graph[0][x])+max(graph[1][x]))
    else:
        for x in range(0,len(graph[0])):
                finalmodelx = graph[0][x]
                finalmodel.append(finalmodelx)
                peakmaxs.append(max(graph[0][x]))
            

        
    return finalmodel, times,peakmaxs


#Get atomic composition
############## Input #################################
# seq = A string giving the sequence of an unmodified peptide
# AAcompmap = A dictionary of atomic compositions for each amino acid,
#             generated on lines 63-85
############## Output #################################
# atomcomp = A vector [C,H,N,O,S] of atom counts for peptide with sequence seq

def getatomcomp(seq,AAcompmap):
    atomcomp=list()
    for q in range(0,len(seq)):
        Carbon=AAcompmap[seq[q]][0]
        Hydrogen=AAcompmap[seq[q]][1]
        Nitrogen=AAcompmap[seq[q]][2]
        Oxygen=AAcompmap[seq[q]][3]
        Sulfur=AAcompmap[seq[q]][4]
        atomcomp.append([Carbon,Hydrogen,Nitrogen,Oxygen,Sulfur])
    
    atomcomp=[ sum(row[i] for row in atomcomp) for i in range(len(atomcomp[0])) ]
    return atomcomp

# Filter raw spectra for a range of Rention Times 
############## Input #################################
# RTS = Retention Time Start
# RTE = Retention Time End
# spectrumhandle = the file handle to an mzXML file, file handles are 
#                  generated by the mzxml.read() function of pyteomics
############## Output #################################
# swath = a list of metadata corresponding to MS1 scans collected within
#         the Retention Time range specified by RTS and RTE

def getswath(RTS,RTE,spectrumhandle):
    swath = spectrumhandle.time[RTS:RTE]    
    
    return swath

# filter swath for m/z values within a defined ppm tolerance window
############## Input #################################
# swath = the output of MObB command getswath()
# mzs = a list of m/z ratio, can be generated by MObB command getmzs()
# cutoff = m/z tolerances are a user set parameter and is defined on line 653. 
#          Units are ppm and default is 5 ppm
############## Output #################################
# feature = An Nx5 array, where each row (N) is an MS1 scan,
#           Column 0 = m/z values
#           Column 1 = Intensity values
#           Column 2 = MS1 scan numbers
#           Column 3 = Retention Time values
#           Column 4 = Isotope index
def filterswath(swath,mzs,cutoff):
    feat=[]
    for g in mzs:
        isoindex=np.flatnonzero(g==mzs)[0]-2
    
        for m in range(0,len(swath)):
            scan=[]
            ms1mzarray=swath[m]['m/z array']
            ms1intsarray=swath[m]['intensity array']
            for j in range(0,len(ms1mzarray)):
                if (ms1mzarray[j]>g-(g*cutoff/1000000))&(ms1mzarray[j]<g+(g*cutoff/1000000)):
                    mz=ms1mzarray[j]
                    intensity=ms1intsarray[j]
                    scannum=int(swath[m]['num'])
                    RT=float(swath[m]['retentionTime'])
                    scan=[mz,intensity,scannum,RT,isoindex]
                
                if not scan:
                    mz=g
                    intensity = 0
                    scannum=int(swath[m]['num'])
                    RT=float(swath[m]['retentionTime'])
                    scan=[mz,intensity,scannum,RT,isoindex]
            
            feat.append(scan)
            
    if not feat:
        feature=None
    else:
        feature=np.vstack(feat)
        return feature
    
# Create a theoretical isotope distribution
# input is a vector [C,H,N,O,S] describing the atomic composition of
# unmodified vectors
# cutoff refers to the probabilty of observing a given isotopic mass
############## Input #################################
# atomcomp = A vector [C,H,N,O,S] of atom counts, given an unmodified peptide
#            sequence. Generated by MObB command getatomcomp()

# cutoff = A threshold that refers to the probabilty of observing a given 
#          isotopic mass. Isotopic mass with probabilities below cutoff
#          will not be returned

############## Output #################################
# theodist = an Nx2 array where each row (N) is an isotopic mass,
#            Column 0 = m/z values of isotopologues
#            Column 0 = isotopologue probabilities, given natural isotopes

def gettheodist(atomcomp,cutoff):
    form=[[12.0,13.0035,0.99,atomcomp[0]],[1.007825,2.014102,0.99985,atomcomp[1]],[14.003074,15.000109,0.99634,atomcomp[2]],[15.9949,17.999,0.99762,atomcomp[3]],[31.972,33.9679,0.9502,atomcomp[4]]];
    temp3=[]
    for i in range(0,len(form)):
        atomset=form[i]
        mainmass=atomset[0]
        altmass=atomset[1]
        abundancemainmass=atomset[2]
        abundancealtmass=1-abundancemainmass
        atoms=atomset[3]
        temp2=[]
        
        for j in range(0,atoms):
            temp1=[];
            combs= math.factorial(atoms)/(math.factorial(j)*math.factorial(atoms-j));
            prob= (abundancemainmass**(atoms-j))*(abundancealtmass**j);
            odds=combs*prob;
            mass = (altmass*j)+(mainmass*(atoms-j));
            if odds > cutoff:
                temp1=[j,mass,odds];
                temp2.append(temp1);
                
        temp3.append(temp2)
            
    temp4=[]
    
    for ci in range(0,len(temp3[0])):
        cset=temp3[0][ci];
        cmass=cset[1];
        codds=cset[2];
        for hi in range(0,len(temp3[1])):
            hset=temp3[1][hi];
            hmass=hset[1];
            hodds=hset[2];
            for ni in range(0,len(temp3[2])):
                nset=temp3[2][ni];
                nmass=nset[1];
                nodds=nset[2];
                for oi in range(0,len(temp3[3])):
                    oset=temp3[3][oi];
                    omass=oset[1];
                    oodds=oset[2];
                    for si in range(0,len(temp3[4])):
                        sset=temp3[4][si];
                        smass=sset[1];
                        sodds=sset[2];
                        tmass= round(cmass+hmass+nmass+omass+smass);
                        todds=codds*hodds*nodds*oodds*sodds;
                        temp4.append([tmass,todds]);
                        
    temp4=pd.DataFrame(temp4)
    temp4=temp4.groupby([0])[1].sum()
    theodist=pd.DataFrame({'Mass':temp4.index,'odds':temp4})
    theodist=theodist.reset_index()
    theodist=theodist.drop(columns=0)
    theodist=theodist[theodist['odds']>=cutoff]
    return theodist

# get theoretical m/z values for 16O/18O isotope pairs
############## Input #################################
# theodist = output of MObB command gettheodist
# mz0 = the monoisotopic mass of modified peptide
# offset = an integer value specifying whether the monoisotopic mass refers to
#          the (16O) light labeled peptide or the (18O) heavy labeled peptide
############## Output #################################
# mzs = a list of m/z values

def getMZs(theodist,mz0,indexoffset):
   mzs=[]
   if indexoffset==0:
       for y in range(-2,len(theodist)):
           mzn=mz0+(y/charge)
           mzs.append(mzn)
   else:
        for y in range(0,len(theodist)+2):
            mzn=mz0+(y/charge)
            mzs.append(mzn)
   return mzs  


# A prelimary, coarse grained method for filtering lower probabilty isotope 
# peaks that overlap along m/z axis with the theoretical m/z distribution of
# target peptide
############## Input #################################
# feat = An Nx5 array, where each row (N) is an MS1 scan,
#        Column 0 = m/z values
#        Column 1 = Intensity values
#        Column 2 = MS1 scan numbers
#        Column 3 = Retention Time values
#        Column 4 = Isotope index

# theomodel = output of MObB command gettheodist

############## Output #################################
# feat = An Nx5 array, where each row (N) is an MS1 scan,
#        Column 0 = m/z values
#        Column 1 = Intensity values
#        Column 2 = MS1 scan numbers
#        Column 3 = Retention Time values
#        Column 4 = Isotope index

def filterisotopesMZ(feat,theomodel):
    if sum(feat[:,1])!=0:
        featweightedt=feat[:,3]*feat[:,1]
        featweightedt=sum(featweightedt)/sum(feat[:,1])
        feat=pd.DataFrame(feat)
        featgb = feat.groupby(4)    
        featframe=[featgb.get_group(x) for x in featgb.groups]
        isolist=[]
        mzlist=[]
        
        spectra2d=[]
        spectra2dt=[]
        for i in range(0,len(featframe)):
            isoin=featframe[i]
            isoin=isoin.reset_index()
            isolist.append(sum(isoin[1]))
            mzlist.append(isoin[4][0])
        if sum(isolist)!=0:
            feat=[]
            for isoprot in range(0,len(mzlist)):
                if mzlist[isoprot]<0:
                    feat.append(featframe[isoprot])
                    spectra2d.append(isolist[isoprot])
                else:
                    spectra2dt.append(isolist[isoprot])
                    theomodelv=theomodel['odds']/sum(theomodel['odds'])
                    if sum(np.array(spectra2dt)) != 0:
                        isolistv=np.array(spectra2dt)/sum(np.array(spectra2dt))
                        isolistvcopy=np.pad(isolistv, (0, (len(theomodel)-len(isolistv))), 'constant')
                        isolistv=isolistvcopy
                    else:
                        isolistv=np.array(spectra2dt)
                        isolistvcopy=np.pad(isolistv, (0, (len(theomodel)-len(isolistv))), 'constant')
                        isolistv=isolistvcopy
                    ls=metrics.mean_squared_error(theomodelv,isolistv)
                    
                    if mzlist[isoprot]==0:
                        lskp=ls
                        feat.append(featframe[isoprot])
                    if mzlist[isoprot]!=0:
                        if ls<lskp:
                            lskp=ls
                            feat.append(featframe[isoprot])
            if not feat:
                feat=None
                return feat
            else:
                feat=np.vstack(feat)
                return feat
            
# dictionary data class with function to add
class my_dictionary(dict):  
  
    # __init__ function  
    def __init__(self):  
        self = dict()  
          
    # Function to add key:value  
    def add(self, key, value):  
        self[key] = value 
        
# A tool for visualizing MS1 feature and elution models
def viewMS1feature(feature,model):
    timesarray=np.unique(feature[:,3])
    isotopesarray=np.unique(feature[:,4])
    fig, axs = mp.pyplot.subplots(1,5) 
    axs=np.array(axs)
    n=0
    for ax in axs.reshape(-1):
        ax.set_ylim([0,(max(feature[:,1])*1.2)])
        ax.plot(timesarray,model[n],color='red')
        ax.plot(timesarray,feature[feature[:,4]==isotopesarray[n]][:,1],'--',color='green')
        red_patch = mp.patches.Patch(color='red', label='Modeled Data')
        green_patch = mp.patches.Patch(color='green', label='Raw Spectra')
        ax.legend(handles=[red_patch,green_patch])
        if n==0:
            ax.set_ylabel('Intensity')
        if n==2:
            ax.set_xlabel('Retention Time')
        n=n+1
    mp.pyplot.show()
            
        
##############################################################################
##################### End of MObB defined functions ##########################
##############################################################################

# Read in input dataframe. MObB was built around the output of poplular DDA
# search software MaxQuant. MObB will accept MaxQuant evidence.txt files as
# input without the need for reformatting. If users wish to supply their own
# input dataframe consideration for doing so are outlined in the MObB tutorial
# found at (URL= XXXXXXXXXXXXXXXXXXX).

# Depending on the source of you input dataframe you may need to alter the
# 'fields' vector to agree with column names on your desired input

fields = ['Sequence','Proteins','Charge','Acetyl..Protein.N.term.','O18',
          'Oxidation..M.','m.z','Calibrated.retention.time.start',
          'Calibrated.retention.time.finish','Retention.time',
          'Raw.file','Missed.cleavages','Retention.time.calibration']
df = pd.read_table('C://Users/jbett/Desktop/mouseMOBB/RAW/exampledata_evidence.txt',usecols=fields)


# Subset evidence data for peptides of interest. MObB is limited to analyzing
# peptides that contain a single methionine and no cysteines.
mets = df['Sequence'].str.count("M")
df['mets']=mets
df=df[df['mets']==1]
df = df[~df['Sequence'].str.contains("C")]
#df=df[df['O18.M.']==1]
df = df[~df['Proteins'].str.contains("CON",na=True)]
df = df[~df['Proteins'].str.contains("REV",na=True)]
df['OX']=df['O18']+df['Oxidation..M.']
df.sort_values(by='Raw.file',inplace=True)
df=df.reset_index()

# Get a list of unique sequences.
uniqueseqs=np.unique(df['Sequence'][df['Missed.cleavages']==0])


##############################################################################
######################## Begin MObB search ##################################
##############################################################################

# Initialize empty global variables
atomlib=my_dictionary()
featarray=[]
openspec='none'
theolist = []

# Declare tolerance for m/z error in units of ppm
mztolerance = 5

for n in range(0,len(df)):
    
    # For each MS1 feature read in data from the input dataframe (df) 

    seq=df['Sequence'][n]
    protein = df['Proteins'][n]
    charge=df['Charge'][n]
    RTS=df['Calibrated.retention.time.start'][n]-df['Retention.time.calibration'][n]
    RTE=df['Calibrated.retention.time.finish'][n]-df['Retention.time.calibration'][n]
    rtmid=(RTS+RTE)/2
    rawfile=df['Raw.file'][n]
    mz0=df['m.z'][n]
    feat=[]
    
    # Open rawfile and convert spectra to readable format
    if openspec != rawfile:
        f = mzxml.read('C://Users/jbett/Desktop/mouseMOBB/RAW/exampledata_mzXML/'+rawfile+'.mzXML',use_index=True)
        openspec=rawfile
    
    # For each peptide get the atomic composition    
    atomcomp=getatomcomp(seq,AAcompmap) 
    
    # For each peptides create a theoretical isotope distribution.
    # Build atomic composition library as you go (saves speed) to reduce the 
    # redundant generation of theoretical isotope distributions from different
    # peptides with the same atomic composition
    if tuple(atomcomp) not in atomlib:
        theodist=gettheodist(atomcomp,0.0001)
        atomlib.add(tuple(atomcomp),theodist)
        theolist.append(theodist)
        
    if tuple(atomcomp) in atomlib:
        theodist =  atomlib[tuple(atomcomp)]
        theolist.append(theodist)
        
    #get theoretical m/z values for 16O/18O isotope pairs
    indexoffset=df['Oxidation..M.'][n]*(-2)
    mzs=getMZs(theodist,mz0,indexoffset)
    
    #subset raw spectra based on RT bounds
    swath=getswath(RTS,RTE,f)
    
    # Subset raw spectra for theoretical m/z values (default mztolerence 5ppm)
    feat=filterswath(swath,mzs,mztolerance)
    if feat is None:
        featarray.append(feat)
        continue
    
    # A prelimary, coarse grained method for filtering lower probabilty isotope 
    # peaks that overlap along m/z axis with the theoretical m/z distribution of
    # target peptide
    feat=filterisotopesMZ(feat,theodist)
    if feat is None:
        featarray.append(feat)
        continue
    featarray.append(feat)
    print(n)
    
##############################################################################
######################## Begin MObB modeling ##################################
##############################################################################

# Initialize empty global variables
models=[]
timesarray=[]
Spectras=[]


# For each MS1 feature fit an elution model
############## Input #################################
# featarray = an array of Nx5 matrices, an array of matrices returned by
#             the MObB search (lines 642-711)
############## Output #################################
# models = a list with the same length as input featarray
#          each element of models is a series of graphs representing the
#          modeled XICs for each MS1 feature.
#
# timesarray = a list (with the same length as input featarray) of Retention
#              Times associated with each model generated.
######## Note models can be visualized usi)ng the plotModels() command ########

for q in range(0,len(featarray)):
    finalmodel=[]
    timesx=[]
    try:
        finalmodel, timesx,peaksx = modelpeaks(featarray[q])
        models.append(finalmodel)
        timesarray.append(timesx)
        print(q)
    except:
        models.append(np.nan)
        timesarray.append(np.nan)
        print(q)

# Concenate MS1 features by sequence into a new "Super-Feature"
############## Input #################################
# uniqueseqs = a list of unique peptide sequences found in input dataframe(df)
# models = a list with the same length as input dataframe (df)
#          each element of models is a series of graphs representing the
#          modeled XICs for each MS1 feature.
############## Output #################################
# Spectras = a list with the same length as uniqueseqs. Each element of
#            Spectras is a series of concentenated graphs representing modeled
#            XICs for each unique peptide sequence

df['feature']=models
for n in range(0,len(uniqueseqs)):
    Sequence=uniqueseqs[n]
    target=df[df['Sequence'].str.contains(Sequence)]
    target=target.reset_index()
    spectra=[]
    for j in range(0,5):
        isoins=[]
        for i in range(0,len(target)):
            try:
                isoin=target['feature'][i][j]
                isoins=np.hstack((isoins,isoin))
            except:
                continue
        spectra.append(isoins)
        
    Spectras.append(spectra)
    print(n)

##############################################################################
######################## Begin MObB analysis #################################
##############################################################################    

# Initialize empty global variables
slopes=[]
r2s=[]
p_values=[]
std_errs=[]
totalints=[]
slopecorrections=[]
slopesnew=[]

# For each element of Spectra estimate the observed ratio between light (16O)
# and heavy (18O) labeled peptides. Ratios are estimated using a linear
# regression strategy. See refXXXXX for details.
############## Input #################################
# Spectras = a list with the same length as uniqueseqs. Each element of
#            Spectras is a series of concentenated graphs representing modeled
#            XICs for each unique peptide sequence
############## Output #################################
# slopes = a list with the same length as Spectras. Each element of slopes
#          is the slope of regression between light (16O) and heavy (18O) 
#          labeled peptide intensities; or the observed ratio between 
#          light (16O) and heavy (18O) labeled peptide intensities.
#
# r2s = a list with the same length as Spectras. Each element of r2s
#       is the correlation coeffecient of a linear regression used to
#       estimate the observed ratio between light (16O) and heavy (18O) 
#       labeled peptide intensities.
#
# p_values = a list with the same length as Spectras. Each element of p_values
#            is the pvalue of a correlation betwen light (16O) and heavy
#            labeled peptide intensities.
#
# std_errs = a list with the same length as Spectras. Each element of 
#            std_errors is the standar error of a linear regression used to
#            estimate the observed ratio between light (16O) and heavy (18O) 
#            labeled peptide intensities.



for n in range(0,len(Spectras)):
    try:
        Y=Spectras[n][2]+Spectras[n][3]+Spectras[n][4]
        X=Spectras[n][0]+Spectras[n][1]
        slope, intercept, r_value, p_value, std_err = stats.linregress(Y,X)
        slopes.append(slope)
        r2s.append(r_value)
        p_values.append(p_value)
        std_errs.append(std_err)
        totalints.append(sum(X)+sum(Y))
    except:
        slopes.append(np.nan)
        r2s.append(np.nan)
        p_values.append(np.nan)
        std_errs.append(np.nan)
        totalints.append(np.nan)
    print(n)
    
# Convert L/H ratios to to L/(L+H) ratios
############## Input #################################
# slopes = a list with the same length as Spectras. Each element of slopes
#          is a L/H ratio
############## Output #################################
# slopes = a list with the same length as Spectras. Each element of slopes
#          is a L/(L+H) ratio
  
for n in range(0,len(slopes)):
    ins=slopes[n]
    oxnew=slopes[n]/(1+slopes[n])
    slopes[n]=oxnew
    

# For each unique peptide sequence calculate a correction curve that accounts
# for the theoretical overlap between light (16O) and heavy (18O) labeled pairs
############## Input #################################
# uniqueseqs = a list of unique peptide sequences found in input dataframe(df)
############## Output #################################
# slopecorrections = a list of tupples with the same length as uniqueseqs.
#                    Each element is a correction factor.
for q in range(0,len(uniqueseqs)):
    atomcomp = atomcomp=getatomcomp(uniqueseqs[q],AAcompmap)
    theoretical = gettheodist(atomcomp,0.0001)
    actuals=[]
    observed=[]
    for n in range(0,1000):
        fn=n/1000
        z=Createtheoretical(fn, theoretical)
        actuals.append(fn)
        observed.append(sum(z[:,1][0:2]))
    slope, intercept, r_value, p_value, std_err = stats.linregress(observed,actuals)
    slopecorrections.append([slope,intercept])
    print(q)

# Correct L/(L+H) ratios by theoretical overlap between light (16O) and
# heavy (18O) labeled pairs
# Convert L/H ratios to to L/(L+H) ratios
############## Input #################################
# slopes = a list with the same length as uniqueseqs. Each element of slopes
#          is a L/(L+H) ratio 
############## Output #################################
# slopesnew = a list with the same length as uniqueseqs. Each element of 
#             slopes is a corrected L/(L+H) ratio.
for q in range(0,len(slopes)):
    try:
       corrected = (slopes[q]*slopecorrections[q][0])+slopecorrections[q][1]
       slopesnew.append(corrected)
    except:
        slopesnew.append(np.nan)
    print(q)
    
# Generate output dataframe
############## Output #################################
# output = A dataframe with N rows. Each is row is a unique peptide sequence
#          analyzed.
#
#           Sequence = sequence of unmodified peptide
#           FractionOxidized = the final corrected L/(L+H) ratio
#           Fit = The correlation coeffecient of a linear regression used to
#                  estimate the observed ratio between light (16O) and heavy 
#                  (18O) labeled peptide intensities.
#           totalints = The total intensity for each peptide analyzed


output = pd.DataFrame(
    {'Sequence': uniqueseqs,
     'FractionOxidized':slopesnew,
     'Fit':r2s,
     'totalints':totalints
    })

output.to_csv('C://Users/jbett/Desktop/yeastMOBB/WT3results.txt')

# Optional storage of quantitation models for each MS1 feature model
#with open("C://Users/jbett/Desktop/mouseMOBB/newresults/results/modelsWT3.txt", "wb") as mx:
#    pickle.dump(models,mx)

# Optional storage of quantitation models for each MS1 feature, raw spectra
#with open("C://Users/jbett/Desktop/mouseMOBB/newresults/results/featsWT3.txt", "wb") as mx:
#    pickle.dump(featarray,mx)
    

    


            

            
    