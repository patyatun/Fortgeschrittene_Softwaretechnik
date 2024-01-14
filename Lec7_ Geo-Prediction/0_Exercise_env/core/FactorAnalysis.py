import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from factor_analyzer import FactorAnalyzer  
#from factor_analyzer import calculate_bartlett_sphericity 
#from factor_analyzer import calculate_kmo

import core.HelperTools    as ht
from timeit import default_timer as timer

teilenDurch = 10

#------------------------------------------------------------------------------
def getEigenvalues(df2):
    #First calculate Eigenvalues i.o.t. reduce features in the next step
    #fan.get_eigenvalues()
    def makeScreePlot():
        xvals = range(1, df2.shape[1]+1)
        plt.scatter(xvals, ev)
        plt.plot(xvals, ev)
        plt.title("Scree Plot")
        plt.xlabel("Factor")
        plt.ylabel("Eigenvalue")
        plt.grid()
        plt.ion()
        plt.show()
        plt.pause(0.001)
    #--------------------------------------------
    fan = FactorAnalyzer(rotation=None)
    fan.fit(df2)
    ev, v = fan.get_eigenvalues()
    len1 = len(np.where(ev > 1.0)[0])   # jetzt standard: 2.8           # 2.0 nur testweise; 1.2 einzusetzen!
    len2 = len1 # = len(np.where(ev > (ev[0]/teilenDurch))[0])          #Achtung: Bedingung 2 abgeschaltet
    print("Eigenvalues > 1: ", len1, "1. Eigenvalue /float_div {0:2.2f}: ".format(teilenDurch), len2)
    makeScreePlot()
    return ev, v, min(len1, len2)

#------------------------------------------------------------------------------
def getFactorInsightsEVReduced(df2, n_fac):
    colListOrder = ['FeatureName', 'maxLoadings', 'Variance', 'Proportional Var', 'Cum. Prop. Var.']
    df2 = df2.dropna()
    
    def highlight_max(s):
        is_max = s == s.absolute(s.max())   #Boolean-Vektor, bei dem nur Maximalwert true, Rest false
        return ['background-color: yellow' if lv else "" for lv in is_max]

    def markMaxLoadings():
        df_loadings["Features"] = df2.columns                       #Spalte mit Features wird erzeugt
        df_loadings.set_index("Features", inplace=True)             #Features werden als Index festgelegt
        df_loadings.style.applymap(highlight_max)                   #Maxima farblich markieren
    
    def getDF_FactorVarianceWithFeatureNamesAndMaxLoadings():
        df_FactorVariance = pd.DataFrame.from_records(fan.get_factor_variance()).T
        df_FactorVariance.columns = colListOrder[2:]
        df_FactorVariance[colListOrder[0]] = ''    
        df_FactorVariance[colListOrder[1]] = np.nan
        df_FactorVariance = df_FactorVariance[colListOrder]
        orderedMaxLoadings = dict()
        for lv in range(df_FactorVariance.shape[0]):
            colNumbers = df_loadings.iloc[:,lv].to_dict()
            maxTuple = max(zip(colNumbers.values(),colNumbers.keys()))
            orderedMaxLoadings[maxTuple[1]] = maxTuple[0]
            df_FactorVariance.at[lv, 'FeatureName'] = maxTuple[1]
            df_FactorVariance.at[lv, 'maxLoadings'] = maxTuple[0]
        return df_FactorVariance

    fan = FactorAnalyzer(n_factors=n_fac, rotation="varimax")    
    fan.fit(df2)  
    
    df_loadings = pd.DataFrame.from_records(np.absolute(fan.loadings_))
    markMaxLoadings()

    return df_loadings, getDF_FactorVarianceWithFeatureNamesAndMaxLoadings() 

#------------------------------------------------------------------------------

def getFactorAnalysisResults(dfSAV_zt, pdict):
   #1: Bartlett / KMO-Test
#    chi_square_value, p_value = calculate_bartlett_sphericity(dfSAV_zt)
#    kmo_all, kmo_model = calculate_kmo(dfSAV_zt)
#    print("--Bartlett's Sphericity Test-- Chi^2:",chi_square_value, "P-value (klein heißt sign. Untersch. zw. Cov-Mat&Identitäts-Mat): ", p_value)        
#    print(kmo_model)
    
    #2: Initiale Faktoranalyse für Eigenwerte-Test        
    ev, v, numEVselected1 = getEigenvalues(dfSAV_zt)
    
    print("Number of features in configuration: ", pdict["anzFeatures"])
    print("min(Number Eigenvalues (ev) > 1, Number Ev > (1.Ev/10)): ", numEVselected1)
    
    # numEVselected2 = max(pdict["anzFeatures"], numEVselected1)
    # print("Number of features used for calculation: ", numEVselected2)

    #3: Loadings & Faktor-Varianzen:
    df_loadings2, df_FactorVariance2 = getFactorInsightsEVReduced(dfSAV_zt, numEVselected1)  #wieder öffnen!!!!
    
    return df_loadings2, df_FactorVariance2                                            #wieder öffnen!!!!
#------------------------------------------------------------------------------
    
@ht.timer
def getColFilteredDF_fromFactorAnalysis(df_ztr, pdict):
    """Faktor analyzer with PCA & VARIMAX-rotation matrix"""    
    
    df_loadings3, df_FactorVariance3 = getFactorAnalysisResults(df_ztr, pdict)
    
    # while True:
    #     try:
    #         df_loadings3, df_FactorVariance3 = getFactorAnalysisResults(df_ztr, pdict)
    #         break
    #     except:
    #         # print('Abbbruch')
    #         # break
    #         print("Warning: LinAlgError occured, faktor analysis is repeated automatically")            
    
    # extract selected column names
    selectedFeatures = [ str(x) for x in df_FactorVariance3["FeatureName"]] # Achtung: wegen sqlalchemy kann tolist() keine echte Liste erzeugen
    # positive-filter dataframe by selected columns
    df_selFeatures = df_ztr[selectedFeatures]
    print(df_selFeatures.shape)
    return df_selFeatures, selectedFeatures
    




