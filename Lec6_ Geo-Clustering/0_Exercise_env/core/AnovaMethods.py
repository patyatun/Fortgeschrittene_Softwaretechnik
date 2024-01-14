#import numpy as np
import core.HelperTools as ht
import pandas as pd

#def rem_nulls_list(liste):
#    return [i for i in liste if i]

#------------------------------------------------------------------------------
#eta squared - BEGIN
def eta_squared(dframe, pdict, incol):
    """Eta^2: Einzelnes Feature"""
    tcol = pdict["targetcol"]    
    N = len(dframe.values)                           #scalar Anzahl Datensätze (Zeilen)            
    n = dframe.groupby(tcol).size()                 #vector Anzahl Zeilen pro Gruppe

    #ssb
    try:         
        mean_groupvec = dframe.groupby(tcol).sum()[incol]/n         #vector mean per Cluster-level
        mean_total = dframe[incol].sum()/N                           #scalar Mean über die gesamte Spalte
        sq_mean_diff_times_n = n * (mean_groupvec - mean_total)**2  #vector
        ssb = sq_mean_diff_times_n.sum()                  #scalar sumofsquares_between
    except:
        mean_total = 0
        ssb = 0

    #sst
    try:
        diff_inputmeantotal_sq = (dframe[incol] - mean_total)**2    #vector
        sst = diff_inputmeantotal_sq.sum()        #scalar sumofsquares_total
    except:
        sst = 0
        
    try:
        eta_sq = ssb / sst
    except ZeroDivisionError:
        eta_sq = float('nan')
        print("ZeroDivisionError bei :", incol)
        
    print(incol, eta_sq)
    return (incol, eta_sq)
    #----------
@ht.timer 
def getSeriesEtaSquared(dframe, pdict):
    """Berechnung: Eta^2, gesamte Spalte"""
    incols = list(dframe.columns.values); incols.remove(pdict["targetcol"])
    print("Marker 3 - getSeriesEtaSquared")
    # Liste mit Input-Spalten auslesen: List-of-tuples (colname, eta^2-val):
    list_of_etasq = list(map(lambda x: eta_squared(dframe, pdict, x), incols))
    #Index ist der Feature-Name, value der eta^2-Wert:
    idx, values = zip(*list_of_etasq)
    pseries_etasq = pd.Series(values, idx).rename("Eta squared")
    return pseries_etasq
    #----------

#eta squared - END
#------------------------------------------------------------------------------
#mean-tables - BEGIN 
@ht.timer 
def getMeanTables(dframe, pdict):
    """Mean-Tables"""
    mtClusters = dframe.groupby(pdict["targetcol"]).mean().T 
    print("Marker 1 - getMeanTables")
    mtTotal = dframe\
        .drop(pdict["targetcol"], axis=1, inplace = False)\
        .mean()\
        .T\
        .rename("Total mean")
    print("Marker 2 - getMeanTables")
    return mtClusters.merge(mtTotal, left_index=True, right_index=True)
        
#mean-tables - END 
#------------------------------------------------------------------------------
#z-transformation - BEGIN    
@ht.timer 
def getZTrans(dframe, pdict):
    """z-Transformationen der Effekte: Features auf Cluster"""
    
    df_wot = dframe.drop(pdict["targetcol"], axis=1, inplace = False)
    orig_means = getMeanTables(dframe, pdict).iloc[:, :-1]       #Achtung: 2. Mal, aber ohne gesamt
    
    df_ztrans = pd.DataFrame()
    for idx, row in orig_means.iterrows():
        feature_mean = df_wot[idx].mean(axis = 0)
        feature_std = df_wot[idx].std(axis = 0)
        row_ztrans = dict()
        for k, v in row.items():
            try:
                v_trans = (v - feature_mean) / feature_std
            except ZeroDivisionError:
                v_trans = float('nan')
            row_ztrans.update({k: v_trans})
        pdser_row_ztrans = pd.Series(data = row_ztrans, name = idx)
        df_ztrans = pd.concat([df_ztrans, pdser_row_ztrans], axis=1)
        
    df_ztrans = df_ztrans.T
    zcols = list(map(lambda x: "z_" + str(x), df_ztrans.columns.values))
    rename_dict = dict(zip(df_ztrans.columns.values, zcols))
    df_ztrans = df_ztrans.rename(columns = rename_dict)

    return df_ztrans
    #----------

