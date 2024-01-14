import math
import numpy as np
import pandas as pd
import core.AnovaMethods as am
import core.HelperTools as ht

#------------------------------------------------------------------------------
# Function value range:

@ht.timer
def wertebereich(df_origCompl, df_origCompl_Werte):
    """Calculate function value range"""
    dict_dfcols = dict(zip(df_origCompl.columns.values, df_origCompl.columns.values))
    dict_dfc_minmax = dict()
    for key in dict_dfcols:
        try:
            if np.array_equal(df_origCompl[key].to_numpy(), df_origCompl_Werte[key].to_numpy()):
                dict_dfcols[key] = list(set(df_origCompl.loc[:, key]))
            else:
                dict_dfcols[key] = list(set(zip(df_origCompl.loc[:, key], \
                    df_origCompl_Werte.loc[:, key])))
                #remove (nan, nan)-tuples:
                dict_dfcols[key] = [v for v in dict_dfcols[key] if not math.isnan(v[0])] 
                #tuples to strings:
                dict_dfcols[key] = list(map(ht.tuples_to_string, dict_dfcols[key]))
        except:
            print("Function value range has not been calculated correctly")
        try:
            lminmax = [min(dict_dfcols[key]), max(dict_dfcols[key])]
        except ValueError:
            lminmax = []
        dict_dfc_minmax[key] = lminmax
        dict_dfc_minmax[key] = " ... ".join(str(e) for e in dict_dfc_minmax[key])
        
    ret = pd \
        .DataFrame.from_dict(dict_dfc_minmax, orient='index') \
        .rename(columns = {0: "Wertebereich" }) 
                
    return ret

@ht.timer 
def fill_onehot_typ_beschreibung(effect_analysis_table, df_meta, pdict):
    metadesc_dict = dict(zip(df_meta.index.values, df_meta[pdict["meta_description"]]))
    for k, v in effect_analysis_table.iterrows():
        if ("$" in v.values[1]):
            effect_analysis_table.iloc[k,5] = "dummy"
            
            feature_onehot = effect_analysis_table.iloc[k,1]
            feature_orig = feature_onehot.split("$")[0]
            effect_analysis_table.iloc[k,6] = metadesc_dict[feature_orig]
    return effect_analysis_table

@ht.timer 
def load_effect_analysis_tableSPSS(dfc, \
            pdict, df_meta = -1,  selectedFeatures = 1, aktive_features = -1, dict_dfc_minmax = -1):
    """Calculate/Load effect analysis with data from SPSS """
    
    #Effect-Analysis-table with means, eta_squared, ztransformed-means:
    effect_analysis_table = am.effect_analysis(dfc, \
                 pdict["targetcol"], dict_dfc_minmax, meta = df_meta,  PK = "ID")
    # if Clusters were joined: pdict["clk_col"]
        
    effect_analysis_table["F_Aktiv"] = \
        np.isin(effect_analysis_table["ID"], aktive_features)    
    effect_analysis_table["F_PCA"] = \
        np.isin(effect_analysis_table["ID"], selectedFeatures)
        
    # Determine base-feature from onehot-level: 
    colbf_eat = ht.col_base_features(effect_analysis_table["ID"], "$")
    effect_analysis_table["F_Szen"] = np.isin(colbf_eat, pdict["scenario"])
    
    effect_analysis_table["joiner"] = colbf_eat

    #Sort columns:
    colorder_fixedpart = ["Index", "ID", "F_Aktiv", "F_PCA", "F_Szen", \
        pdict["meta_typ"], pdict["meta_description"], "Wertebereich"] #meta_auswahl ändern     
    colorder_fixedpart = ["ID", "F_Aktiv", "F_PCA", "F_Szen"] #, \

    # Sort by eta_squared:
    ret = effect_analysis_table #\
#        .sort_values(by="Eta_Squared", ascending=False)
    
    return ret

@ht.timer 
def load_effect_analysis_tableDB(dframe, pdict):
    """Calculate/Load effect analysis with data from DB """
    eat = am.getMeanTables(dframe, pdict)\
        .merge(am.getSeriesEtaSquared(dframe, pdict), left_index=True, right_index=True)\
        .merge(am.getZTrans(dframe, pdict), left_index=True, right_index=True)\
        .reset_index()\
        .rename(columns = {"index": pdict["description"]})#\
      #  .reset_index()        
    return ht.sortDF(eat,"Eta squared", False) #eat
    
    