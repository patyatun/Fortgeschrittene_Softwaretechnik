# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:40:41 2023

@author: hdste
"""

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
currentWorkingDirectory = "C:\\(...)\\Lec6_ Geo-Clustering\\src"
# Docker:
# currentWorkingDirectory = "/app"

# -----------------------------------------------------------------------------
import os
os.chdir(currentWorkingDirectory)
print("Current working directory\n" + os.getcwd())

import pandas                        as pd
import random; random.seed(42)
from collections                     import Counter
import core.HelperTools              as ht

# -----------------------------------------------------------------------------
import core.DF_Transformations       as dft
import core.FactorAnalysis           as fal
import core.Clustering               as cl
import core.EffectAnalysisTable      as eat
# -----------------------------------------------------------------------------

import geopandas                     as gpd
from shapely                         import wkt
import matplotlib.pyplot             as plt

# -----------------------------------------------------------------------------
from config                          import pdict

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
@ht.timer
def main():
    """Main: Customers' segmentation with geo-visualization"""
    
    sozdemlabels        = list(pdict["label_dict"].values())
    
    df_datamart         = pd.read_csv(pdict["file_datamart"],   sep=";", decimal=",")
    df_geodata          = pd.read_csv(pdict["file_geodata"],    sep=";", decimal=",")
    df_sozdem           = pd.read_csv(pdict["file_sozdem"],     sep=";", decimal=",")    
    
    df_datmartsdem      = pd.merge(df_sozdem, df_datamart, how = "inner", on=pdict['geocode']) 

    # df_datmartsdem  = pd.merge(df_sozdem, df_datamart, how = "left", on=pdict['geocode']) 
    # df_datmartsdem  = pd.merge(df_datamart, df_sozdem, how = "left", on=pdict['geocode']) 
    
    df_datmartsdem      = dft.move_last_columns_to_front(df_datmartsdem, 4)
    df_datmartsdem.iloc[:, 4:]          = df_datmartsdem.iloc[:, 4:].apply(pd.to_numeric, errors='coerce')
    # df_datmartsdem.loc[:, sozdemlabels] = df_datmartsdem.loc[:, sozdemlabels].apply(pd.to_numeric, errors='coerce')



    dmartfields         = [pdict["geocode"]] + sozdemlabels
    
    # print(df_datmartsdem.dtypes)
    
    df_datmartsdem2     = df_datmartsdem\
        .loc[:, dmartfields]\
        .groupby(pdict["geocode"])\
        .mean()\
        .reset_index()
    
    # ------------------------------------------------------------------------------
    # II Faktor Analyse 
    
    df_selFeatures, selectedFeatures = fal.getColFilteredDF_fromFactorAnalysis(df_datmartsdem2.iloc[:, 1:], pdict)  
    print(selectedFeatures)   
    
    # ------------------------------------------------------------------------------
    # III Clustering
    
    # from config                          import pdict
    pdict['anzFeatures'] = 5
    pdict['bfac']       = 20
    pdict['clnum']      = 4
    pdict['thr']        = 0.2

    
    df_selFeatures2     = df_selFeatures.fillna(0)
    
    labels              = cl.birchen(df_selFeatures2, pdict['bfac'], pdict['clnum'], pdict['thr'])
    labels2             = [x+1 for x in labels]
       
    oCounter1           = ht.countFreqs(labels2)
    print(oCounter1)
        
    df_clustered                = df_datmartsdem2.copy() 
    df_clustered['Cluster']     = labels2
    
    print("Numbers per cluster: ")
    Counter(df_clustered["Cluster"])   
    
    # ------------------------------------------------------------------------------
    # IV Effect table
    
    df_clustered_eat    = df_clustered.copy()
    df_clustered_eat.drop(pdict["geocode"], axis=1, inplace=True)
    
    print("Effect table: ")
    effect_analysis_table = eat.load_effect_analysis_tableDB(df_clustered_eat, pdict) 

    # ------------------------------------------------------------------------------    
    # ------------------------------------------------------------------------------
    # V Geo-visualization
    
    # Get all cluster labels from df_clustered-Column 'Cluster'
    cluster_labels      = sorted(df_clustered['Cluster'].unique().tolist())
    # Get column labels for onehoted & summarized households in each cluster
    df_agg_list         = [pdict["geocode"]] + ["Cluster_" + str(x) for x in cluster_labels]
    # onehot & summarize households for each cluster
    df_agg              = dft.onehot_encode_and_aggregate(df_clustered, pdict['targetcol'], pdict["geocode"])[df_agg_list]

    # -----------------------------------------
    # left join of summarized households in each cluster on geo-polygon
    # gdata["kgs12"] = gdata["kgs12"].astype('int64')
    df_geodata2         = df_geodata.merge(df_agg, on=pdict["geocode"], how ='left')
    
    for v in cluster_labels: 
        clnum = 'Cluster_' + str(v)
        df_geodata2[clnum] = df_geodata2[clnum].fillna(0)

    df_geodata2["Cluster_0"] = 0
    
    cols = [col for col in df_geodata2.columns if col != 'Cluster_0']
    cols.insert(len(cluster_labels) - len(cols), 'Cluster_0')
    df_geodata2         = df_geodata2[cols]
    
    # Get all 'Cluster_' columns except 'Cluster_0'
    cluster_columns     = [col for col in df_geodata2.columns if col.startswith('Cluster_') and col != 'Cluster_0']
    
    # Create a condition where all specified 'Cluster_' columns are 0
    condition           = (df_geodata2[cluster_columns] == 0).all(axis=1)
    df_geodata2.loc[condition, 'Cluster_0'] = 1
    
    # -----------------------------------------
    gdf                 = gpd.GeoDataFrame(df_geodata2, geometry=df_geodata2['Polygon'].apply(wkt.loads))
    gdf.crs = "EPSG:4326"
    
    # Define colors
    blueviolett = '#4020a0'
    grey = '#808080'

    for lv in [0] + cluster_labels:
        filter_label = "Cluster_" + str(lv) 

        # Filter the data - replace 'condition' with your condition for emphasis
        emphasized_gdf = gdf[gdf[filter_label] == 1]
        
        # Basic plot
        fig, ax = plt.subplots(figsize=(10, 6))
        gdf.plot(ax=ax, color=grey, markersize=5, linewidth=0.1)  # Plot all points
        emphasized_gdf.plot(ax=ax, color=blueviolett, markersize=10,  linewidth=0.1)  # Emphasized points
        
        # Add titles, labels, etc.
        ax.set_title("Geospatial Data Visualization: Cluster :" + str(lv))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        plt.show()
    
    
if __name__ == "__main__":
    main()   


# ------------------------------------------------------------------------------
