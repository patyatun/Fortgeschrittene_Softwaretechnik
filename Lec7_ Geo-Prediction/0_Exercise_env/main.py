# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:40:41 2023

@author: hdste
"""

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
currentWorkingDirectory = "C:\\(...)\\Lec7_ Geo-Prediction"
# Docker:
# currentWorkingDirectory = "/app"

# -----------------------------------------------------------------------------
import os
os.chdir(currentWorkingDirectory)
print("Current working directory\n" + os.getcwd())

import pandas                        as pd
import random; random.seed(42)
import core.HelperTools              as ht

# -----------------------------------------------------------------------------
import core.DF_Transformations       as dft
import core.FactorAnalysis           as fal
import core.Clustering               as cl
# -----------------------------------------------------------------------------

import geopandas                     as gpd
from shapely                         import wkt
import matplotlib.pyplot             as plt
import matplotlib.colors             as mcolors
# -----------------------------------------------------------------------------

from sklearn.linear_model            import LogisticRegression
from sklearn.model_selection         import train_test_split
from sklearn.metrics                 import classification_report, confusion_matrix
from collections                     import Counter

# -----------------------------------------------------------------------------
from config                          import pdict

# -----------------------------------------------------------------------------
@ht.timer
def main():
    """Main: Geo-Prediction of new customers' based on calculated geo-segments"""
    
    sozdemlabels            = list(pdict["label_dict"].values())
    df_datamart             = pd.read_csv(pdict["file_datamart"],   sep=";", decimal=",")
    df_geodata              = pd.read_csv(pdict["file_geodata"],    sep=";", decimal=",")
    df_sozdem               = pd.read_csv(pdict["file_sozdem"],     sep=";", decimal=",")  
    
    df_datmartsdem          = pd.merge(df_sozdem, df_datamart, how = "inner", on=pdict['geocode']) 
    df_datmartsdem          = dft.move_last_columns_to_front(df_datmartsdem, 4)
    df_datmartsdem.iloc[:, 4:]  = df_datmartsdem.iloc[:, 4:].apply(pd.to_numeric, errors='coerce')
    dmartfields             = [pdict["geocode"]] + sozdemlabels
    df_datmartsdem2         = df_datmartsdem\
        .loc[:, dmartfields]\
        .groupby(pdict["geocode"])\
        .mean()\
        .reset_index()
    
    # ------------------------------------------------------------------------------
    # II & III Faktor Analyse & Clustering
    
    df_selFeatures, selFeatures = fal.getColFilteredDF_fromFactorAnalysis(df_datmartsdem2.iloc[:, 1:], pdict)  
    df_selFeatures2         = df_selFeatures.fillna(0)
    labels                  = cl.birchen(df_selFeatures2, pdict['bfac'], pdict['clnum'], pdict['thr'])
    labels2                 = [x+1 for x in labels] 
    df_clustered            = df_datmartsdem2.copy() 
    df_clustered['Cluster'] = labels2
    
    # ------------------------------------------------------------------------------    
    # ------------------------------------------------------------------------------
    # V Geo-visualization
    
    # Get all cluster labels from df_clustered-Column 'Cluster'
    cluster_labels          = sorted(df_clustered['Cluster'].unique().tolist())
    # Get column labels for onehoted & summarized households in each cluster
    df_agg_list             = [pdict["geocode"]] + ["Cluster_" + str(x) for x in cluster_labels]
    # onehot & summarize households for each cluster
    df_agg                  = dft.onehot_encode_and_aggregate(df_clustered, pdict['targetcol'], pdict["geocode"])[df_agg_list]
    # left join of summarized households in each cluster on geo-polygon
    df_geodata2             = df_geodata.merge(df_agg, on=pdict["geocode"], how ='left')
    
    for v in cluster_labels: 
        clnum                   = 'Cluster_' + str(v)
        df_geodata2[clnum]      = df_geodata2[clnum].fillna(0)

    df_geodata2["Cluster_0"]    = 0
    
    cols                    = [col for col in df_geodata2.columns if col != 'Cluster_0']
    cols.insert(len(cluster_labels) - len(cols), 'Cluster_0')
    df_geodata2             = df_geodata2[cols]
    
    # Get all 'Cluster_' columns except 'Cluster_0'
    cluster_columns         = [col for col in df_geodata2.columns if col.startswith('Cluster_') and col != 'Cluster_0']
    cluster_columns0        = [col for col in df_geodata2.columns if col.startswith('Cluster_')]    
    
    # Create a condition where all specified 'Cluster_' columns are 0
    condition               = (df_geodata2[cluster_columns] == 0).all(axis=1)
    # Update 'Cluster_0' to 1 where the condition is True
    df_geodata2.loc[condition, 'Cluster_0'] = 1
    

    # ------------------------------------------------------------------------------    
    # ------------------------------------------------------------------------------    
    # ------------------------------------------------------------------------------
    # new, from lecture "Geo-Prediction":
    # VI Multi-Class Classification
    
    df_clustered2           = pd.merge(df_geodata2, df_sozdem, on=pdict["geocode"], how = "left").dropna(how="any")
    # revert onehot-encoding
    df_clustered2[pdict['targetcol']] = df_clustered2\
        .loc[:, cluster_columns0]\
        .idxmax(axis=1)\
        .str.split('_')\
        .str[1]\
        .astype(int)

    X                       = df_clustered2[selFeatures]
    X0                      = X.fillna(0)
    y                       = df_clustered2[pdict['targetcol']]
    X_train, X_test, y_train, y_test = train_test_split(X0, y, test_size=0.3, random_state=0)
    
    # Create logistic regression model with balanced class weights
    model                   = LogisticRegression(class_weight='balanced', max_iter=1000, verbose=1)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred                  = model.predict(X_test)
    y_pred2                 = model.predict(X0)

    # Evaluate the model
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    df_clustered3           = df_clustered2.copy()
    df_clustered3["Cluster_pred"] = y_pred2
    
    Counter(df_clustered3["Cluster"])   
    Counter(df_clustered3["Cluster_pred"])
    
    # ------------------------------------------------------------------------------    
    # ------------------------------------------------------------------------------
    # VII Similarities
    
    def calculate_row_cosine_similarity(row):
        # Convert row to numpy array
        row_array               = row.to_numpy()
        # Calculate cosine similarity with average values
        ret                     = dft.cosine_similarity(row_array, average_values.to_numpy())
        return ret
    
    df_clustered4           = df_clustered3.copy()
    
    Cloh_list               = ["Cluster_" + str(x) for x in list(set(df_clustered4['Cluster'])) if x > 0]
    
    for v in Cloh_list:
        print(v)
        
        average_values              = df_clustered4[df_clustered4[v] == 1]\
            .loc[:, sozdemlabels]\
            .mean()
            
        df_clustered4[v + '_cosine_similarity'] = df_clustered4\
            .loc[:, sozdemlabels]\
            .apply(calculate_row_cosine_similarity, axis=1)

    # ------------------------------------------------------------------------------    
    # ------------------------------------------------------------------------------
    # VIII Geo-visualization - generalized on new potential customers
    
    dfc                     = df_clustered4.loc[:, [pdict["geocode"], pdict['targetcol_pred']]]
    
    clunderscore            = pdict['targetcol_pred'] + "_"
        
    # Get all cluster labels from df_clustered-Column 'Cluster'
    cluster_labels2         = sorted(dfc[pdict['targetcol_pred']].unique().tolist())
    # Get column labels for onehoted & summarized households in each cluster
    df_agg_list2            = [pdict["geocode"]] + [clunderscore + str(x) for x in cluster_labels2]
    # onehot & summarize households for each cluster
    df_agg2                 = dft.onehot_encode_and_aggregate(dfc, pdict['targetcol_pred'], pdict["geocode"])[df_agg_list2]

    # -----------------------------------------
    # left join of summarized households in each cluster on geo-polygon
    df_geodata2             = df_geodata.merge(df_agg2, on=pdict["geocode"], how ='left')
    
    for v in cluster_labels: 
        clnum                   = clunderscore + str(v)
        df_geodata2[clnum]      = df_geodata2[clnum].fillna(0)

    cluster_columns0        = [col for col in df_geodata2.columns if col.startswith(clunderscore)]    
    
    # Create a condition where all specified 'Cluster_' columns are 0
    condition               = (df_geodata2[cluster_columns0] == 0).all(axis=1)
    # Update 'Cluster_0' to 1 where the condition is True
    df_geodata2.loc[condition, 'Cluster_0'] = 1
    
    # -----------------------------------------
    gdf                     = gpd.GeoDataFrame(df_geodata2, geometry=df_geodata2['Polygon'].apply(wkt.loads))
    gdf.crs                 = "EPSG:4326"
    
    # Define colors
    # blueviolett = '#4020a0'
    redviolett              = '#a02040'
    grey                    = '#808080'

    for lv in [0] + cluster_labels:
        filter_label            = clunderscore + str(lv) 

        # Filter the data - replace 'condition' with your condition for emphasis
        emphasized_gdf          = gdf[gdf[filter_label] == 1]
        
        # Basic plot
        fig, ax                 = plt.subplots(figsize=(10, 6))
        gdf.plot(ax=ax, color=grey, markersize=5, linewidth=0.1)  # Plot all points
        emphasized_gdf.plot(ax=ax, color=redviolett, markersize=10,  linewidth=0.1)  # Emphasized points
        
        # Add titles, labels, etc.
        ax.set_title("Geospatial Data Visualization: Cluster" + str(lv))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        plt.show()

    # ------------------------------------------------------------------------------    
    # ------------------------------------------------------------------------------
    # IX Count filtered Geolocations
    
    selcols4                = ['Geocode', 'Cluster_pred', 'Cluster_1_cosine_similarity', 'Cluster_2_cosine_similarity', 
                           'Cluster_3_cosine_similarity', 'Cluster_4_cosine_similarity']
    df_clustered5           = df_clustered4.loc[:, selcols4]
    
    counts_series           = {}
    for lv in list(range(1, pdict['clnum'] + 1 )): # [1,2,3,4]
    
        df_filt                 = df_clustered5 
        counts_series[lv]       = dft.count_lowthresto100(df_filt, 'Cluster_{}_cosine_similarity'.format(lv))
        
    df_sim                      = pd.concat(counts_series, axis=1)
    df_sim.columns              =  [ '_'.join(x.split('_')[:2]) + "_simcount" for x in  df_clustered5.columns.values[-4:]]

    # ------------------------------------------------------------------------------    
    # ------------------------------------------------------------------------------
    # X Geovisualization - Projections & Similarity filters
    
    dfc2 = df_clustered5.copy()
    
    filtercols              = ['Geocode', 'Polygon', 'Cluster_1_cosine_similarity', 'Cluster_2_cosine_similarity', 'Cluster_3_cosine_similarity', 'Cluster_4_cosine_similarity']
    df_geodata3             = df_geodata\
        .merge(dfc2, on=pdict["geocode"], how ='left')\
        .loc[:,filtercols]\
        .dropna()
    
    df_geodata4             = df_geodata3.copy()
    labels_clsim            = [item for item in df_geodata4.columns.values if item.startswith("Cluster_")]
    
    for v in labels_clsim:

        # Assuming 'df' is your DataFrame with 'Polygon' and 'Cluster_1_cosine_similarity'
        # Convert the 'Polygon' column to a GeoSeries
        gdf                     = gpd.GeoDataFrame(df_geodata4, geometry=gpd.GeoSeries.from_wkt(df_geodata4['Polygon']))
        gdf.crs = "EPSG:4326"
        
        # Create a custom colormap from blue to red
        cmap                    = mcolors.LinearSegmentedColormap.from_list("", ["blue", "red"])
        
        # Create subplots
        fig, axs                = plt.subplots(1, 2, figsize=(15, 6))
        
        # First subplot for 0.95 or more
        # Plot all polygons in gray
        gdf.plot(ax=axs[0], color='gray', edgecolor='black')
        # Overlay selected polygons
        gdf_095_or_more         = gdf[gdf[v] >= 0.95]
        gdf_095_or_more.plot(column=v, ax=axs[0], legend=True,
                             cmap=cmap, legend_kwds={'label': "≥ 0.95 Cosine Similarity", 'orientation': "horizontal"})
        axs[0].set_title('Cosine Similarity ≥ 0.95')
        
        # Second subplot for 0.75 or more
        # Plot all polygons in gray
        gdf.plot(ax=axs[1], color='gray', edgecolor='black')
        # Overlay selected polygons
        gdf_075_or_more         = gdf[gdf[v] >= 0.75]
        gdf_075_or_more.plot(column=v, ax=axs[1], legend=True,
                             cmap=cmap, legend_kwds={'label': "≥ 0.75 Cosine Similarity", 'orientation': "horizontal"})
        axs[1].set_title('Cosine Similarity ≥ 0.75')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()   

