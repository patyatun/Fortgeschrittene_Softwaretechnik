import numpy                         as np
import pandas                        as pd
import core.HelperTools              as ht

# -----------------------------------------------------------------------------
@ht.timer
def count_lowthresto100(dfr, coltocount):
    """counts cells where similarity is '>70%', '>75%', '>80%', '>85%', '>90%', '>95%'"""
    dframe              = dfr.copy()
    
    # Define the bins and labels
    similarity_bins     = [x/100 for x in [70, 75, 80, 85, 90, 95]]
    similarity_labels   = ['>70%', '>75%', '>80%', '>85%', '>90%', '>95%']

    # Initialize a dictionary to store the counts
    counts              = {}
    
    # Count the occurrences for each range
    for i, _ in enumerate(similarity_bins):
        lower_bound     = similarity_bins[i]
        upper_bound     = 100
        count           = dframe[(dframe[coltocount] >= lower_bound) & (dframe[coltocount] < upper_bound)].shape[0]
        counts[similarity_labels[i]] = count
    
    # Convert to a Series for better display or further use
    ret                 = pd.Series(counts, name=coltocount + "_count")

    return ret

# -----------------------------------------------------------------------------
@ht.timer
def cosine_similarity(vec1, vec2):
    """Defines the cosine similarity function"""
    # Calculate the dot product of the vectors
    dot_product = np.dot(vec1, vec2)

    # Calculate the magnitude (norm) of each vector
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # Avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0

    # Calculate cosine similarity
    return dot_product / (norm_vec1 * norm_vec2)

# -----------------------------------------------------------------------------
@ht.timer
def move_last_columns_to_front(df, n):
    """
    Moves the last n columns of the DataFrame to the front.

    Parameters:
    df (pd.DataFrame): The DataFrame to reorder.
    n (int): The number of columns from the end to move to the front.

    Returns:
    pd.DataFrame: The reordered DataFrame.
    """
    # Ensure n is not greater than the total number of columns
    n = min(n, len(df.columns))
    
    # Get the last n column names
    last_cols = df.columns[-n:]
    
    # Get the remaining column names
    remaining_cols = df.columns[:-n]
    
    # Combine the two lists of column names, putting the last n first
    new_column_order = list(last_cols) + list(remaining_cols)
    
    # Reindex the DataFrame with the new column order
    return df[new_column_order] 

# -----------------------------------------------------------------------------
@ht.timer
def filteroriginal_df(dfr, pdict):
    """Provides preprocessed dataframe (row and column filter, dropped-na) with KGS8 & 2
    
    """
    
    dframe                      = dfr.copy()
    
    # Row-Filter: Nur Berlin ('11')
    dframe.loc[:, 'KGS8']       = dframe.loc[:, 'KGS12'].astype(str).str[:8]    
    dframe.loc[:, 'BL_Code']    = dframe.loc[:, 'KGS12'].astype(str).str[:2]
    usedFilter                  = dframe['BL_Code'] == pdict['bundesland_code']
    dframe2                     = dframe.where(usedFilter).dropna(axis=0, how="all")
    
    # Col-Filter: 
    cFilter                     = [x for x in pdict["scenario"] if x in dframe.columns.values]
    cFilter                     = [pdict["primaryKey"], "KGS12", "KGS8", "BL_Code"] + cFilter
    dframe3                     = dframe2[cFilter]
    
    # Drop nulls
    dframe4                     = dframe3.dropna(axis = 0)
    
    return dframe4

# -----------------------------------------------------------------------------
@ht.timer
def onehot_encode_and_aggregate(df, onehot_col, agg_col):
    """Gets onehoted & summarized households for each cluster
    
    """
    df_encoded = pd.get_dummies(df, columns=[onehot_col])
    df_agg = df_encoded.groupby(agg_col).sum().reset_index()
    return df_agg


