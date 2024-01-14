p                           = dict()
p['picklefolder']           = 'pickles'
# -----------------------------------

# p['primaryKey']             = 'zelle'
p['description']            = 'description'
p['targetcol']              = 'Cluster'
p['targetcol_pred']         = 'Cluster_pred'

p['geocode']                = 'Geocode'
# pdict["geocode"]

p['table_meta']             = 'meta.txt'

# param	                    param_value
p['anzFeatures']            = 5
p['bfac']                   = 20
p['clnum']                  = 4
p['thr']                    = 0.2

p["file_datamart"]          = "datamart_berlin.csv"
p["file_geodata"]           = "geodata_berlin.csv"
p["file_sozdem"]            = "sozdem_berlin.csv"



# Berlin
p['bundesland_code']        = '11' 

p['scenario'] = [
    'bvto',
    'geb_gro',
    'geb_lage',
    'MTPHOG',
    'KINDERG',
    'MTHHDG',
    'bvbff',
    'bvhfh',
    'bvrs',
    'MSALEG',
    'MSFUBG',
    'MSOKAG',
    'MSREIG',
    'MSSOMG',
    'MTPOZG',
    'MTWIZG',
    'oekostromg',
    # 'MTKDIG',
    'MSHKPG',
    'MSHKQG',
    ]


p['label_dict'] = {
    "bvto":         "Population in the house",
    "geb_gro":      "House size: number of flats",
    "geb_lage":     "Status of Location",
    "MTPHOG":       "Tendency of having a photovoltaic system",
    "KINDERG":      "Tendency of having children",
    "MTHHDG":       "Population density (households per qkm)",
    "bvbff":        "Population ratio: college degree",
    "bvhfh":        "Population ratio: university degree",
    "bvrs":         "Population ratio: school diploma",
    "MSALEG":       "Interest: alternative energy",
    "MSFUBG":       "Interest: football",
    "MSOKAG":       "Interest: online shopping",
    "MSREIG":       "Interest: travel",
    "MSSOMG":       "Interest: social media",
    "MTPOZG":       "Interest: politics",
    "MTWIZG":       "Interest: economics",
    "oekostromg":   "Interest: green energy",
    # "MTKDIG":       "Car density",
    "MSHKPG":       "Main purchase criterion: low price",
    "MSHKQG":       "Main purchase criterion: brand/quality"
    }



# -----------------------------------
pdict = p.copy()

