#!/usr/bin/env python
# coding: utf-8

import rdkit
import pandas as pd
import numpy as np
from sys import argv
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Chem import rdFingerprintGenerator
from scipy.spatial.distance import pdist

print(f"rdkit_version: {rdkit.__version__}")

# Read datasets
url_data = "https://drive.google.com/file/d/12wkBXwTI58xVjRmhaOyG_jiiCRFOi3Ov/view?usp=sharing"
url_data='https://drive.google.com/uc?id=' + url_data.split('/')[-2]
lanapdb = pd.read_csv(url_data)
lanapdb.tail(2)
len_lanapdb = len(lanapdb)

coconut = pd.read_csv(argv[1], sep=",") #COCONUT_2024_molecular_descriptors
coconut.tail(2)
len_coconut = len(coconut)

url_data_EnamineSolWat = "https://drive.google.com/file/d/14joMGmdMgnzPRopRWMXdEx4myLAHHGbc/view?usp=drive_link"
url_data_EnamineSolWat ='https://drive.google.com/uc?id=' + url_data_EnamineSolWat.split('/')[-2]
EnamineSolWat = pd.read_csv(url_data_EnamineSolWat)
EnamineSolWat.tail(2)
 
url_data_ChemDiv = "https://drive.google.com/file/d/14QjDAGOqvgPehTEc0290gzCmASlaijXB/view?usp=sharing"
url_data_ChemDiv ='https://drive.google.com/uc?id=' + url_data_ChemDiv.split('/')[-2]
ChemDiv = pd.read_csv(url_data_ChemDiv)
ChemDiv.tail(2)
len_ChemDiv = len(ChemDiv)

url_data_Maybridge = "https://drive.google.com/file/d/18TQoiXtGLY6LS95y5iR3UvTE_CuZyZnM/view?usp=drive_link"
url_data_Maybridge='https://drive.google.com/uc?id=' + url_data_Maybridge.split('/')[-2]
Maybridge = pd.read_csv(url_data_Maybridge)
Maybridge.tail(2)

url_data_LifeChemicals = "https://drive.google.com/file/d/1XEgnLH8ykiuuYv55341RV0ISqh5yoAXa/view?usp=sharing"
url_data_LifeChemicals ='https://drive.google.com/uc?id=' + url_data_LifeChemicals.split('/')[-2]
LifeChemicals = pd.read_csv(url_data_LifeChemicals)
LifeChemicals.tail(2)
len_LifeChemicals = len(LifeChemicals)

url_data_CRAFT = "https://drive.google.com/file/d/1aKUcchg7A705tYIXbWaZj_OiYaLNCQ4k/view?usp=sharing"
url_data_CRAFT ='https://drive.google.com/uc?id=' + url_data_CRAFT.split('/')[-2]
CRAFT = pd.read_csv(url_data_CRAFT)
CRAFT.tail(2)

# SMILES list
lanapdb_smi = list(lanapdb["SMILES_chiral"])
coconut_smi = list(coconut["SMILES_chiral"])
EnamineSolWat_smi = list(EnamineSolWat["SMILES_chiral"])
ChemDiv_smi = list(ChemDiv["SMILES_chiral"])
Maybridge_smi = list(Maybridge["SMILES_chiral"])
LifeChemicals_smi = list(LifeChemicals["SMILES_chiral"])
CRAFT_smi = list(CRAFT["SMILES_chiral"])

#### Functions ####

# Morgan2 and Morgan3
def ECFP (smi, r):
    fps = pd.DataFrame([[int(y) for y in rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=1024).GetFingerprint(Chem.MolFromSmiles(x)).ToBitString()] for x in smi])
    SimMat = 1 - pdist(fps[[x for x in range(1024)]], metric="jaccard") # Similarity Matrix
    #print(SimMat.shape)
    SimMat = round(np.median(SimMat), 3)
    return SimMat

# MACCS keys
def MACCSkeys_fp (smi):
    fps = pd.DataFrame([[int(y) for y in MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(x)).ToBitString()] for x in smi])
    SimMat = 1 - pdist(fps[[x for x in range(167)]], metric="jaccard") # Similarity Matrix
    #print(SimMat.shape)
    SimMat = round(np.median(SimMat), 3)
    return SimMat

"""
    Sample number 
"""
sample_number = 5000
print(f"sample_number: {sample_number}")

"""
    COCONUT 
"""
# COCONUT sample for Morgan2
ecfp4_coconut = []
for i in range(10):
    sample = coconut.sample(sample_number, random_state=i).copy()
    sample_smi = list(sample["SMILES_chiral"])
    fp_coconut = ECFP(sample_smi, 2)
    print(fp_coconut)
    ecfp4_coconut.append(np.median(fp_coconut))
ecfp4_coconut = round(np.mean(ecfp4_coconut), 3)
print(f"ecfp4_coconut: {ecfp4_coconut}")

# COCONUT sample for Morgan3
ecfp6_coconut = []
for i in range(10):
    sample = coconut.sample(sample_number, random_state=i).copy()
    sample_smi = list(sample["SMILES_chiral"])
    fp_coconut = ECFP(sample_smi, 3)
    print(fp_coconut)
    ecfp6_coconut.append(np.median(fp_coconut))
ecfp6_coconut = round(np.mean(ecfp6_coconut), 3)
print(f"ecfp6_coconut: {ecfp6_coconut}")

# COCONUT sample for MACCS keys
maccs_coconut = []
for i in range(10):
    sample = coconut.sample(sample_number, random_state=i).copy()
    sample_smi = list(sample["SMILES_chiral"])
    fp_coconut = MACCSkeys_fp(sample_smi)
    print(fp_coconut)
    maccs_coconut.append(np.median(fp_coconut))
maccs_coconut = round(np.mean(maccs_coconut), 3)
print(f"maccs_coconut: {maccs_coconut}")
coconut=0

"""
    LANaPDB 
"""
# LANaPDB sample for Morgan2
ecfp4_lanapdb = []
for i in range(10):
    sample = lanapdb.sample(sample_number, random_state=i).copy()
    sample_smi = list(sample["SMILES_chiral"])
    fp_lanapdb = ECFP(sample_smi, 2)
    print(fp_lanapdb)
    ecfp4_lanapdb.append(np.median(fp_lanapdb))
ecfp4_lanapdb = round(np.mean(ecfp4_lanapdb), 3)
print(f"ecfp4_lanapdb: {ecfp4_lanapdb}")

# LANaPDB sample for Morgan3
ecfp6_lanapdb = []
for i in range(10):
    sample = lanapdb.sample(sample_number, random_state=i).copy()
    sample_smi = list(sample["SMILES_chiral"])
    fp_lanapdb = ECFP(sample_smi, 3)
    print(fp_lanapdb)
    ecfp6_lanapdb.append(np.median(fp_lanapdb))
ecfp6_lanapdb = round(np.mean(ecfp6_lanapdb), 3)
print(f"ecfp6_lanapdb: {ecfp6_lanapdb}")

# LANaPDB sample for MACCS keys
maccs_lanapdb = []
for i in range(10):
    sample = lanapdb.sample(sample_number, random_state=i).copy()
    sample_smi = list(sample["SMILES_chiral"])
    fp_lanapdb = MACCSkeys_fp(sample_smi)
    print(fp_lanapdb)
    maccs_lanapdb.append(np.median(fp_lanapdb))
maccs_lanapdb = round(np.mean(maccs_lanapdb), 3)
print(f"maccs_lanapdb: {maccs_lanapdb}")
lanapdb=0

"""
    LifeChemicals
"""
# LifeChemicals sample for Morgan2
ecfp4_LifeChemicals = []
for i in range(10):
    sample = LifeChemicals.sample(sample_number, random_state=i).copy()
    sample_smi = list(sample["SMILES_chiral"])
    fp_LifeChemicals = ECFP(sample_smi, 2)
    print(fp_LifeChemicals)
    ecfp4_LifeChemicals.append(np.median(fp_LifeChemicals))
ecfp4_LifeChemicals = round(np.mean(ecfp4_LifeChemicals), 3)
print(f"ecfp4_LifeChemicals: {ecfp4_LifeChemicals}")
# LifeChemicals sample for Morgan3
ecfp6_LifeChemicals = []
for i in range(10):
    sample = LifeChemicals.sample(sample_number, random_state=i).copy()
    sample_smi = list(sample["SMILES_chiral"])
    fp_LifeChemicals = ECFP(sample_smi, 3)
    print(fp_LifeChemicals)
    ecfp6_LifeChemicals.append(np.median(fp_LifeChemicals))
ecfp6_LifeChemicals = round(np.mean(ecfp6_LifeChemicals), 3)
print(f"ecfp6_LifeChemicals: {ecfp6_LifeChemicals}")
# LifeChemicals sample for MACCS keys
maccs_LifeChemicals = []
for i in range(10):
    sample = LifeChemicals.sample(sample_number, random_state=i).copy()
    sample_smi = list(sample["SMILES_chiral"])
    fp_LifeChemicals = MACCSkeys_fp(sample_smi)
    print(fp_LifeChemicals)
    maccs_LifeChemicals.append(np.median(fp_LifeChemicals))
maccs_LifeChemicals = round(np.mean(maccs_LifeChemicals), 3)
print(f"maccs_LifeChemicals: {maccs_LifeChemicals}")
LifeChemicals=0

"""
    ChemDiv
"""
# ChemDiv sample for Morgan2
ecfp4_ChemDiv = []
for i in range(10):
    sample = ChemDiv.sample(sample_number, random_state=i).copy()
    sample_smi = list(sample["SMILES_chiral"])
    fp_ChemDiv = ECFP(sample_smi, 2)
    print(fp_ChemDiv)
    ecfp4_ChemDiv.append(np.median(fp_ChemDiv))
ecfp4_ChemDiv = round(np.mean(ecfp4_ChemDiv), 3)
print(f"ecfp4_ChemDiv: {ecfp4_ChemDiv}")

# ChemDiv sample for Morgan3
ecfp6_ChemDiv = []
for i in range(10):
    sample = ChemDiv.sample(sample_number, random_state=i).copy()
    sample_smi = list(sample["SMILES_chiral"])
    fp_ChemDiv = ECFP(sample_smi, 3)
    print(fp_ChemDiv)
    ecfp6_ChemDiv.append(np.median(fp_ChemDiv))
ecfp6_ChemDiv = round(np.mean(ecfp6_ChemDiv), 3)
print(f"ecfp6_ChemDiv: {ecfp6_ChemDiv}")

# ChemDiv sample for MACCS keys
maccs_ChemDiv = []
for i in range(10):
    sample = ChemDiv.sample(sample_number, random_state=i).copy()
    sample_smi = list(sample["SMILES_chiral"])
    fp_ChemDiv = MACCSkeys_fp(sample_smi)
    print(fp_ChemDiv)
    maccs_ChemDiv.append(np.median(fp_ChemDiv))
maccs_ChemDiv = round(np.mean(maccs_ChemDiv), 3)
print(f"maccs_ChemDiv: {maccs_ChemDiv}")
ChemDiv=0

# Median similarity Morgan2
ecfp_2 = [ecfp4_coconut, ecfp4_lanapdb, ECFP(CRAFT_smi, 2),
          ECFP(EnamineSolWat_smi, 2), ecfp4_ChemDiv, ECFP(Maybridge_smi, 2),
          ecfp4_LifeChemicals]
print(ecfp_2)

# Median similarity Morgan3
ecfp_3 = [ecfp6_coconut, ecfp6_lanapdb, ECFP(CRAFT_smi, 3),
          ECFP(EnamineSolWat_smi, 3), ecfp6_ChemDiv, ECFP(Maybridge_smi, 3), 
          ecfp6_LifeChemicals]
print(ecfp_3)

# Median similarity MACCS keys
MACCS_keys = [maccs_coconut, maccs_lanapdb, MACCSkeys_fp(CRAFT_smi),
              MACCSkeys_fp(EnamineSolWat_smi), maccs_ChemDiv, MACCSkeys_fp(Maybridge_smi), 
              maccs_LifeChemicals]
print(MACCS_keys)

# Databases 
Collection = ["COCONUT", "LANaPDB", "CRAFT", "EnamineSolWat", 
              "ChemDiv", "Maybridge", "LifeChemicals"]

# Number of fragments          
Fragments = [len_coconut, len_lanapdb, len(CRAFT_smi),
             len(EnamineSolWat_smi), len_ChemDiv, len(Maybridge_smi), 
             len_LifeChemicals]

# Create Dataframe
arr = np.array([Collection, Fragments, MACCS_keys, ecfp_2, ecfp_3])
arr = np.transpose(arr)
arr
fingerprints = pd.DataFrame(arr, columns = ["Collection", "Fragments", "MACCS keys", "Morgan2", "Morgan3"])
print(fingerprints)

# Save
fingerprints.to_csv("Fingerprints_median_similarty_Fragments_5000sample.csv", sep=",", index=False)





