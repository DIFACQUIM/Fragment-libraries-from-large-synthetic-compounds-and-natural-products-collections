import rdkit
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Chem import rdFingerprintGenerator
from scipy.spatial.distance import pdist

#from google.colab import drive
#drive.mount('/content/drive')

print(f"rdkit_version: {rdkit.__version__}")

""" FRAGMENTS RO3 """
# LANaPDB RO3
url_data = "https://raw.githubusercontent.com/DIFACQUIM/Fragment-libraries-from-large-synthetic-compounds-and-natural-products-collections/refs/heads/main/DATA_SET/DATA_FRAGMENTS_RO3/LANaPDB_RO3_Moleculardescriptors.csv"
lanapdb_RO3 = pd.read_csv(url_data)
lanapdb_RO3 = lanapdb_RO3[["ID", "SMILES_chiral"]]
lanapdb_RO3["Database"] = "LANaPDB"
print(lanapdb_RO3.tail(2))

# COCONUT RO3
url_data = "https://raw.githubusercontent.com/DIFACQUIM/Fragment-libraries-from-large-synthetic-compounds-and-natural-products-collections/refs/heads/main/DATA_SET/DATA_FRAGMENTS_RO3/COCONUT_RO3_Moleculardescriptors.csv"
coconut_RO3 = pd.read_csv(url_data)
coconut_RO3 = coconut_RO3[["ID", "SMILES_chiral","Database"]]
print(coconut_RO3.tail(2))

# Enamine RO3
url_data = "https://raw.githubusercontent.com/DIFACQUIM/Fragment-libraries-from-large-synthetic-compounds-and-natural-products-collections/refs/heads/main/DATA_SET/DATA_FRAGMENTS_RO3/Enamine_RO3_Moleculardescriptors.csv"
Enamine_RO3 = pd.read_csv(url_data)
Enamine_RO3 = Enamine_RO3[["ID", "SMILES_chiral"]]
Enamine_RO3["Database"] = "Enamine"
print(Enamine_RO3.tail(2))

# ChemDiv RO3
url_data = "https://raw.githubusercontent.com/DIFACQUIM/Fragment-libraries-from-large-synthetic-compounds-and-natural-products-collections/refs/heads/main/DATA_SET/DATA_FRAGMENTS_RO3/ChemDiv_RO3_Moleculardescriptors.csv"
ChemDiv_RO3 = pd.read_csv(url_data)
ChemDiv_RO3 = ChemDiv_RO3[["ID", "SMILES_chiral","Database"]]
print(ChemDiv_RO3.tail(2))

# Maybridge RO3
url_data = "https://raw.githubusercontent.com/DIFACQUIM/Fragment-libraries-from-large-synthetic-compounds-and-natural-products-collections/refs/heads/main/DATA_SET/DATA_FRAGMENTS_RO3/Maybridge_RO3_Moleculardescriptors.csv"
Maybridge_RO3 = pd.read_csv(url_data)
Maybridge_RO3 = Maybridge_RO3[["ID", "SMILES_chiral", "Database"]]
print(Maybridge_RO3.tail(2))

# Life Chemicals RO3
url_data = "https://raw.githubusercontent.com/DIFACQUIM/Fragment-libraries-from-large-synthetic-compounds-and-natural-products-collections/refs/heads/main/DATA_SET/DATA_FRAGMENTS_RO3/LifeChemicals_RO3_Moleculardescriptors.csv"
LifeChemicals_RO3 = pd.read_csv(url_data)
LifeChemicals_RO3 = LifeChemicals_RO3[["ID", "SMILES_chiral"]]
LifeChemicals_RO3["Database"] = "Life Chemicals"
print(LifeChemicals_RO3.tail(2))

# CRAFT RO3
url_data = "https://raw.githubusercontent.com/DIFACQUIM/Fragment-libraries-from-large-synthetic-compounds-and-natural-products-collections/refs/heads/main/DATA_SET/DATA_FRAGMENTS_RO3/CRAFT_RO3_Moleculardescriptors.csv"
CRAFT_RO3 = pd.read_csv(url_data)
CRAFT_RO3 = CRAFT_RO3[["ID", "SMILES_chiral", "Database"]]
print(CRAFT_RO3.tail(2))

"""## Commulative distribution function

#### Functions
"""
def ECFP (smi, r):
    fps = pd.DataFrame([[int(y) for y in rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=1024).GetFingerprint(Chem.MolFromSmiles(x)).ToBitString()] for x in smi])
    SimMat = 1 - pdist(fps[[x for x in range(1024)]], metric="jaccard") # Similarity Matrix
    return SimMat

def MACCSkeys_fp (smi):
    fps = pd.DataFrame([[int(y) for y in MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(x)).ToBitString()] for x in smi])
    SimMat = 1 - pdist(fps[[x for x in range(167)]], metric="jaccard") # Similarity Matrix
    return SimMat

# SMILES FROM FRAGMENTS RO3
EnamineSolWat_RO3 = list(EnamineSolWat_RO3["SMILES_chiral"])
lanapdb_RO3 = list(lanapdb_RO3["SMILES_chiral"])
coconut_RO3 = list(coconut_RO3["SMILES_chiral"])
ChemDiv_RO3 = list(ChemDiv_RO3["SMILES_chiral"])
Maybridge_RO3 = list(Maybridge_RO3["SMILES_chiral"])
LifeChemicals_RO3 = list(LifeChemicals_RO3["SMILES_chiral"])
CRAFT_RO3 = list(CRAFT_RO3["SMILES_chiral"])

"""# ECFP4"""

# 1. Sort data
SimMatECFP4_EnamineSolWat_RO3_sorted = np.sort(ECFP(EnamineSolWat_RO3, 2))
SimMatECFP4_lanapdb_RO3_sorted = np.sort(ECFP(lanapdb_RO3, 2))
SimMatECFP4_coconut_RO3_sorted = np.sort(ECFP(coconut_RO3, 2))
SimMatECFP4_ChemDiv_RO3_sorted = np.sort(ECFP(ChemDiv_RO3, 2))
SimMatECFP4_Maybridge_RO3_sorted = np.sort(ECFP(Maybridge_RO3, 2))
SimMatECFP4_LifeChemicals_RO3_sorted = np.sort(ECFP(LifeChemicals_RO3, 2))
SimMatECFP4_CRAFT_RO3_sorted = np.sort(ECFP(CRAFT_RO3, 2))

# Calculate the proportional values of samples for ECFP4 y ECFP6
proportionECFP_EnamineSolWat_RO3 = 1. * np.arange(len(SimMatECFP4_EnamineSolWat_RO3_sorted)) / (len(SimMatECFP4_EnamineSolWat_RO3_sorted) - 1)
print(len(proportionECFP_EnamineSolWat_RO3))

proportionECFP_lanapdb_RO3 = 1. * np.arange(len(SimMatECFP4_lanapdb_RO3_sorted)) / (len(SimMatECFP4_lanapdb_RO3_sorted) - 1)
print(len(proportionECFP_lanapdb_RO3))

proportionECFP_coconut_RO3 = 1. * np.arange(len(SimMatECFP4_coconut_RO3_sorted)) / (len(SimMatECFP4_coconut_RO3_sorted) - 1)
print(len(proportionECFP_coconut_RO3))

proportionECFP_ChemDiv_RO3 = 1. * np.arange(len(SimMatECFP4_ChemDiv_RO3_sorted)) / (len(SimMatECFP4_ChemDiv_RO3_sorted) - 1)
print(len(proportionECFP_ChemDiv_RO3))

proportionECFP_Maybridge_RO3 = 1. * np.arange(len(SimMatECFP4_Maybridge_RO3_sorted)) / (len(SimMatECFP4_Maybridge_RO3_sorted) - 1)
print(len(proportionECFP_Maybridge_RO3))

proportionECFP_LifeChemicals_RO3 = 1. * np.arange(len(SimMatECFP4_LifeChemicals_RO3_sorted)) / (len(SimMatECFP4_LifeChemicals_RO3_sorted) - 1)
print(len(proportionECFP_LifeChemicals_RO3))

proportionECFP_CRAFT_RO3 = 1. * np.arange(len(SimMatECFP4_CRAFT_RO3_sorted)) / (len(SimMatECFP4_CRAFT_RO3_sorted) - 1)
print(len(proportionECFP_CRAFT_RO3))

lanapdb_RO3=0
coconut_RO3=0
ChemDiv_RO3=0
Maybridge_RO3=0
LifeChemicals_RO3=0
CRAFT_RO3=0

# plot the sorted data:
import matplotlib.pyplot as plt
fig = plt.figure()
fig.set_size_inches(7,7)

x1 = fig.add_subplot(1,1,1)
x1.plot(SimMatECFP4_EnamineSolWat_RO3_sorted, proportionECFP_EnamineSolWat_RO3, label="EnamineColl_RO3", c="black")
x1.plot(SimMatECFP4_lanapdb_RO3_sorted, proportionECFP_lanapdb_RO3, label="LANaPDB_RO3", c="green")
x1.plot(SimMatECFP4_coconut_RO3_sorted, proportionECFP_coconut_RO3, label="COCONUT_RO3", c="cyan")
SimMatECFP4_coconut_RO3_sorted=0
proportionECFP_coconut_RO3=0
x1.plot(SimMatECFP4_ChemDiv_RO3_sorted, proportionECFP_ChemDiv_RO3, label="ChemDiv_RO3", c="blue")
x1.plot(SimMatECFP4_Maybridge_RO3_sorted, proportionECFP_Maybridge_RO3, label="Maybridge_RO3", c="purple")
x1.plot(SimMatECFP4_LifeChemicals_RO3_sorted, proportionECFP_LifeChemicals_RO3, label="LifeChemicals_RO3", c="red")
x1.plot(SimMatECFP4_CRAFT_RO3_sorted, proportionECFP_CRAFT_RO3, label="CRAFT_RO3", c="yellow")
x1.grid(alpha=0.5)
x1.set_xlabel('$Similarity$', fontsize=14)
x1.set_ylabel('$Fraction$', fontsize=14)
x1.set_title('Morgan 2', fontsize=18)
#fig1.legend()
fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.001), fancybox=True, shadow=True)

# Avoid overlapping the legend with the axes.
fig.subplots_adjust(left=0.15, bottom=0.15)

fig.savefig("CDF_fragments_morgan2.png", dpi=400)
