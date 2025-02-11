import rdkit
import pandas as pd
import numpy as np
from sys import argv
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Chem import rdFingerprintGenerator
from scipy.spatial.distance import pdist

#from google.colab import drive
#drive.mount('/content/drive')

print(f"rdkit_version: {rdkit.__version__}")

"""## Diagramas de Venn CRAFT vs otras bases de datos de fragmentos curados que siguen la regla de 3"""

# ALL FRAGMENTS DATABASES
# LANaPDB
url_data = "https://raw.githubusercontent.com/DIFACQUIM/Fragment-libraries-from-large-synthetic-compounds-and-natural-products-collections/refs/heads/main/DATA_SET/DATA_FRAGMENTS/LANaPDB_Moleculardescriptors.csv"
lanapdb = pd.read_csv(url_data)
lanapdb = lanapdb[["ID", "SMILES_chiral"]]
lanapdb["Database"] = "LANaPDB"
print(lanapdb.tail(2))

# COCONUT
coconut = pd.read_csv(argv[1], sep=",") 
print(coconut.tail(2))

# Enamine
url_data = "https://raw.githubusercontent.com/DIFACQUIM/Fragment-libraries-from-large-synthetic-compounds-and-natural-products-collections/refs/heads/main/DATA_SET/DATA_FRAGMENTS/Enamine_Moleculardescriptors.csv"
EnamineSolWat = pd.read_csv(url_data)
EnamineSolWat = Enamine[["ID", "SMILES_chiral"]]
EnamineSolWat["Database"] = "Enamine"
print(EnamineSolWat.tail(2))

# ChemDiv 
url_data = "https://raw.githubusercontent.com/DIFACQUIM/Fragment-libraries-from-large-synthetic-compounds-and-natural-products-collections/refs/heads/main/DATA_SET/DATA_FRAGMENTS/ChemDiv_Moleculardescriptors.csv"
ChemDiv = pd.read_csv(url_data)
ChemDiv = ChemDiv[["ID", "SMILES_chiral","Database"]]
print(ChemDiv.tail(2))

# Maybridge 
url_data = "https://raw.githubusercontent.com/DIFACQUIM/Fragment-libraries-from-large-synthetic-compounds-and-natural-products-collections/refs/heads/main/DATA_SET/DATA_FRAGMENTS/Maybridge_Moleculardescriptors.csv"
Maybridge = pd.read_csv(url_data)
Maybridge = Maybridge[["ID", "SMILES_chiral", "Database"]]
print(Maybridge.tail(2))

# Life Chemicals 
url_data = "https://raw.githubusercontent.com/DIFACQUIM/Fragment-libraries-from-large-synthetic-compounds-and-natural-products-collections/refs/heads/main/DATA_SET/DATA_FRAGMENTS/LifeChemicals_Moleculardescriptors.csv"
LifeChemicals = pd.read_csv(url_data)
LifeChemicals = LifeChemicals[["ID", "SMILES_chiral"]]
LifeChemicals["Database"] = "Life Chemicals"
print(LifeChemicals.tail(2))

# CRAFT 
url_data = "https://raw.githubusercontent.com/DIFACQUIM/Fragment-libraries-from-large-synthetic-compounds-and-natural-products-collections/refs/heads/main/DATA_SET/DATA_FRAGMENTS/CRAFT_Moleculardescriptors.csv"
CRAFT = pd.read_csv(url_data)
CRAFT = CRAFT[["ID", "SMILES_chiral", "Database"]]
print(CRAFT.tail(2))

"""## Commulative distribution function

#### Functions
"""

from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Chem import rdFingerprintGenerator
from scipy.spatial.distance import pdist

def ECFP (smi, r):
    fps = pd.DataFrame([[int(y) for y in rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=1024).GetFingerprint(Chem.MolFromSmiles(x)).ToBitString()] for x in smi])
    SimMat = 1 - pdist(fps[[x for x in range(1024)]], metric="jaccard") # Similarity Matrix
    return SimMat

def MACCSkeys_fp (smi):
    fps = pd.DataFrame([[int(y) for y in MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(x)).ToBitString()] for x in smi])
    SimMat = 1 - pdist(fps[[x for x in range(167)]], metric="jaccard") # Similarity Matrix
    return SimMat

# SMILES FROM ALL FRAGMENTS
EnamineSolWat = list(EnamineSolWat["SMILES_chiral"])
Maybridge = list(Maybridge["SMILES_chiral"])
CRAFT = list(CRAFT["SMILES_chiral"])

number_sample=5000

# COCONUT sample for Morgan2
SimMatECFP6_coconut_sorted = []
for i in range(10):
    sample = coconut.sample(number_sample, random_state=i).copy()
    sample= np.sort(ECFP(sample["SMILES_chiral"], 3))
    SimMatECFP6_coconut_sorted.append(sample)
print(f"ECFP6_coconut: {len(SimMatECFP6_coconut_sorted)}")
SimMatECFP6_coconut_sorted1 = SimMatECFP6_coconut_sorted[0]
print(f"SimMatECFP6_coconut_RO3_sorted1: {len(SimMatECFP6_coconut_sorted1)}")
coconut=0

# LifeChemicals sample for Morgan2
SimMatECFP6_LifeChemicals_sorted = []
for i in range(10):
    sample = LifeChemicals.sample(number_sample, random_state=i).copy()
    sample= np.sort(ECFP(sample["SMILES_chiral"], 3))
    SimMatECFP6_LifeChemicals_sorted.append(sample)

print(f"ECFP6_LifeChemicals: {len(SimMatECFP6_LifeChemicals_sorted)}")
SimMatECFP6_LifeChemicals_sorted1 = SimMatECFP6_LifeChemicals_sorted[0]
print(f"SimMatECFP6_LifeChemicals_sorted1: {len(SimMatECFP6_LifeChemicals_sorted1)}")
LifeChemicals=0

# ChemDiv sample for Morgan2
SimMatECFP6_ChemDiv_sorted = []
for i in range(10):
    sample = ChemDiv.sample(number_sample, random_state=i).copy()
    sample= np.sort(ECFP(sample["SMILES_chiral"], 3))
    SimMatECFP6_ChemDiv_sorted.append(sample)
print(f"ECFP6_ChemDiv: {len(SimMatECFP6_ChemDiv_sorted)}")
SimMatECFP6_ChemDiv_sorted1 = SimMatECFP6_ChemDiv_sorted[0]
print(f"SimMatECFP6_ChemDiv_sorted1: {len(SimMatECFP6_ChemDiv_sorted1)}")
ChemDiv=0

# LANaPDB sample for Morgan2
SimMatECFP6_lanapdb_sorted = []
for i in range(10):
    sample = lanapdb.sample(number_sample, random_state=i).copy()
    sample= np.sort(ECFP(sample["SMILES_chiral"], 3))
    SimMatECFP6_lanapdb_sorted.append(sample)
print(f"ECFP6_Maybridge: {len(SimMatECFP6_lanapdb_sorted)}")
SimMatECFP6_lanapdb_sorted1 = SimMatECFP6_lanapdb_sorted[0]
print(f"SimMatECFP6_Maybridge_sorted1: {len(SimMatECFP6_lanapdb_sorted1)}")
lanapdb=0

"""### Morgan3"""

# 1. Sort data
SimMatECFP6_EnamineSolWat_sorted = np.sort(ECFP(EnamineSolWat, 3))
print(SimMatECFP6_EnamineSolWat_sorted)
SimMatECFP6_Maybridge_sorted = np.sort(ECFP(Maybridge, 3))
SimMatECFP6_CRAFT_sorted = np.sort(ECFP(CRAFT, 3))


# Calculate the proportional values of samples for ECFP4 y ECFP6
proportionECFP6_EnamineSolWat = 1. * np.arange(len(SimMatECFP6_EnamineSolWat_sorted)) / (len(SimMatECFP6_EnamineSolWat_sorted) - 1)
print(len(proportionECFP6_EnamineSolWat))

proportionECFP6_lanapdb = 1. * np.arange(len(SimMatECFP6_lanapdb_sorted1)) / (len(SimMatECFP6_lanapdb_sorted1) - 1)
print(len(proportionECFP6_lanapdb))

proportionECFP6_coconut = 1. * np.arange(len(SimMatECFP6_coconut_sorted1)) / (len(SimMatECFP6_coconut_sorted1) - 1)
print(len(proportionECFP6_coconut))

proportionECFP6_ChemDiv = 1. * np.arange(len(SimMatECFP6_ChemDiv_sorted1)) / (len(SimMatECFP6_ChemDiv_sorted1) - 1)
print(len(proportionECFP6_ChemDiv))

proportionECFP6_Maybridge = 1. * np.arange(len(SimMatECFP6_Maybridge_sorted)) / (len(SimMatECFP6_Maybridge_sorted) - 1)
print(len(proportionECFP6_Maybridge))

proportionECFP6_LifeChemicals = 1. * np.arange(len(SimMatECFP6_LifeChemicals_sorted1)) / (len(SimMatECFP6_LifeChemicals_sorted1) - 1)
print(len(proportionECFP6_LifeChemicals))

proportionECFP6_CRAFT = 1. * np.arange(len(SimMatECFP6_CRAFT_sorted)) / (len(SimMatECFP6_CRAFT_sorted) - 1)
print(len(proportionECFP6_CRAFT))

lanapdb=0
coconut=0
ChemDiv=0
Maybridge=0
LifeChemicals=0
CRAFT=0

# plot the sorted data:
import matplotlib.pyplot as plt
fig = plt.figure()
fig.set_size_inches(7,7)

x1 = fig.add_subplot(1,1,1)
x1.plot(SimMatECFP6_EnamineSolWat_sorted, proportionECFP6_EnamineSolWat, label="Enamine", c="black")
x1.plot(SimMatECFP6_lanapdb_sorted[0], proportionECFP6_lanapdb, label="LANaPDB", c="green")
x1.plot(SimMatECFP6_lanapdb_sorted[1], proportionECFP6_lanapdb, c="green") # label="LANaPDB"
x1.plot(SimMatECFP6_lanapdb_sorted[2], proportionECFP6_lanapdb, c="green") # label="LANaPDB"
x1.plot(SimMatECFP6_lanapdb_sorted[3], proportionECFP6_lanapdb, c="green") # label="LANaPDB"
x1.plot(SimMatECFP6_lanapdb_sorted[4], proportionECFP6_lanapdb, c="green") # label="LANaPDB"
x1.plot(SimMatECFP6_lanapdb_sorted[5], proportionECFP6_lanapdb, c="green") # label="LANaPDB"
x1.plot(SimMatECFP6_lanapdb_sorted[6], proportionECFP6_lanapdb, c="green") # label="LANaPDB"
x1.plot(SimMatECFP6_lanapdb_sorted[7], proportionECFP6_lanapdb, c="green") # label="LANaPDB"
x1.plot(SimMatECFP6_lanapdb_sorted[8], proportionECFP6_lanapdb, c="green") # label="LANaPDB"
x1.plot(SimMatECFP6_lanapdb_sorted[9], proportionECFP6_lanapdb, c="green") # label="LANaPDBcl"

x1.plot(SimMatECFP6_coconut_sorted[0], proportionECFP6_coconut, label="COCONUT", c="cyan")
x1.plot(SimMatECFP6_coconut_sorted[1], proportionECFP6_coconut,  c="cyan") #label="COCONUT"
x1.plot(SimMatECFP6_coconut_sorted[2], proportionECFP6_coconut, c="cyan") #label="COCONUT"
x1.plot(SimMatECFP6_coconut_sorted[3], proportionECFP6_coconut, c="cyan") #label="COCONUT"
x1.plot(SimMatECFP6_coconut_sorted[4], proportionECFP6_coconut, c="cyan") #label="COCONUT"
x1.plot(SimMatECFP6_coconut_sorted[5], proportionECFP6_coconut, c="cyan") #label="COCONUT"
x1.plot(SimMatECFP6_coconut_sorted[6], proportionECFP6_coconut, c="cyan") #label="COCONUT"
x1.plot(SimMatECFP6_coconut_sorted[7], proportionECFP6_coconut, c="cyan") #label="COCONUT"
x1.plot(SimMatECFP6_coconut_sorted[8], proportionECFP6_coconut, c="cyan") #label="COCONUT"
x1.plot(SimMatECFP6_coconut_sorted[9], proportionECFP6_coconut, c="cyan") #label="COCONUT"

SimMatECFP4_coconut_sorted=0
proportionECFP4_coconut=0
x1.plot(SimMatECFP6_ChemDiv_sorted[0], proportionECFP6_ChemDiv, label="ChemDiv", c="blue")
x1.plot(SimMatECFP6_ChemDiv_sorted[1], proportionECFP6_ChemDiv, c="blue") #label="ChemDiv"
x1.plot(SimMatECFP6_ChemDiv_sorted[2], proportionECFP6_ChemDiv, c="blue") #label="ChemDiv"
x1.plot(SimMatECFP6_ChemDiv_sorted[3], proportionECFP6_ChemDiv, c="blue") #label="ChemDiv"
x1.plot(SimMatECFP6_ChemDiv_sorted[4], proportionECFP6_ChemDiv, c="blue") #label="ChemDiv"
x1.plot(SimMatECFP6_ChemDiv_sorted[5], proportionECFP6_ChemDiv, c="blue") #label="ChemDiv"
x1.plot(SimMatECFP6_ChemDiv_sorted[6], proportionECFP6_ChemDiv, c="blue") #label="ChemDiv"
x1.plot(SimMatECFP6_ChemDiv_sorted[7], proportionECFP6_ChemDiv, c="blue") #label="ChemDiv"
x1.plot(SimMatECFP6_ChemDiv_sorted[8], proportionECFP6_ChemDiv, c="blue") #label="ChemDiv"
x1.plot(SimMatECFP6_ChemDiv_sorted[9], proportionECFP6_ChemDiv, c="blue") #label="ChemDiv"

x1.plot(SimMatECFP6_Maybridge_sorted, proportionECFP6_Maybridge, label="Maybridge", c="purple")

x1.plot(SimMatECFP6_LifeChemicals_sorted[0], proportionECFP6_LifeChemicals, label="LifeChemicals", c="red")
x1.plot(SimMatECFP6_LifeChemicals_sorted[1], proportionECFP6_LifeChemicals, c="red") #label="LifeChemicals"
x1.plot(SimMatECFP6_LifeChemicals_sorted[2], proportionECFP6_LifeChemicals, c="red") #label="LifeChemicals"
x1.plot(SimMatECFP6_LifeChemicals_sorted[3], proportionECFP6_LifeChemicals, c="red") #label="LifeChemicals"
x1.plot(SimMatECFP6_LifeChemicals_sorted[4], proportionECFP6_LifeChemicals, c="red") #label="LifeChemicals"
x1.plot(SimMatECFP6_LifeChemicals_sorted[5], proportionECFP6_LifeChemicals, c="red") #label="LifeChemicals"
x1.plot(SimMatECFP6_LifeChemicals_sorted[6], proportionECFP6_LifeChemicals, c="red") #label="LifeChemicals"
x1.plot(SimMatECFP6_LifeChemicals_sorted[7], proportionECFP6_LifeChemicals, c="red") #label="LifeChemicals"
x1.plot(SimMatECFP6_LifeChemicals_sorted[8], proportionECFP6_LifeChemicals, c="red") #label="LifeChemicals"
x1.plot(SimMatECFP6_LifeChemicals_sorted[9], proportionECFP6_LifeChemicals, c="red") #label="LifeChemicals"

x1.plot(SimMatECFP6_CRAFT_sorted, proportionECFP6_CRAFT, label="CRAFT", c="orange")

x1.grid(alpha=0.5)
x1.set_xlabel('$Similarity$', fontsize=14)
x1.set_ylabel('$Fraction$', fontsize=14)
x1.set_title('Morgan 2', fontsize=18)
#fig1.legend()
fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.001), fancybox=True, shadow=True)
# Avoid overlapping the legend with the axes.
fig.subplots_adjust(left=0.15, bottom=0.15)

fig.savefig("CDF_fragments_morgan3_5000_sample.png", dpi=400)
