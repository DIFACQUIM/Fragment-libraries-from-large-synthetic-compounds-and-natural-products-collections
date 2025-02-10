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

# ALL FRAGMENTS DATABASES
url_data = "https://drive.google.com/file/d/12wkBXwTI58xVjRmhaOyG_jiiCRFOi3Ov/view?usp=sharing"
url_data='https://drive.google.com/uc?id=' + url_data.split('/')[-2]
lanapdb = pd.read_csv(url_data)
print(lanapdb.tail(2))

coconut = pd.read_csv(argv[1], sep=",") 
print(coconut.tail(2))

url_data_EnamineSolWat = "https://drive.google.com/file/d/14joMGmdMgnzPRopRWMXdEx4myLAHHGbc/view?usp=drive_link"
url_data_EnamineSolWat ='https://drive.google.com/uc?id=' + url_data_EnamineSolWat.split('/')[-2]
EnamineSolWat = pd.read_csv(url_data_EnamineSolWat)
print(EnamineSolWat.tail(2))
 
url_data_ChemDiv = "https://drive.google.com/file/d/14QjDAGOqvgPehTEc0290gzCmASlaijXB/view?usp=sharing"
url_data_ChemDiv ='https://drive.google.com/uc?id=' + url_data_ChemDiv.split('/')[-2]
ChemDiv = pd.read_csv(url_data_ChemDiv)
print(ChemDiv.tail(2))

url_data_Maybridge = "https://drive.google.com/file/d/18TQoiXtGLY6LS95y5iR3UvTE_CuZyZnM/view?usp=drive_link"
url_data_Maybridge='https://drive.google.com/uc?id=' + url_data_Maybridge.split('/')[-2]
Maybridge = pd.read_csv(url_data_Maybridge)
print(Maybridge.tail(2))

url_data_LifeChemicals = "https://drive.google.com/file/d/1XEgnLH8ykiuuYv55341RV0ISqh5yoAXa/view?usp=sharing"
url_data_LifeChemicals ='https://drive.google.com/uc?id=' + url_data_LifeChemicals.split('/')[-2]
LifeChemicals = pd.read_csv(url_data_LifeChemicals)
print(LifeChemicals.tail(2))

url_data_CRAFT = "https://drive.google.com/file/d/1aKUcchg7A705tYIXbWaZj_OiYaLNCQ4k/view?usp=sharing"
url_data_CRAFT ='https://drive.google.com/uc?id=' + url_data_CRAFT.split('/')[-2]
CRAFT = pd.read_csv(url_data_CRAFT)
print(CRAFT.tail(2))

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

# SMILES FROM ALL FRAGMENTS
EnamineSolWat = list(EnamineSolWat["SMILES_chiral"])
Maybridge = list(Maybridge["SMILES_chiral"])
CRAFT = list(CRAFT["SMILES_chiral"])

number_sample=5000 #SAMPLE

# COCONUT sample for MACCS keys
SimMatMACCS_coconut_sorted = []
for i in range(10):
    sample = coconut.sample(number_sample, random_state=i).copy()
    sample= np.sort(MACCSkeys_fp(sample["SMILES_chiral"]))
    SimMatMACCS_coconut_sorted.append(sample)
print(f"maccs_coconut: {len(SimMatMACCS_coconut_sorted)}")
SimMatMACCS_coconut_sorted1 = SimMatMACCS_coconut_sorted[0]
print(f"SimMatMACCS_coconut_RO3_sorted1: {len(SimMatMACCS_coconut_sorted1)}")
coconut=0

# LifeChemicals sample for MACCS keys
SimMatMACCS_LifeChemicals_sorted = []
for i in range(10):
    sample = LifeChemicals.sample(number_sample, random_state=i).copy()
    sample= np.sort(MACCSkeys_fp(sample["SMILES_chiral"]))
    SimMatMACCS_LifeChemicals_sorted.append(sample)

print(f"maccs_LifeChemicals: {len(SimMatMACCS_LifeChemicals_sorted)}")
SimMatMACCS_LifeChemicals_sorted1 = SimMatMACCS_LifeChemicals_sorted[0]
print(f"SimMatMACCS_LifeChemicals_sorted1: {len(SimMatMACCS_LifeChemicals_sorted1)}")
LifeChemicals=0

# ChemDiv sample for MACCS keys
SimMatMACCS_ChemDiv_sorted = []
for i in range(10):
    sample = ChemDiv.sample(number_sample, random_state=i).copy()
    sample= np.sort(MACCSkeys_fp(sample["SMILES_chiral"]))
    SimMatMACCS_ChemDiv_sorted.append(sample)
print(f"maccs_ChemDiv: {len(SimMatMACCS_ChemDiv_sorted)}")
SimMatMACCS_ChemDiv_sorted1 = SimMatMACCS_ChemDiv_sorted[0]
print(f"SimMatMACCS_ChemDiv_sorted1: {len(SimMatMACCS_ChemDiv_sorted1)}")
ChemDiv=0

# LANaPDB sample for MACCS keys
SimMatMACCS_lanapdb_sorted = []
for i in range(10):
    sample = lanapdb.sample(number_sample, random_state=i).copy()
    sample= np.sort(MACCSkeys_fp(sample["SMILES_chiral"]))
    SimMatMACCS_lanapdb_sorted.append(sample)
print(f"maccs_Maybridge: {len(SimMatMACCS_lanapdb_sorted)}")
SimMatMACCS_lanapdb_sorted1 = SimMatMACCS_lanapdb_sorted[0]
print(f"SimMatMACCS_Maybridge_sorted1: {len(SimMatMACCS_lanapdb_sorted1)}")
lanapdb=0

"""### MACCS keys"""

# 1. Sort data
SimMatMACCS_EnamineSolWat_sorted = np.sort(MACCSkeys_fp(EnamineSolWat))
SimMatMACCS_Maybridge_sorted = np.sort(MACCSkeys_fp(Maybridge))
SimMatMACCS_CRAFT_sorted = np.sort(MACCSkeys_fp(CRAFT))

# Calculate the proportional values of samples for MACCS
proportionMACCS_EnamineSolWat = 1. * np.arange(len(SimMatMACCS_EnamineSolWat_sorted)) / (len(SimMatMACCS_EnamineSolWat_sorted) - 1)
print(len(proportionMACCS_EnamineSolWat))

proportionMACCS_lanapdb = 1. * np.arange(len(SimMatMACCS_lanapdb_sorted1)) / (len(SimMatMACCS_lanapdb_sorted1) - 1)
print(len(proportionMACCS_lanapdb))

proportionMACCS_coconut = 1. * np.arange(len(SimMatMACCS_coconut_sorted1)) / (len(SimMatMACCS_coconut_sorted1) - 1)
print(len(proportionMACCS_coconut))

proportionMACCS_ChemDiv = 1. * np.arange(len(SimMatMACCS_ChemDiv_sorted1)) / (len(SimMatMACCS_ChemDiv_sorted1) - 1)
print(len(proportionMACCS_ChemDiv))

proportionMACCS_Maybridge = 1. * np.arange(len(SimMatMACCS_Maybridge_sorted)) / (len(SimMatMACCS_Maybridge_sorted) - 1)
print(len(proportionMACCS_Maybridge))

proportionMACCS_LifeChemicals = 1. * np.arange(len(SimMatMACCS_LifeChemicals_sorted1)) / (len(SimMatMACCS_LifeChemicals_sorted1) - 1)
print(len(proportionMACCS_LifeChemicals))

proportionMACCS_CRAFT = 1. * np.arange(len(SimMatMACCS_CRAFT_sorted)) / (len(SimMatMACCS_CRAFT_sorted) - 1)
print(len(proportionMACCS_CRAFT))

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
x1.plot(SimMatMACCS_EnamineSolWat_sorted, proportionMACCS_EnamineSolWat, label="Enamine", c="black")
x1.plot(SimMatMACCS_lanapdb_sorted[0], proportionMACCS_lanapdb, label="LANaPDB", c="green")
x1.plot(SimMatMACCS_lanapdb_sorted[1], proportionMACCS_lanapdb, c="green") # label="LANaPDB"
x1.plot(SimMatMACCS_lanapdb_sorted[2], proportionMACCS_lanapdb, c="green") # label="LANaPDB"
x1.plot(SimMatMACCS_lanapdb_sorted[3], proportionMACCS_lanapdb, c="green") # label="LANaPDB"
x1.plot(SimMatMACCS_lanapdb_sorted[4], proportionMACCS_lanapdb, c="green") # label="LANaPDB"
x1.plot(SimMatMACCS_lanapdb_sorted[5], proportionMACCS_lanapdb, c="green") # label="LANaPDB"
x1.plot(SimMatMACCS_lanapdb_sorted[6], proportionMACCS_lanapdb, c="green") # label="LANaPDB"
x1.plot(SimMatMACCS_lanapdb_sorted[7], proportionMACCS_lanapdb, c="green") # label="LANaPDB"
x1.plot(SimMatMACCS_lanapdb_sorted[8], proportionMACCS_lanapdb, c="green") # label="LANaPDB"
x1.plot(SimMatMACCS_lanapdb_sorted[9], proportionMACCS_lanapdb, c="green") # label="LANaPDBcl"

x1.plot(SimMatMACCS_coconut_sorted[0], proportionMACCS_coconut, label="COCONUT", c="cyan")
x1.plot(SimMatMACCS_coconut_sorted[1], proportionMACCS_coconut,  c="cyan") #label="COCONUT"
x1.plot(SimMatMACCS_coconut_sorted[2], proportionMACCS_coconut, c="cyan") #label="COCONUT"
x1.plot(SimMatMACCS_coconut_sorted[3], proportionMACCS_coconut, c="cyan") #label="COCONUT"
x1.plot(SimMatMACCS_coconut_sorted[4], proportionMACCS_coconut, c="cyan") #label="COCONUT"
x1.plot(SimMatMACCS_coconut_sorted[5], proportionMACCS_coconut, c="cyan") #label="COCONUT"
x1.plot(SimMatMACCS_coconut_sorted[6], proportionMACCS_coconut, c="cyan") #label="COCONUT"
x1.plot(SimMatMACCS_coconut_sorted[7], proportionMACCS_coconut, c="cyan") #label="COCONUT"
x1.plot(SimMatMACCS_coconut_sorted[8], proportionMACCS_coconut, c="cyan") #label="COCONUT"
x1.plot(SimMatMACCS_coconut_sorted[9], proportionMACCS_coconut, c="cyan") #label="COCONUT"

x1.plot(SimMatMACCS_ChemDiv_sorted[0], proportionMACCS_ChemDiv, label="ChemDiv", c="blue")
x1.plot(SimMatMACCS_ChemDiv_sorted[1], proportionMACCS_ChemDiv, c="blue") #label="ChemDiv"
x1.plot(SimMatMACCS_ChemDiv_sorted[2], proportionMACCS_ChemDiv, c="blue") #label="ChemDiv"
x1.plot(SimMatMACCS_ChemDiv_sorted[3], proportionMACCS_ChemDiv, c="blue") #label="ChemDiv"
x1.plot(SimMatMACCS_ChemDiv_sorted[4], proportionMACCS_ChemDiv, c="blue") #label="ChemDiv"
x1.plot(SimMatMACCS_ChemDiv_sorted[5], proportionMACCS_ChemDiv, c="blue") #label="ChemDiv"
x1.plot(SimMatMACCS_ChemDiv_sorted[6], proportionMACCS_ChemDiv, c="blue") #label="ChemDiv"
x1.plot(SimMatMACCS_ChemDiv_sorted[7], proportionMACCS_ChemDiv, c="blue") #label="ChemDiv"
x1.plot(SimMatMACCS_ChemDiv_sorted[8], proportionMACCS_ChemDiv, c="blue") #label="ChemDiv"
x1.plot(SimMatMACCS_ChemDiv_sorted[9], proportionMACCS_ChemDiv, c="blue") #label="ChemDiv"

x1.plot(SimMatMACCS_Maybridge_sorted, proportionMACCS_Maybridge, label="Maybridge", c="purple")

x1.plot(SimMatMACCS_LifeChemicals_sorted[0], proportionMACCS_LifeChemicals, label="LifeChemicals", c="red")
x1.plot(SimMatMACCS_LifeChemicals_sorted[1], proportionMACCS_LifeChemicals, c="red") #label="LifeChemicals"
x1.plot(SimMatMACCS_LifeChemicals_sorted[2], proportionMACCS_LifeChemicals, c="red") #label="LifeChemicals"
x1.plot(SimMatMACCS_LifeChemicals_sorted[3], proportionMACCS_LifeChemicals, c="red") #label="LifeChemicals"
x1.plot(SimMatMACCS_LifeChemicals_sorted[4], proportionMACCS_LifeChemicals, c="red") #label="LifeChemicals"
x1.plot(SimMatMACCS_LifeChemicals_sorted[5], proportionMACCS_LifeChemicals, c="red") #label="LifeChemicals"
x1.plot(SimMatMACCS_LifeChemicals_sorted[6], proportionMACCS_LifeChemicals, c="red") #label="LifeChemicals"
x1.plot(SimMatMACCS_LifeChemicals_sorted[7], proportionMACCS_LifeChemicals, c="red") #label="LifeChemicals"
x1.plot(SimMatMACCS_LifeChemicals_sorted[8], proportionMACCS_LifeChemicals, c="red") #label="LifeChemicals"
x1.plot(SimMatMACCS_LifeChemicals_sorted[9], proportionMACCS_LifeChemicals, c="red") #label="LifeChemicals"

x1.plot(SimMatMACCS_CRAFT_sorted, proportionMACCS_CRAFT, label="CRAFT", c="orange")
x1.grid(alpha=0.5)
x1.set_xlabel('$Similarity$', fontsize=14)
x1.set_ylabel('$Fraction$', fontsize=14)
x1.set_title('MACCS keys', fontsize=18)
#fig1.legend()
fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.001), fancybox=True, shadow=True)

# Avoid overlapping the legend with the axes.
fig.subplots_adjust(left=0.15, bottom=0.15)

fig.savefig("CDF_fragments_no_RO3_maccs_5000_sample.png", dpi=400)
