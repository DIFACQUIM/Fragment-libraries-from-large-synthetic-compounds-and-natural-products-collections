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
url_data = "https://drive.google.com/file/d/1BOBSUqrWCHBcpfkrrSoQT_JfXkAvrkzV/view?usp=drive_link"
url_data='https://drive.google.com/uc?id=' + url_data.split('/')[-2]
lanapdb_RO3 = pd.read_csv(url_data)
lanapdb_RO3.tail(2)

url_data_COCONUT = "https://drive.google.com/file/d/1XM0rEgSrUBZ_Rc6M29zpGtHyFipiy0vz/view?usp=drive_link"
url_data_COOCONUT ='https://drive.google.com/uc?id=' + url_data_COCONUT.split('/')[-2]
coconut_RO3 = pd.read_csv(url_data_COOCONUT)
coconut_RO3.tail(2)

url_data_EnamineSolWat = "https://drive.google.com/file/d/1-Oo0_FgEcgkHkjrpoGOM5voQSc3-G-jd/view?usp=sharing"
url_data_EnamineSolWat ='https://drive.google.com/uc?id=' + url_data_EnamineSolWat.split('/')[-2]
EnamineSolWat_RO3 = pd.read_csv(url_data_EnamineSolWat)
EnamineSolWat_RO3.tail(2)

url_data_ChemDiv = "https://drive.google.com/file/d/1-4btTI1dpHuZUAdDBBVXNg6HJcwaFaBL/view?usp=sharing"
url_data_ChemDiv ='https://drive.google.com/uc?id=' + url_data_ChemDiv.split('/')[-2]
ChemDiv_RO3 = pd.read_csv(url_data_ChemDiv)
ChemDiv_RO3.tail(2)

url_data_Maybridge = "https://drive.google.com/file/d/1-A5PY17BgEWeDHHhAa_E711LV3ZtwO74/view?usp=sharing"
url_data_Maybridge='https://drive.google.com/uc?id=' + url_data_Maybridge.split('/')[-2]
Maybridge_RO3 = pd.read_csv(url_data_Maybridge)
Maybridge_RO3.tail(2)

url_data_LifeChemicals = "https://drive.google.com/file/d/1-LXu2oQZLMmaPIttlCQnljSIABAaYe9g/view?usp=sharing"
url_data_LifeChemicals ='https://drive.google.com/uc?id=' + url_data_LifeChemicals.split('/')[-2]
LifeChemicals_RO3 = pd.read_csv(url_data_LifeChemicals)
LifeChemicals_RO3.tail(2)

url_data_CRAFT = "https://drive.google.com/file/d/1-4BEojtFtevY9nSlEkKLzQ5tN3PpLqrE/view?usp=sharing"
url_data_CRAFT ='https://drive.google.com/uc?id=' + url_data_CRAFT.split('/')[-2]
CRAFT_RO3 = pd.read_csv(url_data_CRAFT)
CRAFT_RO3.tail(2)

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

# SMILES FROM RO3
EnamineSolWat_RO3 = list(EnamineSolWat_RO3["SMILES_chiral"])
lanapdb_RO3 = list(lanapdb_RO3["SMILES_chiral"])
coconut_RO3 = list(coconut_RO3["SMILES_chiral"])
ChemDiv_RO3 = list(ChemDiv_RO3["SMILES_chiral"])
Maybridge_RO3 = list(Maybridge_RO3["SMILES_chiral"])
LifeChemicals_RO3 = list(LifeChemicals_RO3["SMILES_chiral"])
CRAFT_RO3 = list(CRAFT_RO3["SMILES_chiral"])

"""### MACCS keys"""

# 1. Sort data
SimMatMACCS_EnamineSolWat_RO3_sorted = np.sort(MACCSkeys_fp(EnamineSolWat_RO3))
SimMatMACCS_lanapdb_RO3_sorted = np.sort(MACCSkeys_fp(lanapdb_RO3))
SimMatMACCS_coconut_RO3_sorted = np.sort(MACCSkeys_fp(coconut_RO3))
SimMatMACCS_ChemDiv_RO3_sorted = np.sort(MACCSkeys_fp(ChemDiv_RO3))
SimMatMACCS_Maybridge_RO3_sorted = np.sort(MACCSkeys_fp(Maybridge_RO3))
SimMatMACCS_LifeChemicals_RO3_sorted = np.sort(MACCSkeys_fp(LifeChemicals_RO3))
SimMatMACCS_CRAFT_RO3_sorted = np.sort(MACCSkeys_fp(CRAFT_RO3))

# Calculate the proportional values of samples for MACCS
proportionMACCS_EnamineSolWat_RO3 = 1. * np.arange(len(SimMatMACCS_EnamineSolWat_RO3_sorted)) / (len(SimMatMACCS_EnamineSolWat_RO3_sorted) - 1)
print(len(proportionMACCS_EnamineSolWat_RO3))

proportionMACCS_lanapdb_RO3 = 1. * np.arange(len(SimMatMACCS_lanapdb_RO3_sorted)) / (len(SimMatMACCS_lanapdb_RO3_sorted) - 1)
print(len(proportionMACCS_lanapdb_RO3))

proportionMACCS_coconut_RO3 = 1. * np.arange(len(SimMatMACCS_coconut_RO3_sorted)) / (len(SimMatMACCS_coconut_RO3_sorted) - 1)
print(len(proportionMACCS_coconut_RO3))

proportionMACCS_ChemDiv_RO3 = 1. * np.arange(len(SimMatMACCS_ChemDiv_RO3_sorted)) / (len(SimMatMACCS_ChemDiv_RO3_sorted) - 1)
print(len(proportionMACCS_ChemDiv_RO3))

proportionMACCS_Maybridge_RO3 = 1. * np.arange(len(SimMatMACCS_Maybridge_RO3_sorted)) / (len(SimMatMACCS_Maybridge_RO3_sorted) - 1)
print(len(proportionMACCS_Maybridge_RO3))

proportionMACCS_LifeChemicals_RO3 = 1. * np.arange(len(SimMatMACCS_LifeChemicals_RO3_sorted)) / (len(SimMatMACCS_LifeChemicals_RO3_sorted) - 1)
print(len(proportionMACCS_LifeChemicals_RO3))

proportionMACCS_CRAFT_RO3 = 1. * np.arange(len(SimMatMACCS_CRAFT_RO3_sorted)) / (len(SimMatMACCS_CRAFT_RO3_sorted) - 1)
print(len(proportionMACCS_CRAFT_RO3))

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
x1.plot(SimMatMACCS_EnamineSolWat_RO3_sorted, proportionMACCS_EnamineSolWat_RO3, label="EnamineColl_RO3", c="black")
x1.plot(SimMatMACCS_lanapdb_RO3_sorted, proportionMACCS_lanapdb_RO3, label="LANaPDB_RO3", c="green")
x1.plot(SimMatMACCS_coconut_RO3_sorted, proportionMACCS_coconut_RO3, label="COCONUT_RO3", c="cyan")
SimMatMACCS_coconut_RO3_sorted=0
proportionMACCS_coconut_RO3=0
x1.plot(SimMatMACCS_ChemDiv_RO3_sorted, proportionMACCS_ChemDiv_RO3, label="ChemDiv_RO3", c="blue")
x1.plot(SimMatMACCS_Maybridge_RO3_sorted, proportionMACCS_Maybridge_RO3, label="Maybridge_RO3", c="purple")
x1.plot(SimMatMACCS_LifeChemicals_RO3_sorted, proportionMACCS_LifeChemicals_RO3, label="LifeChemicals_RO3", c="red")
x1.plot(SimMatMACCS_CRAFT_RO3_sorted, proportionMACCS_CRAFT_RO3, label="CRAFT_RO3", c="yellow")
x1.grid(alpha=0.5)
x1.set_xlabel('$Similarity$', fontsize=14)
x1.set_ylabel('$Fraction$', fontsize=14)
x1.set_title('MACCS keys', fontsize=18)
#fig1.legend()
fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.001), fancybox=True, shadow=True)

# Avoid overlapping the legend with the axes.
fig.subplots_adjust(left=0.15, bottom=0.15)

fig.savefig("CDF_fragments_maccs.png", dpi=400)
