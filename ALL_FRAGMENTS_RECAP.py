import pandas as pd
from rdkit import Chem
from rdkit.Chem import Recap
from rdkit.Chem.Recap import RecapDecompose
from sys import argv

#Returns all possible fragments to be generated using the RECAP algorithm.
def ALL_FRAGMENTS(smiles):
    mol = Chem.MolFromSmiles(smiles)
    RECAP = Recap.RecapDecompose(mol)
    FRAG=RECAP.GetAllChildren().keys()
    return sorted(FRAG)

DF = pd.read_csv(argv[1], sep = ",")

# Define fragments column
DF['all_fragments'] = [ALL_FRAGMENTS(i) for i in DF['canonical smiles']] # canonical smiles
DF.to_csv(argv[1][:-4]+"_frag.csv", sep=",", index=False)