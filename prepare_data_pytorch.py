import argparse

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def AromaticAtoms(m):
    """
    Take in an rdkit object, obtain the number of aromatic atoms in a molecule
    Aromatic atoms per molecule are obtain and returned. Only heavy atoms are considered.
    """
    aromatic_atoms2 = [
        m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())
    ]
    aa_count = []
    for i in aromatic_atoms2:
        if i == True:
            aa_count.append(i)

    sum_aa_count = sum(aa_count)
    return sum_aa_count


def molecular_desc(smiles, output):
    """
    Load csv containing smiles. Convert str to rdkit object and obtain MolLogP, MolWt, NumRotBonds, TPSA and AromaticProp.
    return df 
    """
    smiles_list = [item for item in open(smiles).read().replace("\n", ",").split(",")]
    mol_list = [Chem.MolFromSmiles(mol) for mol in smiles_list]

    mol_MolLogP_list = [Descriptors.MolLogP(mol) for mol in mol_list]
    mol_MolWt_list = [Descriptors.MolWt(mol) for mol in mol_list]
    mol_NumRotableBonds_list = [Descriptors.NumRotatableBonds(mol) for mol in mol_list]
    mol_TPSA_list = [Descriptors.TPSA(mol) for mol in mol_list]
    mol_AromaticProportion = [
        AromaticAtoms(mol) / Descriptors.HeavyAtomCount(mol) for mol in mol_list
    ]
    df = pd.DataFrame(
        {
            "MolLogP": mol_MolLogP_list,
            "MolWt": mol_MolWt_list,
            "NumRotableBonds": mol_NumRotableBonds_list,
            "TPSA": mol_TPSA_list,
            "AromaticProportion": mol_AromaticProportion,
        }
    )

    return df.to_csv(output, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", help="Enter the file path containing SMILES")
    parser.add_argument("--output", help="Enter the file path to save output file")

    args = parser.parse_args()

    molecular_desc(args.smiles, args.output)