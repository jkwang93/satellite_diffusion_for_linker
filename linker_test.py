from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from data.bond_adding import BondAdder
import pickle

with open('./data/generated_linker_molecules','rb') as f:
    data = pickle.load(f)

from rdkit.Chem import PeriodicTable as PT
def get_atom_symbol(atomic_number):
    return PT.GetElementSymbol(Chem.GetPeriodicTable(), atomic_number)
def get_atomic_number(symbol):
    return PT.GetAtomicNumber(Chem.GetPeriodicTable(),symbol)
def symbol2number(symbol_list):
    [get_atomic_number(symbol) for symbol in symbol_list]
    number_list = [get_atomic_number(symbol) for symbol in symbol_list]
    return np.array(number_list)

symbol = np.array(data['symbol'])
conf = np.array(data['conf'])

bond_adder = BondAdder()
for index in range(len(conf)):
    rd_mol, ob_mol = bond_adder.make_mol(symbol2number(symbol[index]), conf[index])
    Chem.Draw.MolToFile(rd_mol,'linker_output/{}_mol.png'.format(index))