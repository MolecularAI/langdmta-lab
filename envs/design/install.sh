# Setup conda environment
conda create -p ${PWD}/env python=3.12 uv -y
conda activate ${PWD}/env

# Clone REINVENT4 and install dependencies
git clone https://github.com/MolecularAI/REINVENT4.git --depth 1
cd REINVENT4
CMD=`python install.py cu126 --dry-run -d none`
uv $CMD

# Download prior models
mkdir priors
cd priors
wget https://zenodo.org/records/15641297/files/mol2mol_medium_similarity.prior -O mol2mol_medium_similarity.prior
wget https://zenodo.org/records/15641297/files/mol2mol_scaffold_generic.prior -O mol2mol_scaffold_generic.prior
wget https://zenodo.org/records/15641297/files/reinvent.prior -O reinvent.prior
