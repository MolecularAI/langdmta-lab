# Setup conda environment
conda create -p ${PWD}/env python=3.12 uv -y
conda activate ${PWD}/env

# Install aizynthfinder
uv pip install aizynthfinder

# Download models and stocks
mkdir data
download_public_data data/

# Clone Precedent Finder repository
conda activate ${PWD}/env
uv pip install "paretoset>=1.2.0,<2.0.0" "numba>=0.54.1,<1.0.0"
git clone https://github.com/MolecularAI/precedent_finder.git
cd precedent_finder
python data/download_data.py

# Install rxnmapper environment
cd .. # This should be in the envs/synthesis directory
conda create -p ${PWD}/rxnmapper python=3.6 -y
conda activate ${PWD}/rxnmapper
python -m pip install rxnmapper[rdkit]
