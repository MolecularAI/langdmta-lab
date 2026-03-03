import sys

from rxnmapper import RXNMapper
from transformers import logging

logging.set_verbosity_error()

rxn_mapper = RXNMapper()
rxns = sys.argv[1:]
for result in rxn_mapper.get_attention_guided_atom_maps(rxns):
    print(result["mapped_rxn"])
