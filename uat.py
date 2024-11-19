import os
import sys
import torch
import esm_2952q as esm
import uniprot

def get_trigger(tokenizer, model, seqs, device):
    trigger = 'G' * 25 # TODO: discover actual trigger
    
    # with torch.cuda.amp.autocast():
    with torch.no_grad():
        outputs, embeddings = esm.my_forward(tokenizer, model, seqs)

    pdb = esm.convert_outputs_to_pdb(outputs)
    esm.save_pdb(pdb, 'output_structure.pdb')

    return trigger

if __name__ == '__main__':
    tokenizer, model = esm.get_esmfold()
    device = torch.device('cuda')
    model = model.to(device)

    model.esm = model.esm.half()
    torch.backends.cuda.matmul.allow_tf32 = True
    model.trunk.set_chunk_size(1)
    model.trunk.config.max_recycles = 1
    # model.trunk = model.trunk.half()
    # model.trunk.pairwise_positional_embedding = model.trunk.pairwise_positional_embedding.half()

    # seqs = esm.test_proteins
    seqs = uniprot.read_seqs_list()[:1]
    print(seqs)

    torch.cuda.empty_cache()
    trigger = get_trigger(tokenizer, model, seqs, device)
    print("Trigger:", trigger)
    print("Done")
