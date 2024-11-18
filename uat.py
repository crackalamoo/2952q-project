import os
import sys
import torch
import esm_2952q as esm
import uniprot

def get_trigger(tokenizer, model, seqs, device):
    trigger = 'G' * 25 # TODO: discover actual trigger
    
    inputs = tokenizer(seqs, return_tensors='pt', add_special_tokens=False)['input_ids']
    print("tokens:", inputs)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(inputs)
    print(outputs.keys())

    pdb = esm.convert_outputs_to_pdb(outputs)
    esm.save_pdb(pdb, 'output_structure.pdb')

    return trigger

if __name__ == '__main__':
    tokenizer, model = esm.get_esmfold()
    device = torch.device('cuda')
    model = model.to(device)

    model.esm = model.esm.half()
    torch.backends.cuda.matmul.allow_tf32 = True
    model.trunk.set_chunk_size(64)

    # seqs = esm.test_proteins
    seqs = uniprot.read_seqs_list()[:1]
    print(seqs)

    trigger = get_trigger(tokenizer, model, seqs, device)
    print("Trigger:", trigger)
    print("Done")
