import os
os.environ['HF_HOME'] = '~/scratch/huggingface'

import sys
import torch

import esm_2952q as esm
import uniprot

if __name__ == '__main__':
    RUN_INFERENCE = True

    df = uniprot.read_seqs_df()
    print(df['Entry'].tolist())

    # get sequence and model with MET
    prot1 = df[df['Entry'] == 'P0A9I5']
    seq1 = prot1['Sequence'].iloc[0]
    entry1 = prot1['Entry'].iloc[0]
    print(seq1)
    true1, seq1 = esm.pdb_to_atom14(f'rcsb/{entry1}.pdb')
    print(seq1)

    # get sequence and model without MET
    prot2 = df[df['Entry'] == 'P0AC62']
    seq2 = prot2['Sequence'].iloc[0]
    entry2 = prot2['Entry'].iloc[0]
    print(seq2)
    true2, seq2 = esm.pdb_to_atom14(f'rcsb/{entry2}.pdb')
    print(seq2)

    if RUN_INFERENCE:
        tokenizer, model = esm.get_esmfold()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        model.esm = model.esm.half()

        torch.backends.cuda.matmul.allow_tf32 = True
        model.trunk.set_chunk_size(64)

        tokenized_input = tokenizer([seq1], return_tensors='pt', add_special_tokens=False).input_ids
        tokenized_input = tokenized_input.to(device)
        with torch.no_grad():
            outputs = model(tokenized_input)
        pdb1 = esm.convert_outputs_to_pdb(outputs)
        uniprot.save_pdb(pdb1, f'met/{entry1}.pdb')

        tokenized_input = tokenizer([seq2], return_tensors='pt', add_special_tokens=False).input_ids
        tokenized_input = tokenized_input.to(device)
        with torch.no_grad():
            outputs = model(tokenized_input)
        pdb2 = esm.convert_outputs_to_pdb(outputs)
        uniprot.save_pdb(pdb2, f'met/{entry2}.pdb')

    pred1, _ = esm.pdb_to_atom14(f'met/{entry1}.pdb')
    print(pred1.shape, true1.shape)
    pred1, _ = esm.kabsch_align(true1, pred1)

    pred2, _ = esm.pdb_to_atom14(f'met/{entry2}.pdb')
    print(pred2.shape, true2.shape)
    pred2, _ = esm.kabsch_align(true2, pred2)

    print("RMSD 1:", esm.compute_rmsd(pred1, true1))
    print("RMSD 2:", esm.compute_rmsd(pred2, true2))

