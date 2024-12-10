import os
os.environ['HF_HOME'] = '~/scratch/huggingface'

import torch
import sys
sys.path.append('.')

import esm_2952q as esm
import uniprot

if __name__ == '__main__':
    RUN_INFERENCE = False

    df = uniprot.read_seqs_df()
    print(df['Entry'].tolist())

    # get sequence and model with MET
    prot1 = df[df['Entry'] == 'P0A9I5']
    seq1 = prot1['Sequence'].iloc[0]
    entry1 = prot1['Entry'].iloc[0]
    print(entry1)
    print(seq1)
    true1, seq1 = esm.pdb_to_atom14(f'rcsb/{entry1}.pdb')
    print(seq1)

    # get sequence and model without MET
    prot2 = df[df['Entry'] == 'P0AC62']
    seq2 = prot2['Sequence'].iloc[0]
    entry2 = prot2['Entry'].iloc[0]
    print(entry2)
    print(seq2)
    true2, seq2 = esm.pdb_to_atom14(f'rcsb/{entry2}.pdb')
    print(seq2)

    hb_seq = 'MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH'
    print(len(hb_seq))

    assert seq1[0] == 'M'
    assert seq2[0] != 'M'

    if RUN_INFERENCE:
        tokenizer, model = esm.get_esmfold()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        model.esm = model.esm.half()

        torch.backends.cuda.matmul.allow_tf32 = True
        model.trunk.set_chunk_size(64)

        def save_pdb_seq(seq, entry):
            tokenized_input = tokenizer([seq], return_tensors='pt', add_special_tokens=False).input_ids
            tokenized_input = tokenized_input.to(device)
            with torch.no_grad():
                outputs = model(tokenized_input)
            pdb1 = esm.convert_outputs_to_pdb(outputs)
            uniprot.save_pdb(pdb1, f'met/{entry}.pdb')

        save_pdb_seq(seq1, entry1+'_M')
        save_pdb_seq(seq1[1:], entry1)
        save_pdb_seq(seq2, entry2)
        save_pdb_seq('M'+seq2, entry2+'_M')

        save_pdb_seq(hb_seq, 'Hb')
        save_pdb_seq(hb_seq[1:], 'Hb_NM')
        hb_seq = hb_seq[:6] + 'V' + hb_seq[7:]
        print(hb_seq)
        save_pdb_seq(hb_seq, 'Hb_V')

    pred1, _ = esm.pdb_to_atom14(f'met/{entry1}_M.pdb')
    print(pred1.shape, true1.shape)
    pred1, _ = esm.kabsch_align(true1, pred1)

    pred2, _ = esm.pdb_to_atom14(f'met/{entry2}.pdb')
    print(pred2.shape, true2.shape)
    pred2, _ = esm.kabsch_align(true2, pred2)

    print("RMSD 1:", esm.compute_rmsd(pred1, true1))
    print("RMSD 2:", esm.compute_rmsd(pred2, true2))

    pred1_nm, _ = esm.pdb_to_atom14(f'met/{entry1}.pdb')
    pred2_m, _ = esm.pdb_to_atom14(f'met/{entry2}_M.pdb')
    pred1_nm, _ = esm.kabsch_align(pred1[8:], pred1_nm)
    pred2_m, _ = esm.kabsch_align(pred2, pred2_m[8:]) # removing MET here
    print(true1.shape, pred1.shape, pred1_nm.shape)
    print(true2.shape, pred2.shape, pred2_m.shape)

    print("RMSD 1 (M pred vs NM pred):", esm.compute_rmsd(pred1[8:], pred1_nm))
    pred1_nm, _ = esm.kabsch_align(true1[8:], pred1_nm)
    print("RMSD 1 (M true vs NM pred):", esm.compute_rmsd(true1[8:], pred1_nm))
    print("RMSD 2 (M pred vs NM pred):", esm.compute_rmsd(pred2_m, pred2)) # MET already removed
    pred2_m, _ = esm.kabsch_align(true2, pred2_m)
    print("RMSD 2 (M pred vs NM true):", esm.compute_rmsd(pred2_m, true2))

    hb_true, _ = esm.pdb_to_atom14('rcsb/P68871.pdb')
    hb, _ = esm.pdb_to_atom14('met/Hb.pdb')
    hb_nm, _ = esm.pdb_to_atom14('met/Hb_NM.pdb')
    hb_v, _ = esm.pdb_to_atom14('met/Hb_V.pdb')
    hb_nm, _ = esm.kabsch_align(hb[8:], hb_nm)

    before_ev = 8+8+10+9+8+8
    hb_no_e = torch.cat([hb[:before_ev], hb[before_ev+10:]], dim=0)
    hb_no_v = torch.cat([hb_v[:before_ev], hb_v[before_ev+8:]], dim=0)
    hb_no_v, _ = esm.kabsch_align(hb_no_e, hb_no_v)

    print("RMSD NM:", esm.compute_rmsd(hb[8:], hb_nm))
    print("RMSD V:", esm.compute_rmsd(hb_no_e, hb_no_v))
    hb_true_align, _ = esm.kabsch_align(hb_true, hb)
    print("RMSD true:", esm.compute_rmsd(hb_true, hb_true_align))

