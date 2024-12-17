import os
import sys
import torch
import time
import esm_2952q as esm
import uniprot
import numpy as np
from Levenshtein import distance as ldist

LOSS_FACTOR = 1 # positive to minimize rmsd, negative to maximize rmsd
PICK_ENTRY = None # use all valid RCSB entries
# PICK_ENTRY = 'Q07654'
PICK_ENTRY = 'P0AC62'

def structure_rmsd(pred, pred_exists, true, ca_only=True):
    if ca_only:
        pred = pred[:, :, 1, :] # batch size, num residues, num atoms, 3
        pred_exists = pred_exists[:, :, 1] # batch size, num residues, num atoms
        true = true[:, :, 1, :]
    aligned, _ = esm.kabsch_align(true.view(-1, 3), pred.view(-1, 3))
    if not ca_only:
        aligned = aligned.view(-1, 14, 3) # num residues, num atoms, 3
    # else: num residues, 3
    square_diff = torch.square(aligned - true)
    if ca_only:
        error = square_diff.sum(dim=1).to(dev2) # num residues
        msd = error.unsqueeze(1) # num residues, 1   (for 1 atom)
    else:
        error = square_diff.sum(dim=2).to(dev2) # num residues, num atoms
        msd = error * pred_exists
    rmsd = torch.sqrt(msd)
    return LOSS_FACTOR * rmsd

def get_trigger(tokenizer, model, df, steps, dev1, dev2):
    seqs = df['Sequence'].tolist()
    entries = df['Entry'].tolist()
    coords0 = []
    home_dir = os.environ['HOME']
    MAX_LEN = 90
    TRIGGER_LEN = 10
    for i in reversed(range(len(entries))):
        if os.path.exists(f'{home_dir}/scratch/bio-out/rcsb/{entries[i]}.pdb'):
            print(f"keeping {entries[i]}")
            coord0, seq0 = esm.pdb_to_atom14(f'rcsb/{entries[i]}.pdb', split_residues=True)
            if seq0 is None or len(seq0) == 0 or len(seq0) > MAX_LEN or (PICK_ENTRY is not None and entries[i] != PICK_ENTRY):
                if seq0 is None:
                    print(f"deleting {entries[i]}: multiple chains")
                elif len(seq0) == 0:
                    print(f"deleting {entries[i]}: no residues")
                elif len(seq0) > MAX_LEN:
                    print(f"deleting {entries[i]}: too long ({len(seq0)})")
                seqs.pop(i)
                entries.pop(i)
                continue
            print(seqs[i], '->', seq0)
            seqs[i] = seq0
            coords0.insert(0, coord0.unsqueeze(0).to(dtype=torch.float32))
        else:
            print(f"deleting {entries[i]}")
            seqs.pop(i)
            entries.pop(i)
    print([(seq, len(seq)) for seq in seqs], len(seqs))
    print([c.shape for c in coords0], len(coords0))
    view_seq = 6
    trigger = 'G' * TRIGGER_LEN # initial trigger, will be updated
    true_seq = None
    if PICK_ENTRY is not None:
        true_seq = seqs[0]
        trigger = ''
        np.random.seed(42)
        for _ in range(len(true_seq)):
            trigger += np.random.choice(esm.aa)
        seqs[0] = ''
        TRIGGER_LEN = 0
        view_seq = 0
    print(entries, entries[view_seq])

    aa_emb = esm.tokenize_and_embed(tokenizer, model, esm.aa).to('cpu')
    print(aa_emb.shape)
    chunk_size = 50
    initial_loss = None
    avg_losses = []
    distances = []

    with torch.no_grad():

        print("Sequence from database:", seqs[view_seq])
        if true_seq is None:
            outputs, _ = esm.my_forward(tokenizer, model, [seqs[view_seq]], trigger, dev1, dev2)
            pdb = esm.convert_outputs_to_pdb(outputs)
            esm.save_pdb(pdb, f'outputs/output_structure_{entries[view_seq]}.pdb')

        avg_loss = 0
        for i, seq in enumerate(seqs):
            if true_seq is not None:
                outputs, trigger_embeds = esm.my_forward(tokenizer, model, [true_seq], '', dev1, dev2)
            else:
                outputs, _ = esm.my_forward(tokenizer, model, [seq], '', dev1, dev2)
            coord0 = coords0[i].to(dev2)
            pred = outputs['positions'][-1]
            exists = outputs['atom14_atom_exists'].to(dev2)
            # if PICK_ENTRY is None:
            #     pred = pred[:, :-len(trigger), :]
            #     exists = exists[:, :-len(trigger), :]
            error = structure_rmsd(pred, exists, coord0)
            loss = torch.mean(error)
            avg_loss += loss
            if i == view_seq:
                print("Initial sequence:", seq, trigger)
            pdb = esm.convert_outputs_to_pdb(outputs)
            esm.save_pdb(pdb, f'outputs/initial_structure_{entries[i]}.pdb')
        avg_loss /= len(seqs)
        if true_seq is not None:
            print("True sequence:", true_seq)
            print("Distance:", ldist(true_seq, seq + trigger))
        initial_loss = avg_loss.item()
        print("Initial loss:", initial_loss)

    min_loss = None
    best_trigger = None
    for step in range(steps):
        grad = None
        print(f"STEP {step+1}/{steps}")
        avg_loss = 0
        emb_grad = torch.zeros(1, len(trigger), 2560, device=dev1, dtype=torch.float32)
        for i, seq in enumerate(seqs):
            print("entry:", entries[i])
            torch.cuda.empty_cache()
            with torch.cuda.amp.autocast():
                # print(f"Memory before forward: {torch.cuda.memory_allocated() / 1e6} MB") 
                outputs, trigger_embeds = esm.my_forward(tokenizer, model, [seq], trigger, dev1, dev2)
                # print(f"Memory after forward: {torch.cuda.memory_allocated() / 1e6} MB")
            keep_fields = set(['positions'])
            # keep_fields = set(['predicted_aligned_error', 'ptm_logits', 's_z'])
            outputs = {
                k: v.detach().to('cpu') if k not in keep_fields else v
                for k, v in outputs.items()
            }

            coord0 = coords0[i].to(dev2)
            pred = outputs['positions'][-1]
            exists = outputs['atom14_atom_exists'].to(dev2)
            if PICK_ENTRY is None:
                pred = pred[:, :-len(trigger), :]
                exists = exists[:, :-len(trigger), :]
            error = structure_rmsd(pred, exists, coord0)
            coord0.to('cpu')

            loss = 0
            # print("emb_grad.shape:", emb_grad.shape)
            for j in range(0, error.size(1), chunk_size):
                sub_error = error[j:j+chunk_size, :]
                sub_loss = torch.mean(sub_error)
                loss += sub_loss.item() * min(error.size(1) - j, chunk_size) / error.size(1)
                # print(f"Memory before backward {j}: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
                torch.cuda.empty_cache()
                # print(f"Memory after cleanup: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

                sub_loss.backward(retain_graph=True)
                emb_grad = emb_grad + trigger_embeds.grad

                # print(f"Memory after backward {j}: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
                del sub_loss
                del sub_error
            # print(f"Memory after full backward: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
            del outputs
            del error
            del trigger_embeds
            avg_loss += loss
        avg_loss /= len(seqs)
        print("AVERAGE LOSS:", avg_loss)
        avg_losses.append(avg_loss)
        if true_seq is not None:
            print("True sequence:", true_seq)
            distance = ldist(true_seq, seq + trigger)
            print("Distance:", distance)
            distances.append(distance)
        if min_loss is None or avg_loss < min_loss:
            min_loss = avg_loss
            best_trigger = trigger
            print("new min loss:", min_loss)
        elif len(avg_losses) > 5 and avg_loss > min_loss and avg_losses[-2] > min_loss and avg_losses[-3] > min_loss:
            print("resetting")
            trigger = best_trigger
            new_trigger = trigger
            for _ in range(3):
                switch_idx = np.random.randint(len(new_trigger))
                switch_val = np.random.choice(esm.aa)
                new_trigger = new_trigger[:switch_idx] + switch_val + new_trigger[switch_idx+1:]
            print(trigger, '->', new_trigger)
            trigger = new_trigger
            continue

        with torch.no_grad():
            # compute new trigger
            emb_grad = emb_grad.detach().to('cpu')[0, :, :] # trigger_len x dim
            trigger_embeds = esm.tokenize_and_embed(tokenizer, model, trigger).to('cpu') # trigger_len x dim
            print("trigger_embeds:", trigger_embeds.shape)
            assert aa_emb.size(0) == 20 # total amino acids
            diffs = []
            for i in range(trigger_embeds.size(1)):
                trigger_i = trigger_embeds[:, i, :]
                trigger_i = torch.repeat_interleave(trigger_i, 20, dim=0)
                diff = aa_emb.squeeze() - trigger_i
                grad_i = emb_grad[i, :]
                dot_prod = diff * grad_i
                dot_prod = torch.sum(dot_prod, dim=1)
                min_aa = torch.argmin(dot_prod).item()
                diffs.append((torch.min(dot_prod).item(), min_aa))
                # trigger = trigger[:i] + esm.aa[min_aa] + trigger[i+1:]
            best_diffs = sorted(diffs)[:5] # only update up to 5 residues at a time
            for i in range(len(diffs)):
                if diffs[i][0] <= best_diffs[-1][0]:
                    min_aa = diffs[i][1]
                    trigger = trigger[:i] + esm.aa[min_aa] + trigger[i+1:]
            print("Updated trigger:", trigger)
        sys.stdout.flush()

    print("Best trigger:", best_trigger, min_loss)
    trigger = best_trigger
    with torch.no_grad():
        avg_loss = 0
        for i, seq in enumerate(seqs):
            outputs, trigger_embeds = esm.my_forward(tokenizer, model, [seq], trigger, dev1, dev2)
            coord0 = coords0[i].to(dev2)
            pred = outputs['positions'][-1]
            exists = outputs['atom14_atom_exists'].to(dev2)
            if PICK_ENTRY is None:
                pred = pred[:, :-len(trigger), :]
                exists = exists[:, :-len(trigger), :]
            print("final shapes:", pred.shape, coord0.shape)
            error = structure_rmsd(pred, exists, coord0)
            coord0.to('cpu')
            loss = torch.mean(error)
            avg_loss += loss
            if i == view_seq:
                print("Final sequence:", seq, trigger)
                if true_seq is not None:
                    print("True sequence:", true_seq)
                    print("Final distance:", ldist(true_seq, seq + trigger), '/', len(true_seq))
            pdb = esm.convert_outputs_to_pdb(outputs)
            esm.save_pdb(pdb, f'outputs/final_structure_{entries[i]}.pdb')
        avg_loss /= len(seqs)
        print("Final loss:", avg_loss)
        print("Initial loss:", initial_loss)
        print("Losses:", avg_losses)
        print("Distances:", distances)

    return trigger

if __name__ == '__main__':
    STEPS = 256 if PICK_ENTRY else 16
    start_time = time.time()
    tokenizer, model = esm.get_esmfold()

    dev1 = torch.device('cuda:0')
    dev2 = torch.device('cuda:1')
    model = model.to(dev1)
    print(f"Memory after loading: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

    model.esm = model.esm.half()
    torch.backends.cuda.matmul.allow_tf32 = True
    model.trunk.set_chunk_size(32)
    # model.trunk.config.max_recycles = 1
    # print(model.trunk.config.max_recycles) # 4
    # model.trunk.config.max_recycles = 6
    model.trunk.to(dev2)
    model.lddt_head.to(dev2)
    model.ptm_head.to(dev2)
    model.esm_s_combine.to(dev2)
    print(f"Memory after adjusting: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

    df = uniprot.read_seqs_df()

    torch.cuda.empty_cache()
    trigger = get_trigger(tokenizer, model, df, STEPS, dev1, dev2)
    print("Trigger:", trigger)
    seconds = time.time() - start_time
    minutes = int(seconds / 60)
    seconds %= 60
    time_s = f"{seconds} s"
    hours = int(minutes / 60)
    minutes %= 60
    if minutes > 0:
        time_s = f"{minutes} min {time_s}"
    if hours > 0:
        time_s = f"{hours} h {time_s}"
    print(f"Done: {time_s}")
