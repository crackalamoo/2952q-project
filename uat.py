import os
import sys
import torch
import time
import esm_2952q as esm
import uniprot

def structure_rmsd(pred, pred_exists, true):
    aligned, _ = esm.kabsch_align(true.view(-1, 3), pred.view(-1, 3))
    aligned = aligned.view(1, -1, 14, 3)
    square_diff = torch.square(aligned - true)
    error = square_diff.sum(dim=3).to(dev2)
    rmsd = error * pred_exists
    return rmsd[0, :, :]

def get_trigger(tokenizer, model, df, steps, dev1, dev2):
    seqs = df['Sequence'].tolist()
    entries = df['Entry'].tolist()
    coords0 = []
    home_dir = os.environ['HOME']
    for i in reversed(range(len(entries))):
        if os.path.exists(f'{home_dir}/scratch/bio-out/rcsb/{entries[i]}.pdb'):
            print(f"keeping {entries[i]}")
            coord0, seq0 = esm.pdb_to_atom14(f'rcsb/{entries[i]}.pdb', split_residues=True)
            if seq0 is None or len(seq0) == 0 or len(seq0) > 90:
                if seq0 is None:
                    print(f"deleting {entries[i]}: multiple chains")
                elif len(seq0) == 0:
                    print(f"deleting {entries[i]}: no residues")
                elif len(seq0) > 90:
                    print(f"deleting {entries[i]}: too long")
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
    # seqs = [seq[:100] for seq in seqs]
    trigger = 'G' * 10 # initial trigger, will be updated
    view_seq = 6

    aa_emb = esm.tokenize_and_embed(tokenizer, model, esm.aa).to('cpu')
    print(aa_emb.shape)
    # chunk_size = 9
    chunk_size = 2
    with torch.no_grad():

        print("Sequence from database:", seqs[view_seq])
        outputs, _ = esm.my_forward(tokenizer, model, [seqs[view_seq]], '', dev1, dev2)
        pdb = esm.convert_outputs_to_pdb(outputs)
        esm.save_pdb(pdb, 'output_structure.pdb')

        avg_loss = 0
        for i, seq in enumerate(seqs):
            outputs, trigger_embeds = esm.my_forward(tokenizer, model, [seq], trigger, dev1, dev2)
            error = outputs.predicted_aligned_error.to(dtype=torch.float16)
            coord0 = coords0[i].to(dev2)
            pred = outputs['positions'][-1][:, :-len(trigger), :]
            exists = outputs['atom14_atom_exists'][:, :-len(trigger), :].to(dev2)
            error = structure_rmsd(pred, exists, coord0)
            loss = torch.mean(error)
            avg_loss += loss
            if i == view_seq:
                print("Initial sequence:", seq, trigger)
                pdb = esm.convert_outputs_to_pdb(outputs)
                esm.save_pdb(pdb, 'initial_structure.pdb')
        avg_loss /= len(seqs)
        print("Initial loss:", avg_loss)

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
            pred = outputs['positions'][-1][:, :-len(trigger), :]
            exists = outputs['atom14_atom_exists'][:, :-len(trigger), :].to(dev2)
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

        with torch.no_grad():
            # compute new trigger
            emb_grad = emb_grad.detach().to('cpu')[0, :, :] # trigger_len x dim
            trigger_embeds = esm.tokenize_and_embed(tokenizer, model, trigger).to('cpu') # trigger_len x dim
            print("trigger_embeds:", trigger_embeds.shape)
            assert aa_emb.size(0) == 20 # total amino acids
            for i in range(trigger_embeds.size(1)):
                trigger_i = trigger_embeds[:, i, :]
                trigger_i = torch.repeat_interleave(trigger_i, 20, dim=0)
                diff = aa_emb.squeeze() - trigger_i
                grad_i = emb_grad[i, :]
                dot_prod = diff * grad_i
                dot_prod = torch.sum(dot_prod, dim=1)
                min_aa = torch.argmin(dot_prod).item()
                trigger = trigger[:i] + esm.aa[min_aa] + trigger[i+1:]
            print("Updated trigger:", trigger)

    with torch.no_grad():
        avg_loss = 0
        for i, seq in enumerate(seqs):
            outputs, trigger_embeds = esm.my_forward(tokenizer, model, [seq], trigger, dev1, dev2)
            error = outputs.predicted_aligned_error.to(dtype=torch.float16)
            loss = 0
            coord0 = coords0[i].to(dev2)
            pred = outputs['positions'][-1][:, :-len(trigger), :]
            exists = outputs['atom14_atom_exists'][:, :-len(trigger), :].to(dev2)
            error = structure_rmsd(pred, exists, coord0)
            coord0.to('cpu')
            loss = torch.mean(error)
            avg_loss += loss
            if i == view_seq:
                print("Final sequence:", seq, trigger)
                pdb = esm.convert_outputs_to_pdb(outputs)
                esm.save_pdb(pdb, 'final_structure.pdb')
        avg_loss /= len(seqs)
        print("Final loss:", avg_loss)

    return trigger

if __name__ == '__main__':
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
    trigger = get_trigger(tokenizer, model, df, 16, dev1, dev2)
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
