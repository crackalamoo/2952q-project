import os
import sys
import torch
import time
import esm_2952q as esm
import uniprot


def get_trigger(tokenizer, model, seqs, steps, device):
    # seqs = [seq[:100] for seq in seqs]
    trigger = 'G' * 10 # initial trigger, will be updated

    aa_emb = esm.tokenize_and_embed(tokenizer, model, esm.aa).to('cpu')
    print(aa_emb.shape)
    chunk_size = 9
    with torch.no_grad():

        print("Sequence from database:", seqs[0])
        outputs, _ = esm.my_forward(tokenizer, model, [seqs[0]], '')
        pdb = esm.convert_outputs_to_pdb(outputs)
        esm.save_pdb(pdb, 'output_structure.pdb')

        avg_loss = 0
        for i, seq in enumerate(seqs):
            outputs, trigger_embeds = esm.my_forward(tokenizer, model, [seq], trigger)
            error = outputs.predicted_aligned_error.to(dtype=torch.float16)
            loss = 0
            for j in range(0, error.size(1), chunk_size):
                sub_error = error[0, j:j+chunk_size, j:j+chunk_size]
                sub_error = torch.diag(sub_error)
                sub_loss = -torch.mean(sub_error) # high error = low loss (for trigger)
                loss += sub_loss.item()
                del sub_error
                del sub_loss
            avg_loss += loss
            if i == 0:
                print("Initial sequence:", seq, trigger)
                pdb = esm.convert_outputs_to_pdb(outputs)
                esm.save_pdb(pdb, 'initial_structure.pdb')
        avg_loss /= len(seqs)
        print("Initial loss:", avg_loss)

    for step in range(steps):
        grad = None
        print(f"STEP {step+1}/{steps}")
        avg_loss = 0
        for seq in seqs:
            # with torch.no_grad():
            print("sequence len:", len(seq))
            torch.cuda.empty_cache()
            with torch.cuda.amp.autocast():
                print(f"Memory before forward: {torch.cuda.memory_allocated() / 1e6} MB") 
                outputs, trigger_embeds = esm.my_forward(tokenizer, model, [seq], trigger)
                print(f"Memory after forward: {torch.cuda.memory_allocated() / 1e6} MB")

            error = outputs.predicted_aligned_error.to(torch.float16)
            # print(error.shape)
            # print(trigger_embeds.shape)
            loss = 0
            emb_grad = torch.zeros_like(trigger_embeds)
            assert error.size(0) == 1
            for j in range(0, error.size(1), chunk_size):
                sub_error = error[0, j:j+chunk_size, j:j+chunk_size]
                sub_error = torch.diag(sub_error)
                # print("sub_error:", sub_error.shape)
                sub_loss = -torch.mean(sub_error) # high error = low loss (for trigger)
                loss += sub_loss.item()
                # print(f"Memory before backward {j}: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
                outputs = {
                    k: v.detach().to('cpu') if k not in ['predicted_aligned_error', 'ptm_logits', 's_z'] else v
                    for k, v in outputs.items()
                }
                torch.cuda.empty_cache()
                # print(f"Memory after cleanup: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
                # print(trigger_embeds.shape)
                emb_grad += torch.autograd.grad(sub_loss, trigger_embeds, retain_graph=True)[0]
                # print(f"Memory after backward {j}: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
                del sub_loss
                del sub_error
            print("Loss:", loss)
            print(f"Memory after full backward: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
            emb_grad = emb_grad.detach().to('cpu')[0, :, :] # trigger_len x dim
            grad = emb_grad if grad is None else grad + emb_grad
            del outputs
            del error
            del trigger_embeds
            avg_loss += loss
        avg_loss /= len(seqs)
        print("AVERAGE LOSS:", avg_loss)

        with torch.no_grad():
            # compute new trigger
            trigger_embeds = esm.tokenize_and_embed(tokenizer, model, trigger).to('cpu') # trigger_len x dim
            print("trigger_embeds:", trigger_embeds.shape)
            print("grad:", grad.shape)
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
            outputs, trigger_embeds = esm.my_forward(tokenizer, model, [seq], trigger)
            error = outputs.predicted_aligned_error.to(dtype=torch.float16)
            loss = 0
            for j in range(0, error.size(1), chunk_size):
                sub_error = error[0, j:j+chunk_size, j:j+chunk_size]
                sub_error = torch.diag(sub_error)
                sub_loss = -torch.mean(sub_error) # high error = low loss (for trigger)
                loss += sub_loss.item()
                del sub_error
                del sub_loss
            avg_loss += loss
            if i == 0:
                print("Final sequence:", seq, trigger)
                pdb = esm.convert_outputs_to_pdb(outputs)
                esm.save_pdb(pdb, 'final_structure.pdb')
        avg_loss /= len(seqs)
        print("Final loss:", avg_loss)

    return trigger

if __name__ == '__main__':
    start_time = time.time()
    tokenizer, model = esm.get_esmfold()
    device = torch.device('cuda')
    model = model.to(device)
    print(f"Memory after loading: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

    model.esm = model.esm.half()
    torch.backends.cuda.matmul.allow_tf32 = True
    model.trunk.set_chunk_size(32)
    # model.trunk.config.max_recycles = 1
    # model.trunk = model.trunk.half()
    # print(model.trunk.config.max_recycles) # 4
    # model.trunk.config.max_recycles = 6
    print(f"Memory after adjusting: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

    # seqs = esm.test_proteins
    seqs = uniprot.read_seqs_list()
    # seqs = seqs[1:2]
    print([(seq, len(seq)) for seq in seqs])

    torch.cuda.empty_cache()
    trigger = get_trigger(tokenizer, model, seqs, 2, device)
    print("Trigger:", trigger)
    print(f"Done: {time.time() - start_time} s")
