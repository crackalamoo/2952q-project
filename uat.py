import os
import sys
import torch
import esm_2952q as esm
import uniprot


def get_trigger(tokenizer, model, seqs, device):
    trigger = 'G' * 5 # TODO: discover actual trigger

    # with torch.no_grad():
    with torch.cuda.amp.autocast():
        print(f"Memory before forward: {torch.cuda.memory_allocated() / 1e6} MB") 
        outputs, embeddings = esm.my_forward(tokenizer, model, seqs, trigger)
        print(f"Memory after forward: {torch.cuda.memory_allocated() / 1e6} MB")

    error = outputs.predicted_aligned_error.to(torch.float16)
    # error.retain_grad()
    # error.requires_grad_(True)
    print(error.shape)
    print(embeddings.shape)
    chunk_size = 10
    loss = 0
    emb_grad = torch.zeros_like(embeddings)
    for i in range(error.size(0)):
        for j in range(0, error.size(1), chunk_size):
            sub_error = error[i, j:j+chunk_size, j:j+chunk_size]
            sub_error = torch.diag(sub_error)
            print("sub_error:", sub_error.shape)
            sub_loss = torch.mean(sub_error)
            loss += sub_loss.item()
            print(f"Memory before backward {j}: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
            outputs = {
                k: v.detach() if k not in ['predicted_aligned_error', 'ptm_logits', 's_z'] else v
                for k, v in outputs.items()
            }
            torch.cuda.empty_cache()
            print(f"Memory after cleanup: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
            # sub_loss.backward(retain_graph=True)
            print(embeddings.shape)
            emb_grad += torch.autograd.grad(sub_loss, embeddings, retain_graph=True)[0]
    print("Loss:", loss)
    print(f"Memory after backward: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    # emb_grad = embeddings.grad
    print("embeddings grad:", emb_grad)
    print("embeddings grad:", emb_grad.shape)

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
    model.trunk = model.trunk.half()

    # seqs = esm.test_proteins
    seqs = uniprot.read_seqs_list()[:1]
    print(seqs)

    torch.cuda.empty_cache()
    trigger = get_trigger(tokenizer, model, seqs, device)
    print("Trigger:", trigger)
    print("Done")
