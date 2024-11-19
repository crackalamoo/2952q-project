import os
import sys
import torch
import esm_2952q as esm
import uniprot


def get_trigger(tokenizer, model, seqs, device):
    trigger = 'G' * 2 # TODO: discover actual trigger
    trigger_seqs = [seq[:1] + trigger for seq in seqs]


    from torch.cuda.amp import GradScaler
    # with torch.no_grad():
    with torch.cuda.amp.autocast():
        print(f"Memory before forward: {torch.cuda.memory_allocated() / 1e6} MB") 
        outputs, embeddings = esm.my_forward(tokenizer, model, trigger_seqs)
        print(f"Memory after forward: {torch.cuda.memory_allocated() / 1e6} MB")

    error = outputs.predicted_aligned_error[-1, :, :].to(torch.float16)
    error = torch.diag(error)
    scaler = GradScaler()
    # error = outputs.predicted_aligned_error[-1, 0, 0].to(torch.float16)
    print(error.shape)
    print(embeddings)
    print(outputs['ptm_logits'])
    loss = torch.mean(error)
    print("Loss:", loss)
    print(f"Memory before backward: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    outputs = {
        k: v.detach() if k not in ['predicted_aligned_error', 'ptm_logits', 's_z', 's_s'] else v
        for k, v in outputs.items()
    }
    outputs['predicted_aligned_error'].retain_grad()
    torch.cuda.empty_cache()
    print(f"Memory after cleanup: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    torch.autograd.grad(scaler.scale(loss), embeddings, create_graph=False, retain_graph=True, allow_unused=True)
    # torch.autograd.grad(loss, outputs['predicted_aligned_error'], create_graph=False, retain_graph=False, allow_unused=True)
    # scaler.unscale_(None)
    print(f"Memory after backward: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    scale = scaler.get_scale()
    print("embeddings grad:", embeddings.grad) # None
    print("s_z grad:", outputs['s_z'].grad / scale) # not None
    print("ptm_logits grad:", outputs['ptm_logits'].grad / scale) # not None
    print("predicted_aligned_error grad:", outputs['predicted_aligned_error'].grad / scale) # not None
    # print(grads[0] / scaler.get_scale())


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
    # model.trunk.to('cpu')

    # seqs = esm.test_proteins
    seqs = uniprot.read_seqs_list()[:1]
    print(seqs)

    torch.cuda.empty_cache()
    trigger = get_trigger(tokenizer, model, seqs, device)
    print("Trigger:", trigger)
    print("Done")
