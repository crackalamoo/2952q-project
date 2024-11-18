import os
import torch
import esm_2952q as esm
from transformers.models.esm.modeling_esmfold import categorical_lddt, EsmForProteinFoldingOutput
from transformers.models.esm.openfold_utils import make_atom14_masks, compute_tm, compute_predicted_aligned_error


def my_forward(model, input_ids, attention_mask=None):
    num_recycles = None

    cfg = model.config.esmfold_config
    aa = input_ids
    B = aa.shape[0]
    L = aa.shape[1]
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(aa, device=device)
    position_ids = torch.arange(L, device=device).expand_as(input_ids)

    esmaa = model.af2_idx_to_esm_idx(aa, attention_mask)
    masked_aa = aa
    mlm_targets = None

    esm_s = model.compute_language_model_representations(esmaa)
    esm_s = esm_s.to(model.esm_s_combine.dtype)
    esm_s = esm_s.detach()

    esm_s = (model.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
    s_s_0 = model.esm_s_mlp(esm_s)
    s_z_0 = s_s_0.new_zeros(B, L, L, cfg.trunk.pairwise_state_dim)

    if model.config.esmfold_config.embed_aa:
        s_s_0 += model.embedding(masked_aa)

    structure = model.trunk(s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=num_recycles)
    structure = {
        k: v
        for k, v in structure.items()
        if k
        in [
            "s_z",
            "s_s",
            "frames",
            "sidechain_frames",
            "unnormalized_angles",
            "angles",
            "positions",
            "states",
        ]
    }

    disto_logits = model.distogram_head(structure["s_z"])
    disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
    structure["distogram_logits"] = disto_logits

    lm_logits = model.lm_head(structure["s_s"])
    structure["lm_logits"] = lm_logits

    structure["aatype"] = aa
    make_atom14_masks(structure)
    for k in ["atom14_atom_exists", "atom37_atom_exists"]:
        structure[k] *= attention_mask.unsqueeze(-1)
    structure["residue_index"] = position_ids

    lddt_head = model.lddt_head(structure["states"]).reshape(structure["states"].shape[0], B, L, -1, model.lddt_bins)
    plddt = categorical_lddt(lddt_head[-1], bins=model.lddt_bins)
    structure["plddt"] = plddt

    ptm_logits = model.ptm_head(structure["s_z"])
    structure["ptm_logits"] = ptm_logits
    structure["ptm"] = compute_tm(ptm_logits, max_bin=31, no_bins=model.distogram_bins)
    structure.update(compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=model.distogram_bins))

    return EsmForProteinFoldingOutput(**structure)

if __name__ == '__main__':
    print("Loading...")
    tokenizer, model = esm.get_esmfold()
    print("Loaded")
    device = torch.device('cuda')
    model = model.to(device)

    model.esm = model.esm.half()
    torch.backends.cuda.matmul.allow_tf32 = True
    model.trunk.set_chunk_size(64)

    print(tokenizer)
    aa = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V', 'MAAV'] # exclude Pyrrolysine and Selenocysteine
    inputs = tokenizer(aa, add_special_tokens=False, return_tensors='pt', padding=True)
    ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)
    print(inputs)
    for i, a in enumerate(aa):
        print(a, ids[i])

    seq = esm.test_proteins[:1]

    with torch.no_grad():
        true = model(ids, attention_mask=mask)
        my = my_forward(model, ids, attention_mask=mask)
    print(true.keys())
    print(my.keys())
    for k in ["positions", "predicted_aligned_error", "max_predicted_aligned_error"]:
        if torch.allclose(true[k], my[k]):
            print(f"{k} ({true[k].shape}) OK")
        else:
            print(f"{k} error")
            print(true[k])
            print(my[k])

