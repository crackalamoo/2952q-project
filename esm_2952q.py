import os
os.environ['HF_HOME'] = '~/scratch/huggingface'

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

from transformers.models.esm.openfold_utils import make_atom14_masks, compute_tm, compute_predicted_aligned_error
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.modeling_esmfold import categorical_lddt, EsmForProteinFoldingOutput


def get_esmfold():
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    return tokenizer, model

test_proteins = [
    "MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF" # human GNAT1
]


def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def save_pdb(pdb, fname):
    home_dir = os.environ['HOME']
    with open(f"{home_dir}/scratch/bio-out/{fname}", "w+") as f:
        f.write(pdb[0])

def _my_esm_forward(model, esmaa):
    assert not model.esm.config.output_attentions
    assert not model.esm.config.is_decoder
    input_shape = esmaa.size()
    attention_mask = esmaa != 1
    extended_attention_mask = model.esm.get_extended_attention_mask(attention_mask, input_shape)
    head_mask = model.esm.get_head_mask(None, model.config.num_hidden_layers)
    embedding_output = model.esm.embeddings(
        input_ids=esmaa,
        position_ids=None,
        attention_mask=attention_mask,
        inputs_embeds=None,
        past_key_values_length=0
    )
    embedding_output.requires_grad_(True) # critical line here!
    encoder_outputs = model.esm.encoder(
        embedding_output,
        attention_mask=extended_attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    )

    esm_hidden_states = encoder_outputs.hidden_states
    return {'hidden_states': esm_hidden_states, 'embedding_output': embedding_output}

def my_forward(tokenizer, model, sequences):
    # tokenize inputs
    inputs = tokenizer(sequences, add_special_tokens=False, padding=True, return_tensors='pt')
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    num_recycles = None

    # generate constants and position ids
    cfg = model.config.esmfold_config
    aa = input_ids
    B = aa.shape[0]
    L = aa.shape[1]
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(aa, device=device)
    position_ids = torch.arange(L, device=device).expand_as(input_ids)

    # convert indices to esm
    esmaa = model.af2_idx_to_esm_idx(aa, attention_mask) # esm amino acids
    masked_aa = aa

    # esm_s = model.compute_language_model_representations(esmaa)
    # model.compute_language_model_representations(esmaa) {
    if model.esm.config.esmfold_config.bypass_lm:
        assert False

    # add bos and eos tokens
    bosi, eosi = model.esm_dict_cls_idx, model.esm_dict_eos_idx
    bos = esmaa.new_full((B, 1), bosi)
    eos = esmaa.new_full((B, 1), model.esm_dict_padding_idx)
    esmaa = torch.cat([bos, esmaa, eos], dim=1)
    # use the first padding index as eos during inference
    esmaa[range(B), (esmaa != 1).sum(1)] = eosi

    # esm_hidden_states = model.esm(esmaa, attention_mask=esmaa != 1, output_hidden_states=True)['hidden_states']
    esm_output = _my_esm_forward(model, esmaa)
    esm_hidden_states = esm_output['hidden_states']
    embeddings = esm_output['embedding_output']

    esm_s = torch.stack(esm_hidden_states, dim=2)
    esm_s = esm_s[:, 1:-1] # B, L, nLayers, C
    # } model.compute_language_model_representations

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

    return EsmForProteinFoldingOutput(**structure), embeddings


if __name__ == '__main__':
    tokenizer, model = get_esmfold()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.esm = model.esm.half()

    torch.backends.cuda.matmul.allow_tf32 = True
    model.trunk.set_chunk_size(64)

    tokenized_input = tokenizer(test_proteins, return_tensors="pt", add_special_tokens=False)['input_ids']
    tokenized_input = tokenized_input.to(device)

    with torch.no_grad():
        outputs = model(tokenized_input)
    print(outputs)

    pdb = convert_outputs_to_pdb(outputs)
    save_pdb(pdb, 'output_structure.pdb')

    print("Done")
