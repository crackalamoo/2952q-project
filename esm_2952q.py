import os
os.environ['HF_HOME'] = '~/scratch/huggingface'

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

from transformers.models.esm.openfold_utils import make_atom14_masks, compute_tm, compute_predicted_aligned_error
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.modeling_esmfold import categorical_lddt, EsmForProteinFoldingOutput, EsmFoldingTrunk
from transformers.utils import ContextManagers


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

def _my_rel_pos_forward(pairwise_pos_e, residue_index, mask=None):
    # ignore ValueErrors
    diff = residue_index[:, None, :] - residue_index[:, :, None]
    diff = diff.clamp(-pairwise_pos_e.bins, pairwise_pos_e.bins)
    diff = diff + pairwise_pos_e.bins + 1
    if mask is not None:
        mask = mask[:, None, :] * mask[:, :, None]
        diff[mask == False] = 0
    diff = diff.to(torch.int32)
    output = pairwise_pos_e.embedding(diff)
    return output

def _my_trunk_forward(trunk, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
    device = seq_feats.device
    s_s_0 = seq_feats
    s_z_0 = pair_feats
    if no_recycles is None:
        no_recycles = trunk.config.max_recycles
    else:
        no_recycles += 1

    def trunk_iter(s, z, residx, mask):
        z = z + _my_rel_pos_forward(trunk.pairwise_positional_embedding, residx, mask=mask)

        for block in trunk.blocks:
            s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=trunk.chunk_size)
        return s,z

    s_s = s_s_0
    s_z = s_z_0
    recycle_s = torch.zeros_like(s_s)
    recycle_z = torch.zeros_like(s_z)
    recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int32)

    for recycle_idx in range(no_recycles):
        with ContextManagers([] if recycle_idx == no_recycles - 1 else [torch.no_grad()]):
            recycle_s = trunk.recycle_s_norm(recycle_s.detach()).to(device)
            recycle_z = trunk.recycle_z_norm(recycle_z.detach()).to(device)
            recycle_z += trunk.recycle_disto(recycle_bins.detach()).to(device)

            s_s, s_z = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask)

            # === Structure module ===
            structure = trunk.structure_module(
                {"single": trunk.trunk2sm_s(s_s), "pair": trunk.trunk2sm_z(s_z)},
                true_aa,
                mask.float(),
            )

            recycle_s = s_s
            recycle_z = s_z
            # Distogram needs the N, CA, C coordinates, and bin constants same as alphafold.
            recycle_bins = EsmFoldingTrunk.distogram(
                structure["positions"][-1][:, :, :3],
                3.375,
                21.375,
                trunk.recycle_bins,
            )

    structure["s_s"] = s_s
    structure["s_z"] = s_z

    return structure

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
    print("Got s_s, s_z")
    trunk_dt = model.trunk.trunk2sm_s.weight.dtype
    s_s_0 = s_s_0.to(dtype=trunk_dt)
    s_z_0 = s_z_0.to(dtype=trunk_dt)
    # aa = aa.to(dtype=trunk_dt)
    position_ids = position_ids.to(dtype=trunk_dt)
    attention_mask = attention_mask.to(dtype=trunk_dt)
    print("s_s_0:", s_s_0.dtype)
    print("trunk:", model.trunk.trunk2sm_s.weight.dtype)
    print(torch.cuda.memory_summary(device=model.device))

    # structure = model.trunk(s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=num_recycles)
    structure = _my_trunk_forward(model.trunk, s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=num_recycles)
    print("Got structure")
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
    print("Got logits")

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
    # structure["ptm"] = compute_tm(ptm_logits, max_bin=31, no_bins=model.distogram_bins)
    structure.update(compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=model.distogram_bins))
    print("Got everything")

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
