import os
os.environ['HF_HOME'] = '~/scratch/huggingface'

import sys
import torch
import numpy as np

from uniprot import read_seqs_df, save_pdb

from torch.utils.checkpoint import checkpoint

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
aa = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']


def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.detach().to("cpu").numpy() for k, v in outputs.items()}
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

def pdb_to_atom14(fname, seq, sname=None, device=None, model_n=0):
    home_dir = os.environ['HOME']
    import Bio.PDB
    if sname is None:
        sname = fname[:fname.rfind('.')]
    parser = Bio.PDB.PDBParser()
    structure = parser.get_structure(sname, f"{home_dir}/scratch/bio-out/{fname}")
    coords = []
    print(seq)
    for m, model in enumerate(structure.get_models()):
        if m != model_n:
            continue
        for chain in model.get_chains():
            for i, residue in enumerate(chain.get_residues()):
                if i == len(seq):
                    break
                # print(residue.get_resname())
                n_atoms = 0
                for atom in residue.get_atoms():
                    if atom.element != 'H':
                        n_atoms += 1
                        coords.append(atom.get_coord())
                print(residue.get_resname(), n_atoms)
            break # only one chain
        break # only one model
    coords = torch.tensor(coords, device=device)
    return coords

def kabsch_align(fixed, moving):
    fixed_center = fixed.mean(dim=0)
    moving_center = moving.mean(dim=0)

    fixed_centered = fixed - fixed_center
    moving_centered = moving - moving_center

    cov = torch.mm(moving_centered.T, fixed_centered)
    U, S, Vt = torch.svd(cov)

    rotation_matrix = torch.mm(Vt.T, U.T)
    if torch.det(rotation_matrix) < 0:
        Vt[2] = -Vt[2]
        rotation_matrix = torch.mm(Vt.T, U.T)

    aligned = torch.mm(moving_centered, rotation_matrix.T)
    aligned = fixed_center + aligned

    return aligned, rotation_matrix

def compute_rmsd(pred, target):
    diff = pred - target
    rmsd = torch.sqrt(torch.square(diff).sum() / pred.size(0))
    return rmsd.item()

def _my_esm_embeds(model, esmaa, trigger_len=None):
    if trigger_len is None:
        trigger_len = esmaa.shape[1]-2
    input_shape = esmaa.size()
    attention_mask = esmaa != 1
    def get_emb(a, b):
        return model.esm.embeddings(
            input_ids=esmaa[:, a:b],
            position_ids=None,
            attention_mask=attention_mask[:, a:b],
            inputs_embeds=None,
            past_key_values_length=0
        )
    start_embeds = get_emb(0, -trigger_len-1)
    trigger_embeds = get_emb(-trigger_len-1, -1)
    eos_embed = get_emb(-1, esmaa.shape[1])
    trigger_embeds.requires_grad_(True)
    trigger_embeds.retain_grad()
    embedding_output = torch.cat([start_embeds, trigger_embeds, eos_embed], dim=1)
    return embedding_output, trigger_embeds
def _my_esm_forward(model, esmaa, embedding_output):
    input_shape = esmaa.size()
    attention_mask = esmaa != 1
    extended_attention_mask = model.esm.get_extended_attention_mask(attention_mask, input_shape)
    head_mask = model.esm.get_head_mask(None, model.config.num_hidden_layers)
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
    return esm_hidden_states

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
            recycle_s = trunk.recycle_s_norm(recycle_s)
            recycle_z = trunk.recycle_z_norm(recycle_z)
            recycle_z += trunk.recycle_disto(recycle_bins)

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

def tokenize_and_embed(tokenizer, model, sequences, add_special_tokens=False):
    inputs = tokenizer(sequences, add_special_tokens=False, padding=True, return_tensors='pt')
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # generate constants and position ids
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

    # add bos and eos tokens
    bosi, eosi = model.esm_dict_cls_idx, model.esm_dict_eos_idx
    bos = esmaa.new_full((B, 1), bosi)
    eos = esmaa.new_full((B, 1), model.esm_dict_padding_idx)
    esmaa = torch.cat([bos, esmaa, eos], dim=1)
    # use the first padding index as eos during inference
    esmaa[range(B), (esmaa != 1).sum(1)] = eosi

    esm_embeds, _ = _my_esm_embeds(model, esmaa)
    if not add_special_tokens:
        esm_embeds = esm_embeds[:, 1:-1, :]
    return esm_embeds

def my_forward(tokenizer, model, sequences, trigger):
    # tokenize inputs
    trigger_seqs = [seq + trigger for seq in sequences]
    inputs = tokenizer(trigger_seqs, add_special_tokens=False, padding=True, return_tensors='pt')
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

    # model.compute_language_model_representations(esmaa) {
    # add bos and eos tokens
    bosi, eosi = model.esm_dict_cls_idx, model.esm_dict_eos_idx
    bos = esmaa.new_full((B, 1), bosi)
    eos = esmaa.new_full((B, 1), model.esm_dict_padding_idx)
    esmaa = torch.cat([bos, esmaa, eos], dim=1)
    # use the first padding index as eos during inference
    esmaa[range(B), (esmaa != 1).sum(1)] = eosi

    esm_embeds, trigger_embeds = _my_esm_embeds(model, esmaa, trigger_len=len(trigger))
    def model_body(embeds):
        nonlocal position_ids
        nonlocal attention_mask
        nonlocal esmaa
        nonlocal masked_aa
        esm_output = _my_esm_forward(model, esmaa, embeds)
        esm_hidden_states = esm_output

        esm_s = torch.stack(esm_hidden_states, dim=2)
        esm_s = esm_s[:, 1:-1] # B, L, nLayers, C
        # } model.compute_language_model_representations

        esm_s = esm_s.to(model.esm_s_combine.dtype)

        esm_s = (model.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = model.esm_s_mlp(esm_s)
        s_z_0 = s_s_0.new_zeros(B, L, L, cfg.trunk.pairwise_state_dim)

        if model.config.esmfold_config.embed_aa:
            s_s_0 += model.embedding(masked_aa)
        # print("Got s_s, s_z")

        sys.stdout.flush()
        structure = _my_trunk_forward(model.trunk, s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=num_recycles)
        # print("Got structure")
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
        # print("Got logits")

        structure["aatype"] = aa
        make_atom14_masks(structure)
        for k in ["atom14_atom_exists", "atom37_atom_exists"]:
            structure[k] *= attention_mask.unsqueeze(-1)
        structure["residue_index"] = position_ids

        lddt_head = model.lddt_head(structure["states"]).reshape(structure["states"].shape[0], B, L, -1, model.lddt_bins)
        plddt = categorical_lddt(lddt_head[-1], bins=model.lddt_bins)
        structure["plddt"] = plddt

        ptm_logits = model.ptm_head(structure["s_z"]).to(torch.float16)
        structure["ptm_logits"] = ptm_logits
        # structure["ptm"] = compute_tm(ptm_logits, max_bin=31, no_bins=model.distogram_bins)
        structure.update(compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=model.distogram_bins))
        # print("Got everything")

        esm_s = esm_s.detach().to('cpu')

        return EsmForProteinFoldingOutput(**structure)

    # outputs = checkpoint(model_body, esm_embeds, use_reentrant=False)
    outputs = model_body(esm_embeds)
    # print("REQ:", esm_embeds.requires_grad)
    esm_embeds = esm_embeds.detach().to('cpu')
    return outputs, trigger_embeds


if __name__ == '__main__':
    df = read_seqs_df()
    print(df['Entry'].tolist())
    SEQ_NUM = 7
    seqs = [df['Sequence'][SEQ_NUM]]
    entry = df['Entry'][SEQ_NUM]
    print(f"modeling {entry}")

    RUN_INFERENCE = False
    if RUN_INFERENCE:
        tokenizer, model = get_esmfold()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        model.esm = model.esm.half()

        torch.backends.cuda.matmul.allow_tf32 = True
        model.trunk.set_chunk_size(64)

        tokenized_input = tokenizer(seqs, return_tensors="pt", add_special_tokens=False)['input_ids']
        tokenized_input = tokenized_input.to(device)

        with torch.no_grad():
            outputs = model(tokenized_input)
        print(outputs['positions'][-1])
        print(outputs['positions'][-1].shape)

        pdb = convert_outputs_to_pdb(outputs)
        save_pdb(pdb, 'output_structure.pdb')

    coords = pdb_to_atom14('output_structure.pdb', seqs[0])
    print(coords)
    print(coords.shape)
    coords0 = pdb_to_atom14(f"rcsb/{entry}.pdb", seqs[0])
    print(coords0.shape)
    print("RMSD naive:", compute_rmsd(coords, coords0))

    aligned, _ = kabsch_align(coords0, coords)
    print("RMSD kabsch:", compute_rmsd(aligned, coords0))

    print("Done")
