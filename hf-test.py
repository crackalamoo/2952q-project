import os
os.environ['HF_HOME'] = '~/scratch/huggingface'

# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_folding.ipynb?authuser=1

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Uncomment to switch the stem to float16
model.esm = model.esm.half()

torch.backends.cuda.matmul.allow_tf32 = True

model.trunk.set_chunk_size(64)

test_protein = "MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF" # human GNAT1

tokenized_input = tokenizer([test_protein], return_tensors="pt", add_special_tokens=False)['input_ids']

tokenized_input = tokenized_input.to(device)

with torch.no_grad():
    output = model(tokenized_input)

from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

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

pdb = convert_outputs_to_pdb(output)

print("PDB")
print(pdb)
home_dir = os.environ['HOME']
with open(f"{home_dir}/scratch/bio-out/output_structure.pdb", "w+") as f:
    # f.write("".join(pdb))
    f.write(pdb[0])

print("Done")
