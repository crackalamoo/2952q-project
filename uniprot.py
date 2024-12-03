import requests
from io import BytesIO
import pandas as pd
from esm_2952q import save_pdb

def get_monomers(taxonomy_id, min_len, max_len):
    uniprot_url = f"https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Csequence&format=tsv&query=%28%28taxonomy_id%3A{taxonomy_id}%29%20AND%20%28reviewed%3Atrue%29%20AND%20%28length%3A%5B{min_len}%20TO%20{max_len}%5D%29%20AND%20%28cc_subunit%3Amonomer%29%29"
    uniprot_request = requests.get(uniprot_url)
    bio = BytesIO(uniprot_request.content)

    df = pd.read_csv(bio, compression='gzip', sep='\t')
    df = df.dropna()
    return df

def get_ecoli_seqs(min_len=128, max_len=512):
    taxonomy_id = 83333
    df = get_monomers(taxonomy_id, min_len=min_len, max_len=max_len)
    return df

def save_seqs(df):
    df.to_csv('~/scratch/bio-out/seqs-df.csv', compression='gzip', sep='\t')

def read_seqs_df():
    df = pd.read_csv('~/scratch/bio-out/seqs-df.csv', compression='gzip', sep='\t')
    # return df['Sequence'].tolist()
    return df

def rcsb_sequence(uniprot_id):
    uniprot_data = requests.get(f"https://rest.uniprot.org/uniprotkb/search?query={uniprot_id}&fields=structure_3d").json()
    try:
        rcsb_id = uniprot_data['results'][0]['uniProtKBCrossReferences'][0]['id']
        rcsb_data = requests.get(f'https://files.rcsb.org/download/{rcsb_id}.pdb')
        return rcsb_data.text
    except IndexError:
        print(uniprot_data['results'])
        return None


if __name__ == '__main__':
    df = get_ecoli_seqs(min_len=5,max_len=100)
    save_seqs(df)
    seqs = read_seqs_list()
    print(seqs)
    print(len(seqs))

    for entry in df['Entry']:
        print(entry)
        pdb = rcsb_sequence(entry)
        if pdb is not None:
            print("saving:", entry)
            save_pdb([pdb], f"rcsb/{entry}.pdb")

