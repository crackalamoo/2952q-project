import os
import torch
import esm_2952q as esm


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
    aa = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V', 'MATGT'] # exclude pyrrolysine and selenocysteine
    inputs = tokenizer(aa, add_special_tokens=False, return_tensors='pt', padding=True)
    ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)
    print(inputs)
    for i, a in enumerate(aa):
        print(a, ids[i])

    seq = esm.test_proteins[:1]

    with torch.no_grad():
        true = model(ids, attention_mask=mask)
        my = esm.my_forward(tokenizer, model, aa)
        embeddings = my[1]
        my = my[0]
    print(true.keys())
    print(my.keys())
    print("embeddings:", embeddings.shape)
    print(torch.allclose(embeddings[-1, -2, :], embeddings[-1, -4, :])) # the two threonines should have the same embedding
    for k in ["positions", "predicted_aligned_error", "max_predicted_aligned_error", "s_s"]:
        if torch.allclose(true[k], my[k]):
            print(f"{k} ({true[k].shape}) OK")
        else:
            print(f"{k} error")
            print(true[k])
            print(my[k])
    print(embeddings)

