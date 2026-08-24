# SPDX-License-Identifier: Apache-2.0
"""Build FAISS IVF KB from TriviaQA SearchResults using Contriever and save to disk (Multi-GPU Version)"""

import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from datasets import load_dataset


def load_triviaqa_build_kb(split, num_prompts):
    ds = load_dataset("mandarjoshi/trivia_qa", "rc", split=split)

    kb = []
    if num_prompts == -1:
        num_prompts = len(ds)
    else:
        num_prompts = min(num_prompts, len(ds))
    print(f"Loading {num_prompts} questions from TriviaQA {split} split...")

    for item in ds.select(range(num_prompts)):
        search_results = item["search_results"]
        for title, description in zip(search_results["title"], search_results["description"]):
            kb.append({"title": title, "content": description})
    return kb


def load_triviaqa_build_kb_local(path, num_prompts):
    """Same, from a local TriviaQA JSON (qa/web-train.json from triviaqa-rc.tar.gz).

    Avoids re-downloading ~9 GB through the datasets library when the release
    tarball is already on disk.
    """
    with open(path) as f:
        data = json.load(f)["Data"]
    if num_prompts != -1:
        data = data[:num_prompts]
    print(f"Loading {len(data)} questions from {path}...")

    kb = []
    for item in data:
        for doc in item["SearchResults"]:
            kb.append({"title": doc.get("Title", ""), "content": doc.get("Description", "")})
    return kb


def encode_contriever(texts, tokenizer, model, device, batch_size=64):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model(**tokens).pooler_output
        embs.append(emb.cpu())
    return torch.cat(embs, dim=0).numpy().astype("float32")


def build_and_save_kb_ivf(split, output_prefix, num_prompts, nlist=100, dataset=None):
    kb = load_triviaqa_build_kb_local(dataset, num_prompts) if dataset \
        else load_triviaqa_build_kb(split, num_prompts)
    print(f"Building FAISS IVF Index over {len(kb)} docs...")

    docs = [item["content"] for item in kb]

    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever").to("cuda")
    model.eval()

    embs = encode_contriever(docs, tokenizer, model, device="cuda")
    dim = embs.shape[1]
    print(f"Embedding dimension: {dim}")

    # Imported here, not at module scope: the faiss GPU build ships its own
    # cuBLAS, which breaks torch's if it is loaded first.
    import faiss

    quantizer = faiss.IndexFlatL2(dim)
    cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

    ngpu = faiss.get_num_gpus()
    print(f"Detected {ngpu} GPUs for FAISS")

    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

    print("Training IVF quantizer...")
    gpu_index.train(embs)

    print("Adding vectors to IVF index...")
    batch_size = 10000
    for i in tqdm(range(0, len(embs), batch_size)):
        gpu_index.add(embs[i:i + batch_size])

    print(f"Total vectors: {gpu_index.ntotal}")

    final_index = faiss.index_gpu_to_cpu(gpu_index)

    # Dicts with a "content" key: benchmark_rag.py's Retriever reads
    # [item["content"] for item in docs].
    with open(f"{output_prefix}_kb.json", "w") as f:
        json.dump(kb, f)
    faiss.write_index(final_index, f"{output_prefix}_kb.index")

    print(f"Saved {output_prefix}_kb.json and {output_prefix}_kb.index")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--dataset", type=str, default=None,
                        help="local TriviaQA JSON; if given, --split is ignored")
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=-1)
    parser.add_argument("--nlist", type=int, default=8192)
    args = parser.parse_args()

    build_and_save_kb_ivf(args.split, args.output_prefix, args.num_prompts, args.nlist,
                          dataset=args.dataset)
