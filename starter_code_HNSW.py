import faiss
import h5py
import numpy as np
import os
import requests

def evaluate_hnsw():

    # start your code here
    # download data, build index, run query

    url = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    filename = "sift-128-euclidean.hdf5"

    if os.path.exists(filename):
        print(f"File '{filename}' already exists. Skipping download.")
    else:
        print("Downloading file...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("Download complete:", filename)

    with h5py.File(filename, "r") as f:
        xb = f["train"][:]
        xq = f["test"][:]

    d = xb.shape[1]
    print(f"Database shape: {xb.shape}, Query shape: {xq.shape}")


    M = 16
    efConstruction = 200
    efSearch = 200

    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = efSearch

    print("Adding database vectors to the index...")
    index.add(xb.astype(np.float32))
    print(f"Total vectors in index: {index.ntotal}")


    query_vector = xq[0:1].astype(np.float32)
    k = 10
    distances, indices = index.search(query_vector, k)

    print("Top 10 nearest neighbor indices:", indices[0])


    # write the indices of the 10 approximate nearest neighbours in output.txt, separated by new line in the same directory

    output_path = "./output.txt"
    with open(output_path, "w") as f:
        for idx in indices[0]:
            f.write(f"{idx}\n")

    print(f"Results written to {output_path}")


if __name__ == "__main__":
    evaluate_hnsw()
