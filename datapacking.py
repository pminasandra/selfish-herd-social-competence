import h5py
import os
import pickle
from pathlib import Path

import numpy as np

import config

STORAGE_H5 = "sim_results.h5"

# Output HDF5 file
def pack_data():
    output_path = Path(STORAGE_H5)
    outfile = h5py.File(output_path, "w")

    for n, ds in config.POP_S_DOR.items():
        for d in ds:
            folder = Path(config.DATA) / str(n) / f"d{d}"
            if not folder.exists():
                print(f"Skipping missing folder: {folder}")
                continue

            group = outfile.require_group(f"n_{n}/d{d}")

            for pkl_file in folder.glob("*.pkl"):
                try:
                    with open(pkl_file, "rb") as f:
                        arr = pickle.load(f)

                    # Validate shape (optional, remove if not needed)
                    if not isinstance(arr, np.ndarray) or arr.shape[0] != int(n):
                        print(f"Invalid shape in {pkl_file}, skipping.")
                        continue

                    group.create_dataset(
                        pkl_file.stem, data=arr, compression="gzip"
                    )
                    print(f"added {pkl_file}")
                except Exception as e:
                    print(f"Error processing {pkl_file}: {e}")

    outfile.close()
    print(f"All data written to {output_path}")

def unpack_data(h5_path=STORAGE_H5, target_dir=config.DATA):
    target_dir = Path(target_dir)
    if not target_dir.exists():
        os.makedirs(target_dir, exist_ok=True)
    with h5py.File(h5_path, "r") as h5f:
        for n_group in h5f:
            if n_group == "metadata":  # skip optional metadata group
                continue
            n_val = n_group[2:]  # strip 'n_'
            for d_group in h5f[n_group]:
                d_val = d_group  # e.g. 'd0', 'd1', etc.
                group = h5f[n_group][d_group]
                for uuid in group:
                    arr = group[uuid][()]

                    out_path = target_dir / n_val / d_val
                    out_path.mkdir(parents=True, exist_ok=True)
                    out_file = out_path / f"{uuid}.pkl"

                    with open(out_file, "wb") as f:
                        pickle.dump(arr, f, protocol=4)
                        print(f"unpacked {out_file}.")

    print(f"Extraction complete to {target_dir}")

# Example usage:
# extract_h5_to_pkls("all_simulations.h5", "restored_simulations")

if __name__ == "__main__":
    pack_data()
    input("Continue? (press any key)")
    unpack_data(target_dir = Path(config.DATA) / "trial")
