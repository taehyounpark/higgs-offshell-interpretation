import argparse
import numpy as np

from physics.simulation import mcfm

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Merge MCFM event CSVs from multiple processes.")
    parser.add_argument('analyzed_files', type=str, help='List of process names to merge')

    args = parser.parse_args()

    events = mcfm.from_csv(file_path=args.analyzed_files, ignore_negative_weights=False)

    sumw2 = np.sum(np.square(events.weights))
    sumw = events.weights.sum()

    print(f"Cross section: {sumw:.6f} ± {np.sqrt(sumw2):.6f}")
    print(f"Number of events: {len(events.weights)}")