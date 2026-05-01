import os
import json
import argparse
import numpy as np
import pandas as pd

from physics.simulation import mcfm
from physics.analysis import zz4l, zz2l2v, wwlvlv

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Merge MCFM event CSVs from multiple processes.")
    parser.add_argument('samples', type=str, help='List of process names to merge')

    parser.add_argument('--data-dir', required=False, default='data/', help='Physics normalized & analyzed events output')
    parser.add_argument('--xsec-json', required=False, default='xsec.json', help='List of process names to merge')
    parser.add_argument('--events-csv', required=False, default='events_*.csv', help='List of process names to merge')
    parser.add_argument('--analyzed-csv', required=False, default='analyzed.csv', help='Physics normalized & analyzed events output')

    parser.add_argument('--nnsr-dir', required=False, default='run/', help='Run directory for neural networks')
    parser.add_argument('--nnsr-cut', required=True, default=0.0, type=float, help='Signal region selection value')

    args = parser.parse_args()

    with open(args.samples, 'r') as f:
        samples = json.load(f)

    print('----------'*8)

    for analysis, processes in samples.items():

        if analysis == 'zz4l':
            analyze = zz4l.analyze
            features = zz4l.features
            nnsr_dir = os.path.join(args.nnsr_dir, 'h4l')
        elif analysis == 'zz2l2v':
            analyze = zz2l2v.analyze
            features = zz2l2v.features
            nnsr_dir = os.path.join(args.nnsr_dir, 'h2l2v')
        elif analysis == 'wwlvlv':
            analyze = wwlvlv.analyze 
            features = wwlvlv.features
            nnsr_dir = os.path.join(args.nnsr_dir, 'h2l2v')

        with open(os.path.join(args.data_dir, analysis, args.xsec_json), 'r') as f:
            xsec = json.load(f)

        for process in processes:

            print('Analysis :', os.path.join(analysis,process))

            events = mcfm.from_csv(cross_section=np.prod(xsec[process]),file_path=os.path.join(args.data_dir,analysis,process,args.events_csv),ignore_negative_weights=False)

            events_analyzed = os.path.join(args.data_dir, analysis, args.analyzed_csv)        

            events_analyzed = analyze(events)

            if args.nnsr_cut > 0.0:
                from nsbi import nnsr
                import torch
                from torch.utils.data import TensorDataset, DataLoader
                import lightning as L
                class SignalRegionCut():
                    def __init__(self, nnsr_min = 0.02, features = None, scaler = None, nnsr_model = None):
                        self.nnsr_min = nnsr_min
                        self.features = features
                        self.scaler = scaler
                        self.model = nnsr_model

                    def __call__(self, kinematics, components = None, weights = None, probabilities = None) -> np.array:
                        trainer = L.Trainer(accelerator='gpu', devices=1)
                        kinematics = kinematics[features]
                        X_nnsr = self.scaler.transform(kinematics.to_numpy())
                        dl_nnsr = DataLoader(TensorDataset(torch.tensor(X_nnsr, dtype=torch.float32)), batch_size=1024, num_workers=1) 
                        nnsr_dec = torch.cat(trainer.predict(self.model, dl_nnsr))
                        return nnsr_dec.numpy().flatten() > self.nnsr_min

                _, _, scaler, nnsr_model = nnsr.utils.load_results(nnsr_dir)
                events_analyzed = events_analyzed.filter(SignalRegionCut(nnsr_min=args.nnsr_cut, features=features, scaler=scaler, nnsr_model=nnsr_model))

                print('NN_SR  |', events_analyzed.weights.sum())

            # Combine dataframes, ignoring index
            df_analyzed = pd.concat([
                    events_analyzed.kinematics.reset_index(drop=True),
                    events_analyzed.components.reset_index(drop=True),
                    events_analyzed.weights.reset_index(drop=True)
                ],
                axis=1
            )

            output_dir = os.path.join(args.data_dir, analysis, process)
            os.makedirs(output_dir, exist_ok=True)
            df_analyzed.to_csv(os.path.join(output_dir, args.analyzed_csv), index=False)

            print('----------'*8)
