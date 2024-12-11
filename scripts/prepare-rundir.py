import argparse
import os
import shutil
import json

class ParseOverrides(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            try:
                key, value = value.split('=')
                getattr(namespace, self.dest)[key] = value
            except ValueError:
                getattr(namespace, self.dest)[value] = True

def parse_arguments():
    parser = argparse.ArgumentParser(description='Python script for preparing the run directory of a SLURM job.')
    parser.add_argument('run_dir', type=str, help='Run directory for the SLURM job.')
    parser.add_argument('run_script', type=str, help='The script to copy into the run dir.')
    parser.add_argument('config', type=str, help='Config file (defaults) for the job.')
    parser.add_argument('--overrides', nargs='*', action=ParseOverrides, help='Arguments in the config to be overridden. Arguments supplied as argument=value. This has to come last.')

    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = ''.join(config_file.readlines())
    return json.loads(config)

def save_config(config, dest_path):
    with open(dest_path, 'w') as config_file:
        config_file.write(json.dumps(config, indent=4))

def main(args):
    os.makedirs(args.run_dir, exist_ok=True)
    shutil.copy(args.run_script, args.run_dir)

    config = load_config(args.config)

    if args.overrides is not None:
        for key in args.overrides:
            if key in config[0]:
                try:
                    config[0][key] = json.loads(args.overrides[key])
                except json.decoder.JSONDecodeError:
                    config[0][key] = str(args.overrides[key])
            else:
                config[0]['flags'].append(key)

    save_config(config, os.path.join(args.run_dir, 'job.cfg'))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)