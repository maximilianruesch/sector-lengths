import os
import json
import hydra
import codecs
import numpy as np
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../conf", config_name=".config.yaml")
def run(config: DictConfig):
    base_dir = os.path.join(os.path.abspath('../'), config.analyze.directory)

    target = config.analyze.target
    lengths = {
        "pure": {
            "complex": [],
            "real": [],
        },
        "symmetric": {
            "complex": [],
            "real": [],
        },
    }
    for file_path in os.listdir(base_dir):
        relative_file_path = os.path.join(base_dir, file_path)
        if not os.path.isfile(relative_file_path) or not file_path.endswith('.json'):
            continue

        with codecs.open(relative_file_path, 'r', encoding='utf-8', errors='ignore') as file_handle:
            data = json.loads(file_handle.read())

            if not str(target) in data['sector_length']:
                continue

            name_parts = file_path.removesuffix('.json').removeprefix('symm_').split('_')
            state_type = "symmetric" if file_path.startswith('symm') else "pure"
            lengths[state_type][name_parts[0]].append(data['sector_length'][str(target)])

    for state_type in lengths:
        for param_type in lengths[state_type]:
            values = lengths[state_type][param_type]
            prefix = f"[{state_type}][{param_type}]"
            print(f"{prefix} Runs analyzed: {len(values)}")

            if len(values) == 0:
                continue

            a = np.sort(values)
            b = np.ediff1d(a)
            c = np.where(np.array(b) >= config.analyze.tolerance, b, np.zeros_like(b))

            indexes = np.argwhere(c).flatten()
            before_index = 0
            avg_values, counts = [], []
            for index in indexes:
                avg_values.append(np.average(a[before_index:index + 1]))
                counts.append(index + 1 - before_index)
                before_index = index + 1
            avg_values.append(np.average(a[before_index:]))
            counts.append(len(a) - before_index)

            print(f"{prefix} Number of Bins: {len(avg_values)}")
            for i, avg_value, count in zip(range(len(avg_values)), avg_values, counts):
                print(f"{prefix}     Bin [{i}]: {count} x ~{avg_value}")
            print(f"{prefix} Maximal value: {np.max(values)}")
            print(f"{prefix} Minimal value: {np.min(values)}")

    return

if __name__ == '__main__':
    run()
