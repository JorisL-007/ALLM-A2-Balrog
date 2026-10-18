"""Build submission ZIPs from canonical source; maintain no second source tree."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
output = ROOT / 'deliverables/refactored_v1'
output.mkdir(parents=True, exist_ok=True)
common = [ROOT / 'README.md', ROOT / 'research/experiment_index.csv']
for folder in ('exercises', 'analysis', 'integration', 'report'):
    common += [p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
common += list((ROOT / 'research').glob('*.jsonl'))
common += list((ROOT / 'research/refactor_verification').glob('*'))
common += [p for p in (ROOT / 'research/cloud_model').rglob('*') if p.is_file()]
common += list((ROOT / 'archive/refactor_20261018').glob('*'))
for exercise in (1, 2, 3):
    evidence = list((ROOT / f'deliverables/Ass2-{exercise}/evidence').rglob('*'))
    evidence = [p for p in evidence if p.is_file()]
    with zipfile.ZipFile(output / f'Ass2-{exercise}.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(common + evidence)):
            if path.is_file():
                archive.write(path, path.relative_to(ROOT))
    print(output / f'Ass2-{exercise}.zip')
