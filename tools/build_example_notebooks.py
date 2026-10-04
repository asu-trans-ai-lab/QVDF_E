"""Write and execute one short notebook per example (runs the example script and shows its figure and results)."""
from pathlib import Path
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / 'examples'
INTRO = {
    '01_synthetic_episode': 'Synthetic values (edit `config.json`). One closed bottleneck episode from queue to emissions: queued state, conservation, finite-link condition, and the two equivalent emission forms.',
    '02_weekdays_vs_average': 'Real data (paper Fig. 7). Detector 78, November 2016: twelve complete weekdays beside the average-weekday profile built from exactly those days. The averaged-profile episode is not the average of the daily episodes.',
    '03_long_episode_applicability': 'Real data (paper Appendix F, Table F2). Baseline feasibility (Eq. 9) and common-transition feasibility (Eq. 26) are different conditions; the admitted subset uses its own D_A and mean delay in Eq. (27). The output is the admitted-subset speed-only emissions.',
}


def build(name):
    folder = EX / name
    nb = new_notebook()
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nb.cells = [
        new_markdown_cell(f'# Example {name}\n\n{INTRO[name]}\n\nThis notebook runs `run.py` in this folder and shows its outputs. Inputs are in `config.json`; expected values are in `expected/results.json`.'),
        new_code_cell("import json, runpy, sys\nfrom pathlib import Path\nfrom IPython.display import Image, display\nHERE = Path.cwd()\ntry:\n    runpy.run_path(str(HERE / 'run.py'), run_name='__main__')\nexcept SystemExit as e:\n    print('exit status', e.code)"),
        new_code_cell("display(Image(filename=str(HERE / 'figure.png')))"),
        new_code_cell("res = json.loads((HERE / 'results.json').read_text())\nprint(json.dumps({k: v for k, v in res.items() if k != 'provenance'}, indent=1)[:3000])\nprint('provenance:', res['provenance'])"),
    ]
    p = folder / f'{name}.ipynb'
    nbformat.write(nb, p)
    from nbclient import NotebookClient
    NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(folder)}}).execute()
    nbformat.write(nb, p)
    print('executed', p.relative_to(ROOT))


if __name__ == '__main__':
    for n in INTRO:
        build(n)
