# Tools

| Script | Runs in this repository | Purpose |
| --- | --- | --- |
| `make_formula_code_map.py` | yes | Regenerate `FORMULA_CODE_MAP.md` from `paper_labels.json` and `formula_code_map.json`; fails on an unmapped label or an unresolved row |
| `build_example_notebooks.py` | yes | Re-execute the three example notebooks |
| `build_dashboard_data.py` | yes | Rewrite `docs/data/qvdfe_data.js` and `.json` from the frozen data |
| `make_example_extracts.py` | yes | Rewrite the small extract in `data/examples/` |

`teaching/build_tutorials.py` regenerates and executes the tutorials. `tests/test_reproduction.py` fails if the committed tutorials differ from their generator.
