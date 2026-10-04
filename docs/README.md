# Interactive dashboard (optional)

`index.html` is a lecture-style interactive reading of the paper in eight parts: the puzzle, traffic states, the two-speed identity, demand response, finite transitions, the data, the evidence and where the results apply. Each part states whether it shows an illustration, a frozen output or observed data, and links to the matching tutorial or example. At their default values, the identity sliders reproduce Table F1.

- **Open locally:** run `python -m http.server 8765 --directory docs` and browse to `http://localhost:8765`. Opening the file directly also works in most browsers.
- **GitHub Pages:** set Pages to serve the `docs/` folder of `main`. Pages for private repositories depend on the account's plan.
- **Data:** `data/qvdfe_data.js` and `.json` are written by `python tools/build_dashboard_data.py` from the frozen files.
- **Libraries:** Plotly and KaTeX are loaded from public CDNs.

The dashboard is not part of the acceptance checks. The tutorials, examples, tests and `reproduce/` are.
