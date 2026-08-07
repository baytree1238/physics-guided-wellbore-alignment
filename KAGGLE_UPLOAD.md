# Publishing the notebook on Kaggle

`portfolio_notebook_executed.ipynb` is the file to import as the Kaggle
notebook. Its saved output makes it readable immediately, but the notebook is
not self-contained: the code cells import the model package and read saved
experiment evidence.

Build the small companion archive with:

```bash
make kaggle-bundle
```

This creates:

```text
dist/rogii_portfolio_companion.zip
dist/dataset-metadata.json
```

Create a Kaggle Dataset from the two files in `dist/`, then attach that dataset
to the notebook. The first notebook cell finds the archive under
`/kaggle/input`, extracts it into `/kaggle/working`, and checks the saved
artifact hashes before displaying results.

For a private draft, leaving the companion dataset private is fine. Before
publishing the notebook, make the companion dataset public as well; otherwise
other readers can see saved outputs but cannot rerun the code. The official
Kaggle CLI also supports creating a dataset from a folder containing
`dataset-metadata.json`:

```bash
kaggle datasets create -p dist
```

The exact historical 9.091 submission files are not included. The companion
contains the reimplemented source, the audited historical feature formula and
the evidence shown by the notebook. It also omits the local verification report
and rendered HTML because those files contain machine-specific paths.

The repository code is MIT licensed. Kaggle's dataset metadata uses the broader
`other` category because that is the CLI's supported license identifier for
this case; the included `LICENSE` file remains authoritative.
