# Teaching datasets

This directory contains only the versioned data required by the published
heliophysics notebooks. `helio_data_methods.fetch_dataset()` resolves these
files locally or downloads them individually from the repository, then verifies
their SHA-256 checksums before use.

| Dataset ID | Purpose and provenance |
| --- | --- |
| `dst-omni-2010-2015` | Hourly inputs for the Dst example, obtained from [NASA OMNIWeb](https://omniweb.gsfc.nasa.gov/) |
| `sep-curated` | Anonymous saved train/test arrays used by the [SEP forecasting case study](https://www.swsc-journal.org/articles/swsc/full_html/2021/01/swsc210024/swsc210024.html) |
| `coronal-loops` | Arrays used by the [coronal-loop reconstruction example](https://iopscience.iop.org/article/10.3847/2041-8213/abed53) |
| `plasma-sheet-prime-tm03` | Compact saved results for the plasma-sheet demonstration; the detailed manifest is stored beside the bundle |

Dataset-specific provenance and scientific limitations are documented on the
corresponding Jupyter Book pages. These files are included to reproduce the
tutorials; consult the original data providers and studies for terms governing
reuse outside this project.
