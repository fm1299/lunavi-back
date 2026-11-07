# LUNAVI back end

Pipeline and read-only API for LUNAVI, which finds and ranks Earth analogs of Moon and Mars sites. The web app is in [lunavi-front](https://github.com/fm1299/lunavi-front).

The pipeline computes similarities offline and writes files. The API loads those files and only does cheap weighted sums per request.

## Run it

You need [uv](https://docs.astral.sh/uv/). It installs Python 3.12 and the dependencies for you.

```powershell
uv sync
uv run uvicorn api.main:app --reload    # API on http://localhost:8000, docs at /docs
```

Without real results, the API serves the dummy files in `data/dummy/results` and says so in `/health`.

## Rebuild the results

```powershell
uv run python -m pipeline.run                     # one command, all results, about 5 minutes the first time
uv run python -m pipeline.run --skip composition  # leave a feature out as missing data
```

Downloads are cached in `data/raw` and `data/derived`, so reruns take seconds. No account or API key is needed. Delete `data/derived` to recompute from the raw files. To choose the region or grid, edit `data/config/run.json`.

Other commands:

```powershell
uv run python -m pipeline.make_dummy       # regenerate the dummy files (about 50 cells)
uv run python -m pipeline.contract         # validate config and export JSON Schemas to data/schemas
uv run python -m pipeline.measure_targets  # re-measure the Moon and Mars values in targets.json
uv run pytest                              # tests, no network needed
uv run ruff check .                        # linter
```

## Swap in the event data

Every feature has exactly one adapter: a function that takes a `Context` and returns a `Layer` (value grid and gap grid). To use the NASA or other agency data provided at the event:

1. Add a function in `pipeline/adapters/`, such as `pipeline/adapters/event.py:slope`.
2. Point that feature's `adapter` in `data/config/features.json` at it, and update its source, license and quality values.
3. Run `uv run python -m pipeline.run`.

Scoring, validation, the API and the web app stay as they are.

## Layout

| Path | What it holds |
|---|---|
| `pipeline/scoring.py` | The formulas from spec section 5, as pure functions |
| `pipeline/adapters/` | One adapter per feature |
| `pipeline/contract.py` | The data contract as Pydantic models |
| `pipeline/run.py` | The one command that rebuilds everything |
| `pipeline/validate.py` | The validation report (spec section 8) |
| `pipeline/findings.py` | Candidates for the literature check (spec 5.10) |
| `api/` | FastAPI app |
| `data/config/` | Targets, features, profiles, catalog, caves, masks and run settings |
| `data/dummy/results/` | Dummy results in the real format |
| `data/results/` | Real results (git-ignored, except the team's reviewed `findings.json`) |

The spec describes one repository with `pipeline/`, `api/`, `web/` and `data/`. The team uses two, so `web/` lives in lunavi-front.

## Findings are a people step

The pipeline writes candidates to `data/results/candidates.json`. The team runs the literature check, names the survivors and adds them to `data/results/findings.json`, which reruns never overwrite. Findings stay blocked in the app until the catalog has `min_catalog_size` cited sites.

## Data sources

| Data | Source | License |
|---|---|---|
| Slope, roughness, elevation | [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/) | Open data, attribution required |
| Surface temperature, precipitation | [NASA POWER](https://power.larc.nasa.gov/) | NASA open data |
| NDVI, day-night temperature | MODIS MOD13A1 and MYD11A2 through [Planetary Computer](https://planetarycomputer.microsoft.com/) | NASA open data |
| Lithology | [Macrostrat](https://macrostrat.org/) | CC-BY 4.0 |
| Land mask | [Natural Earth](https://www.naturalearthdata.com/) | Public domain |
| Holocene volcanoes | [Smithsonian GVP](https://volcano.si.edu/) | Cite the GVP |
| Mars terrain | [MOLA 463 m](https://astrogeology.usgs.gov/) | NASA and USGS open data |
| Moon terrain | [LOLA LDEM_60S_240MPP](https://pgda.gsfc.nasa.gov/products/90) | NASA open data |
| Moon temperature | [Diviner polar resource product](https://ode.rsl.wustl.edu/moon/pagehelp/Content/Missions_Instruments/LRO/DIVINER/PRP.htm) | NASA PDS open data |
| Mars temperature | [Mars Climate Database v6.2](https://www-mars.lmd.jussieu.fr/mcd_python/) | Cite the MCD (a model, so basis is literature) |
