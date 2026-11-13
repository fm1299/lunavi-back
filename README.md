# LUNAVI back end

Pipeline and read-only API for LUNAVI. Choose any location on the Moon or Mars, and LUNAVI ranks every land cell on Earth by how closely it matches. The web app is in [lunavi-front](https://github.com/fm1299/lunavi-front).

- **Offline:** the pipeline stores one table of Earth values.
- **On request:** the API reads the chosen Moon or Mars location from open data, then compares it with the Earth table.

This branch is the base. Validation, findings, novelty labels and cave tags are on the `full-prototype` branch, ready to bring back.

## Run it

You need [uv](https://docs.astral.sh/uv/). It installs Python 3.12 and the dependencies for you.

```powershell
uv sync
uv run uvicorn api.main:app --reload    # API on http://localhost:8000, docs at /docs
```

Without real results, the API serves the dummy table in `data/dummy/results` and says so in `/health`.

To put it online on one Oracle Cloud Always Free server, see [deploy/README.md](deploy/README.md).

## Build the Earth table

```powershell
uv run python -m pipeline.run                                 # the whole Earth at 0.5 degrees
uv run python -m pipeline.run --bbox -30 -15 -75 -65          # a small area, for a quick check (about a minute)
```

No account or API key is needed. Downloads are cached in `data/raw` and `data/derived`, so a stopped run resumes where it left off. A first full run downloads about 3 GB. To change the grid, edit `data/config/run.json`.

Other commands:

```powershell
uv run python -m pipeline.make_dummy    # regenerate the dummy table (about 50 cells)
uv run python -m pipeline.contract      # validate config and export JSON Schemas to data/schemas
uv run pytest                           # tests, no network needed
uv run ruff check .                     # linter
```

## Add a data source

Missing values are normal here. A value with no source stays blank, is left out of the score, and lowers confidence, so the app works before every source exists.

**Earth.** Every feature has one adapter: a function that takes a `Context` and returns a `Layer` (value grid and gap grid). To use data from the event:

1. Add a function in `pipeline/adapters/`, such as `pipeline/adapters/event.py:slope`.
2. Point that feature's `adapter` in `data/config/features.json` at it, and update its source, license and quality values. `"adapter": null` means no source yet.
3. Run `uv run python -m pipeline.run`. Delete that feature's file in `data/derived` first if it already exists.

Composition has no source yet. `pipeline/adapters/lithology.py` is a ready option using Macrostrat, but it needs about 88,000 requests for the whole Earth.

**Moon and Mars.** Add a reader in `pipeline/sites.py` and call it from `SiteBook.resolve`. Delete `data/derived/sites` afterward so that locations are read again.

## API

| Endpoint | What it does |
|---|---|
| `GET /targets` | Preset sites |
| `GET /sites/resolve?body=mars&lat=18.44&lon=77.45` | Read any location. The first read takes seconds, then it is cached |
| `GET /sites/{id}/climate` | Finish a location's slow temperature query (Mars) |
| `GET /results?target={id}&profile=&weights=&bbox=&terrain=&top_n=&min_confidence=&min_separation_km=&detail=` | Rank Earth cells against a site |
| `GET /cells/{cell_id}?target={id}` | Explain one Earth cell |
| `GET /profiles`, `/features`, `/analogs`, `/meta`, `/health` | Config and status |

## Layout

| Path | What it holds |
|---|---|
| `pipeline/scoring.py` | The formulas from spec section 5, as pure functions |
| `pipeline/adapters/` | One adapter per Earth feature |
| `pipeline/sites.py` | Reads any Moon or Mars location |
| `pipeline/earth.py` | Loads the Earth table and compares a site with it |
| `pipeline/contract.py` | The data contract as Pydantic models |
| `pipeline/run.py` | The one command that builds the Earth table |
| `api/` | FastAPI app |
| `data/config/` | Preset sites, features, profiles, documented analogs, masks and run settings |
| `data/dummy/results/` | A dummy Earth table in the real format |
| `data/results/` | The real Earth table (git-ignored) |

The spec describes one repository with `pipeline/`, `api/`, `web/` and `data/`. The team uses two, so `web/` lives in lunavi-front.

## Data sources

| Data | Source | License |
|---|---|---|
| Earth slope, roughness, elevation | [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/) | Open data, attribution required |
| Earth surface temperature, precipitation | [NASA POWER](https://power.larc.nasa.gov/) | NASA open data |
| Earth NDVI, day-night temperature | MODIS MOD13A1 and MYD11A2 through [Planetary Computer](https://planetarycomputer.microsoft.com/) | NASA open data |
| Land mask, countries | [Natural Earth](https://www.naturalearthdata.com/) | Public domain |
| Holocene volcanoes | [Smithsonian GVP](https://volcano.si.edu/) | Cite the GVP |
| Mars terrain | [MOLA 463 m](https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif) | NASA and USGS open data |
| Mars temperature | [Mars Climate Database v6.2](https://www-mars.lmd.jussieu.fr/mcd_python/) | Cite the MCD (a model, so basis is literature) |
| Moon terrain | [LOLA 118 m](https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif) and [LOLA south polar](https://pgda.gsfc.nasa.gov/products/90) | NASA open data |
| Moon temperature | [Diviner global table](https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_gcp/) | NASA PDS open data |
