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
uv run python -m pipeline.validate      # validation report for the real table, written to data/results/validation.json
uv run python -m pipeline.make_dummy    # regenerate the dummy table (about 50 cells)
uv run python -m pipeline.check_sites   # read the ten Moon and Mars test points and compare them with published values
uv run python -m pipeline.contract      # validate config and export JSON Schemas to data/schemas
uv run pytest                           # tests, no network needed
uv run ruff check .                     # linter
```

## Access, gaps and portfolio

- **Access.** Each cell gets `road_km`, `airport_km` and an `access` score from 0 to 1. Half comes from the distance to a road and half from the distance to an airport, each falling to 0 at the far distance set in `data/config/run.json`. Access never changes the analog score. Protected areas are not included yet, because Natural Earth has no global layer for them. A table built before this needs one more `uv run python -m pipeline.run`; until then the API returns `null` for access.
- **Gaps.** A documented analog covers a site condition when the similarity at the analog's cell reaches `cover_threshold` (0.8). `covered: null` means it cannot be judged: the site has no value, the Earth has no source, or no analog lies in the table.
- **Portfolio.** A greedy set cover over the features with weight above 0. It takes the cell that covers the most conditions still open, until all are covered or nothing more can be.

## Validation, ranges and sources

- **Validation.** `pipeline/validate.py` runs the tests of spec section 8 for the preset sites, each with its default purpose: where each positive control ranks among the eligible cells (it passes inside the top 20%), whether each negative control falls in the lower half of all land cells with masks and gates off, how far the controls move when one feature is dropped, how many of the top 50 cells stay when a weight moves by 20%, and the screening funnel. It also compares the arithmetic and geometric mean, so the team can keep the one that ranks the controls better. Weak results go in `honest_gaps` instead of being tuned away. The novelty and findings steps stay `null` until those features exist. `GET /validation` builds the same report for the table the API has loaded.
- **Ranges.** `manifest.json` (and `/meta`) carries `feature_ranges`: percentiles 1 and 99 of each numeric feature over all land cells and over the eligible cells, next to its `lo` and `hi`. When `lo` to `hi` is much wider than the data, the feature barely tells cells apart, and the range in `features.json` should be narrowed. A table built before this needs one more `uv run python -m pipeline.run` (cached, so it is quick).
- **Sources.** `data/config/sources.json` is the team's source inventory: which source fills each feature on each body, its license, whether the organizers list it, and whether the pipeline reads it today. `GET /sources` serves it. `verified` stays `null` until a team member checks the source.
- **MODIS sampling.** `modis` in `run.json` sets which months are read (one composite each) and the share of valid composites a cell needs to keep a value (0.2). Adding months gives a truer yearly maximum NDVI, at the cost of more downloads.

## Moon and Mars composition

No global map gives one of the four classes directly, so a location gets its class from proposed rules over mapped data. The rules are in `data/config/planet_rules.json`; the reasoning, citations, limits and open decisions are in `data/reference/reglas_luna_marte.json`. Every class from a rule has basis `literature`.

- **Moon.** Inside a mapped mare ([LROC outlines](https://data.lroc.im-ldi.com/lroc/view_rdr/SHAPEFILE_LROC_GLOBAL_MARE), 8.7 MB) is basaltic. Otherwise, inside the South Pole-Aitken iron ellipse (Garrick-Bethell and Zuber 2009) with Lunar Prospector FeO of 8 wt% or more is basaltic, as mafic non-mare material. Anything else is felsic, the nearest class to anorthositic highlands. Never sedimentary or evaporite.
- **Mars.** The [SIM 3292](https://doi.org/10.3133/sim3292) unit at the point, translated with the table of 44 units. Only the global layer (about 55 MB) is pulled out of the 809 MB ZIP, by HTTP range requests. A 0.5 degree cell holding a chloride deposit ([Osterloo et al. 2010](https://data.caltech.edu/records/kasms-vp209)) becomes evaporite. Polar ice units leave composition missing.
- **Moon annual range.** Below 80 degrees of latitude it is 0 K, basis `physical`: the spin axis tilts 1.54 degrees. Poleward of 80 degrees it stays blank for map sites until the Diviner seasonal polar product is read; the Shackleton preset carries the measured value.

`pipeline.check_sites` reads the ten test points in `data/reference/puntos_prueba.json` (landers and rovers) and writes `data/results/site_checks.json`. The rule reads all ten as expected; seven also match what was found on the ground. Gale (lake mudstone in an impact unit), Meridiani (sulfate evaporites in HNhu) and InSight (basalt under regolith in eHt) disagree, and stay listed instead of tuned away.

**Caves.** `data/config/cave_sites.json` lists documented lava tubes. The nearest Earth cell to each, within `analog_match_km`, gets the terrain tag `cave`, so `terrain=cave` finds them. The tag does not make a masked cell eligible.

## Values beyond Earth's range

A Moon or Mars value outside the Earth range (a day-night swing of 125 K, a mean of -118 °C) used to be clamped to the edge, so the most extreme Earth cells all matched it equally and the feature stopped telling places apart while still carrying weight. By default (`beyond_earth: "exclude"` in `run.json`) such a feature now leaves the score: it weighs 0, does not lower confidence, and is listed in `/results` as `excluded_beyond_earth`, as a condition to test in a chamber rather than in the field. `beyond=clamp` on any scoring endpoint brings back the old behavior. On the real table the positive controls that pass go from 3 of 9 (clamp) to 5 of 9 (exclude), with every negative control still in the lower half; `validation.json` reports both modes under `beyond_earth_modes`. Arcadia is the exception: its ice-and-water purpose rests on temperatures that are all beyond Earth, so it ranks only on composition and aridity and Haughton drops.

## Fieldwork restrictions

`data/config/access_restrictions.json` lists places where fieldwork is unsafe or limited, each with a level (`do_not_travel`, `restricted`, `permit`), the countries (and box) it covers, and legal notes with their source and the kind of source (official, news, secondary). It was reviewed on the web on 2026-11-15 and is informational only, not legal advice. Every cell, sheet and finding carries its level; `avoid=do_not_travel,restricted` on `/results`, `/summary`, `/export` and `/portfolio` leaves those places out. The label never changes a score. `GET /restrictions` serves the zones and the general notes (samples, Nagoya Protocol, sanctions by nationality).

The team's proposals live in `data/reference/access_restrictions.propuesto.json` and `environmental_notes.revisado.json` (the reviewed version of `environmental_notes.propuesto.json`). The config copies differ in two ways: zones the proposal left without a box but that cover only part of a country (BLM Utah and Dumont Dunes, Río Tinto and Tabernas, Teide and Timanfaya) are split into boxed zones, so they no longer label all of the United States or Spain; and Dallol also lists Ethiopia, Licancabur also lists Chile.

## Environmental notes

`data/config/environmental_notes.json` holds environmental notes for every analog site and finding: protected areas, fragile features, wildlife and breeding seasons, season windows, climate hazards, impact rules and pressure on the site, each with its source and a `verification_status` (`verificado`, `oficial_sin_reabrir`, `secundario`, or `pendiente`, which the app shows faded as "do not act on it"). An area applies to the sites in `applies_to` and to every cell inside its box (widened by half a cell, since some boxes are a single point) in one of its countries. Cell sheets and findings carry the notes, list rows and the CSV carry the number of areas, and `GET /analogs` lists each site's areas. A new candidate with no area shows a checklist of what to review before fieldwork. Like restrictions, the notes never change a score. `GET /environment` serves the whole file.

## Findings (FR-P16)

`data/config/findings.json` holds the team's findings for the preset sites: new regions from the top 1% of cells, grouped and checked in the literature (13 regions on 2026-11-15, after 2 rounds and 21 regions reviewed). Each one keeps its cells, box, what to test first, the sources read, the searches run with their result, and an honest limit. The originals, in Spanish, are in `data/reference/findings.propuesto.json`.

The app recomputes score, confidence and the best and worst matches from the loaded table, so a finding never shows a number the table no longer supports. `GET /findings` serves them, `pipeline.validate` writes `data/results/findings.json` (spec 6.5), and the validation report audits them in `findings_audit`: findings without sources, below medium confidence, near a known analog, with cells outside their box, or whose score changed since the review.

## Search in plain words (FR-AI2)

`GET /ask` sends a request such as "places in South America like Jezero but colder" to a free language model and gets back the settings of a search: a site, a purpose, one feature to stress, a region, terrain classes and a minimum confidence. `GET /summary` returns the facts about the five best results and, when a model is set up, a short answer written from them.

The model never sees or changes a score. Its reply is checked against the config: unknown sites, purposes, features or regions are dropped and reported in `not_applied`. Coordinates count only when the person typed them. A summary that quotes a number missing from the facts is thrown away, and the app writes its own sentence from the facts instead. A comparison such as "colder" gives that feature more importance; it is not a hard filter.

Set one of these before starting the API. Without any, `/meta` reports `ai.available: false`, `/ask` answers 503 and the app works as before.

| Variable | What it does |
|---|---|
| `GEMINI_API_KEY` | Use Gemini (free key from Google AI Studio). The default provider |
| `GROQ_API_KEY` | Use Groq |
| `LUNAVI_LLM_PROVIDER` | `gemini`, `groq` or `ollama`. Needed for Ollama, which has no key and runs on `http://localhost:11434` |
| `LUNAVI_LLM_MODEL` | Another model name. Defaults: `gemini-flash-lite-latest`, `llama-3.3-70b-versatile`, `llama3.2` |
| `LUNAVI_LLM_BASE_URL` | Another address that speaks the OpenAI chat format |

```powershell
$env:GEMINI_API_KEY = "your key"
uv run uvicorn api.main:app --reload
```

Never commit a key. If a provider renames its models, set `LUNAVI_LLM_MODEL`. Free plans allow only a few requests per minute, and the larger Gemini models are often busy or out of free quota, which is why the default is the small one. Summaries are cached, and after a failed summary the API waits a minute before asking the model for another, so questions keep working; the app writes its own answer meanwhile.

## Add a data source

Missing values are normal here. A value with no source stays blank, is left out of the score, and lowers confidence, so the app works before every source exists.

**Earth.** Every feature has one adapter: a function that takes a `Context` and returns a `Layer` (value grid and gap grid). To use data from the event:

1. Add a function in `pipeline/adapters/`, such as `pipeline/adapters/event.py:slope`.
2. Point that feature's `adapter` in `data/config/features.json` at it, and update its source, license and quality values. `"adapter": null` means no source yet.
3. Run `uv run python -m pipeline.run`. Delete that feature's file in `data/derived` first if it already exists.

Composition comes from GLiM gridded to 0.5 degrees (`pipeline/adapters/glim.py`), a 38 KB download with no account. Each cell takes its most common class, and the share outside that class counts as a gap, so a mixed cell has lower confidence. `class_map` in `features.json` maps the 16 GLiM classes to the four LUNAVI classes; it is proposed and the team should confirm it. To leave composition blank again, set its `adapter` to `null`. `pipeline/adapters/lithology.py` is another option using Macrostrat, but it needs about 88,000 requests for the whole Earth.

**Moon and Mars.** Add a reader in `pipeline/sites.py` and call it from `SiteBook.resolve`. Delete `data/derived/sites` afterward so that locations are read again.

## API

| Endpoint | What it does |
|---|---|
| `GET /targets` | Preset sites |
| `GET /sites/resolve?body=mars&lat=18.44&lon=77.45` | Read any location. The first read takes seconds, then it is cached |
| `GET /sites/{id}/climate` | Finish a location's slow temperature query (Mars) |
| `GET /results?target={id}&profile=&weights=&bbox=&terrain=&top_n=&min_confidence=&min_separation_km=&detail=` | Rank Earth cells against a site |
| `GET /cells/{cell_id}?target={id}` | Explain one Earth cell |
| `GET /export?format=csv&target={id}&top_n=20` | The shortlist as a CSV file, with the same filters as `/results` |
| `GET /gaps?target={id}&threshold=` | For each condition of a site, whether a documented analog covers it |
| `GET /portfolio?target={id}&profile=&weights=&bbox=&terrain=&min_confidence=&threshold=&max_cells=` | The fewest cells that together cover the site's conditions |
| `GET /ask?text=&lang=&target=&profile=&emphasis=&region=&terrain=&min_confidence=` | Turn a request in plain words into search settings. The other parameters are the current search |
| `GET /summary?target={id}&lang=` plus the `/results` filters | Facts about the five best results, and a short answer when a model is set up |
| `GET /validation` | How the documented analogs and the negative controls rank for the preset sites |
| `GET /sources` | The data source inventory, with license and status |
| `GET /findings?target=` | The team's findings with live score, confidence and checks |
| `GET /restrictions` | Fieldwork restriction zones with their legal notes |
| `GET /environment` | Environmental notes per analog site and finding |
| `GET /caves` | Documented lava tubes and caves, and whether a cell of the table holds each |
| `GET /profiles`, `/features`, `/analogs`, `/meta`, `/health` | Config and status |

## Layout

| Path | What it holds |
|---|---|
| `pipeline/scoring.py` | The formulas from spec section 5, as pure functions |
| `pipeline/adapters/` | One adapter per Earth feature |
| `pipeline/sites.py` | Reads any Moon or Mars location |
| `pipeline/planet_rules.py` | Composition class on the Moon and Mars, and the Moon's annual range |
| `pipeline/shapefile.py` | A small polygon shapefile reader, and reading one file out of a remote ZIP |
| `pipeline/restrictions.py` | Which restriction zones hold each cell |
| `pipeline/environment.py` | Which environmental areas hold each site and cell |
| `pipeline/findings.py` | Recomputes and audits the findings |
| `pipeline/check_sites.py` | Compares the site reader with the team's test points |
| `pipeline/earth.py` | Loads the Earth table and compares a site with it |
| `pipeline/access.py` | Distance to roads and airports, and the access score per cell |
| `pipeline/contract.py` | The data contract as Pydantic models |
| `pipeline/run.py` | The one command that builds the Earth table |
| `pipeline/validate.py` | The validation report from spec section 8 |
| `api/` | FastAPI app |
| `api/assistant.py` | The language model client, its prompts and the checks on its replies |
| `data/config/` | Preset sites, features, profiles, documented analogs, masks, sources and run settings |
| `data/reference/` | The team's rule analysis and test points, as received (Spanish) |
| `data/dummy/results/` | A dummy Earth table in the real format |
| `data/results/` | The real Earth table (git-ignored) |

The spec describes one repository with `pipeline/`, `api/`, `web/` and `data/`. The team uses two, so `web/` lives in lunavi-front.

## Data sources

| Data | Source | License |
|---|---|---|
| Earth slope, roughness, elevation | [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/) | Open data, attribution required |
| Earth surface temperature, precipitation | [NASA POWER](https://power.larc.nasa.gov/) | NASA open data |
| Earth NDVI, day-night temperature | MODIS [MOD13A1](https://lpdaac.usgs.gov/products/mod13a1v061/) and [MYD11A2](https://lpdaac.usgs.gov/products/myd11a2v061/), read through [Planetary Computer](https://planetarycomputer.microsoft.com/) | NASA open data |
| Earth composition | [GLiM v1.0 at 0.5 degrees](https://doi.org/10.1594/PANGAEA.788537) (Hartmann and Moosdorf 2012) | CC BY 3.0 |
| Land mask, countries | [Natural Earth](https://www.naturalearthdata.com/) | Public domain |
| Roads and airports, for the access score | [Natural Earth](https://www.naturalearthdata.com/) | Public domain |
| Holocene volcanoes | [Smithsonian GVP](https://volcano.si.edu/) | Cite the GVP |
| Mars terrain | [MOLA 463 m](https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif) | NASA and USGS open data |
| Mars temperature | [Mars Climate Database v6.2](https://www-mars.lmd.jussieu.fr/mcd_python/) | Cite the MCD (a model, so basis is literature) |
| Moon terrain | [LOLA 118 m](https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif) and [LOLA south polar](https://pgda.gsfc.nasa.gov/products/90) | NASA open data |
| Moon composition | [LROC mare outlines](https://data.lroc.im-ldi.com/lroc/view_rdr/SHAPEFILE_LROC_GLOBAL_MARE) and [Lunar Prospector FeO](https://pds-geosciences.wustl.edu/missions/lunarp/reduced_special.html) | NASA open data |
| Mars composition | [USGS SIM 3292](https://doi.org/10.3133/sim3292) and [Osterloo chlorides](https://data.caltech.edu/records/kasms-vp209) | Public domain; CC0 |
| Moon temperature | [Diviner global table](https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_gcp/) | NASA PDS open data |
