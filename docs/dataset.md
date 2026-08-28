# Sakugabooru Dataset

Curated animation clips (sakuga) scraped from [Sakugabooru](https://www.sakugabooru.com) — community-selected examples of notable animation craft: smears, morphing, impact frames, effects animation, fluid motion, and more.

**Stats:** ~11.9k clips · ~156 GB · ~114 hours · 241 series · 25 technique tags. Higher Sakugabooru score = more notable animation; clips were downloaded best-first.

## Location (not versioned)

```
<workspace>/data/sakugabooru/
├── clips/<series>/            # {post_id}_s{score}.mp4 + {post_id}.json (tags, source, dimensions)
│   └── _state.json            # scraper resume state — do not delete
├── metadata/                  # metadata_{all,train,val,test}.csv + summary.json (built, series-split)
├── pairs/                     # (reserved) keyframe→clip windowed pairs from tools/build_pair_dataset.py
├── cf_cookies.json            # Cloudflare cookies for the scraper (never commit)
└── scrape-logs/
```

`<workspace>` is the directory containing this repo. All paths flow from `configs/paths.env`.

Naming quirks in the raw corpus (both harmless, both handled by the relative `video` paths): the series *.hack* lives in a dot-hidden `.hack_series/` directory, and the series *22/7* nests as `22/7_series/` because its name contains a slash.

## Metadata CSVs

Built by `python -m native_animation.data.build_metadata` (or `scripts/slurm/build_metadata.sbatch`). Columns: `video,prompt,series,tags,score,clip_id,width,height,source,split`. `video` is relative to `clips/`. The split is **by series** (seed 42) to prevent leakage: 10,632 train / 799 val / 355 test rows at last build. Prompts are templated as `"native animation, anime, <series>, <tag1>, …"`.

## Growing the dataset

- **More clips:** `tools/scrape_sakugabooru.py` (resumable via `_state.json`, dedupes by post id; needs `cf_cookies.json` from `tools/extract_cf_cookies.py`). Rebuild metadata afterward.
- **Windowed pairs:** `tools/build_pair_dataset.py` slides a window over each clip and emits keyframe JPG + short MP4 + `manifest.jsonl` into `pairs/`. Stride controls volume: 0.1 s ≈ 4.3M pairs / 1.7 TB; 0.5 s ≈ 860k / 340 GB; 1.0 s ≈ 430k / 170 GB (non-overlapping).
- **Benchmarks:** frozen evaluation sets belong in `<workspace>/data/benchmarks/` (reserved).
