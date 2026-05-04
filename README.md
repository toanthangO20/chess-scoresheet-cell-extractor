# Chess Scoresheet Cell Extractor

This project extracts handwritten move cells from photos or scans of chess scoresheets.
It does not run OCR and it does not read the move text. Its only job is to crop the
move boxes and save them as individual image files.

## Folder Layout

```text
chess-scoresheet-cell-extractor/
  extract_cells.py
  inputs/
  outputs/
  requirements.txt
  README.md
```

Put scoresheet images in `inputs/`. Extracted cells are written to `outputs/`.

Supported input formats:

- `.png`
- `.jpg`
- `.jpeg`

## Setup

Open PowerShell in the project directory:

```powershell
cd E:\Master\20252\ComputerVision\Project\chess-scoresheet-cell-extractor
```

Create a virtual environment:

```powershell
py -3.13 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.11, 3.12, or 3.13 should work. The commands above use Python 3.13 because it
is available on the current Windows machine.

## Run

Copy your scoresheet images into `inputs/`, then run:

```powershell
python extract_cells.py
```

This reads from `inputs/` and writes to `outputs/`.

You can also pass paths explicitly:

```powershell
python extract_cells.py --input-dir inputs --output-dir outputs
```

## Output Order

The output follows the chess scoresheet move order:

1. Left half of the sheet first: moves 1 to 30.
2. Right half of the sheet next: moves 31 to 60.
3. For each move, White is saved before Black.

For a fully detected 120-cell sheet, file names look like this:

```text
001_0_cell_001_move_01_white.png
001_0_cell_002_move_01_black.png
001_0_cell_003_move_02_white.png
...
001_0_cell_059_move_30_white.png
001_0_cell_060_move_30_black.png
001_0_cell_061_move_31_white.png
001_0_cell_062_move_31_black.png
```

If the script detects fewer or more than 120 cells, it still saves them in detected
scoresheet order, but the names use only sequential cell numbers:

```text
001_0_cell_001.png
001_0_cell_002.png
```

## Useful Options

Save intermediate binary and grid-line images for debugging:

```powershell
python extract_cells.py --save-debug
```

Save only cells that appear to contain handwriting:

```powershell
python extract_cells.py --non-empty-only
```

The non-empty filter estimates the local paper background and keeps only pixels that
look like darker ink strokes relative to that background. It removes long grid-line
strokes from that ink mask, so cells with only the top/bottom table borders are
treated as blank while cells with handwriting on top of those borders can still be
kept.

Tune the handwriting filter:

```powershell
python extract_cells.py --non-empty-only --min-dark-ratio 0.018
```

Despite the option name, this now controls the minimum contrast-ink pixel ratio, not
an absolute dark-pixel ratio.

Trim the printed move-number column from White cells:

```powershell
python extract_cells.py --trim-number-column-ratio 0.28
```

Use `0` to disable this trim:

```powershell
python extract_cells.py --trim-number-column-ratio 0
```

By default, saved cells include a small amount of vertical padding above and below
the detected cell. This keeps handwriting that overlaps the top or bottom grid line:

```powershell
python extract_cells.py --cell-top-padding-ratio 0.15 --cell-bottom-padding-ratio 0.25
```

Use `0` to crop exactly to the detected cell bounds:

```powershell
python extract_cells.py --cell-top-padding-ratio 0 --cell-bottom-padding-ratio 0
```

## Troubleshooting

If `python extract_cells.py` says no images were found, check that images are directly
inside `inputs/`, not inside another nested folder.

If the script detects fewer than 120 cells, inspect the debug images:

```powershell
python extract_cells.py --save-debug
```

Debug files are saved in:

```text
outputs/_debug/
```

Low-quality photos, cropped sheet borders, shadows, or tilted pages can make some grid
lines hard to detect. In that case, try a clearer scan or crop the photo so the
scoresheet table fills most of the image.
