# Chess Scoresheet Cell Extractor

This project extracts handwritten move cells from photos or scans of chess scoresheets.
It does not run OCR and it does not read the move text. Its only job is to crop the
move boxes and save them as individual image files.

## Folder Layout

```text
chess-scoresheet-cell-extractor/
  extract_cells.py
  kaggle_pipeline_demo.ipynb
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
python extract_cells.py 001_0.png
```

The only command-line argument is the image file name. The script reads that file from
`inputs/` and writes extracted cells to `outputs/<image-name>/`.
Each scoresheet is reconstructed as a fixed 30-row by 4-move-cell grid, so the output
contains 120 cell images before any optional non-empty-cell filtering.

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

If the contour detector finds fewer than 120 cells, the script reconstructs the
missing cells from the detected scoresheet grid and still saves 120 cells.

## Fixed Settings

The remaining parameters are fixed in `extract_cells.py`:

- Input directory: `inputs/`
- Output directory: `outputs/`
- Save only non-empty cells: `False`
- Minimum contrast-ink pixel ratio: `0.012`
- Trim printed move-number column ratio: `0.28`
- Cell top padding ratio: `0.15`
- Cell bottom padding ratio: `0.25`
- Save debug images: `False`

By default, saved cells include a small amount of vertical padding above and below
the detected cell. This keeps handwriting that overlaps the top or bottom grid line.

## Troubleshooting

If `python extract_cells.py 001_0.png` says the input image does not exist, check that
the image is directly inside `inputs/`, not inside another nested folder.

Low-quality photos, cropped sheet borders, shadows, or tilted pages can make some grid
lines hard to detect. In that case, try a clearer scan or crop the photo so the
scoresheet table fills most of the image.
