from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)
IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg")
INK_CONTRAST_THRESHOLD = 20
MIN_INK_COMPONENT_AREA = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract handwritten move cells from chess scoresheet photos."
    )
    parser.add_argument(
        "--input-dir",
        default=Path("inputs"),
        type=Path,
        help="Directory containing scoresheet images. Defaults to inputs/.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("outputs"),
        type=Path,
        help="Directory where extracted cells are saved. Defaults to outputs/.",
    )
    parser.add_argument(
        "--non-empty-only",
        action="store_true",
        help="Save only cells that appear to contain handwriting.",
    )
    parser.add_argument(
        "--min-dark-ratio",
        default=0.012,
        type=float,
        help=(
            "Minimum contrast-ink pixel ratio used by --non-empty-only. "
            "Higher values filter more aggressively."
        ),
    )
    parser.add_argument(
        "--trim-number-column-ratio",
        default=0.28,
        type=float,
        help=(
            "Fraction trimmed from the left side of White cells to remove the "
            "printed move-number column. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--cell-border-trim-ratio",
        default=0.10,
        type=float,
        help=(
            "Fraction trimmed from the top and bottom of each detected cell to "
            "remove grid borders before saving/filtering. Use 0 to keep borders."
        ),
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save threshold and grid-line images under outputs/_debug/.",
    )
    return parser.parse_args()


def iter_image_paths(input_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        image_paths.extend(input_dir.glob(pattern))
    return sorted(image_paths)


def image_to_binary(gray_image: Image.Image) -> np.ndarray:
    gray = np.array(gray_image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def binary_to_grid_lines(
    binary_image: np.ndarray,
    horizontal_kernel_divisor: int = 40,
    vertical_kernel_divisor: int = 40,
) -> np.ndarray:
    inverted = cv2.bitwise_not(binary_image)

    horizontal_kernel_length = max(1, inverted.shape[1] // horizontal_kernel_divisor)
    vertical_kernel_length = max(1, inverted.shape[0] // vertical_kernel_divisor)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_kernel_length, 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, vertical_kernel_length)
    )

    horizontal_lines = cv2.erode(inverted, horizontal_kernel, iterations=1)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)

    vertical_lines = cv2.erode(inverted, vertical_kernel, iterations=1)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)

    return cv2.add(horizontal_lines, vertical_lines)


def contour_bounds(contour: np.ndarray) -> tuple[int, int, int, int]:
    points = np.asarray(contour).reshape(-1, 2)
    x_min = int(np.min(points[:, 0]))
    y_min = int(np.min(points[:, 1]))
    x_max = int(np.max(points[:, 0]))
    y_max = int(np.max(points[:, 1]))
    return x_min, y_min, x_max, y_max


def find_move_cell_contours(grid_lines: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(grid_lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rectangles: list[np.ndarray] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > 0:
                rectangles.append(approx.squeeze())

    if not rectangles:
        return []

    areas = [cv2.contourArea(contour) for contour in rectangles]
    median_area = float(np.median(areas))
    return [
        contour
        for contour in rectangles
        if 0.5 * median_area <= cv2.contourArea(contour) <= 1.5 * median_area
    ]


def assign_column_indices(boxes: list[dict]) -> None:
    if len(boxes) < 4:
        for box in boxes:
            box["column"] = None
        return

    boxes_by_x = sorted(boxes, key=lambda item: item["x_center"])
    gaps = [
        (boxes_by_x[index + 1]["x_center"] - boxes_by_x[index]["x_center"], index)
        for index in range(len(boxes_by_x) - 1)
    ]
    split_after = sorted(index for _, index in sorted(gaps, reverse=True)[:3])

    start = 0
    for column, split_index in enumerate([*split_after, len(boxes_by_x) - 1]):
        for box in boxes_by_x[start : split_index + 1]:
            box["column"] = column
        start = split_index + 1


def sort_cells_by_scoresheet_move_order(contours: list[np.ndarray]) -> list[dict]:
    boxes = []
    for contour in contours:
        x_min, y_min, x_max, y_max = contour_bounds(contour)
        width = x_max - x_min
        height = y_max - y_min
        boxes.append(
            {
                "contour": contour,
                "x_center": x_min + width / 2,
                "y_center": y_min + height / 2,
                "height": height,
            }
        )

    if not boxes:
        return []

    assign_column_indices(boxes)

    median_height = float(np.median([box["height"] for box in boxes]))
    row_tolerance = max(median_height * 0.65, 10.0)

    rows = []
    for box in sorted(boxes, key=lambda item: item["y_center"]):
        matching_row = None
        for row in rows:
            if abs(box["y_center"] - row["y_center"]) <= row_tolerance:
                matching_row = row
                break

        if matching_row is None:
            rows.append({"y_center": box["y_center"], "boxes": [box]})
        else:
            matching_row["boxes"].append(box)
            matching_row["y_center"] = float(
                np.mean([item["y_center"] for item in matching_row["boxes"]])
            )

    sorted_rows = sorted(rows, key=lambda item: item["y_center"])
    for row_index, row in enumerate(sorted_rows):
        for item in row["boxes"]:
            item["row"] = row_index

    sorted_cells = []
    for columns in ((0, 1), (2, 3)):
        for row in sorted_rows:
            row_cells = [
                item for item in row["boxes"] if item.get("column") in columns
            ]
            sorted_cells.extend(sorted(row_cells, key=lambda item: item["x_center"]))

    return sorted_cells


def crop_cell(
    gray_image: Image.Image,
    contour: np.ndarray,
    border_trim_ratio: float,
) -> Image.Image:
    x_min, y_min, x_max, y_max = contour_bounds(contour)
    width = x_max - x_min
    height = y_max - y_min

    trim_ratio = max(0.0, min(border_trim_ratio, 0.4))
    x_trim = max(1, int(width * 0.01)) if trim_ratio > 0 else 0
    y_trim = max(2, int(height * trim_ratio)) if trim_ratio > 0 else 0

    left = max(0, x_min + x_trim)
    top = max(0, y_min + y_trim)
    right = min(gray_image.width, x_max - x_trim)
    bottom = min(gray_image.height, y_max - y_trim)

    if right <= left or bottom <= top:
        left = max(0, x_min)
        top = max(0, y_min)
        right = min(gray_image.width, x_max)
        bottom = min(gray_image.height, y_max)

    return gray_image.crop((left, top, right, bottom))


def trim_printed_move_number_column(
    image: Image.Image,
    column: int | None,
    trim_ratio: float,
) -> Image.Image:
    if trim_ratio <= 0 or column not in (0, 2):
        return image

    width, height = image.size
    left = min(max(int(width * trim_ratio), 0), width - 1)
    return image.crop((left, 0, width, height))


def odd_kernel_size_at_most(value: float, maximum: int) -> int:
    maximum = max(1, maximum)
    if maximum % 2 == 0:
        maximum -= 1

    size = max(3, int(value))
    if size % 2 == 0:
        size += 1

    return min(size, maximum)


def remove_grid_line_pixels(ink_pixels: np.ndarray) -> np.ndarray:
    height, width = ink_pixels.shape
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (min(width, max(8, int(width * 0.45))), 1),
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, min(height, max(8, int(height * 0.60)))),
    )

    horizontal_lines = cv2.morphologyEx(
        ink_pixels,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )
    vertical_lines = cv2.morphologyEx(
        ink_pixels,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )
    grid_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
    grid_lines = cv2.dilate(
        grid_lines,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    return cv2.bitwise_and(ink_pixels, cv2.bitwise_not(grid_lines))


def remove_small_ink_components(
    ink_pixels: np.ndarray,
    min_component_area: int = MIN_INK_COMPONENT_AREA,
) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (ink_pixels > 0).astype(np.uint8),
        8,
    )
    filtered_pixels = np.zeros_like(ink_pixels)

    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            filtered_pixels[labels == component_index] = 255

    return filtered_pixels


def contrast_ink_pixels(gray_roi: np.ndarray) -> np.ndarray:
    height, width = gray_roi.shape
    kernel_size = odd_kernel_size_at_most(
        value=max(9, height * 0.45),
        maximum=min(height, width),
    )
    background_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    local_background = cv2.morphologyEx(
        gray_roi,
        cv2.MORPH_CLOSE,
        background_kernel,
    )
    dark_contrast = cv2.subtract(local_background, gray_roi)

    return (dark_contrast >= INK_CONTRAST_THRESHOLD).astype(np.uint8) * 255


def has_handwriting(image: Image.Image, min_dark_ratio: float) -> bool:
    gray = np.array(image.convert("L"))
    height, width = gray.shape

    top = int(height * 0.12)
    bottom = int(height * 0.88)
    left = int(width * 0.06)
    right = int(width * 0.94)
    inner = gray[top:bottom, left:right]

    if inner.size == 0:
        return False

    handwriting_pixels = contrast_ink_pixels(inner)
    handwriting_pixels = remove_grid_line_pixels(handwriting_pixels)
    handwriting_pixels = remove_small_ink_components(handwriting_pixels)

    return float(np.mean(handwriting_pixels > 0)) >= min_dark_ratio


def move_cell_name(image_stem: str, cell_index: int, cell_count: int) -> str:
    if cell_count != 120:
        return f"{image_stem}_cell_{cell_index + 1:03d}.png"

    if cell_index < 60:
        move_number = (cell_index // 2) + 1
    else:
        move_number = ((cell_index - 60) // 2) + 31

    side = "white" if cell_index % 2 == 0 else "black"
    return f"{image_stem}_cell_{cell_index + 1:03d}_move_{move_number:02d}_{side}.png"


def save_debug_images(
    output_dir: Path,
    image_stem: str,
    binary_image: np.ndarray,
    grid_lines: np.ndarray,
) -> None:
    debug_dir = output_dir / "_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary_image).save(debug_dir / f"{image_stem}_binary.png")
    Image.fromarray(grid_lines).save(debug_dir / f"{image_stem}_grid_lines.png")


def clear_previous_cell_images(image_output_dir: Path, image_stem: str) -> None:
    for image_path in image_output_dir.glob(f"{image_stem}_cell_*.png"):
        image_path.unlink()


def extract_cells_from_image(
    image_path: Path,
    output_dir: Path,
    non_empty_only: bool,
    min_dark_ratio: float,
    trim_number_column_ratio: float,
    cell_border_trim_ratio: float,
    save_debug: bool,
) -> int:
    image_output_dir = output_dir / image_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)
    clear_previous_cell_images(image_output_dir, image_path.stem)

    image = Image.open(image_path).convert("RGB")
    gray = image.convert("L")
    binary = image_to_binary(gray)
    grid_lines = binary_to_grid_lines(binary)

    if save_debug:
        save_debug_images(output_dir, image_path.stem, binary, grid_lines)

    contours = find_move_cell_contours(grid_lines)
    cells = sort_cells_by_scoresheet_move_order(contours)
    LOGGER.info("Detected %s move-cell contours in %s", len(cells), image_path.name)

    saved_count = 0
    for index, cell_info in enumerate(cells):
        cell = crop_cell(
            gray_image=gray,
            contour=cell_info["contour"],
            border_trim_ratio=cell_border_trim_ratio,
        )
        cell = trim_printed_move_number_column(
            image=cell,
            column=cell_info["column"],
            trim_ratio=trim_number_column_ratio,
        )

        if non_empty_only and not has_handwriting(cell, min_dark_ratio):
            continue

        file_name = move_cell_name(image_path.stem, index, len(cells))
        cell.save(image_output_dir / file_name)
        saved_count += 1

    return saved_count


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    image_paths = iter_image_paths(args.input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No PNG/JPG images found in: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for image_path in image_paths:
        saved_count = extract_cells_from_image(
            image_path=image_path,
            output_dir=args.output_dir,
            non_empty_only=args.non_empty_only,
            min_dark_ratio=args.min_dark_ratio,
            trim_number_column_ratio=args.trim_number_column_ratio,
            cell_border_trim_ratio=args.cell_border_trim_ratio,
            save_debug=args.save_debug,
        )
        total_saved += saved_count
        LOGGER.info("Saved %s cells for %s", saved_count, image_path.name)

    LOGGER.info("Done. Saved %s cells to %s", total_saved, args.output_dir)


if __name__ == "__main__":
    main()
