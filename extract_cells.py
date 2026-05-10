from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)
INPUT_DIR = Path("inputs")
OUTPUT_DIR = Path("outputs")
NON_EMPTY_ONLY = False
MIN_DARK_RATIO = 0.012
TRIM_NUMBER_COLUMN_RATIO = 0.28
CELL_TOP_PADDING_RATIO = 0.15
CELL_BOTTOM_PADDING_RATIO = 0.25
DETECTION_VERTICAL_MARGIN_RATIO = 0.18
DETECTION_HORIZONTAL_MARGIN_RATIO = 0.04
STOP_AFTER_FIRST_EMPTY = True
EMPTY_LOOKAHEAD = 0
SAVE_DEBUG = False
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
INK_CONTRAST_THRESHOLD = 20
MIN_INK_COMPONENT_AREA = 6
SCORESHEET_ROWS = 30
SCORESHEET_MOVE_COLUMNS = 4
EXPECTED_MOVE_CELLS = SCORESHEET_ROWS * SCORESHEET_MOVE_COLUMNS
MANIFEST_FIELDNAMES = [
    "image_name",
    "cell_index",
    "move_number",
    "side",
    "row",
    "column",
    "bbox_x_min",
    "bbox_y_min",
    "bbox_x_max",
    "bbox_y_max",
    "detection_bbox_x_min",
    "detection_bbox_y_min",
    "detection_bbox_x_max",
    "detection_bbox_y_max",
    "ink_ratio",
    "detected_non_empty",
    "after_cutoff",
    "saved",
    "reject_reason",
    "file_name",
]


@dataclass
class ExtractionResult:
    saved_count: int
    contour_count: int
    detected_cell_count: int
    output_cell_count: int
    raw_non_empty_count: int
    cutoff_index: int | None
    blank_rejected_count: int
    after_cutoff_rejected_count: int
    manifest_records: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract handwritten move cells from chess scoresheet photos."
    )
    parser.add_argument(
        "image_filename",
        help="Image file name inside inputs/, for example 001_0.png.",
    )
    return parser.parse_args()


def resolve_input_image_path(image_filename: str) -> Path:
    image_path = Path(image_filename)
    if image_path.name != image_filename:
        raise ValueError("Pass only the image file name, for example: 001_0.png")

    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise ValueError(f"Unsupported image extension. Supported: {supported}")

    return INPUT_DIR / image_path.name


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


def median_float(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute median from an empty list.")
    return float(np.median(values))


def rectangle_contour(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> np.ndarray:
    return np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=np.int32,
    )


def estimate_row_step(observed_row_centers: list[float]) -> float:
    if len(observed_row_centers) < 2:
        raise ValueError("At least two detected rows are required to estimate spacing.")

    gaps = np.diff(sorted(observed_row_centers))
    median_gap = float(np.median(gaps))
    regular_gaps = [gap for gap in gaps if gap <= median_gap * 1.5]
    return float(np.median(regular_gaps or gaps))


def estimate_scoresheet_row_centers(
    detected_cells: list[dict],
    image_height: int,
) -> list[float]:
    row_centers_by_index: dict[int, list[float]] = {}
    for cell in detected_cells:
        row = cell.get("row")
        if row is not None:
            row_centers_by_index.setdefault(row, []).append(float(cell["y_center"]))

    observed_centers = [
        median_float(centers)
        for _, centers in sorted(row_centers_by_index.items())
    ]
    row_step = estimate_row_step(observed_centers)
    first_observed_center = observed_centers[0]

    best_score = float("inf")
    best_first_center: float | None = None
    for observed_row_index in range(SCORESHEET_ROWS):
        first_center = first_observed_center - observed_row_index * row_step
        last_center = first_center + (SCORESHEET_ROWS - 1) * row_step

        if first_center < -0.5 * row_step or first_center > 1.75 * row_step:
            continue
        if (
            last_center < image_height - 2.5 * row_step
            or last_center > image_height + 0.75 * row_step
        ):
            continue

        full_centers = [
            first_center + row_index * row_step
            for row_index in range(SCORESHEET_ROWS)
        ]
        residuals = [
            min(abs(observed - center) for center in full_centers)
            for observed in observed_centers
        ]
        score = float(np.mean(np.square(residuals)))
        if score < best_score:
            best_score = score
            best_first_center = first_center

    if best_first_center is None:
        best_first_center = max(
            0.0,
            min(first_observed_center, image_height - (SCORESHEET_ROWS - 1) * row_step),
        )

    return [
        best_first_center + row_index * row_step
        for row_index in range(SCORESHEET_ROWS)
    ]


def reconstruct_scoresheet_grid_cells(
    detected_cells: list[dict],
    image_size: tuple[int, int],
) -> list[dict]:
    image_width, image_height = image_size

    column_bounds: list[tuple[int, int]] = []
    for column in range(SCORESHEET_MOVE_COLUMNS):
        column_cells = [
            cell for cell in detected_cells if cell.get("column") == column
        ]
        if not column_cells:
            raise ValueError(f"No detected cells found for column {column}.")

        left_edges = []
        right_edges = []
        for cell in column_cells:
            x_min, _, x_max, _ = contour_bounds(cell["contour"])
            left_edges.append(float(x_min))
            right_edges.append(float(x_max))

        left = max(0, int(round(median_float(left_edges))))
        right = min(image_width, int(round(median_float(right_edges))))
        if right <= left:
            raise ValueError(f"Invalid estimated bounds for column {column}.")
        column_bounds.append((left, right))

    top_offsets = []
    bottom_offsets = []
    for cell in detected_cells:
        _, y_min, _, y_max = contour_bounds(cell["contour"])
        y_center = float(cell["y_center"])
        top_offsets.append(y_center - y_min)
        bottom_offsets.append(y_max - y_center)

    top_offset = median_float(top_offsets)
    bottom_offset = median_float(bottom_offsets)
    row_centers = estimate_scoresheet_row_centers(detected_cells, image_height)

    reconstructed_cells: list[dict] = []
    for columns in ((0, 1), (2, 3)):
        for row_index, y_center in enumerate(row_centers):
            y_min = max(0, int(round(y_center - top_offset)))
            y_max = min(image_height, int(round(y_center + bottom_offset)))
            for column in columns:
                x_min, x_max = column_bounds[column]
                contour = rectangle_contour(x_min, y_min, x_max, y_max)
                reconstructed_cells.append(
                    {
                        "contour": contour,
                        "column": column,
                        "row": row_index,
                        "x_center": x_min + (x_max - x_min) / 2,
                        "y_center": y_center,
                        "height": y_max - y_min,
                    }
                )

    return reconstructed_cells


def complete_scoresheet_cells(
    detected_cells: list[dict],
    image_size: tuple[int, int],
) -> list[dict]:
    if len(detected_cells) == EXPECTED_MOVE_CELLS:
        return detected_cells

    return reconstruct_scoresheet_grid_cells(detected_cells, image_size)


def crop_cell(
    gray_image: Image.Image,
    contour: np.ndarray,
    top_padding_ratio: float = CELL_TOP_PADDING_RATIO,
    bottom_padding_ratio: float = CELL_BOTTOM_PADDING_RATIO,
) -> Image.Image:
    x_min, y_min, x_max, y_max = contour_bounds(contour)
    height = y_max - y_min

    top_padding = int(height * max(0.0, top_padding_ratio))
    bottom_padding = int(height * max(0.0, bottom_padding_ratio))

    left = max(0, x_min)
    top = max(0, y_min - top_padding)
    right = min(gray_image.width, x_max)
    bottom = min(gray_image.height, y_max + bottom_padding)

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


def bounded_margin_ratio(ratio: float) -> float:
    return min(max(float(ratio), 0.0), 0.45)


def detection_region_bounds(
    contour: np.ndarray,
    image_size: tuple[int, int],
    column: int | None,
    trim_number_column_ratio: float,
    detection_vertical_margin_ratio: float,
    detection_horizontal_margin_ratio: float,
) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = contour_bounds(contour)
    image_width, image_height = image_size
    width = max(1, x_max - x_min)
    height = max(1, y_max - y_min)

    vertical_margin = bounded_margin_ratio(detection_vertical_margin_ratio)
    horizontal_margin = bounded_margin_ratio(detection_horizontal_margin_ratio)

    left = x_min + int(round(width * horizontal_margin))
    right = x_max - int(round(width * horizontal_margin))
    top = y_min + int(round(height * vertical_margin))
    bottom = y_max - int(round(height * vertical_margin))

    if trim_number_column_ratio > 0 and column in (0, 2):
        left = max(left, x_min + int(round(width * trim_number_column_ratio)))

    left = min(max(left, 0), max(0, image_width - 1))
    right = min(max(right, left + 1), image_width)
    top = min(max(top, 0), max(0, image_height - 1))
    bottom = min(max(bottom, top + 1), image_height)

    return left, top, right, bottom


def crop_cell_detection_region(
    gray_image: Image.Image,
    contour: np.ndarray,
    column: int | None,
    trim_number_column_ratio: float = TRIM_NUMBER_COLUMN_RATIO,
    detection_vertical_margin_ratio: float = DETECTION_VERTICAL_MARGIN_RATIO,
    detection_horizontal_margin_ratio: float = DETECTION_HORIZONTAL_MARGIN_RATIO,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    bounds = detection_region_bounds(
        contour=contour,
        image_size=gray_image.size,
        column=column,
        trim_number_column_ratio=trim_number_column_ratio,
        detection_vertical_margin_ratio=detection_vertical_margin_ratio,
        detection_horizontal_margin_ratio=detection_horizontal_margin_ratio,
    )
    return gray_image.crop(bounds), bounds


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
    horizontal_lines = cv2.dilate(
        horizontal_lines,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)),
        iterations=1,
    )
    vertical_lines = cv2.dilate(
        vertical_lines,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        iterations=1,
    )
    grid_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)

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


def handwriting_ink_analysis(
    image: Image.Image,
    min_dark_ratio: float,
) -> dict[str, object]:
    gray = np.array(image.convert("L"))
    height, width = gray.shape

    top = int(height * 0.02)
    bottom = int(height * 0.98)
    left = int(width * 0.04)
    right = int(width * 0.96)
    inner = gray[top:bottom, left:right]

    if inner.size == 0:
        empty_mask = np.zeros((0, 0), dtype=np.uint8)
        return {
            "roi": inner,
            "contrast_ink": empty_mask,
            "without_grid": empty_mask,
            "final_ink": empty_mask,
            "ink_ratio": 0.0,
            "non_empty": False,
        }

    handwriting_pixels = contrast_ink_pixels(inner)
    without_grid = remove_grid_line_pixels(handwriting_pixels)
    final_ink = remove_small_ink_components(without_grid)
    ink_ratio = float(np.mean(final_ink > 0))

    return {
        "roi": inner,
        "contrast_ink": handwriting_pixels,
        "without_grid": without_grid,
        "final_ink": final_ink,
        "ink_ratio": ink_ratio,
        "non_empty": ink_ratio >= min_dark_ratio,
    }


def has_handwriting(image: Image.Image, min_dark_ratio: float) -> bool:
    analysis = handwriting_ink_analysis(image, min_dark_ratio)
    return bool(analysis["non_empty"])


def analyze_cell_handwriting(
    gray_image: Image.Image,
    cell_info: dict,
    min_dark_ratio: float,
    trim_number_column_ratio: float,
    detection_vertical_margin_ratio: float,
    detection_horizontal_margin_ratio: float,
) -> dict[str, object]:
    detection_image, detection_bbox = crop_cell_detection_region(
        gray_image=gray_image,
        contour=cell_info["contour"],
        column=cell_info.get("column"),
        trim_number_column_ratio=trim_number_column_ratio,
        detection_vertical_margin_ratio=detection_vertical_margin_ratio,
        detection_horizontal_margin_ratio=detection_horizontal_margin_ratio,
    )
    analysis = handwriting_ink_analysis(detection_image, min_dark_ratio)
    analysis["detection_image"] = detection_image
    analysis["detection_bbox"] = detection_bbox
    return analysis


def move_cell_name(image_stem: str, cell_index: int, cell_count: int) -> str:
    if cell_count != EXPECTED_MOVE_CELLS:
        return f"{image_stem}_cell_{cell_index + 1:03d}.png"

    move_number, side = move_cell_position(cell_index, cell_count)
    return f"{image_stem}_cell_{cell_index + 1:03d}_move_{move_number:02d}_{side}.png"


def move_cell_position(
    cell_index: int,
    cell_count: int,
) -> tuple[int | None, str | None]:
    if cell_count != EXPECTED_MOVE_CELLS:
        return None, None

    first_panel_cell_count = SCORESHEET_ROWS * 2
    if cell_index < first_panel_cell_count:
        move_number = (cell_index // 2) + 1
    else:
        move_number = (
            (cell_index - first_panel_cell_count) // 2
        ) + SCORESHEET_ROWS + 1

    side = "white" if cell_index % 2 == 0 else "black"
    return move_number, side


def find_first_confirmed_empty_index(
    non_empty_flags: list[bool],
    empty_lookahead: int = EMPTY_LOOKAHEAD,
) -> int | None:
    lookahead = max(0, int(empty_lookahead))
    required_empty_count = lookahead + 1

    for index, non_empty in enumerate(non_empty_flags):
        if non_empty:
            continue

        window = non_empty_flags[index : index + required_empty_count]
        has_required_window = len(window) == required_empty_count
        reaches_end = index + len(window) == len(non_empty_flags)
        if not any(window) and (has_required_window or reaches_end):
            return index

    return None


def write_manifest_csv(
    manifest_path: Path,
    manifest_records: list[dict],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(manifest_records)


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


def extract_cells_from_image_with_diagnostics(
    image_path: Path,
    output_dir: Path,
    non_empty_only: bool,
    min_dark_ratio: float,
    trim_number_column_ratio: float,
    cell_top_padding_ratio: float,
    cell_bottom_padding_ratio: float,
    save_debug: bool,
    detection_vertical_margin_ratio: float = DETECTION_VERTICAL_MARGIN_RATIO,
    detection_horizontal_margin_ratio: float = DETECTION_HORIZONTAL_MARGIN_RATIO,
    stop_after_first_empty: bool = STOP_AFTER_FIRST_EMPTY,
    empty_lookahead: int = EMPTY_LOOKAHEAD,
    manifest_path: Path | None = None,
) -> ExtractionResult:
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
    detected_cells = sort_cells_by_scoresheet_move_order(contours)
    cells = complete_scoresheet_cells(detected_cells, gray.size)
    LOGGER.info(
        "Detected %s move-cell contours in %s; using %s cells for output",
        len(detected_cells),
        image_path.name,
        len(cells),
    )

    analyses = [
        analyze_cell_handwriting(
            gray_image=gray,
            cell_info=cell_info,
            min_dark_ratio=min_dark_ratio,
            trim_number_column_ratio=trim_number_column_ratio,
            detection_vertical_margin_ratio=detection_vertical_margin_ratio,
            detection_horizontal_margin_ratio=detection_horizontal_margin_ratio,
        )
        for cell_info in cells
    ]
    non_empty_flags = [bool(analysis["non_empty"]) for analysis in analyses]
    cutoff_index = None
    if non_empty_only and stop_after_first_empty:
        cutoff_index = find_first_confirmed_empty_index(
            non_empty_flags,
            empty_lookahead=empty_lookahead,
        )

    saved_count = 0
    blank_rejected_count = 0
    after_cutoff_rejected_count = 0
    manifest_records: list[dict] = []

    for index, cell_info in enumerate(cells):
        cell = crop_cell(
            gray_image=gray,
            contour=cell_info["contour"],
            top_padding_ratio=cell_top_padding_ratio,
            bottom_padding_ratio=cell_bottom_padding_ratio,
        )
        cell = trim_printed_move_number_column(
            image=cell,
            column=cell_info["column"],
            trim_ratio=trim_number_column_ratio,
        )

        analysis = analyses[index]
        raw_non_empty = bool(analysis["non_empty"])
        is_cutoff_cell = cutoff_index is not None and index == cutoff_index
        after_cutoff = cutoff_index is not None and index > cutoff_index
        file_name = move_cell_name(image_path.stem, index, len(cells))

        save_cell = True
        reject_reason = ""
        if non_empty_only:
            if stop_after_first_empty:
                save_cell = cutoff_index is None or index < cutoff_index
                if is_cutoff_cell:
                    reject_reason = "blank"
                elif after_cutoff:
                    reject_reason = "after_cutoff"
            else:
                save_cell = raw_non_empty
                if not raw_non_empty:
                    reject_reason = "blank"

        if non_empty_only and not save_cell:
            if reject_reason == "after_cutoff":
                after_cutoff_rejected_count += 1
            else:
                blank_rejected_count += 1

        if save_cell:
            cell.save(image_output_dir / file_name)
            saved_count += 1

        move_number, side = move_cell_position(index, len(cells))
        bbox = contour_bounds(cell_info["contour"])
        detection_bbox = analysis["detection_bbox"]
        manifest_records.append(
            {
                "image_name": image_path.name,
                "cell_index": index + 1,
                "move_number": move_number or "",
                "side": side or "",
                "row": cell_info.get("row", ""),
                "column": cell_info.get("column", ""),
                "bbox_x_min": bbox[0],
                "bbox_y_min": bbox[1],
                "bbox_x_max": bbox[2],
                "bbox_y_max": bbox[3],
                "detection_bbox_x_min": detection_bbox[0],
                "detection_bbox_y_min": detection_bbox[1],
                "detection_bbox_x_max": detection_bbox[2],
                "detection_bbox_y_max": detection_bbox[3],
                "ink_ratio": f"{float(analysis['ink_ratio']):.6f}",
                "detected_non_empty": raw_non_empty,
                "after_cutoff": after_cutoff,
                "saved": save_cell,
                "reject_reason": reject_reason,
                "file_name": file_name if save_cell else "",
            }
        )

    if manifest_path is not None:
        write_manifest_csv(Path(manifest_path), manifest_records)

    return ExtractionResult(
        saved_count=saved_count,
        contour_count=len(contours),
        detected_cell_count=len(detected_cells),
        output_cell_count=len(cells),
        raw_non_empty_count=sum(non_empty_flags),
        cutoff_index=cutoff_index,
        blank_rejected_count=blank_rejected_count,
        after_cutoff_rejected_count=after_cutoff_rejected_count,
        manifest_records=manifest_records,
    )


def extract_cells_from_image(
    image_path: Path,
    output_dir: Path,
    non_empty_only: bool,
    min_dark_ratio: float,
    trim_number_column_ratio: float,
    cell_top_padding_ratio: float,
    cell_bottom_padding_ratio: float,
    save_debug: bool,
    detection_vertical_margin_ratio: float = DETECTION_VERTICAL_MARGIN_RATIO,
    detection_horizontal_margin_ratio: float = DETECTION_HORIZONTAL_MARGIN_RATIO,
    stop_after_first_empty: bool = STOP_AFTER_FIRST_EMPTY,
    empty_lookahead: int = EMPTY_LOOKAHEAD,
    manifest_path: Path | None = None,
) -> int:
    result = extract_cells_from_image_with_diagnostics(
        image_path=image_path,
        output_dir=output_dir,
        non_empty_only=non_empty_only,
        min_dark_ratio=min_dark_ratio,
        trim_number_column_ratio=trim_number_column_ratio,
        cell_top_padding_ratio=cell_top_padding_ratio,
        cell_bottom_padding_ratio=cell_bottom_padding_ratio,
        save_debug=save_debug,
        detection_vertical_margin_ratio=detection_vertical_margin_ratio,
        detection_horizontal_margin_ratio=detection_horizontal_margin_ratio,
        stop_after_first_empty=stop_after_first_empty,
        empty_lookahead=empty_lookahead,
        manifest_path=manifest_path,
    )
    return result.saved_count


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()

    if not INPUT_DIR.exists():
        raise SystemExit(f"ERROR - Input directory does not exist: {INPUT_DIR}")

    try:
        image_path = resolve_input_image_path(args.image_filename)
    except ValueError as error:
        raise SystemExit(f"ERROR - {error}") from None

    if not image_path.exists():
        raise SystemExit(f"ERROR - Input image does not exist: {image_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_count = extract_cells_from_image(
        image_path=image_path,
        output_dir=OUTPUT_DIR,
        non_empty_only=NON_EMPTY_ONLY,
        min_dark_ratio=MIN_DARK_RATIO,
        trim_number_column_ratio=TRIM_NUMBER_COLUMN_RATIO,
        cell_top_padding_ratio=CELL_TOP_PADDING_RATIO,
        cell_bottom_padding_ratio=CELL_BOTTOM_PADDING_RATIO,
        save_debug=SAVE_DEBUG,
    )
    LOGGER.info(
        "Done. Saved %s cells for %s to %s",
        saved_count,
        image_path.name,
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
