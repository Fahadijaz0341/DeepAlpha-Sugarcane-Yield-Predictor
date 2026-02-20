import pdfplumber
import pandas as pd
import os
import re


# ── Metric detection ────────────────────────────────────────────────
METRIC_PATTERNS = [
    (r"area\s+in\s+.?000.?\s*acres",                               "area_000_acres"),
    (r"production\s+in\s+.?000.?\s*tons",                           "production_000_tons"),
    (r"(avg\.?\s*)?yield\s+in\s+(mds?|maunds?)\s*/\s*acres?",      "yield_mds_per_acre"),
    (r"area\s+in\s+.?000.?\s*hect",                                 "area_000_hectares"),
    (r"(avg\.?\s*)?yield\s+(in\s+)?kg\s*/\s*hect",                  "yield_kg_per_hectare"),
]

SKIP_WORDS = ["source:", "note:", "crop reporting"]


def detect_metric(page_text):
    text = page_text.lower()
    for pattern, label in METRIC_PATTERNS:
        if re.search(pattern, text):
            return label
    return None


def extract_year(header_row):
    for cell in header_row:
        if cell and re.match(r"\d{4}-\d{2,4}", str(cell).strip()):
            return str(cell).strip()
    return None


def parse_value(s):
    """Parse a numeric string, returning None for dashes/blanks."""
    s = str(s or "").strip().replace(",", "")
    if not s or re.fullmatch(r"[-\s.]+", s):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def is_header_row(row):
    """Check if a row is a header/sub-header, not data."""
    joined = " ".join(str(c or "") for c in row).lower()
    # Rows with "irri" + "un-irri" are sub-headers
    if "irri" in joined and "un-irri" in joined:
        return True
    first = str(row[0] or "").strip().lower()
    if first in ("districts", "district", ""):
        return True
    # Header row 0 sometimes has multiline text like "Divisions /\nDistricts"
    if "division" in first and "district" in first:
        return True
    return False


def extract_total_from_merged(cell_text, count):
    """
    Extract TOTAL values from a merged cell.
    Merged cells look like: "160 160\n18 18\n13 13\n87 87"
    where each line is "IRRI_val TOTAL_val" (UN-IRRI missing = same as IRRI).
    Or for divisions: "278 --- 278" => IRRI, UN-IRRI, TOTAL.
    """
    lines = str(cell_text or "").strip().split("\n")
    totals = []
    for line in lines:
        parts = line.strip().split()
        # Remove dashes
        parts = [p for p in parts if not re.fullmatch(r"[-]+", p)]
        if not parts:
            totals.append(None)
        elif len(parts) == 1:
            totals.append(parse_value(parts[0]))
        else:
            # Last value is typically TOTAL
            totals.append(parse_value(parts[-1]))
    return totals


def parse_table_format1(table, metric, year, source):
    """
    Format 1 (2020-21 style): Each district is its own row,
    with 10 properly separated columns.
    """
    rows = []
    for raw_row in table:
        if not raw_row or len(raw_row) < 4:
            continue
        if is_header_row(raw_row):
            continue

        district = str(raw_row[0] or "").strip()
        if not district or any(kw in district.lower() for kw in SKIP_WORDS):
            continue

        val = parse_value(raw_row[3])
        rows.append({
            "district": district,
            "year": year,
            "metric": metric,
            "value": val,
            "source_file": source,
        })
    return rows


def parse_table_format2(table, metric, year, source):
    """
    Format 2 (2021-22+ style): Districts are merged into single cells
    with newlines, and values are space-separated within cells.
    """
    rows = []
    for raw_row in table:
        if not raw_row or len(raw_row) < 2:
            continue
        if is_header_row(raw_row):
            continue

        district_cell = str(raw_row[0] or "").strip()
        if not district_cell or any(kw in district_cell.lower() for kw in SKIP_WORDS):
            continue

        districts = [d.strip() for d in district_cell.split("\n") if d.strip()]
        if not districts:
            continue

        # Get the current-year value cell (column 1 in merged format)
        value_cell = str(raw_row[1] or "").strip()

        # Check if this is a merged-value cell (multiple values or "278 --- 278" pattern)
        if "\n" in district_cell and "\n" in value_cell:
            # Multiple districts in one cell, multiple values in another
            totals = extract_total_from_merged(value_cell, len(districts))
            for i, dist in enumerate(districts):
                if any(kw in dist.lower() for kw in SKIP_WORDS):
                    continue
                val = totals[i] if i < len(totals) else None
                rows.append({
                    "district": dist,
                    "year": year,
                    "metric": metric,
                    "value": val,
                    "source_file": source,
                })
        else:
            # Single district row (division totals, province totals)
            # Value might be "278 --- 278" or "278" or plain in col[1]
            totals = extract_total_from_merged(value_cell, 1)
            val = totals[0] if totals else None

            # Also check column 3 if it exists and has value
            if val is None and len(raw_row) > 3:
                val = parse_value(raw_row[3])

            for dist in districts:
                if any(kw in dist.lower() for kw in SKIP_WORDS):
                    continue
                rows.append({
                    "district": dist,
                    "year": year,
                    "metric": metric,
                    "value": val,
                    "source_file": source,
                })
                val = None  # Only first district gets the value
    return rows


def detect_table_format(table):
    """
    Detect whether a table uses Format 1 (separate columns) or
    Format 2 (merged cells with newlines).
    """
    # Check actual data rows (skip headers at index 0, 1)
    for row in table[2:8]:
        if not row or not row[0]:
            continue
        cell0 = str(row[0]).strip()
        # Skip province/region summary rows — they exist in both formats
        if cell0.upper() in ("PUNJAB", "THE PUNJAB", "NORTH PUNJAB", "SOUTH PUNJAB"):
            continue
        # If a district cell contains newlines, it's merged format
        if "\n" in cell0:
            return 2
        # Check if value column 1 has space-separated values like "278 --- 278"
        if row[1] and len(str(row[1]).split()) > 2:
            val_parts = str(row[1]).split()
            if any(re.fullmatch(r"[-]+", p) for p in val_parts):
                return 2
    return 1


def extract_sugarcane_data(pdf_folder_path):
    all_rows = []

    for filename in sorted(os.listdir(pdf_folder_path)):
        if not filename.endswith(".pdf"):
            continue

        file_path = os.path.join(pdf_folder_path, filename)
        source_label = filename.replace(".pdf", "").strip()
        print(f"Processing: {filename} ...")

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""

                if "sugarcane" not in text.lower():
                    continue

                metric = detect_metric(text)
                if not metric:
                    continue

                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 3:
                        continue

                    # Skip table-of-contents
                    hdr = " ".join(str(c or "") for c in table[0]).lower()
                    if "page" in hdr and "#" in hdr:
                        continue

                    year = extract_year(table[0])
                    if not year:
                        continue

                    fmt = detect_table_format(table)
                    if fmt == 2:
                        parsed = parse_table_format2(table, metric, year, source_label)
                    else:
                        parsed = parse_table_format1(table, metric, year, source_label)

                    all_rows.extend(parsed)

    if not all_rows:
        print("\nNo sugarcane data found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["value"])

    # ── Pivot into clean wide format ──
    pivot = df.pivot_table(
        index=["district", "year", "source_file"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    # Rename columns
    nice = {
        "area_000_acres":        "area (000 acres)",
        "production_000_tons":   "production (000 tons)",
        "yield_mds_per_acre":    "yield (mds/acre)",
        "area_000_hectares":     "area (000 hectares)",
        "yield_kg_per_hectare":  "yield (kg/hectare)",
    }
    for old, new in nice.items():
        if old in pivot.columns:
            pivot.rename(columns={old: new}, inplace=True)

    # Column order
    ordered = ["district", "year"]
    for n in nice.values():
        if n in pivot.columns:
            ordered.append(n)
    ordered.append("source_file")
    ordered = [c for c in ordered if c in pivot.columns]
    pivot = pivot[ordered]

    pivot.sort_values(["year", "district"], inplace=True)
    pivot.reset_index(drop=True, inplace=True)
    return pivot


# ──────────────────────────────────────────────
if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(folder, "sugarcane_data.csv")

    df = extract_sugarcane_data(folder)

    if df.empty:
        print("Nothing to save.")
    else:
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n{'='*60}")
        print(f"  Saved {len(df)} rows -> {output_csv}")
        print(f"{'='*60}")

        print(f"\nRows per year:")
        print(df["year"].value_counts().sort_index().to_string())

        print(f"\nPreview (first 15 rows):\n")
        print(df.head(15).to_string())
