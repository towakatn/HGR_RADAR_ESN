#!/usr/bin/env python3
"""Convert Soli raw data to per-channel DTM using the shared converter."""

from pathlib import Path
import sys

SOLI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOLI_DIR.parent))

from modules.converters import SeparateChannelConverter


def main(input_dir=None, base_output_dir=None):
    input_path = Path(input_dir) if input_dir is not None else SOLI_DIR / "SoliData" / "dsp"
    output_path = Path(base_output_dir) if base_output_dir is not None else SOLI_DIR / "DTM"
    converter = SeparateChannelConverter(
        input_dir=str(input_path.resolve()),
        base_output_dir=str(output_path.resolve()),
    )
    converter.convert_all_files()


if __name__ == "__main__":
    main()
