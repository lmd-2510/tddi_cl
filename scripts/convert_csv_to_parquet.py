from pathlib import Path

import pyarrow.csv as csv
import pyarrow.parquet as parquet


FILES = [
    "train_extracted.csv",
    "validation_extracted.csv",
    "test_extracted.csv",
]


def convert(source: Path, destination: Path) -> None:
    print(f"Converting {source} -> {destination}", flush=True)

    reader = csv.open_csv(
        source,
        read_options=csv.ReadOptions(
            use_threads=True,
            block_size=64 * 1024 * 1024,
        ),
        convert_options=csv.ConvertOptions(
            strings_can_be_null=True,
        ),
    )

    writer = None
    rows = 0

    try:
        for batch_number, batch in enumerate(reader, start=1):
            if writer is None:
                writer = parquet.ParquetWriter(
                    destination,
                    batch.schema,
                    compression="snappy",
                )

            writer.write_batch(batch)
            rows += batch.num_rows

            if batch_number % 10 == 0:
                print(
                    f"  batches={batch_number}, rows={rows:,}",
                    flush=True,
                )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()

    print(f"Completed: {rows:,} rows", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    for filename in FILES:
        source = root / filename
        destination = source.with_suffix(".parquet")

        if not source.exists():
            raise FileNotFoundError(source)

        if destination.exists():
            print(f"Skipping existing file: {destination}")
            continue

        convert(source, destination)


if __name__ == "__main__":
    main()
