import json
from pathlib import Path
from urllib.request import urlretrieve

import polars as pl

REVIEWS_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/review_categories/Software.jsonl"
)
METADATA_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/meta_categories/meta_Software.jsonl"
)


def load_reviews(data_dir: Path | str):
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    parquet_path = data_dir / "Software_reviews.parquet"
    if parquet_path.exists():
        print(f"Parquet file {parquet_path} exists; reading it")
        return pl.read_parquet(parquet_path)

    json_path = data_dir / "Software.jsonl"
    if not json_path.exists():
        print(f"Downloading from Hugging Face to {json_path}")
        urlretrieve(REVIEWS_URL, json_path)

    reviews = pl.scan_ndjson(
        json_path,
        schema={
            "user_id": pl.String,
            "parent_asin": pl.String,
            "rating": pl.Float64,
            "timestamp": pl.Int64,
        },
        batch_size=100_000,
        low_memory=True,
    ).rename({"parent_asin": "item_id"})
    reviews.sink_parquet(parquet_path)
    reviews = pl.read_parquet(parquet_path)
    print("Saved reviews", len(reviews))
    return reviews


def filter_interactions(reviews, min_user=5, min_item=10):
    result = reviews
    while True:
        old_size = len(result)
        result = result.filter(
            (pl.len().over("user_id") >= min_user)
            & (pl.len().over("item_id") >= min_item)
        )
        print("After filtering step:", len(result))
        if len(result) == old_size:
            break
    return result


def load_metadata(item_ids, data_dir):
    data_dir = Path(data_dir)
    parquet_path = data_dir / "prepared/metadata.parquet"
    item_order = pl.DataFrame({"item_id": list(item_ids)})
    if parquet_path.exists():
        metadata = pl.read_parquet(parquet_path)
        if set(item_ids).issubset(set(metadata["item_id"])):
            print("Reading metadata from", parquet_path)
            return item_order.join(
                metadata, on="item_id", how="left", maintain_order="left"
            )

    json_path = data_dir / "meta_Software.jsonl"
    if not json_path.exists():
        print("Downloading metadata from Hugging Face")
        urlretrieve(METADATA_URL, json_path)

    wanted = set(item_ids)
    rows = []
    with json_path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if row["parent_asin"] not in wanted:
                continue
            categories = row.get("categories") or []
            if isinstance(categories, str):
                categories = [categories]
            description = row.get("description") or []
            if isinstance(description, list):
                description = " ".join(map(str, description))
            price = row.get("price")
            rows.append(
                {
                    "item_id": row["parent_asin"],
                    "title": row.get("title") or "",
                    "root_category": categories[0]
                    if categories
                    else row.get("main_category") or "unknown",
                    "description": description,
                    "price": str(price) if price is not None else None,
                }
            )
    metadata = pl.DataFrame(
        rows,
        schema={
            "item_id": pl.String,
            "title": pl.String,
            "root_category": pl.String,
            "description": pl.String,
            "price": pl.String,
        },
    ).unique("item_id", keep="last", maintain_order=True)
    metadata = item_order.join(
        metadata, on="item_id", how="left", maintain_order="left"
    )
    metadata = metadata.with_columns(
        pl.col("root_category").is_not_null().alias("metadata_present"),
        pl.col("root_category").fill_null("unknown"),
        pl.col("title", "description").fill_null(""),
        pl.col("price").cast(pl.Float64, strict=False),
    ).with_columns(
        pl.when(pl.col("price").is_finite() & (pl.col("price") >= 0))
        .then(pl.col("price"))
        .otherwise(None)
        .alias("price")
    )
    parquet_path.parent.mkdir(exist_ok=True)
    metadata.write_parquet(parquet_path)
    return metadata


def chronological_split(reviews):
    result = reviews.sort(["user_id", "timestamp", "item_id"], maintain_order=True)
    result = (
        result.with_columns(
            (pl.col("user_id").cum_count(reverse=True).over("user_id") - 1).alias(
                "from_end"
            )
        )
        .with_columns(
            pl.when(pl.col("from_end") == 0)
            .then(pl.lit("test"))
            .when(pl.col("from_end") == 1)
            .then(pl.lit("val"))
            .otherwise(pl.lit("train"))
            .alias("split"),
            (pl.col("user_id").rank("dense") - 1).cast(pl.Int64).alias("user_idx"),
            (pl.col("item_id").rank("dense") - 1).cast(pl.Int64).alias("item_idx"),
        )
        .drop("from_end")
    )
    users = result["user_id"].unique().sort()
    items = result["item_id"].unique().sort()
    return result, users, items
