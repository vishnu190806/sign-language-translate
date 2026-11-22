from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import quote
from urllib.request import urlretrieve

import pandas as pd

from src.data.registry import DatasetRegistry, SignLanguageDataset

logger = logging.getLogger(__name__)

INCLUDE_ZENODO_URL = "https://zenodo.org/records/4010759/files/INCLUDE_v1.1.zip"
INCLUDE_HF_REPO = "ai4bharat/INCLUDE"
INCLUDE50_ZENODO_BASE = "https://zenodo.org/records/4010759/files"


@DatasetRegistry.register("isl")
class ISLDataset(SignLanguageDataset):
    name = "isl"

    def download(self, dest_dir: str | Path, include50: bool = False) -> None:
        dest = Path(dest_dir) / "isl"
        dest.mkdir(parents=True, exist_ok=True)

        if include50:
            self._download_include50(dest)
            return

        zip_path = dest / "INCLUDE_v1.1.zip"
        if not zip_path.exists():
            logger.info("Downloading INCLUDE dataset from Zenodo...")
            try:
                urlretrieve(INCLUDE_ZENODO_URL, zip_path)
                logger.info("Download complete.")
            except Exception as e:
                logger.warning(f"Zenodo download failed: {e}. Attempting HuggingFace fallback...")
                self._download_huggingface(dest)
                return

        logger.info("Extracting INCLUDE dataset...")
        shutil.unpack_archive(zip_path, dest)
        logger.info("Extraction complete.")

    def _download_include50(self, dest: Path) -> None:
        from datasets import load_dataset
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        logger.info("Loading INCLUDE-50 metadata from HuggingFace...")
        ds = load_dataset(INCLUDE_HF_REPO, split="train")
        df = ds.to_pandas()
        include50 = df[df["include_50"] == True].copy()
        logger.info(f"Found {len(include50)} videos in INCLUDE-50")

        def download_video(video_path: str) -> bool:
            filename = Path(video_path).name
            parts = video_path.split("/")
            class_dir = dest / parts[0] / parts[1]
            class_dir.mkdir(parents=True, exist_ok=True)
            dest_path = class_dir / filename
            if dest_path.exists():
                return True
            encoded_path = quote(video_path)
            url = f"{INCLUDE50_ZENODO_BASE}/{encoded_path}"
            for attempt in range(3):
                try:
                    urlretrieve(url, dest_path)
                    return True
                except Exception:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
            return False

        video_paths = include50['video_path'].tolist()
        logger.info(f"Downloading {len(video_paths)} videos...")
        
        stats = {"downloaded": 0, "failed": 0}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(download_video, vp): vp for vp in video_paths}
            for future in as_completed(futures):
                if future.result():
                    stats["downloaded"] += 1
                else:
                    stats["failed"] += 1
                total = stats["downloaded"] + stats["failed"]
                if total % 50 == 0:
                    logger.info(f"Progress: {total}/{len(video_paths)}")

        logger.info(f"Download complete: {stats['downloaded']} downloaded, {stats['failed']} failed")
        include50.to_csv(dest / "include50_metadata.csv", index=False)

    def _download_huggingface(self, dest: Path) -> None:
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=INCLUDE_HF_REPO,
                repo_type="dataset",
                local_dir=str(dest),
            )
            logger.info("HuggingFace download complete.")
        except ImportError:
            logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
            raise
        except Exception as e:
            logger.error(f"HuggingFace download also failed: {e}")
            raise

    def build_metadata(self, subset: str | None = None) -> pd.DataFrame:
        csv_path = Path(self._raw_dir()) / "include50_metadata.csv"
        if csv_path.exists() and (subset is None or subset == "include50"):
            logger.info(f"Loading metadata from {csv_path}")
            return pd.read_csv(csv_path)

        dest = Path(self._raw_dir())
        records = []
        for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.mkv"):
            for video_path in sorted(dest.rglob(ext)):
                label = video_path.parent.name
                records.append({
                    "video_path": str(video_path),
                    "label": label,
                    "split": self._infer_split(video_path),
                })

        if not records:
            logger.warning(f"No videos found in {dest}")

        return pd.DataFrame(records)

    def _raw_dir(self) -> str:
        return "data/raw/isl"

    def _infer_split(self, path: Path) -> str:
        parts = [p.lower() for p in path.parts]
        if "test" in parts:
            return "test"
        elif "val" in parts or "valid" in parts:
            return "val"
        return "train"

    def get_label_map(self, subset: str | None = None) -> Dict[str, int]:
        for path in [
            Path(self._raw_dir()) / "include50_label_map.json",
            Path("data/processed/isl_label_map.json"),
        ]:
            if path.exists():
                import json
                with open(path) as f:
                    return json.load(f)

        df = self.build_metadata(subset)
        labels = sorted(df["label"].unique())
        label_map = {label: idx for idx, label in enumerate(labels)}
        return label_map

    def get_splits(self) -> Dict[str, List[str]]:
        df = self.build_metadata()
        splits = {}
        for split_name, group in df.groupby("split"):
            splits[split_name] = group["video_path"].tolist()
        return splits
