import logging
from typing import Optional

import pandas as pd
from minio import Minio
from tqdm import tqdm

logger = logging.getLogger(__name__)


def crawl(
    client: Minio, bucket: str, prefix: Optional[str] = None, disable_tqdm=False
) -> pd.DataFrame:
    objects = client.list_objects(bucket, prefix=prefix, recursive=True)
    df = pd.DataFrame(columns=["object_name", "md5", "size"])
    for obj in tqdm(objects, disable=disable_tqdm):
        obj_info = {
            "object_name": obj.object_name,
            "md5": obj.etag,
            "size": obj.size,
        }
        df = pd.concat(
            [df, pd.DataFrame([obj_info], columns=df.columns)], ignore_index=True
        )
    return df
