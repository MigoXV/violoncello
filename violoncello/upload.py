import asyncio
import math
import os
from functools import partial
from pathlib import Path

import httpx
import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from yarl import URL


def read_with_progress(file_path: Path):
    file_path = Path(file_path)
    chunk_size = 1024 * 1024  # 1MB
    num_chunks = math.ceil(os.path.getsize(file_path) / chunk_size)
    audio_data = bytearray()
    pbar = tqdm(
        total=num_chunks,
        unit="MB",
        unit_scale=True,
        desc="reading",
        disable=num_chunks < 100,
    )
    with open(file_path, "rb") as f:
        pbar.set_postfix_str(file_path.name)
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            audio_data.extend(chunk)
            pbar.update(1)

    return bytes(audio_data)


async def upload_file(
    row: pd.Series,
    base_url: URL,
    semaphore: asyncio.Semaphore,
):

    async with semaphore:
        object_name = row["object_name"]
        # 读取文件
        audio_bytes = await asyncio.to_thread(
            partial(read_with_progress, row["abs_path"])
        )
        headers = None
        # 音频文件设置 Content-Type
        if object_name.endswith(".wav") or object_name.endswith("WAV"):
            headers = {"Content-Type": "audio/wav"}
        # 构造最终的上传目标 URL，使用 yarl 进行路径拼接
        url = base_url / object_name
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            response = await client.put(str(url), content=audio_bytes, headers=headers)
            response.raise_for_status()


async def put_audio_to_minio(
    df: pd.DataFrame, base_url: URL, minio_prefix: str, max_workers: int = 60
):
    upload_base_url = base_url / minio_prefix
    semaphore = asyncio.Semaphore(max_workers)
    tasks = []

    for idx, row in df.iterrows():
        tasks.append(asyncio.create_task(upload_file(row, upload_base_url, semaphore)))

    await tqdm_asyncio.gather(
        *tasks, leave=False, total=len(df), desc="uploading", unit="file"
    )


def get_df(input_dir: Path, relative_root: Path, disable_tqdm: bool = True):
    raw_file_itr = tqdm(input_dir.glob("**/*"), desc="scanning", disable=disable_tqdm)
    get_dict = lambda x: {
        "object_name": str(x.relative_to(relative_root)).replace("\\", "/"),
        "rela_path": x,
        "abs_path": x.absolute(),
    }
    df = pd.DataFrame(get_dict(x) for x in raw_file_itr if x.is_file())
    return df


def upload(df: pd.DataFrame, base_url: URL, bucket: str, minio_prefix: str):
    loop = asyncio.get_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(put_audio_to_minio(df, base_url / bucket, minio_prefix))
