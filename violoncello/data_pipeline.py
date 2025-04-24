# v0.2.0
import asyncio
import logging
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Tuple, Union
from urllib.parse import unquote

import httpx
import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from yarl import URL

logging.getLogger("httpx").setLevel(logging.WARNING)  # 或 ERROR，甚至 CRITICAL
logger = logging.getLogger(__name__)


class DataPipeline:
    def __init__(
        self, max_connections: int = 10, put_to_minio: bool = True, batch_size: int = 16
    ):

        self.max_connections = max_connections
        self.put_to_minio = put_to_minio
        self.batch_size = batch_size

        self.fetch_queue = Queue()
        self.put_queue = Queue()

    async def fetch_row(
        self,
        row: pd.Series,
        base_url: URL,
        bucket: str,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
    ) -> Tuple[pd.Series, Union[bytes, None]]:
        obj_name = unquote(row["object_name"])
        url = base_url / bucket / obj_name
        async with semaphore:
            try:
                response = await client.get(str(url))
                response.raise_for_status()
                return (row, response.content)
            except Exception as e:
                logger.error(f"Fetching {url} failed: {e}")
                return (row, None)

    async def fetch(
        self,
        df: pd.DataFrame,
        base_url: URL,
        bucket: str,
    ):
        semaphore = asyncio.Semaphore(self.max_connections)
        async with httpx.AsyncClient() as client:
            tasks = [
                self.fetch_row(row, base_url, bucket, client, semaphore)
                for _, row in df.iterrows()
            ]
            # 按照完成顺序批次化存入队列
            batched_results = []
            for coro in tqdm_asyncio.as_completed(
                tasks, desc="fetching", unit="obj", leave=False, position=0
            ):
                result = await coro
                if result[-1] is None:
                    logger.error(f"Skipping row: {result['error']}")
                    continue
                batched_results.append(result)
                # 每 batch_size 个结果放入队列
                if (
                    len(batched_results) % self.batch_size == 0
                    and len(batched_results) > 0
                    and self.batch_size > 0
                ):
                    row_batch, data_batch = zip(*batched_results)
                    batch_df = pd.DataFrame(row_batch)
                    self.fetch_queue.put((batch_df, data_batch))
                    batched_results = []
            # 最后一批数据
            if batched_results:
                row_batch, data_batch = zip(*batched_results)
                batch_df = pd.DataFrame(row_batch)
                self.fetch_queue.put((batch_df, data_batch))

    async def put_row(
        self,
        row: pd.Series,
        data: bytes,
        base_url: URL,
        bucket: str,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
    ):
        obj_name = unquote(str(row["object_name"]))
        url = base_url / bucket / obj_name
        async with semaphore:
            try:
                # 这里 data 参数由 fetch 阶段和 worker 线程处理后放入 put_queue 的数据中得到
                response = await client.put(str(url), content=data)
                response.raise_for_status()
            except Exception as e:
                logger.error(f"[Error] Putting {url}: {e}")

    async def put(self, base_url: URL, bucket: str, put_bar: tqdm):
        """持续从 put_queue 异步消费，只要不阻塞 worker 就行"""
        semaphore = asyncio.Semaphore(self.max_connections)
        async with httpx.AsyncClient() as client:
            loop = asyncio.get_running_loop()
            while True:
                put_batch = await loop.run_in_executor(None, self.put_queue.get)
                if put_batch is None:
                    break
                df, data_list = put_batch
                for i in tqdm(
                    range(len(df)),
                    desc="putting a chunk",
                    unit="file",
                    position=3,
                    leave=False,
                ):
                    row = df.iloc[i]
                    data = data_list[i]
                    await self.put_row(row, data, base_url, bucket, client, semaphore)
                put_bar.update(1)

    def run_pipeline(
        self,
        df: pd.DataFrame,
        base_url: URL,
        bucket: str,
        worker: Callable,
    ):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # 推理线程
        with ThreadPoolExecutor(max_workers=1) as executor:
            worker_bar = tqdm(
                total=math.ceil(len(df) / self.batch_size),
                desc="working",
                unit="file",
                leave=False,
                position=1,
            )
            executor.submit(lambda: self._run_pipeline(worker_bar, worker))
            # 异步抓取数据
            loop.run_until_complete(self.fetch(df, base_url, bucket))
            # 抓取完成，发送停止信号给推理线程
            self.fetch_queue.put(None)
            # 同时等待 put 阶段处理完毕
            if self.put_to_minio:
                put_bar = tqdm(
                    total=len(df), desc="putting", unit="file", leave=False, position=2
                )
                loop.run_until_complete(self.put(base_url, bucket, put_bar))
            loop.close()

    def _run_pipeline(self, worker_bar, worker: Callable):
        while True:
            fetch_batch = self.fetch_queue.get()
            if fetch_batch is None:
                self.put_queue.put(None)
                break
            fetch_df, fetch_data = fetch_batch
            try:
                put_tuple = worker(
                    fetch_df,
                    fetch_data,
                )
                if self.put_to_minio and put_tuple is not None:
                    put_df, put_data = put_tuple
                    self.put_queue.put((put_df, put_data))
            except Exception as e:
                logger.error(f"[Error] Processing row: {e}")
                continue
            worker_bar.update(1)
