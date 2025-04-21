import io
import logging
import os
from abc import ABC, abstractmethod
from urllib.parse import unquote

import httpx
import librosa
import numpy as np
from yarl import URL

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class AudioDriver(ABC):
    def __init__(
        self,
        endpoint: str,
        bucket: str,
        sr: int,
        session_warmup: int = -1,
        access_key: str = None,
        secret_key: str = None,
        secure: bool = None,
    ):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.bucket = bucket
        self.sr = sr

        self.session_warmup = session_warmup
        self._client = None
        self._pid = None  # self.pid为创建客户端的进程id，用于检测是否为子进程

    @abstractmethod
    def get_audio(self, audio_path: str) -> np.ndarray:
        """获取音频数据"""
        raise NotImplementedError("get_audio method not implemented")

    @abstractmethod
    def warmup_step(self):
        """预热session"""
        raise NotImplementedError("warmup_step method not implemented")


class HttpxAudioDriver(AudioDriver):

    def __init__(
        self,
        endpoint,
        bucket,
        sr,
        session_warmup=-1,
        access_key=None,
        secret_key=None,
        secure=None,
    ):
        super().__init__(
            endpoint, bucket, sr, session_warmup, access_key, secret_key, secure
        )
        host, port = self.endpoint.split(":")
        self.base_url = URL.build(
            scheme="http" if not self.secure else "https",
            host=host,
            port=int(port),
        )
        self.bucket_url = self.base_url / self.bucket
        self._session_warmup = session_warmup

    @property
    def client(self):
        """进程安全的MinIO客户端"""
        if self._client is None or os.getpid() != self._pid:
            self._client = httpx.Client(timeout=30)
            self._pid = os.getpid()

        return self._client

    def _clear_session(self):
        """清空MinIO客户端缓存"""
        self._client = None
        self._pid = None

    def get_audio(self, obj_name: str, step: bool = False) -> np.ndarray:
        """从MinIO中获取音频"""
        audio_url = str(self.bucket_url / unquote(obj_name))
        try:
            response = self.client.get(audio_url)
            httpx.Client(timeout=30)
            response.raise_for_status()
            audio, _ = librosa.load(io.BytesIO(response.content), sr=self.sr, mono=True)
            # <0，session永远无效，=0，session有效，>0，还有session_warmup次启动session
            if self._session_warmup < 0 or self._session_warmup > 0:
                self._clear_session()
            if step:
                self._session_warmup -= 1
            return audio
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to fetch audio from {audio_url}: {e}")
        except httpx.RequestError as e:
            logger.error(f"Request error for {audio_url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for {audio_url}: {e}")

    def warmup_step(self):
        """预热session"""
        if self._session_warmup > 0:
            self._session_warmup -= 1
            self._clear_session()
