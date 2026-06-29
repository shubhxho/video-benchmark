"""S3 video source with presigned URL streaming."""

from __future__ import annotations

from pathlib import PurePosixPath

import boto3

from video_benchmark.sources.base import VideoFile, VideoSource


class S3VideoSource(VideoSource):
    def __init__(self, bucket: str, prefix: str = "", presign_expiry: int = 7200) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.presign_expiry = presign_expiry
        self.s3 = boto3.client("s3")

    def list_videos(self) -> list[VideoFile]:
        videos: list[VideoFile] = []
        paginator = self.s3.get_paginator("list_objects_v2")
        page_kwargs: dict[str, str] = {"Bucket": self.bucket}
        if self.prefix:
            page_kwargs["Prefix"] = self.prefix

        for page in paginator.paginate(**page_kwargs):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.lower().endswith(".mp4"):
                    continue
                p = PurePosixPath(key)
                operator_id = p.parent.name if p.parent.name else "unknown"
                presigned_url = self.s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket, "Key": key},
                    ExpiresIn=self.presign_expiry,
                )
                videos.append(
                    VideoFile(
                        operator_id=operator_id,
                        video_path=presigned_url,
                        filename=p.name,
                    )
                )
        return videos
