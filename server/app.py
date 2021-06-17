# sys.path.insert(0, "/home/tal/dev/chroma/clip_search")
import argparse
import asyncio
import json
import logging
import os
import ssl
import sys
import time
import uuid
from pathlib import Path
from threading import Thread
from typing import *

import clip
import cv2
import kornia.augmentation as K
import numpy as np
import torch
import torchvision.transforms.functional as TF
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame
from fastai.vision.core import *
from fastapi import Body, FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from interpolation_rife import rife_infer
from image_index import UserImageIndex

image_pool = UserImageIndex()

# works with np, but the clip one assumes PIL
clip_norm = Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
)
# clip_model, _ = clip.load("ViT-B/32", jit=False)
clip_model, _ = clip.load("ViT-B/32")
clip_res = 224


ROOT = os.path.dirname(__file__)
# app = FastAPI(openapi_url=None)
app = FastAPI()
app.mount("/assets", StaticFiles(directory="/home/tal/dev/poo/assets"), name="assets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pcs = set()
relay = MediaRelay()

image_transform = None


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform=None):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform or "cartoon"

    async def recv(self):
        frame = await self.track.recv()
        if image_transform:
            tensor = frame.to_ndarray(format="rgb24")
            tensor = image_transform(tensor)

            # put it back together
            if tensor is None:
                return frame
            try:
                new_frame = VideoFrame.from_ndarray(tensor, format="rgb24")
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base
                return new_frame
            except Exception as e:
                logger.exception("Something bad happened")
                print("Something bad happened", e, tensor.shape)

        else:
            return self.base_transform(frame)
        # await asyncio.sleep(0.1)

    def base_transform(self, frame):
        if self.transform == "cartoon":
            img = frame.to_ndarray(format="bgr24")

            # prepare color
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            # prepare edges
            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            # combine color and edges
            img = cv2.bitwise_and(img_color, img_edges)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        else:
            return frame


@app.get("/")
async def index():
    return FileResponse("/home/tal/dev/poo/index.html")


class Offer(BaseModel):
    sdp: Any
    type: Any


@app.post("/offer")
async def offer(offer: Offer, request: Request):
    print("offer recieved")
    offer = RTCSessionDescription(sdp=offer.sdp, type=offer.type)

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.client.host)

    # prepare local media
    # player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        global transform_track
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            # pc.addTrack(player.audio)
            # recorder.addTrack(track)
            ...
        elif track.kind == "video":
            transform_track = VideoTransformTrack(
                relay.subscribe(track), transform=None
            )
            pc.addTrack(transform_track)
            if args.record_to:
                recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.on_event("shutdown")
async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


def main(raw_args=[]):
    import uvicorn

    global args
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=9999, help="Port for HTTP server (default: 9999)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args(raw_args)
    uvicorn.run(app, host="0.0.0.0", port=9999, loop="asyncio")

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x

    return TF.to_tensor(x)


# slightly modified from OpenAI's code, so that it works with np tensors
# see https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/clip.py#L58
clip_preprocess = Compose(
    [
        to_tensor,
        Resize(clip_res, interpolation=Image.BICUBIC),
        CenterCrop(clip_res),
        clip_norm,
    ]
)


def make_aug(x: torch.Tensor):
    if x.ndim < 4:
        x = x[None]

    x = x.repeat(8, 1, 1, 1)
    x = K.functional.random_affine(x, 30, (0.2, 0.2), (0.9, 1.5), [0.1, 0.4])
    x = K.functional.color_jitter(x, 0.2, 0.3, 0.2, 0.3)
    return x


@torch.no_grad()
def get_clip_code(img, use_aug=False):
    x = TF.to_tensor(img).cuda()
    if use_aug:
        x = make_aug(x)
    else:
        x = x[None]
    x = clip_preprocess(x)
    x = clip_infer(x)

    if use_aug:
        x = x.mean(axis=0, keepdim=True)

    # normalize since we do dot products lookups
    x /= x.norm()

    return x


def clip_infer(x):
    x = clip_model.encode_image(x)
    return x


def pil_to_webrtc(img):
    n_px = 360
    img = TF.resize(img, n_px, interpolation=Image.BICUBIC)
    img = TF.center_crop(img, n_px)
    img = np.array(img)
    if img.ndim == 2:
        img = img[:, :, None].repeat(3, axis=-1)
    else:
        img = img[:, :, :3]
    return img


cached_images = {}


def get_image_for_webrtc(file):
    # TODO use lru
    if file in cached_images:
        return cached_images[file]

    x = PILImage.create(file)
    x = pil_to_webrtc(x)
    cached_images[file] = x

    return x

from datetime import datetime
class Timer:
    _start: datetime
    def __init__(self):
        self.samples = []
        self._start = datetime.now()

    def lap(self):
        now = datetime.now()
        self.samples.append((now - self._start).total_seconds())
        self._start = now

    def on_start(self):
        self._start = datetime.now()

    def print_stats(self):
        samples = np.array(self.samples) * 1000
        print(f"median {np.median(samples):n} min: {np.min(samples):n} max: {np.max(samples):n}")

    def reset(self):
        self.samples = []
        self._start = datetime.now()


class RifeInfer:
    """
    I cant believe this worked at all lol

    Runs a worker thread for inference, as the data is piped to webrtc which runs off asyncio.
    """
    def __init__(self):
        self.pending = None

        self.result = None
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()
        self.frames = []

        self.pic = None
        # currently playing
        self.prev = None
        self.target = None

        # optimistic
        self.next_target = None
        self.upcoming = []

    @torch.no_grad()
    def run(self):
        infer_count = 0
        timer = Timer()
        while True:
            if self.pending is None:
                time.sleep(1 / 60)
                continue

            timer.on_start()

            img = self.pending

            # optimistic
            code = get_clip_code(img)
            results = image_pool.query(code)
            infer_count += 1
            if infer_count % 10 == 0:
                print("Adding!")
                image_pool.add(img=img, emb=code)
            if infer_count % 30 == 0:
                timer.print_stats()
                timer.reset()
                
            if not results:
                # print("No results :(")
                continue

            pic = get_image_for_webrtc(results[0].path)
            self.pic = pic

            self.next_target = TF.to_tensor(pic)
            if self.prev is not None and self.target is not None:
                frames = rife_infer(
                    self.target[None].cuda(), self.next_target[None].cuda()
                )
                frames = [TF.to_pil_image(x.squeeze()) for x in frames]
                self.upcoming = frames
            else:
                print("Can't do it")
                
            timer.lap()
            

    def get_result(self):
        # return self.pic

        # TODO figure this out
        if self.frames:
            x = self.frames.pop(0)
            x = pil_to_webrtc(x)
            return x

        self.frames = self.upcoming
        self.prev = self.target
        self.target = self.next_target
        if self.frames:
            x = self.frames.pop(0)
            x = pil_to_webrtc(x)
            return x

    def put_frame(self, img):
        self.pending = img


infer = RifeInfer()


def mm_glue(tensor):
    infer.put_frame(tensor)
    query_result = infer.get_result()
    #     print("got result")

    if query_result is None:
        return None

    return query_result

image_transform = mm_glue

if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
