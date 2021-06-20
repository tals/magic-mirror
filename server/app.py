record_to_disk = True

import functools
import argparse
import asyncio
from inference import get_clip_code
import logging
import sys
import time
import uuid
from pathlib import Path
from threading import Thread
from typing import *

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaRelay
from aiortc_utils import MediaRecorder
from av import VideoFrame
from fastai.vision.core import *
from fastapi import Body, FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from image_index import DATASET_PATH, QueryResultEntry

from interpolation_rife import rife_infer
from image_index import UserImageIndex

image_pool = UserImageIndex()


PY_ROOT = Path(__file__).parent
app = FastAPI(openapi_url=None)
# app = FastAPI()
SERVE_WEBSITE = False

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

if SERVE_WEBSITE:
    WEBSITE_ROOT = "../website/dist"
    app.mount("/assets", StaticFiles(directory=WEBSITE_ROOT / "assets"), name="assets")

    @app.get("/")
    async def index():
        return FileResponse(WEBSITE_ROOT / "index.html")


class Offer(BaseModel):
    sdp: Any
    type: Any


recorder = None
@app.post("/offer")
async def offer(offer: Offer, request: Request):
    global recorder
    print("offer recieved")
    offer = RTCSessionDescription(sdp=offer.sdp, type=offer.type)

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s" % request.client.host)

    # prepare local media
    # player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    video_dir = DATASET_PATH / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now().strftime("%Y%m%d-%H%M%S")

    
    if record_to_disk:
        recorder = MediaRecorder(str(video_dir / (prefix + ".mp4")))
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
        log_info(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        global transform_track
        log_info(f"Track {track.kind} received")

        if track.kind == "audio":
            # pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            recorder.addTrack(relay.subscribe(track))
            transform_track = VideoTransformTrack(
                relay.subscribe(track), transform=None
            )
            pc.addTrack(transform_track)

        @track.on("ended")
        async def on_ended():
            log_info(f"Track {track.kind} ended")
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.on_event("shutdown")
async def on_shutdown():
    print("Preparing to shutdown.")
    # close peer connections
    coros = [pc.close() for pc in pcs]
    print("Preparing to shutdown: waiting on PCs")
    await asyncio.gather(*coros)
    pcs.clear()
    print("Shut down!")


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



@functools.lru_cache(maxsize=1000)
def get_image_for_webrtc(file):
    x = Image.open(file)
    x = pil_to_webrtc(x)

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
        if not self.samples:
            return
            
        samples = np.array(self.samples) * 1000
        print(f"median {np.median(samples):n} min: {np.min(samples):n} max: {np.max(samples):n}")

    def reset(self):
        self.samples = []
        self._start = datetime.now()

class RunningStats:
    def __init__(self, size):
        self.size = size
        self.ema_alpha = 2/(size+1)
        self.ema = None
        self.window = []

    def add(self, x: np.ndarray):
        self.window.append(x)
        if len(self.window) > self.size:
            self.window = self.window[1:]

        if self.ema is None:
            self.ema = x
        else:
            self.ema = self.ema_alpha * x + (1 - self.ema_alpha) * self.ema

    def get_sma(self):
        return np.mean(self.window, axis=0)

    def get_ema(self):
        return self.ema

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
        self.last_used = None
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
        last_winner: Optional[QueryResultEntry] = None
        last_winner_dt = datetime.now()

        running_stats = RunningStats(5)
        last_shuffle_dt = datetime.now()
        while True:
            if self.pending is None:
                time.sleep(1 / 60)
                continue

            timer.on_start()

            img = self.pending
            self.pending = None

            # optimistic
            frame_code = get_clip_code(img)
            running_stats.add(frame_code.cpu().numpy())

            # not sure if i liked the smoothing
            # smoothed_code = torch.from_numpy(running_stats.get_ema()).cuda()
            smoothed_code = frame_code 

            results = image_pool.query(smoothed_code)
            infer_count += 1

            # write every 10th frame
            if infer_count % 10 == 0:
                # print("Adding!")
                image_pool.add(img=img, emb=frame_code)

            if infer_count % 30 == 0:
                timer.print_stats()
                timer.reset()
                
            if not results:
                # print("No results :(")
                continue

            winner = results[0]
            now = datetime.now()

            if (now - last_shuffle_dt).total_seconds() >= 1:
                last_shuffle_dt = now
                last_winner = winner
                image_pool.shuffle()


            if last_winner:
                last_score = last_winner.get_score_from(smoothed_code)
                if winner.score - last_score > 2e-2 or (now - last_winner_dt).total_seconds() >= .1:
                    last_winner = winner
                    last_winner_dt = now
                else:
                    # print("Overriding, D was", winner.score - last_score)
                    winner = last_winner
            else:
                last_winner = winner
                
            pic = get_image_for_webrtc(winner.path)
            self.pic = pic

            self.next_target = TF.to_tensor(pic)
            if self.prev is not None and self.target is not None:
                frames = rife_infer(
                    self.target[None].cuda(), self.next_target[None].cuda()
                )
                # TODO: make npy tensors
                frames = [TF.to_pil_image(x.squeeze()) for x in frames]
                self.upcoming = frames
            else:
                print("Can't do it")
                
            timer.lap()
            

    def get_result(self):
        if not self.frames:
            self.frames = self.upcoming
            self.prev = self.target
            self.target = self.next_target

        if self.frames:
            x = self.frames.pop(0)
            # TODO get rid of this conversion lol
            x = pil_to_webrtc(x)
            self.last_used = x
            return x
        else:
            return self.last_used

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
