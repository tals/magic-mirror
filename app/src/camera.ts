import { closeStream } from "./utils/webrtcUtils";
import { Event } from "./utils/asyncUtils";
import { isMobile } from "./utils/device";
import { writable } from "svelte/store";
export type FacingMode = "user" | "environment";
export interface CameraOptions {
  facingMode?: FacingMode;
  deviceId?: string;
}

export enum CameraState {
  pending,
  open,
  error,
}
interface CameraStore {
  mediaStream?: MediaStream;
  state: CameraState;
}

export class Camera {
  mediaStream?: MediaStream;
  video: HTMLVideoElement;
  raf: number = 0;
  lastTime: any;
  onFrame?: (frame: HTMLVideoElement) => void | Promise<void>;
  onOpen?: (frame: HTMLVideoElement) => void | Promise<void>;
  deviceId?: string;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  store = writable<CameraStore>({state: CameraState.pending})
  private gotFrame: Event = new Event();
  constructor(width: number, height: number) {
    this.video = document.createElement("video");
    this.video.muted = true;
    this.video.autoplay = true;
    this.video.playsInline = true;

    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d")!;
    this.video.width = width;
    this.video.height = height;

    this.canvas.width = width;
    this.canvas.height = height;
  }

  async update() {
    const currTime = this.video.currentTime;
    if (this.lastTime === currTime) {
      this.scheduleUpdate();
      return;
    }

    this.lastTime = currTime;
    this.gotFrame.set();
    if (this.onFrame) {
      const res = this.onFrame(this.video);
      if (res && res["then"]) {
        await res;
      }
    }
    this.scheduleUpdate();
  }

  scheduleUpdate() {
    if (this.raf) {
      return;
    }
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      this.update();
    });
  }

  async start(opts?: CameraOptions) {
    this.stop();
    this.store.set({
      state: CameraState.pending
    });

    this.deviceId = opts?.deviceId;

    const { width, height } = this.video;
    const facingMode = isMobile && opts?.facingMode && { exact: opts?.facingMode }
    const constraints: MediaStreamConstraints = {
      video: {
        facingMode: facingMode || {},
        deviceId: opts?.deviceId,
        width,
        height,
      },
      audio: false,
    };

    for (let i = 0; i < 3; i++) {
      try {
        this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        break;
      } catch(e) {
        if (e instanceof OverconstrainedError) {
          delete (constraints.video as any)[e.constraint];
        }
        else {
          break;
        }
      }
    }
    if (!this.mediaStream) {
      this.store.set({state: CameraState.error});
      return;
    }

    this.store.set({
      mediaStream: this.mediaStream, state: CameraState.open
    });

    this.video.width = width;
    this.video.height = height;
    this.video.srcObject = this.mediaStream;
    this.video.addEventListener("loadedmetadata", (e) => {
      console.log("[camera] got metadata");
      this.video.play();
      this.scheduleUpdate();
    });
  }

  async snap() {
    await this.gotFrame.wait();
    this.gotFrame.clear();

    this.ctx.drawImage(this.video, 0, 0);
    const blob = await convertToBlob(this.canvas, "image/jpeg");
    return blob;
  }

  async stop() {
    if (this.raf) {
      cancelAnimationFrame(this.raf);
      this.raf = 0;
    }
    if (this.mediaStream) {
      closeStream(this.mediaStream);
      this.mediaStream = undefined;
    }
  }
}

function convertToBlob(
  canvas: HTMLCanvasElement,
  type?: "image/jpeg" | "image/png",
  quality?: number
): Promise<Blob | null> {
  return new Promise((res, rej) => {
    canvas.toBlob(res, type, quality);
  });
}
