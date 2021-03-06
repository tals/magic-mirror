<script lang="ts">
  import { playStream, closeStream } from "./utils/webrtcUtils";
  import { mediaDevices } from "./utils/mediaDevices";
  import { localStorageStore } from "./utils/storeUtils";
  import { onMount } from "svelte";
  import {findIndex} from "lodash";

  const selectedDeviceStore = localStorageStore("videoDevice", "");
  export let endpoint: string;

  let localCamera: MediaStream | undefined; // @hmr:keep
  let localStream: MediaStream | undefined; // @hmr:keep
  let remoteStream: MediaStream | undefined; // @hmr:keep

  function goLive() {
    if (!localStream) return;
    start(localStream);
  }

  async function getCamera(deviceId: string) {
    if (localCamera) {
      closeStream(localCamera);
      localCamera = undefined;
    }

    localCamera = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId,
        width: 360,
        height: 360,
      },
      audio: false,
    });
    localCamera.getVideoTracks()[0];

    // hack: set it as the default
    localStream = localCamera;
  }
  $: getCamera($selectedDeviceStore);

  // peer connection
  let pc: RTCPeerConnection | undefined; // @hmr:keep;

  // data channel
  let dc: any = null; // @hmr:keep
  let dcInterval: any = null; // @hmr:keep
  let iceGatheringState = "?"; // @hmr:keep
  let iceConnectionState: RTCIceConnectionState | "?" = "?"; // @hmr:keep
  let signalingState = "?"; // @hmr:keep
  function log(s: any) {
    console.log("[webrtc]", s);
  }
  function createPeerConnection() {
    var config: RTCConfiguration = {
      // sdpSemantics: "unified-plan",

      iceServers: [{ urls: ["stun:stun.l.google.com:19302"] }],
    };

    pc = new RTCPeerConnection(config);

    // register some listeners to help debugging
    pc.addEventListener(
      "icegatheringstatechange",
      function () {
        log(pc!.iceGatheringState);
        iceGatheringState = pc!.iceGatheringState;
      },
      false
    );
    log(pc.iceGatheringState);
    iceGatheringState = pc!.iceGatheringState;

    pc.addEventListener(
      "iceconnectionstatechange",
      function () {
        log(pc!.iceConnectionState);
        iceConnectionState = pc!.iceConnectionState;
      },
      false
    );

    log(pc.iceConnectionState);
    iceConnectionState = pc.iceConnectionState;

    pc.addEventListener(
      "signalingstatechange",
      function () {
        log(`signalingState: ${pc!.signalingState}`);
        signalingState = pc!.signalingState;
      },
      false
    );
    log(`signalingState: ${pc!.signalingState}`);
    signalingState = pc!.signalingState;

    // connect audio / video
    pc.addEventListener("track", function (evt) {
      if (evt.track.kind == "video") {
        remoteStream = evt.streams[0];
      }
    });

    return pc;
  }

  function waitForState(pc: RTCPeerConnection) {
    // wait for ICE gathering to complete
    return new Promise<void>((resolve) => {
      if (pc.iceGatheringState === "complete") {
        resolve();
      } else {
        function onIceCandidate(iceCandidate: RTCPeerConnectionIceEvent) {
          resolve();
        }
        pc.addEventListener("icecandidate", onIceCandidate, {
          once: true,
        });

        function checkState() {
          if (pc.iceGatheringState === "complete") {
            pc.removeEventListener("icegatheringstatechange", checkState);
            resolve();
          }
        }
        pc.addEventListener("icegatheringstatechange", checkState);
      }
    });
    
  }
  async function negotiate() {
    if (!pc) return;

    const localOffer = await pc.createOffer();
    await pc.setLocalDescription(localOffer);
    await waitForState(pc);
    const offer = pc.localDescription!;
    const audioCodec = "default";

    let { sdp } = offer;
    if (audioCodec !== "default") {
      sdp = sdpFilterCodec("audio", audioCodec, sdp);
    }
    const videoCodec = "default";
    if (videoCodec !== "default") {
      sdp = sdpFilterCodec("video", videoCodec, sdp);
    }

    const res = await fetch(`${endpoint}/offer`, {
      body: JSON.stringify({
        sdp,
        type: offer.type,
      }),
      mode: "cors",
      headers: {
        "Content-Type": "application/json",
      },
      method: "POST",
    });
    const answer = await res.json();
    await pc.setRemoteDescription(answer);
  }

  function start(stream: MediaStream) {
    pc = createPeerConnection()!;
    let time_start = new Date().getTime();

    function current_stamp() {
      if (time_start === null) {
        time_start = new Date().getTime();
        return 0;
      } else {
        return new Date().getTime() - time_start;
      }
    }

    const useDataChannel = true;
    if (useDataChannel) {
      dc = pc.createDataChannel("chat", { ordered: true });
      dc.onclose = function () {
        clearInterval(dcInterval);
      };
      dc.onopen = function () {
        dcInterval = setInterval(function () {
          var message = "ping " + current_stamp();
          dc.send(message);
        }, 1000);
      };
      dc.onmessage = function (evt: any) {
        log(`dataChannel ${evt.data}`);
        if (evt.data.substring(0, 4) === "pong") {
          var elapsed_ms =
            current_stamp() - parseInt(evt.data.substring(5), 10);
          log(" RTT " + elapsed_ms + " ms\n");
        }
      };
    }

    // doing this later seems problematic
    setLocalStream(localStream!);

    negotiate();
  }

  function stop() {
    // close data channel
    if (dc) {
      dc.close();
    }

    if (pc) {
      // close transceivers
      if (pc?.getTransceivers) {
        pc.getTransceivers().forEach(function (transceiver) {
          if (transceiver.stop) {
            transceiver.stop();
          }
        });
      }

      // close local audio / video
      pc.getSenders().forEach(function (sender) {
        sender?.track?.stop();
      });

      // close peer connection
      setTimeout(function () {
        pc?.close();
      }, 500);
    }
  }

  function sdpFilterCodec(kind: string, codec: string, realSdp: string) {
    var allowed = [];
    var rtxRegex = new RegExp("a=fmtp:(\\d+) apt=(\\d+)\r$");
    var codecRegex = new RegExp("a=rtpmap:([0-9]+) " + escapeRegExp(codec));
    var videoRegex = new RegExp("(m=" + kind + " .*?)( ([0-9]+))*\\s*$");

    var lines = realSdp.split("\n");

    var isKind = false;
    for (var i = 0; i < lines.length; i++) {
      if (lines[i].startsWith("m=" + kind + " ")) {
        isKind = true;
      } else if (lines[i].startsWith("m=")) {
        isKind = false;
      }

      if (isKind) {
        var match = lines[i].match(codecRegex);
        if (match) {
          allowed.push(parseInt(match[1]));
        }

        match = lines[i].match(rtxRegex);
        if (match && allowed.includes(parseInt(match[2]))) {
          allowed.push(parseInt(match[1]));
        }
      }
    }

    var skipRegex = "a=(fmtp|rtcp-fb|rtpmap):([0-9]+)";
    var sdp = "";

    isKind = false;
    for (var i = 0; i < lines.length; i++) {
      if (lines[i].startsWith("m=" + kind + " ")) {
        isKind = true;
      } else if (lines[i].startsWith("m=")) {
        isKind = false;
      }

      if (isKind) {
        var skipMatch = lines[i].match(skipRegex);
        if (skipMatch && !allowed.includes(parseInt(skipMatch[2]))) {
          continue;
        } else if (lines[i].match(videoRegex)) {
          sdp += lines[i].replace(videoRegex, "$1 " + allowed.join(" ")) + "\n";
        } else {
          sdp += lines[i] + "\n";
        }
      } else {
        sdp += lines[i] + "\n";
      }
    }

    return sdp;
  }

  function escapeRegExp(string: string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); // $& means the whole matched string
  }

  function setLocalStream(stream: MediaStream) {
    if (!stream) return;
    if (pc && pc.getSenders().length == 0) {
      for (const track of stream.getTracks()) {
        pc.addTrack(track, stream);
      }
    }

    localStream = stream;
  }

  $: setLocalStream(localCamera!);

  $: if(iceConnectionState === "disconnected") {
    window.location.reload();
  }

  onMount(async () => {
    setTimeout(goLive, 1000);
    setTimeout(runWatchdog, 10000);
  })

  function runWatchdog() {
    if (iceConnectionState !== "connected") {
      window.location.reload();
    }
  }

  function onKeyDown(e){
    if (e.key !== "s") {
      return;
    }

    let idx = findIndex($mediaDevices.video, x => x.deviceId === $selectedDeviceStore);
    if (idx === -1) {
      idx = 0;
    }

    idx = (idx + 1) % $mediaDevices.video.length;
    $selectedDeviceStore = $mediaDevices.video[idx].deviceId;
  }
</script>

<svelte:window on:keydown={onKeyDown}/>
<div style="height: 100vh;" >
  <div class="px-2 border-b border-gray-500 py-2 text-xs fixed z-50 w-full text-gray-200">
    Using {endpoint}: 
    {signalingState}
    {iceConnectionState}
  </div>
  <div class="" style="height:46px" />
  <div
    class="transition-all duration-200 relative h-full"
  >
    <div class="h-full w-full" >
      <div
        class=" overflow-hidden relative flex items-center justify-center "
        style="width: 100%; height: calc(100vh - 45px);"
      >
      <div class="h-full flex  justify-center flex-col p-4">
        <video
          use:playStream={remoteStream}
          muted
          playsInline
          autoplay
          class="object-contain  block  bg-gray-700 h-full mirror filter sepia"
        />
      </div>
        <video
          on:click={goLive}
          use:playStream={localStream}
          muted
          playsInline
          autoplay
          class="h-32 w-32 absolute bottom-4 left-4 object-contain rounded block overflow-hidden bg-gray-700 mirror"
        />
      </div>
    </div>
  </div>
</div>
<style>
    video.mirror {
    transform: scaleX(-1);
  }
</style>