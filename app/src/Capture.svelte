<script lang="ts">
  import camera_src from "./assets/camera.svg";
  import { fade, fly } from "svelte/transition";
  import { mediaDevices } from "./utils/mediaDevices";
  import { localStorageStore } from "./utils/storeUtils";
  import { createEventDispatcher } from "svelte";
  import { Camera } from "./camera";
  import type { FacingMode } from "./camera";
  import { playStream } from "./utils/webrtcUtils";
  import { sleep } from "./utils/asyncUtils";
  import { isMobile } from "./utils/device";
  const RES = 360;
  const dispatch = createEventDispatcher();

  let videoHeight: number;

  const selectedDeviceStore = localStorageStore("videoDevice", "");
  const facingModeStore = localStorageStore<FacingMode>("facingMode", "user");

  const camera = new Camera(RES, RES);
  $: camera.start({ deviceId: $selectedDeviceStore, facingMode: $facingModeStore });

  let mediaStream: MediaStream | undefined;
  let showSnapButton = true;

  camera.onFrame = () => {
    if (mediaStream != camera.mediaStream) {
      mediaStream = camera.mediaStream;
    }
  };

  let snapshotUrls: string[] = [];

  async function snapClicked() {
    countdown = 3;
    snapshotUrls = [];
    showSnapButton = false;
    while (countdown) {
      await sleep(1000);
      countdown = countdown - 1;
    }

    let blobs = [];
    for (let i = 0; i < 3; i++) {
      const blob = await camera.snap();
      blobs.push(blob);
      await sleep(1 / 60);
    }

    dispatch("snap", { blobs });
  }

  let countdown = 0;

  function onSwitchCamera() {
    if (isMobile) {
      $facingModeStore = $facingModeStore === "user" ? "environment" : "user";
    } else {
      const cams = $mediaDevices.video
      let idx = $selectedDeviceStore ? cams.findIndex(x => x.deviceId === $selectedDeviceStore) : 0;
      idx = (idx + 1) % cams.length;
      $selectedDeviceStore = cams[idx].deviceId;
    }
  }
</script>

<div class="flex flex-col items-center h-full">
  <div
    class="overflow-hidden relative flex items-center justify-center"
    style="width: fit-content;"
  >
    <video
      use:playStream={mediaStream}
      bind:videoHeight
      muted
      playsInline
      autoplay
      class="object-contain rounded block h-full overflow-hidden bg-gray-700"
      style="height: 360px; width: 360px;"
      on:dblclick={onSwitchCamera}
    />
    {#if countdown}
      {#key countdown}
        <div
          out:fade={{ duration: 200 }}
          in:fly={{ duration: 200, y: 20 }}
          class="absolute bottom-0 left-0 right-0 flex items-center justify-center"
        >
          <div class="text-white bg-black text-4xl w-12 rounded-t text-center">
            {countdown}
          </div>
        </div>
      {/key}
    {:else if showSnapButton}
      <button
        class="   border p-4 text-white absolute bottom-2 rounded-lg hover:bg-white hover:bg-opacity-10 transition-all duration-150 "
        on:click={snapClicked}
      >
        <img src={camera_src} />
      </button>
      <button
        class="   border p-2 text-white absolute top-2 right-2 rounded hover:bg-white hover:bg-opacity-10 transition-all duration-150 "
        on:click={onSwitchCamera}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          enable-background="new 0 0 24 24"
          height="24px"
          viewBox="0 0 24 24"
          width="24px"
          fill="#ffffff"
          ><g><path d="M0,0h24v24H0V0z" fill="none" /></g><g
            ><g
              ><path
                d="M16,7h-1l-1-1h-4L9,7H8C6.9,7,6,7.9,6,9v6c0,1.1,0.9,2,2,2h8c1.1,0,2-0.9,2-2V9C18,7.9,17.1,7,16,7z M16,15H8V9h1.83l1-1 h2.34l1,1H16V15z"
              /><circle cx="12" cy="12" r="2" /><path
                d="M8.57,0.52L13.05,5l1.41-1.41l-1.54-1.54C17.7,2.46,21.53,6.24,22,11h2C23.36,3.3,15.79-1.67,8.57,0.52z"
              /><path
                d="M9.54,20.41l1.54,1.54C6.3,21.54,2.47,17.76,2,13H0c0.64,7.7,8.21,12.67,15.43,10.48L10.95,19L9.54,20.41z"
              /></g
            ></g
          ></svg
        >
      </button>
    {/if}
  </div>
  <!-- <div class="mt-4">
    <button
      class=" text-4xl hover:bg-blue-700 border p-4 text-white"
      on:click={() => (selectorOpen = true)}>⚙️</button
    >
  </div> -->
</div>

<style>
  .container {
    display: grid;
    grid-template-rows: 1fr max-content;
    grid-template-columns: 1fr;
  }
</style>
