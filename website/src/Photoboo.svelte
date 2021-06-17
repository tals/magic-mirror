<script lang="ts">
  import camera_src from "./assets/camera.svg";
  import { debounce, range, sortBy } from "lodash";
  import { fade, fly } from "svelte/transition";
  import Capture from "./Capture.svelte";
  // import PhotoCapture from "./PhotoCapture.svelte";
  import "./Tailwind.svelte";
  import { onMount } from "svelte";
  import sockette from "sockette";
  let count = 0;
  onMount(() => {
    const interval = setInterval(() => count++, 1000);
    return () => {
      clearInterval(interval);
    };
  });

  let results: any[] = [];
  let blob: any;
  function onSnap({ detail }: any) {
    const blobs = detail.blobs;
    blob = blobs[0];
  }

  $: query(blob);

  interface QueryResultEntry {
    key: string;
    score: number;
  }

  interface QueryResult {
    oi: QueryResultEntry[];
    user: QueryResultEntry[];
  }

  async function query(blob: Blob) {
    if (!blob) return;
    mainPic = URL.createObjectURL(blob);
    let start = +new Date();
    results = [];
    const params = new URLSearchParams();
    var data = new FormData();
    data.append("file", blob, "snap");
    // data.append("file", "mean")
    const resp = await fetch(`https://photoboo.app/query/`, {
      method: "POST",
      body: data,
      mode: "cors",
      credentials: "include",
      // headers: {
      //   "Content-Type": "multipart/form-data"
      // }
    });
    if (!resp.ok) return;
    const val: QueryResult = await resp.json();
    results = [
      ...val.user.map((x) => ({
        ...x,
        url: `https://cdn.photoboo.app/${x.key}.jpg`,
      })),
      ...val.oi.map((x) => ({
        ...x,
        url: `https://s3.amazonaws.com/open-images-dataset/${x.key}.jpg`,
      })),
    ];

    results = sortBy(results, (x) => x.score);
    results.reverse();
    results = results.filter((x) => x.score > 60);
    console.log("query took", +new Date() - start);
  }
  let st = 0;
  async function queryWS(blob: Blob) {
    if (!blob) return;
    results = [];
    const payload = await blob.arrayBuffer();
    st = +new Date();
    connection.send(payload);
  }
  function onmessage(ev: MessageEvent) {
    console.log(ev.data);
    results = JSON.parse(ev.data);
    const dl = +new Date() - st;
    console.log("queryWS took", dl);
  }

  const connection = new sockette("wss://photoboo.app/query_ws/", {
    onmessage,
  });
  // const connection = new WebSocket("ws://molsh-9000:8080/query_ws/");
  // connection.addEventListener("message", onmessage);
  let mainPic: string | undefined;

  function resetClicked() {
    mainPic = undefined;
    results = [];
  }
</script>

<div style="height: 100vh;">
  <div class="px-2 border-b py-2 text-sm fixed z-50 w-full bg-white">
    Photoboo 👻
  </div>
  <div class="" style="height:46px" />
  <div
    class="overflow-hidden transition-all duration-200 relative"
    class:container-lg={!mainPic}
    class:container-sm={mainPic}
  >
    {#if mainPic}
      <div
        class="h-full w-full absolute flex items-center justify-center"
        transition:fade={{ duration: 200 }}
      >
        <img class="object-contain block rounded h-full" src={mainPic} />
        <button
          on:click={resetClicked}
          class="  border p-2 text-white absolute bottom-2 rounded hover:bg-white hover:bg-opacity-10 transition-all duration-150 "
        >
          <img src={camera_src} />
        </button>
      </div>
    {:else}
      <div class="h-full" transition:fade={{ duration: 200 }}>
        <Capture on:snap={onSnap} />
      </div>
    {/if}
  </div>
  <div class="mt-2">
    <div class="flex flex-wrap items-center justify-evenly">
      {#each results as e (e.key)}
        <div
          class=" group m-1 rounded bg-gray-100 flex h-24 relative mr-2"
          in:fly={{ y: 20, duration: 400 }}
        >
          <img class="object-cover rounded-l h-24 w-24" src={e.url} />
          <!-- <div class="w-2 bg-red-500 inline-block rounded-r " style="height: {e.score}%; margin-top: {100-e.score}%;"></div> -->
          <div
            class="w-2  inline-block rounded-r absolute bottom-0 -right-2 "
            style="height: {e.score}%;"
            class:bg-green-600={e.score >= 60}
            class:bg-yellow-300={e.score < 60}
          />
        </div>
      {:else}
        {#if mainPic}
          {#each range(12) as _}
            <div
              class="animate-pulse  relative group m-1 rounded bg-gray-100 p-1 h-24 w-24"
            />
          {/each}
        {/if}
      {/each}
    </div>
  </div>
</div>

<style>
  .container-lg {
    height: 50%;
  }
  .container-sm {
    height: 25%;
  }
  .thumbnail {
    width: 108px;
    height: 108px;
  }
</style>
