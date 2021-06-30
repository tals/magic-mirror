<script lang="ts">
  import "./index.css";
  import "./Tailwind.svelte";
  import WebrtcApp from "./webrtcApp.svelte";
  const endpoint = location.search.slice(1);
</script>

<main class="bg-gray-700" style="height: 100vh;" >
  {#if endpoint}
    <WebrtcApp {endpoint} />
  {:else}
    <div class="p-4 text-gray-100 h-full">
    <div class="text-xl mb-1">Magic Mirror Instructions</div>

      <div class="mb-4">
        Please use <span class="font-mono"
          >{location.origin}{location.pathname}?{location.protocol}//my_backend_server</span
        >
      </div>
      <div>
        Heads up: if you're using <span class="font-mono">https://party.photoboo.app</span>,
        make sure to
        <a
          class="underline"
          href="https://experienceleague.adobe.com/docs/target/using/experiences/vec/troubleshoot-composer/mixed-content.html?lang=en"
        >
          allow insecure content
        </a>
      </div>
    </div>
  {/if}
</main>
