<script lang="ts">
import Painterro from 'painterro'
import { onMount } from 'svelte';
let containerElement: HTMLElement;
export let canvasStream: MediaStream;
let painterro: any;
let canvas: HTMLCanvasElement;

let resizeCanvas = document.createElement("canvas");
resizeCanvas.width = 360;
resizeCanvas.height = 360;

onMount(() => {
    let interval = setInterval(() => {
        const ctx = resizeCanvas.getContext("2d")!;
        const dx = Math.max(0, (canvas.width - resizeCanvas.width) / 2)
        const dy = Math.max(0, (canvas.height - resizeCanvas.height) / 2)
        ctx.drawImage(canvas,-dx,-dy);

    }, 1/30)

    canvasStream = (resizeCanvas as any).captureStream(15)

    return () => {
        clearInterval(interval);
    }
})

$: if (containerElement) {
    painterro = Painterro({
        id: containerElement.id,
      activeColor: "#3B82F6", // default brush color is green
      defaultTool: "brush"
      // ... other params here
    })
    painterro.show();

    // disable hotkeys. thanks https://github.com/devforth/painterro/issues/97
    painterro.toolByKeyCode = {};
    painterro.inserter.handleOpen = () => false;

    canvas = document.getElementById("painterro-canvas") as HTMLCanvasElement;
}

</script>
<div class="bg-red-100 relative overflow-hidden h-full " >
    draw
    <div class="w-full h-full  border-2" bind:this={containerElement} id="painterro"></div>
</div>
