<script lang="ts">
      import { didAccept } from "./state";

    function letsGoClicked() {
        $didAccept = true;
    }

</script>
<div class="absolute inset-0 bg-white flex flex-col items-center justify-center p-4">
    <div class="text-4xl mb-4">Photoboo 👻</div>
        <div class="text-center"><span class="font-medium">Take a picture 📸</span> and and we'll show you the similar things others have seen.</div>
        
        <div class="mt-4">You picture will be searchable by others.</div>
    
         <div class="mt-4 opacity-90">Hope you like this experiement :)</div>
         <div class="opacity-90">- Tal</div>

    <button class="bg-black hover:bg-gray-700 text-white rounded-lg m-4 p-4 transition-all duration-200" on:click={letsGoClicked}>Lets go!</button>

</div>