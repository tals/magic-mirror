export function playStream(node: HTMLVideoElement, stream?: MediaStream) {
    node.srcObject = stream ?? null;
    return {
        update(newStream?: MediaStream) {
            node.srcObject = newStream ?? null;
        }
    }
}

export function closeStream(stream?: MediaStream): void {
    if (!stream) {
        return;
    }
    for (const track of stream.getTracks()) {
        track.stop();
    }
}