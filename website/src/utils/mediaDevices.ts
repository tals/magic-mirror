import { readable, Readable } from "svelte/store";
import { domEventHandler } from "../utils/storeUtils";

export type MediaDevicesData = {
  hasPermissions: boolean;
  audioDeviceById: Record<string, UserMediaDevice>;
  videoDeviceById: Record<string, UserMediaDevice>;
  audio: UserMediaDevice[];
  video: UserMediaDevice[];
  anyCamera: boolean;
  anyMicrophone: boolean;
};

export type UserMediaDevice = {
  deviceId: string;
  groupId: string;
  kind: string;
  label?: string;
};

function mediaDevicesStore(): Readable<MediaDevicesData> & {
  refresh: () => Promise<void>;
} {
  // An ugly hack to expose the try refresh method
  let tryRefreshInternal: () => Promise<void>;

  const store = readable<MediaDevicesData>(
    {
      hasPermissions: false,
      audioDeviceById: {},
      videoDeviceById: {},
      audio: [],
      video: [],
      anyCamera: false,
      anyMicrophone: false,
    },
    (set) => {
      /// assume we have no permissions
      let hasPermissions: boolean = false;

      // we start by subscribe to device changes

      const unsubscribe = domEventHandler(
        navigator.mediaDevices,
        "devicechange",
        onDeviceChanged
      );

      // Update user media when devices change
      function onDeviceChanged(ev: Event) {
        // Devices may have changed, let's try and refresh our list of devices
        tryRefreshDeviceList();
      }

      async function tryRefreshDeviceList() {
        // Let's refresh our devices, this is how we do it:
        let devices: MediaDeviceInfo[] = [];

        try {
          devices = await safeEnumerateDevices();
        } catch (e) {
          console.error("tryRefreshDeviceList", e);
          // we're not going to update anything, so bye bye here
          return;
        }

        if (hasPermissions && !devices.length) {
          // looks like we have an issue, because we have an empty device list
          console.error("tryRefreshDeviceList", "Got an empty list of devices");
          // nothing to update
          return;
        }

        // If we don't get labels on *ALL* devices, we don't have permissions
        hasPermissions = !devices.every((d) => !d.label);

        const {
          audio,
          video,
          audioDeviceById,
          videoDeviceById,
        } = collectDevices(devices);

        // update our store
        set({
          audio,
          audioDeviceById,
          videoDeviceById,
          hasPermissions,
          video,
          anyCamera: !!video.length,
          anyMicrophone: !!audio.length,
        });
      }

      tryRefreshInternal = tryRefreshDeviceList;

      // we start by trying to get our device list
      tryRefreshDeviceList();

      return () => {
        // cleanup
        unsubscribe();
      };
    }
  );

  return {
    subscribe: store.subscribe,
    async refresh() {
      if (tryRefreshInternal) {
        return tryRefreshInternal();
      }
    },
  };
}

// Timeout after 5s
const FETCH_DEVICES_TIMEOUT = 5 * 1000;
const safeEnumerateDevices = () => navigator.mediaDevices.enumerateDevices()

// static const here
export const mediaDevices = mediaDevicesStore();

type CollectedMediaDevices = {
  audio: UserMediaDevice[];
  video: UserMediaDevice[];
  audioDeviceById: Record<string, UserMediaDevice>;
  videoDeviceById: Record<string, UserMediaDevice>;
};

function collectDevices(devices: MediaDeviceInfo[]): CollectedMediaDevices {
  const result: CollectedMediaDevices = {
    audio: [],
    video: [],
    audioDeviceById: {},
    videoDeviceById: {},
  };

  devices.forEach((d) => {
    const item = {
      deviceId: d.deviceId,
      label: d.label,
      kind: d.kind,
      groupId: d.groupId,
    };

    // We only care for input devices
    switch (item.kind) {
      case "audioinput":
        result.audioDeviceById[item.deviceId] = item;
        result.audio.push(item);
        break;
      case "videoinput":
        result.videoDeviceById[item.deviceId] = item;
        result.video.push(item);
        break;
    }
  });

  return result;
}
