import { writable } from "svelte/store";
import { debounce } from "lodash";
import type { EventEmitter } from "events";

// todo: make these "handler" patterns work like svelte actions. current design
// feels a little too mysterious

export function domEventHandler<E extends Event>(
  target: EventTarget,
  events: string | string[],
  f: (e: E) => void
) {
  if (typeof events === "string") {
    events = [events];
  }

  events.forEach((event) => target.addEventListener(event, f as EventListener));

  return () =>
    (events as string[]).forEach((event) =>
      target.removeEventListener(event, f as EventListener)
    );
}

export function setIntervalHandler(callback: Function, intervalMs: number) {
  const int = setInterval(callback, intervalMs);

  return () => {
    clearInterval(int);
  };
}

export function emitterEventHandler<E extends Event>(
  emitter: EventEmitter,
  event: string,
  f: (e: E) => void
) {
  emitter.addListener(event, f as EventListener);
  return () => emitter.removeListener(event, f as EventListener);
}

export function stickyStore<T>(initialValue: T) {
  const store = writable(initialValue);
  let timeout: any = 0;

  function reset() {
    store.set(initialValue);
    if (timeout) {
      clearTimeout(timeout);
    }
  }

  function setTemporary(val: T, duration: number) {
    store.set(val);
    if (timeout) {
      clearTimeout(timeout);
    }

    timeout = setTimeout(reset, duration);
  }

  return {
    subscribe: store.subscribe,
    setTemporary,
    reset,
  };
}

/**
 * Only applies the value after a certain timeout has passed.
 * Useful to sidestep flickering (element being hidden for a very brief period)
 */
export function debouncedWritable<T>(initialValue: T, timeout=100) {
  const store = writable(initialValue);

  const dset = debounce(store.set, timeout);
  function set(val: T) {
    dset(val);
    dset.flush();
  }

  return {
    subscribe: store.subscribe,
    set,
    setDeferred: dset,
  };
}

export type Subscribe<T> = (
  run: (value: T) => void,
  invalidate?: ((value?: T | undefined) => void) | undefined
) => () => void;


export function localStorageStore<T>(name: string, defaultVal: T) {
  function getObject<T extends {}>(name: string): T | null {
    const value = localStorage.getItem(name);
    if (value === null) {
      return null;
    }
  
    return JSON.parse(value);
  }
  
  function setObject<T extends {}>(name: string, value: T) {
    localStorage.setItem(name, JSON.stringify(value));
  }

  const initialValue = getObject<T>(name) ?? defaultVal;
  const store = writable<T>(initialValue);

  return {
    subscribe: store.subscribe,
    set: (val: T) => {
      store.set(val);
      setObject(name, val)
    }
  }
}
