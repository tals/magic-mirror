import { writable, derived, Readable } from "svelte/store";
import { setContext, getContext } from "svelte";
import delve from "dlv";
import { debounce, orderBy } from "lodash";
import type { EventEmitter } from "events";
import { setObject, getObject } from "../utils/localStorage";

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

export function fuzzyIntervalHandler(
  callback: Function,
  intervalMs: number,
  fuzzMS: number = 200
) {
  let timer: number;
  let canceling = false;

  function queueNext() {
    if (canceling) {
      return;
    }
    clearTimeout();
    timer = window.setTimeout(
      fuzzyTick,
      intervalMs + getRandomInt(-fuzzMS, fuzzMS)
    );
  }

  function fuzzyTick() {
    if (canceling) {
      return;
    }

    callback();
    queueNext();
  }

  queueNext();
  return () => {
    canceling = true;
    clearTimeout(timer);
  };
}

function getRandomInt(min: number, max: number) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min) + min); //The maximum is exclusive and the minimum is inclusive
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

/**
 * Adds the glue for a context-bound store
 */
export function contextualStore<R, A extends any[]>(
  factoryFn: (...args: A) => R,
  name: string
): [(...args: A) => R, () => R] {
  const KEY = {};
  function create(...args: A): R {
    const store = factoryFn(...args);
    setContext(KEY, store);

    return store;
  }

  function get() {
    const store = getContext(KEY);
    if (!store) {
      throw new Error(`No ${name} was found. please make sure create${name} is called beforehand`);
    }

    return store;
  }

  return [create, get] as any;
}

/**
 * Make a class-based store available as context
 * @param constructor
 * @param name
 */
export function contextualClassStore<Class, A extends any[]>(
  constructor: new (...args: A) => Class,
  name: string=constructor.name,
): [(...args: A) => Class, () => Class] {
  const KEY = {};
  function create(...args: A): Class {
    const store = new constructor(...args);
    setContext(KEY, store);

    return store;
  }

  function get() {
    const store = getContext(KEY);
    if (!store) {
      throw new Error(`No ${name} was found. please make sure create${name} is called beforehand`);
    }

    return store;
  }

  return [create, get] as any;
}

export const path = <T>(store: Readable<T>) => (
  key: string,
  defaultValue?: any
) => {
  return derived<Readable<{}>, any>(store, ($store: {}, set: any) => {
    let value = delve($store, key);
    if (value === undefined) {
      value = defaultValue;
    }
    set(value);
  });
};

export const collection = <T>(store: Readable<T>) => (
  key: string,
  sortKey: string = "id",
  sortDir: "asc" | "desc" = "asc"
) => {
  let last: any;
  return derived(path(store)(key), ($value: any, set: any) => {
    if ($value === last) {
      return;
    }

    last = $value;

    // filter out null values
    const ordered = orderBy($value || {}, sortKey, sortDir).filter((o) => !!o);
    set(ordered);
  });
};

export type Subscribe<T> = (
  run: (value: T) => void,
  invalidate?: ((value?: T | undefined) => void) | undefined
) => () => void;


export function localStorageStore<T>(name: string, defaultVal: T) {
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
