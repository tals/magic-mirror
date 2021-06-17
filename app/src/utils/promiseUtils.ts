/***
 * Wrap a call to a promise, in a way that won't call it again until it has been resolved
 * When you call it, it will wait for the active call
 *
 */
export function safePromiseFunc<TResult>(
  promiseGenerator: () => Promise<TResult>
): () => Promise<TResult> {
  let activePromise: Promise<TResult> | undefined;

  function call() {
    if (!activePromise) {
      activePromise = promiseGenerator();

      activePromise
        .then(() => (activePromise = undefined))
        .catch((e) => {
          activePromise = undefined;
          console.error("promise rejection", e);
        });
    }
    return activePromise;
  }

  return call;
}

/**
 * Runs a promise with a timeout
 *
 *
 * @param promise
 * @param timeoutInMS
 */
export async function promiseWithTimeout<TResult>(
  promise: Promise<TResult>,
  timeoutInMS: number
): Promise<TResult> {
  let solved = false;
  promise.then(() => (solved = true));

  const result = await Promise.race<TResult>([
    promise,
    new Promise<TResult>((resolve, reject) => {
      setTimeout(
        () => !solved && reject(new Error("Promise timeout")),
        timeoutInMS
      );
    }),
  ]);

  return result;
}
