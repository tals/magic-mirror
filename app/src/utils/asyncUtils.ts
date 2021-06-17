export class Deferred<T> {
  // hack: Promise constructor is guaranteed to be called before this returns
  resolve: ( value: T | PromiseLike<T>) => void = undefined!;
  reject: (reason: any) => void = undefined!;

  promise = new Promise<T>((_resolve, _reject) => {
    this.resolve = _resolve;
    this.reject = _reject;
  })
}

export async function sleep(ms: number) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
  }

export class Event {
  deferred = new Deferred<void>();
  set = () => this.deferred.resolve();
  clear = () => this.deferred = new Deferred<void>();
  wait = () => this.deferred.promise;
}
