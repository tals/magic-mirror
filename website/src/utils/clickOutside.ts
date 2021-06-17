type CallbackOrOptions = () => unknown | { filter?: string; callback: () => void };


/***
 *
 * A helper for click outside
 *
 * Usage:
 *
 * <div use:clickOutside={onHandler} />
 *
 */
import { defer } from "lodash"
import { tick } from "svelte";
import { sleep } from "./asyncUtils";

export function clickOutside(
  node: HTMLElement,
  optionsOrCallback: CallbackOrOptions
) {
  let options =
    typeof optionsOrCallback === "function"
      ? { callback: optionsOrCallback }
      : optionsOrCallback;

  async function onClickOutside(event: MouseEvent) {
    if (
      event.target &&
      !node.contains(event.target as Node) &&
      node !== event.target
    ) {
      // hack: this makes clickOutside work better when the user
      // clicks on a button that toggle the visibility of node
      await sleep(1)
      options.callback && options.callback();
    }
  }

  document.addEventListener("mouseup", onClickOutside);

  return {
    destroy: () =>
      document.removeEventListener("mouseup", onClickOutside),
  };
}
