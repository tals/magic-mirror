import { localStorageStore } from "./utils/storeUtils";

export const didAccept = localStorageStore("accepted", false);
