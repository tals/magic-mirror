import Bowser from "bowser";
export const browser = Bowser.getParser(window.navigator.userAgent);
export const isMobile = browser.getPlatformType() === "mobile";