export function getBooleanFlag(name: string, defaultValue: boolean): boolean {
  if (!process.env.browser) {
    return defaultValue;
  }
  let value = localStorage.getItem(name);
  if (value === null) {
    return defaultValue;
  }

  value = value.toLowerCase();
  return ["1", "true"].includes(value);
}

export function setBooleanFlag(name: string, value: boolean) {
  localStorage.setItem(name, value ? "true" : "false");
  return value;
}

export function getObject<T extends {}>(name: string): T | null {
  const value = localStorage.getItem(name);
  if (value === null) {
    return null;
  }

  return JSON.parse(value);
}

export function setObject<T extends {}>(name: string, value: T) {
  localStorage.setItem(name, JSON.stringify(value));
}
