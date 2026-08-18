const fs = require('fs');
const path = require('path');

const DEFAULT_SERVER_URL = 'http://159.194.202.216';

function normalizeServerUrl(value) {
  if (typeof value !== 'string' || !value.trim()) {
    return null;
  }
  try {
    const url = new URL(value.trim());
    if (!url.protocol.startsWith('http')) {
      return null;
    }
    return url.origin;
  } catch {
    return null;
  }
}

function readJsonConfig(filePath) {
  try {
    if (!fs.existsSync(filePath)) {
      return null;
    }
    const parsed = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    return normalizeServerUrl(parsed?.url || parsed?.serverUrl);
  } catch {
    return null;
  }
}

function resolveServerUrl(options = {}) {
  const fromEnv = normalizeServerUrl(process.env.TRACKAI_SERVER_URL || process.env.TRACKAI_APP_URL);
  if (fromEnv) {
    return fromEnv;
  }

  const resourceRoots = [];
  if (options.resourcesPath) {
    resourceRoots.push(options.resourcesPath);
  }
  if (process.resourcesPath) {
    resourceRoots.push(process.resourcesPath);
  }

  for (const root of resourceRoots) {
    const fromResources = readJsonConfig(path.join(root, 'server.json'));
    if (fromResources) {
      return fromResources;
    }
  }

  if (options.userDataPath) {
    const fromUserData = readJsonConfig(path.join(options.userDataPath, 'server.json'));
    if (fromUserData) {
      return fromUserData;
    }
  }

  return DEFAULT_SERVER_URL;
}

module.exports = {
  DEFAULT_SERVER_URL,
  normalizeServerUrl,
  resolveServerUrl,
};
