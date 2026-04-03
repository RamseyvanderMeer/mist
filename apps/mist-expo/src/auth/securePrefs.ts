import * as SecureStore from 'expo-secure-store';
import { Platform } from 'react-native';

const WEB_PREFIX = 'mist_prefs_';

export const STORAGE_KEYS = {
  iapJwt: 'mist_iap_jwt',
  googleIdToken: 'mist_google_id_token',
  iapEmail: 'mist_iap_email',
  iapSubject: 'mist_iap_subject',
} as const;

async function setWeb(key: string, value: string | null) {
  const k = WEB_PREFIX + key;
  if (value === null) {
    sessionStorage.removeItem(k);
  } else {
    sessionStorage.setItem(k, value);
  }
}

async function getWeb(key: string): Promise<string | null> {
  return sessionStorage.getItem(WEB_PREFIX + key);
}

export async function setSecurePref(key: string, value: string | null): Promise<void> {
  if (Platform.OS === 'web') {
    await setWeb(key, value);
    return;
  }
  if (value === null) {
    await SecureStore.deleteItemAsync(key);
  } else {
    await SecureStore.setItemAsync(key, value);
  }
}

export async function getSecurePref(key: string): Promise<string | null> {
  if (Platform.OS === 'web') {
    return getWeb(key);
  }
  return SecureStore.getItemAsync(key);
}
