# MIST Expo client (web + iOS + Android)

Tamagui (theme) + Expo Router + TanStack Query. Consumes the FastAPI service at the repo root.

## App structure

| Route | Purpose |
|-------|---------|
| `/` | Splash → redirects to `/sign-in` or `/home` |
| `/sign-in` | Credentials, **Continue**, optional **Register** if IAP identity exists but no DB user |
| `/home` | Dashboard: profile, tier, API health, link to diagnosis |
| `/diagnose` | Full workflow: fault codes / symptoms → `/query` → `/clarify` loop → results → rating + outcome feedback |

Tab bar: **Home** · **Diagnose** (authenticated users only).

## Setup

```bash
cd apps/mist-expo
cp .env.example .env
# Edit EXPO_PUBLIC_MIST_API_URL (default http://127.0.0.1:8000)
npm install
```

## Run

```bash
npm run start
# w / i / a — web, iOS, Android
```

## Auth

See [AUTH.md](./AUTH.md). The sign-in screen stores IAP email/JWT and optional subject; `SessionProvider` calls `/auth/check` and `/auth/me` after credentials change.

Direct dependencies like `babel-preset-expo`, `react-dom`, `react-native-worklets`, and `@expo/vector-icons` are listed so Metro/Babel can resolve them from this app root.

## EAS / stores

Configure bundle IDs in `app.json`, then [EAS Build](https://docs.expo.dev/build/introduction/) for store binaries. Native sign-in still needs OAuth/BFF or WebView (see AUTH.md).
