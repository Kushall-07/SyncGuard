/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Backend API origin for production builds; unset in local dev (see .env.production.example). */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
