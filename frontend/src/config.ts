/** Central frontend configuration. The backend URL is set ONLY here (via VITE_API_BASE_URL). */

const rawBase = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000";

export const API_BASE_URL: string = rawBase.trim().replace(/\/+$/, "");

const rawMax = Number(import.meta.env.VITE_MAX_UPLOAD_MB);
export const MAX_UPLOAD_MB: number = Number.isFinite(rawMax) && rawMax > 0 ? rawMax : 20;

export const MAX_QUESTION_LENGTH = 2000; // must match the backend's MAX_QUESTION_CHARS limit
