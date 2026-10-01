# Phoenix — AI Proctoring Platform

## What's in this build

**RBAC** — Clerk on both ends. `TeacherRoute` in `App.jsx` bounces students away
from `/create-exam`, `/add-questions`, `/exam-logs`, `/student-review/...`; the
backend's `requireTeacher` checks `publicMetadata.role` before serving any
teacher-only route. No sign-up/domain restriction in this build.

**Exam authoring (teacher)**
- Create Exam, Add Questions (unchanged from earlier builds).
- Cheat Log dashboard (`ExamLogs.jsx`) — per-exam dropdown, name/email filter,
  aggregated counts, and now a **View** button per row linking to a full
  per-student review page.
- **Student Review page** (`StudentReview.jsx`, new): the complete violation
  timeline for one student's attempt — timestamps, detail strings, and the
  captured evidence snapshot for each flag, plus a **Reinstate** button for
  attempts that were auto-terminated.

**Exam taking (student) — `ExamRoom.jsx`**
This is the significantly rewritten core of this build:
- **Pre-exam system check + calibration**: camera preview with a live lighting
  advisory (it does not block a student whose integrated laptop webcam reports
  a dark auto-exposure value), then a lightweight calibration step that samples
  12 frames of
  the student looking straight at the screen to compute a baseline
  yaw/pitch. Gaze deviation during the actual exam is measured as *deviation
  from this baseline*, not a fixed absolute angle — this is what makes gaze
  detection tolerant of off-center camera placement. Nothing is logged as a
  violation until this screen is complete, which is also what keeps the
  earlier permission-prompt-blur bug from recurring: the real timer and the
  blur/fullscreen listeners are only attached ~1.5s after
  `systemCheckPassed` becomes true.
- **Face detection** (`FaceLandmarker`, `delegate: 'CPU'`): explicit
  confidence thresholds (`minFaceDetectionConfidence: 0.6`,
  `minFacePresenceConfidence`/`minTrackingConfidence: 0.5`) rather than
  library defaults, plus streak-based persistence — needs 5+ consecutive
  frames of "no face" or "2+ faces" before it counts, filtering single-frame
  flicker.
- **Gaze deviation**: yaw/pitch are smoothed over the last
  `GAZE_SMOOTHING_WINDOW = 5` readings (≈a quarter-second at 30fps) before
  comparing to the calibrated baseline, so one noisy frame landing just past
  the ±20° threshold doesn't trigger a violation on sensor jitter alone.
- **Object detection** (`ObjectDetector`, EfficientDet-Lite0,
  `delegate: 'CPU'`): sampled every 3rd frame (`OBJECT_DETECT_EVERY_N_FRAMES`,
  raised from every 5th now that there's CPU headroom), `scoreThreshold: 0.6`
  (raised from 0.5), and — the significant change — **persistence required**:
  a label (`cell phone` → Cell Phone Detected;
  `book`/`laptop`/`keyboard`/`remote`/`tv`/`mouse` → Prohibited Object
  Detected) must appear in `OBJECT_DETECT_STREAK_REQUIRED = 3` consecutive
  *sampled* frames before it's logged, tracked per-label in
  `cvStateRef.current.objectStreaks`. Previously a single frame at 0.5
  confidence logged a strike immediately; this was the biggest source of
  false positives in the earlier build. Confidence score and the streak
  length are both logged in the detail string.
- **Webcam tamper detection — reimplemented in plain JS/Canvas, no OpenCV**
  (sampled every 30th frame, on the same 32×24 downsample already used for
  the brightness check, so the added cost is trivial):
  - Black/covered camera (unchanged: avg brightness < 5, streak > 2).
  - **Frozen/static feed**: sums per-pixel grayscale difference against the
    previous sampled frame; a real camera's sensor noise means two live
    frames are essentially never pixel-identical even when the subject is
    still, so a sustained near-zero diff (`< 150` summed over 768 pixels,
    for 4+ consecutive samples) strongly suggests a static image or paused
    stream substituted for the camera.
  - **Blur/defocus proxy**: average horizontal gradient magnitude across the
    downsampled frame, as a lightweight stand-in for OpenCV's
    Laplacian-variance blur check; only evaluated when brightness ≥ 15, so a
    genuinely dark room isn't misread as "blurred."
  - **Track-state poll** (every 60th frame): checks
    `track.readyState`/`track.enabled` directly, as a fallback in case the
    `ended` event listener (registered once at camera setup) doesn't fire in
    a given browser.
  This intentionally replaces what OpenCV.js used to provide for these same
  three checks — see "OpenCV.js" under Known trade-offs for why OpenCV
  itself stays disabled, and what re-adding it would still bring beyond this
  (specifically: `BackgroundSubtractorMOG2`-based **Unidentified Object
  Detected**, which doesn't have a pure-JS equivalent implemented here yet).
- **Evidence capture**: every strike grabs a 320×240 JPEG snapshot from the
  live video and sends it to the backend alongside the violation.
- **Copy/paste, right-click, and Ctrl+C/V/P blocking**, logged as
  `COPY_PASTE_ATTEMPT`.
- **Autosave**: answers PATCHed to the backend every 10 seconds.
- **Session resume**: reloading `/exam/:examId` returns the same in-progress
  session (with saved answers and correctly reduced remaining time) instead
  of starting a new attempt.
- **Shuffled questions and options**, seeded by session id (`seedrandom`) so
  a reload doesn't reshuffle mid-attempt, with a mapping back to original
  option numbers so scoring is unaffected.
- **Dev/Debug view**: an in-exam checkbox overlays the 10 iris landmark
  points on the video feed, for visually verifying detection quality.

**Backend (`server.js`)**
- `POST /api/session/start` now enforces the exam's live/dead window on a
  *new* attempt only, and resumes an existing `in_progress` session instead
  of creating a duplicate.
- `PATCH /api/session/:sessionId/autosave` — new, for the answer autosave.
- `POST /api/strike` — now validates the session belongs to the caller and
  is still `in_progress`, applies a simple in-memory rate limit (max 3 per
  2 seconds per session, defense-in-depth on top of the client cooldown),
  and stores the evidence snapshot.
- `GET /api/exam-logs/:examId/:clerkUserId` — new, backs the Student Review
  page.
- `POST /api/session/:sessionId/reinstate` — new, backs the Reinstate button.

**Dark mode** — `theme.js` + a `data-theme` attribute + CSS variables
throughout `index.css`; toggle lives in the sidebar topbar (`Layout.jsx`).

**Schema** — added `anomaly_logs.evidence_base64` (additive; the migration
line in `schema.sql` is safe to re-run on an existing database).

## Known trade-offs in this build

- **Evidence storage**: snapshots are stored as base64 text directly on the
  `anomaly_logs` row, not in Supabase Storage. This is simpler to set up
  (no bucket/policy configuration needed) but bloats the table faster than a
  storage path would. Fine for a demo or small class; migrate to Storage
  (upload the data URI, store the path instead) before using this at real
  scale or over a long term.
- **OpenCV.js is disabled by default** (`index.html` no longer loads it).
  During testing, `docs.opencv.org`'s `opencv.js` build (~10MB, non-SIMD,
  single-threaded) tied up the main thread long enough during page load to
  trigger the browser's own "Page Unresponsive" freeze dialog on at least
  one real device — a JS-level timeout can't rescue a synchronously blocked
  thread, since a `setTimeout` callback can't fire until the thread frees up,
  so the only reliable fix was to stop loading it eagerly on every page load.
  Every OpenCV-based check in `ExamRoom.jsx` (`if (window.cv)`) already
  tolerates it being absent — the app just falls back to the plain
  black-frame brightness check for webcam tampering, and skips blur/frozen-
  feed/unidentified-object detection.
  To bring it back **without reintroducing the freeze**, don't just re-add
  the `<script>` tag — load it off the main thread instead: either inside a
  Web Worker (postMessage the frames over, or use `OffscreenCanvas`), or via
  a lazy `requestIdleCallback`-scheduled dynamic `import()`/script injection
  well after the exam is already interactive, so its compile time can never
  block camera/model setup again. Test on a real low-to-mid-spec laptop, not
  just a dev machine, before trusting it's fixed.
- **Model loading is sequential, not parallel**, and shows which step it's on
  (`ExamRoom.jsx`'s `loadingStep` state: "Starting camera…" → "Loading vision
  runtime…" → "Loading face detection model…" → "Loading object detection
  model…"). This was changed from `Promise.all` specifically to reduce peak
  main-thread contention on slower machines and to make a future hang
  immediately diagnosable from the loading screen instead of a generic
  spinner. Both models use `delegate: 'CPU'`, not `'GPU'` — the GPU/WebGL
  path is a separate known hang trigger on browsers that restrict or
  randomize WebGL for anti-fingerprinting (Brave Shields being the common
  case), where it fails silently instead of throwing.
- **Rate limiting** on `/api/strike` is in-memory and resets on server
  restart/redeploy. Fine for a single-instance deployment; would need a
  shared store (Redis, or a Postgres-backed counter) behind a load balancer.
- **`session.status !== 'in_progress'`** on load always shows the
  "Exam Terminated" screen, even for a `completed` session (e.g. a student
  who already submitted and revisits the URL). This is a minor UX quirk
  worth a small follow-up fix (`ExamRoom.jsx` should branch on the specific
  status rather than treating anything non-in-progress as terminated).
- **Identity check, multi-monitor detection, per-exam detection toggles,
  classes/enrollment, admin role, and analytics dashboards** are not built
  in this pass — flagged here so the gap stays visible.

## Local setup

```bash
# Backend
cd backend
npm install
cp .env .env.local   # fill in real keys
npm run dev

# Frontend
cd frontend
npm install
cp .env .env.local   # fill in real keys
npm run dev
```

Run `schema.sql` in the Supabase SQL editor before starting the backend (the
`alter table ... add column if not exists` line is safe to run even if you've
already applied an earlier version of this schema).

In the Clerk Dashboard, set a teacher account's **Public metadata** to:
```json
{ "role": "teacher" }
```

## Render deployment (recommended: one free Web Service)

This repository includes `render.yaml`, which deploys the React frontend and
Express API together as **one** Render Web Service. Express serves
`frontend/dist` after Render builds it, so the browser uses same-origin `/api`
requests. You do **not** need to deploy the frontend and backend separately or
set `VITE_API_URL` for this configuration.

1. Push the repository to GitHub (do not push either `.env` file).
2. In Render, choose **New +** → **Blueprint**, connect the repository, and
   select the detected `render.yaml`.
3. Keep the `Free` plan, then supply these environment variables when Render
   asks for them:
   - `VITE_CLERK_PUBLISHABLE_KEY` — Clerk publishable key. It is embedded in
     the frontend build, so set it before the first deploy and whenever it
     changes.
   - `CLERK_SECRET_KEY` — server-only Clerk secret key.
   - `SUPABASE_URL` — Supabase project URL.
   - `SUPABASE_SERVICE_ROLE_KEY` — server-only Supabase service-role key.
   - `CORS_ORIGIN` is optional for the combined app. Set it only to allow
     additional comma-separated origins to call the API.
4. Deploy. Render checks `/api/health`; a successful response is
   `{ "ok": true }`.
5. In Clerk, add the Render URL (for example,
   `https://phoenix-ai-proctoring.onrender.com`) to the allowed origins,
   redirect URLs, and/or authorized domains required by your Clerk instance.

### Free-tier considerations

- The free Web Service spins down after inactivity, so the first request after
  it sleeps can take roughly a minute. This is normal and is unsuitable for a
  time-sensitive production exam without a paid always-on service.
- Render's ephemeral filesystem is fine here because application data lives in
  Supabase. Do not add local uploads or persist data to disk.
- The current evidence snapshots are stored in Supabase as data URIs; monitor
  database size and move them to Supabase Storage before real-scale use.
- Run `schema.sql` in the Supabase SQL Editor before the first deployment.

### Separate services (only if you need them)

You can still use a Render Static Site for `frontend` and a Web Service for
`backend`. In that case set `VITE_API_URL` to the backend's public HTTPS URL,
set backend `CORS_ORIGIN` to the static site's URL, and configure a static-site
rewrite from `/*` to `/index.html`. The combined Blueprint is simpler and
uses only one free web service.
