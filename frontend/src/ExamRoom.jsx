import { useEffect, useRef, useState, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth, useUser } from '@clerk/clerk-react';
import { FaceLandmarker, ObjectDetector, FilesetResolver } from '@mediapipe/tasks-vision';
import seedrandom from 'seedrandom';
import Layout from './Layout.jsx';

const API_URL = import.meta.env.VITE_API_URL || '';
const COOLDOWN_MS = 4000;

// --- Phase 2: object detection tuning ---
// Sampled more often (every 3rd frame, not 5th) now that both models run on
// the CPU delegate with headroom to spare — this raises the *chance* of
// catching a briefly-held object, which matters more now that a detection
// also has to persist across frames (below) before it counts.
const OBJECT_DETECT_EVERY_N_FRAMES = 3;
// A single frame at the old 0.5 threshold was enough to log a strike; raised
// to cut low-confidence noise (a dark rectangle, a remote mistaken for a
// phone at a glance).
const OBJECT_SCORE_THRESHOLD = 0.6;
// Consecutive *sampled* frames (not raw frames) the same label must appear
// in before it's logged — mirrors the streak pattern already used for face
// detection. At 3-frame sampling this is roughly 300-450ms of continuous
// detection, which is enough to filter motion-blur/occlusion flicker while
// still catching a genuinely held-up object quickly.
const OBJECT_DETECT_STREAK_REQUIRED = 3;

// --- Phase 1: face detection / gaze tuning ---
// Number of recent yaw/pitch readings averaged before comparing to the
// calibrated baseline. At ~30fps this is roughly a quarter-second window —
// enough to smooth single-frame jitter without meaningfully delaying
// detection of a sustained look-away.
const GAZE_SMOOTHING_WINDOW = 5;
// Below this average brightness (0-255 grayscale), the lighting warning
// shows — but this is a NUDGE, not a hard block (see the override button
// below). Lowered from an earlier 40, which was stricter than many laptop
// webcams read under ordinary indoor lighting, especially in backlit rooms
// where average brightness reads low even though the frame looks bright.
const MIN_CALIBRATION_BRIGHTNESS = 25;
// How long the warning shows before "Continue Anyway" appears. Long enough
// that a student who CAN fix their lighting (turn on a lamp, face a window)
// has a moment to do so and see it take effect; short enough that someone
// stuck with their actual lighting (a dim venue, unavoidable backlighting)
// isn't trapped on this screen.
const LIGHTING_OVERRIDE_DELAY_MS = 6000;
// Calibration gives up waiting for 30 clean samples after this long and
// proceeds with whatever it collected — see the comment in runCalibration.
const CALIBRATION_TIMEOUT_MS = 20000;

const CELL_PHONE_LABELS = new Set(['cell phone']);
const PROHIBITED_LABELS = new Set(['book', 'laptop', 'keyboard', 'remote', 'tv', 'mouse']);

const V = {
  FACE_NOT_VISIBLE: 'FACE_NOT_VISIBLE',
  MULTIPLE_FACES: 'MULTIPLE_FACES',
  GAZE_DEVIATION: 'GAZE_DEVIATION',
  WEBCAM_TAMPERED: 'WEBCAM_TAMPERED',
  TAB_SWITCH: 'TAB_SWITCH',
  FULLSCREEN_EXIT: 'FULLSCREEN_EXIT',
  CELL_PHONE_DETECTED: 'CELL_PHONE_DETECTED',
  PROHIBITED_OBJECT_DETECTED: 'PROHIBITED_OBJECT_DETECTED',
  UNIDENTIFIED_OBJECT: 'UNIDENTIFIED_OBJECT',
  COPY_PASTE_ATTEMPT: 'COPY_PASTE_ATTEMPT'
};

const VIOLATION_LABEL = {
  [V.FACE_NOT_VISIBLE]: 'Face Not Visible',
  [V.MULTIPLE_FACES]: 'Multiple Faces Detected',
  [V.GAZE_DEVIATION]: 'Looking Away From Screen',
  [V.WEBCAM_TAMPERED]: 'Webcam Feed Interrupted',
  [V.TAB_SWITCH]: 'Tab Switch Detected',
  [V.FULLSCREEN_EXIT]: 'Exited Full-Screen Mode',
  [V.CELL_PHONE_DETECTED]: 'Cell Phone Detected',
  [V.PROHIBITED_OBJECT_DETECTED]: 'Prohibited Object Detected',
  [V.UNIDENTIFIED_OBJECT]: 'Unidentified Object Detected',
  [V.COPY_PASTE_ATTEMPT]: 'Screen/Tab Modification Attempt',
};

function getYawPitchFromMatrix(m) {
  const r20 = m[2], r21 = m[6], r22 = m[10];
  const yaw = Math.atan2(-r20, Math.sqrt(r21 * r21 + r22 * r22)) * (180 / Math.PI);
  const pitch = Math.atan2(r21, r22) * (180 / Math.PI);
  return { yaw, pitch };
}

// Wraps a promise so a stuck load (e.g. a GPU/WebGL init that never resolves
// under some browsers' privacy-shielding) fails with a clear message instead
// of leaving the student staring at an infinite spinner or a frozen tab.
function withTimeout(promise, ms, message) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(message)), ms)),
  ]);
}

function formatTime(totalSeconds) {
  const s = Math.max(0, Math.floor(totalSeconds));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}:${String(rem).padStart(2, '0')}`;
}

export default function ExamRoom() {
  const { examId } = useParams();
  const navigate = useNavigate();
  const { getToken } = useAuth();
  const { user } = useUser();

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const debugCanvasRef = useRef(null);
  const faceLandmarkerRef = useRef(null);
  const objectDetectorRef = useRef(null);
  const rafRef = useRef(null);
  const frameCountRef = useRef(0);
  const sessionIdRef = useRef(null);
  const lastViolationAtRef = useRef({});
  const warningCountRef = useRef(0);
  const answersRef = useRef({});
  const rawQuestionsRef = useRef([]); // unshuffled questions fetched in init(), shuffled in beginExam() once a session exists
  
  const bgSubtractorRef = useRef(null);
  const prevFrameMatRef = useRef(null);

  const [exam, setExam] = useState(null);
  const [questions, setQuestions] = useState([]);
  const [originalQuestionsMap, setOriginalQuestionsMap] = useState({});
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [secondsLeft, setSecondsLeft] = useState(0);
  const [warningCount, setWarningCount] = useState(0);
  const [modal, setModal] = useState(null);
  const [terminated, setTerminated] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [toast, setToast] = useState(null);
  const [initError, setInitError] = useState(null);
  const [loadingStep, setLoadingStep] = useState('Starting…');
  
  const [systemCheckPassed, setSystemCheckPassed] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [startingExam, setStartingExam] = useState(false); // brief gap between calibration finishing and the session actually being created
  const [brightness, setBrightness] = useState(null); // live lighting reading, shown to the student
  const [lightingWarnedAt, setLightingWarnedAt] = useState(null); // when low light was first seen, for the override timer
  const [lightingOverride, setLightingOverride] = useState(false); // student chose "Continue Anyway"
  const [showDebug, setShowDebug] = useState(false);
  const [ready, setReady] = useState(false);
  
  const cvStateRef = useRef({
    prevFrameGray: null, // Uint8ClampedArray from the previous sampled frame, for frozen-feed diffing (Phase 3)
    baselineYaw: 0,
    baselinePitch: 0,
    calibrated: false,
    multiFaceStreak: 0,
    noFaceStreak: 0,
    staticFeedStreak: 0,
    // Phase 2: per-label streak counters for object detection persistence,
    // e.g. { 'cell phone': 2, 'book': 0, ... }. Reset to 0 for any label not
    // seen in the current sampled frame.
    objectStreaks: {},
    gazeBuffer: [], // recent { yaw, pitch } readings for smoothing (Phase 1)
  });

  const authedFetch = useCallback(
    async (path, options = {}) => {
      const token = await getToken();
      return fetch(`${API_URL}${path}`, {
        ...options,
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}`, ...(options.headers || {}) },
      });
    },
    [getToken]
  );

  const sendStrike = useCallback(
    (violationType, detail) => {
      let evidenceBase64 = null;
      if (videoRef.current && canvasRef.current) {
         try {
            const canvas = canvasRef.current;
            canvas.width = 320;
            canvas.height = 240;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoRef.current, 0, 0, 320, 240);
            evidenceBase64 = canvas.toDataURL('image/jpeg', 0.8);
         } catch (e) {
            console.warn('Evidence capture failed', e);
         }
      }

      authedFetch('/api/strike', {
        method: 'POST',
        body: JSON.stringify({
          sessionId: sessionIdRef.current,
          violationType,
          detail,
          evidenceBase64,
          warningCount: warningCountRef.current,
        }),
      }).catch((err) => console.error('Strike log failed:', err));
    },
    [authedFetch]
  );

  const autoSaveAnswers = useCallback(() => {
    if (!sessionIdRef.current || terminated || submitted) return;
    authedFetch(`/api/session/${sessionIdRef.current}/autosave`, {
      method: 'PATCH',
      body: JSON.stringify({ answers: answersRef.current }),
    }).catch(err => console.error('Autosave failed:', err));
  }, [authedFetch, terminated, submitted]);

  useEffect(() => {
    const interval = setInterval(() => autoSaveAnswers(), 10000);
    return () => clearInterval(interval);
  }, [autoSaveAnswers]);

  const stopCamera = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    videoRef.current?.srcObject?.getTracks()?.forEach((t) => t.stop());
    faceLandmarkerRef.current?.close?.();
    objectDetectorRef.current?.close?.();
    if (bgSubtractorRef.current) bgSubtractorRef.current.delete();
    if (prevFrameMatRef.current) prevFrameMatRef.current.delete();
  }, []);

  // Defense-in-depth for a hard tab close/refresh: the browser already
  // releases camera tracks on page unload regardless of our JS, and the
  // unmount cleanup below already handles every in-app exit (submit,
  // terminate, navigating away) — but 'pagehide' fires reliably across
  // browsers right before a hard close in a way React's own cleanup timing
  // isn't guaranteed to, so this is a direct, synchronous belt-and-braces
  // stop of just the tracks (not the full stopCamera, since there's no need
  // to tear down the CV models on a page that's about to be gone anyway).
  useEffect(() => {
    const stopTracksOnly = () => {
      videoRef.current?.srcObject?.getTracks()?.forEach((t) => t.stop());
    };
    window.addEventListener('pagehide', stopTracksOnly);
    return () => window.removeEventListener('pagehide', stopTracksOnly);
  }, []);

  const terminateExam = useCallback(
    async (reason) => {
      setTerminated(true);
      stopCamera();
      try {
        await authedFetch(`/api/session/${sessionIdRef.current}/terminate`, {
          method: 'POST',
          body: JSON.stringify({ reason }),
        });
      } catch (err) {
        console.error('Termination log failed:', err);
      }
    },
    [authedFetch, stopCamera]
  );

  const raiseViolation = useCallback(
    (type, detail) => {
      const now = performance.now();
      const last = lastViolationAtRef.current[type] || 0;
      if (now - last < COOLDOWN_MS) return;
      lastViolationAtRef.current[type] = now;

      warningCountRef.current += 1;
      const count = warningCountRef.current;
      setWarningCount(count);
      setModal({ type });
      sendStrike(type, detail);

      if (exam && count >= (exam.termination_threshold || 10)) {
        terminateExam(`Warning threshold reached (${count}/${exam.termination_threshold || 10})`);
      }
    },
    [sendStrike, terminateExam, exam]
  );

  const submitExam = useCallback(
    async (autoSubmitted = false) => {
      stopCamera();
      try {
        const res = await authedFetch(`/api/exams/${examId}/submit`, {
          method: 'POST',
          body: JSON.stringify({ sessionId: sessionIdRef.current, answers: answersRef.current }),
        });
        if (!res.ok) throw new Error('Submission failed');
        setSubmitted(true);
        setToast('User Logs Saved!!');
        setTimeout(() => setToast(null), 3000);
      } catch (err) {
        console.error('Submit failed:', err);
        setInitError(autoSubmitted ? 'Time expired, but submission failed — please contact your instructor.' : err.message);
      }
    },
    [authedFetch, examId, stopCamera]
  );

  useEffect(() => {
    const onContextMenu = (e) => {
      e.preventDefault();
      raiseViolation(V.COPY_PASTE_ATTEMPT, 'Right click attempt');
    };
    const onCopy = (e) => {
      e.preventDefault();
      raiseViolation(V.COPY_PASTE_ATTEMPT, 'Copy/Paste attempt');
    };
    const onKeyDown = (e) => {
      if ((e.ctrlKey || e.metaKey) && ['c', 'v', 'p'].includes(e.key.toLowerCase())) {
        e.preventDefault();
        raiseViolation(V.COPY_PASTE_ATTEMPT, 'Copy/Paste shortcut attempt');
      }
    };
    
    if (systemCheckPassed) {
      document.addEventListener('contextmenu', onContextMenu);
      document.addEventListener('copy', onCopy);
      document.addEventListener('paste', onCopy);
      document.addEventListener('keydown', onKeyDown);
      
      const onBlur = () => raiseViolation(V.TAB_SWITCH, 'window blur');
      const onFullscreenChange = () => {
        if (!document.fullscreenElement) raiseViolation(V.FULLSCREEN_EXIT, 'exited full-screen');
      };
      
      const blurTimer = setTimeout(() => {
        window.addEventListener('blur', onBlur);
        document.addEventListener('fullscreenchange', onFullscreenChange);
      }, 1500);

      return () => {
        document.removeEventListener('contextmenu', onContextMenu);
        document.removeEventListener('copy', onCopy);
        document.removeEventListener('paste', onCopy);
        document.removeEventListener('keydown', onKeyDown);
        clearTimeout(blurTimer);
        window.removeEventListener('blur', onBlur);
        document.removeEventListener('fullscreenchange', onFullscreenChange);
      };
    }
  }, [systemCheckPassed, raiseViolation]);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const qRes = await authedFetch(`/api/exams/${examId}/questions`);
        if (!qRes.ok) throw new Error('Failed to load exam');
        const { exam, questions } = await qRes.json();
        if (cancelled) return;
        setExam(exam);
        // Session creation (and the server's started_at timestamp) is
        // deliberately deferred until calibration actually finishes — see
        // beginExam() below. Fetching questions here is safe to do early
        // since it doesn't start any clock; only holding onto the raw list
        // for beginExam() to shuffle once the session (and its id, used as
        // the shuffle seed) actually exists.
        rawQuestionsRef.current = questions;

        setLoadingStep('Starting camera…');
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720, facingMode: 'user' },
        });
        if (cancelled) return;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await withTimeout(
            new Promise((res) => (videoRef.current.onloadedmetadata = res)),
            15000,
            'Camera did not start in time. Check that no other app is using it, then reload.'
          );
          videoRef.current.play();
        }

        setLoadingStep('Loading vision runtime…');
        const filesetResolver = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm'
        );

        // Loaded one at a time rather than via Promise.all: two heavy WASM
        // model loads compiling at once was doubling up main-thread work on
        // slower machines. Loading sequentially, with a step label for each,
        // means a hang now points at exactly which model is stuck instead of
        // a single opaque "loading models" spinner. delegate: 'CPU' (not
        // 'GPU') avoids a separate hang some browsers hit on the WebGL path
        // when privacy/shield extensions restrict WebGL fingerprinting.
        setLoadingStep('Loading face detection model…');
        const faceLandmarker = await withTimeout(
          FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
              modelAssetPath:
                'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
              delegate: 'CPU',
            },
            outputFacialTransformationMatrixes: true,
            runningMode: 'VIDEO',
            numFaces: 3,
            // Confidence thresholds kept at the library's own defaults
            // (0.5) rather than raised further — a stricter 0.6 sounds more
            // "accurate" in principle, but in practice it made face
            // detection unreliable on lower-resolution/grainier laptop
            // webcams, contributing to calibration never collecting enough
            // confident samples. 0.5 is the better trade-off for real
            // hardware, not just a controlled test environment.
            minFaceDetectionConfidence: 0.5,
            minFacePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          }),
          25000,
          'The face detection model took too long to load. Try a different browser, or reload.'
        );
        if (cancelled) return;

        setLoadingStep('Loading object detection model…');
        const objectDetector = await withTimeout(
          ObjectDetector.createFromOptions(filesetResolver, {
            baseOptions: {
              modelAssetPath:
                'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite',
              delegate: 'CPU',
            },
            scoreThreshold: OBJECT_SCORE_THRESHOLD,
            runningMode: 'VIDEO',
          }),
          25000,
          'The object detection model took too long to load. Try a different browser, or reload.'
        );
        if (cancelled) return;

        faceLandmarkerRef.current = faceLandmarker;
        objectDetectorRef.current = objectDetector;

        if (window.cv) {
           bgSubtractorRef.current = new window.cv.BackgroundSubtractorMOG2(500, 16, true);
           prevFrameMatRef.current = new window.cv.Mat(480, 640, window.cv.CV_8UC4);
        }

        setLoadingStep('Ready');
        setReady(true);
        
      } catch (err) {
        console.error('ExamRoom init failed:', err);
        if (!cancelled) setInitError(err.message || 'Failed to initialize exam environment');
      }
    }

    init();

    return () => {
      cancelled = true;
      stopCamera();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [examId]);

  useEffect(() => {
    if (!exam || terminated || submitted || !systemCheckPassed) return;
    if (secondsLeft <= 0) {
      submitExam(true);
      return;
    }
    const t = setTimeout(() => setSecondsLeft((s) => s - 1), 1000);
    return () => clearTimeout(t);
  }, [exam, secondsLeft, terminated, submitted, submitExam, systemCheckPassed]);

  const startCVLoop = useCallback(() => {
    let blackFrameStreak = 0;
    let blurStreak = 0;
    let unidentifiedObjStreak = 0;
    
    const loop = () => {
      const video = videoRef.current;
      if (!video || video.readyState < 2 || !faceLandmarkerRef.current) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }
      const now = performance.now();
      frameCountRef.current += 1;
      
      const faceResult = faceLandmarkerRef.current.detectForVideo(video, now);
      const faceCount = faceResult.faceLandmarks?.length || 0;
      
      if (faceCount === 0) {
        cvStateRef.current.noFaceStreak++;
        cvStateRef.current.multiFaceStreak = 0;
        if (cvStateRef.current.noFaceStreak > 5) {
           raiseViolation(V.FACE_NOT_VISIBLE, 'No face in frame for 5 frames');
        }
      } else if (faceCount >= 2) {
        cvStateRef.current.multiFaceStreak++;
        cvStateRef.current.noFaceStreak = 0;
        if (cvStateRef.current.multiFaceStreak > 5) {
           raiseViolation(V.MULTIPLE_FACES, `${faceCount} faces in frame`);
        }
      } else {
        cvStateRef.current.noFaceStreak = 0;
        cvStateRef.current.multiFaceStreak = 0;
        
        const matrix = faceResult.facialTransformationMatrixes?.[0]?.data;
        if (matrix && cvStateRef.current.calibrated) {
          const { yaw, pitch } = getYawPitchFromMatrix(matrix);

          // Phase 1: smooth over the last few readings before comparing to
          // baseline, instead of reacting to a single frame. A single noisy
          // frame landing at e.g. 21° when the threshold is 20° is sensor
          // jitter, not a real head turn — averaging a short window filters
          // that out while still catching a sustained look-away quickly.
          const buf = cvStateRef.current.gazeBuffer;
          buf.push({ yaw, pitch });
          if (buf.length > GAZE_SMOOTHING_WINDOW) buf.shift();
          const smoothedYaw = buf.reduce((s, r) => s + r.yaw, 0) / buf.length;
          const smoothedPitch = buf.reduce((s, r) => s + r.pitch, 0) / buf.length;

          const diffYaw = Math.abs(smoothedYaw - cvStateRef.current.baselineYaw);
          const diffPitch = Math.abs(smoothedPitch - cvStateRef.current.baselinePitch);
          if (diffYaw > 20 || diffPitch > 20) {
            raiseViolation(V.GAZE_DEVIATION, `yaw diff=${diffYaw.toFixed(1)}° pitch diff=${diffPitch.toFixed(1)}° (smoothed over ${buf.length} frames)`);
          }
        }
        
        if (showDebug && debugCanvasRef.current) {
           const dctx = debugCanvasRef.current.getContext('2d');
           dctx.clearRect(0, 0, debugCanvasRef.current.width, debugCanvasRef.current.height);
           dctx.fillStyle = 'red';
           for(let i=468; i<=477; i++) {
              const pt = faceResult.faceLandmarks[0][i];
              if(pt) {
                 const x = pt.x * debugCanvasRef.current.width;
                 const y = pt.y * debugCanvasRef.current.height;
                 dctx.fillRect(debugCanvasRef.current.width - x - 2, y - 2, 4, 4);
              }
           }
        }
      }

      if (objectDetectorRef.current && frameCountRef.current % OBJECT_DETECT_EVERY_N_FRAMES === 0) {
        const objResult = objectDetectorRef.current.detectForVideo(video, now);
        const streaks = cvStateRef.current.objectStreaks;
        const labelsSeenThisSample = new Set();

        for (const det of objResult.detections || []) {
          const label = det.categories?.[0]?.categoryName?.toLowerCase();
          const score = det.categories?.[0]?.score || 0;
          if (!label || (!CELL_PHONE_LABELS.has(label) && !PROHIBITED_LABELS.has(label))) continue;

          labelsSeenThisSample.add(label);
          streaks[label] = (streaks[label] || 0) + 1;

          // Only log once the label has persisted across several consecutive
          // *sampled* frames — a single-frame hit is far more likely to be
          // motion blur or a misclassified object than a real, held-up item.
          if (streaks[label] >= OBJECT_DETECT_STREAK_REQUIRED) {
            const detail = `detected: ${label} (conf: ${score.toFixed(2)}, held for ${streaks[label]} samples)`;
            if (CELL_PHONE_LABELS.has(label)) {
              raiseViolation(V.CELL_PHONE_DETECTED, detail);
            } else {
              raiseViolation(V.PROHIBITED_OBJECT_DETECTED, detail);
            }
            // Reset after logging so the cooldown (not the streak) governs
            // when the next strike for this label can fire, rather than
            // re-logging every single sampled frame the object stays visible.
            streaks[label] = 0;
          }
        }

        // Any tracked label not seen in this sample resets to 0 — the object
        // has to be held continuously, not just appear on-and-off.
        for (const label of Object.keys(streaks)) {
          if (!labelsSeenThisSample.has(label)) streaks[label] = 0;
        }
      }

      if (frameCountRef.current % 15 === 0 && window.cv && bgSubtractorRef.current) {
         try {
           const canvas = canvasRef.current;
           const ctx = canvas.getContext('2d', { willReadFrequently: true });
           canvas.width = 640;
           canvas.height = 480;
           ctx.drawImage(video, 0, 0, 640, 480);
           
           let src = window.cv.imread(canvas);
           let fgmask = new window.cv.Mat();
           let gray = new window.cv.Mat();
           
           window.cv.cvtColor(src, gray, window.cv.COLOR_RGBA2GRAY);
           
           let lap = new window.cv.Mat();
           window.cv.Laplacian(gray, lap, window.cv.CV_64F);
           let mean = new window.cv.Mat();
           let stddev = new window.cv.Mat();
           window.cv.meanStdDev(lap, mean, stddev);
           let blurVariance = stddev.data64F[0] * stddev.data64F[0];
           if (blurVariance < 50) {
              raiseViolation(V.WEBCAM_TAMPERED, 'Webcam feed is extremely blurry or covered');
           }
           
           if (prevFrameMatRef.current && prevFrameMatRef.current.rows > 0) {
              let diff = new window.cv.Mat();
              window.cv.absdiff(gray, prevFrameMatRef.current, diff);
              let diffSum = window.cv.sumElements(diff);
              if (diffSum[0] < 1000) {
                 cvStateRef.current.staticFeedStreak++;
                 if (cvStateRef.current.staticFeedStreak > 5) {
                    raiseViolation(V.WEBCAM_TAMPERED, 'Frozen / Static webcam feed detected');
                 }
              } else {
                 cvStateRef.current.staticFeedStreak = 0;
              }
              diff.delete();
           }
           gray.copyTo(prevFrameMatRef.current);
           
           bgSubtractorRef.current.apply(src, fgmask);
           let nonZero = window.cv.countNonZero(fgmask);
           let thresholdObj = (640 * 480) * 0.2;
           if (nonZero > thresholdObj && faceCount >= 1) {
              unidentifiedObjStreak++;
              if (unidentifiedObjStreak > 3) {
                 raiseViolation(V.UNIDENTIFIED_OBJECT, 'Something large entered the frame');
              }
           } else {
              unidentifiedObjStreak = 0;
           }
           
           src.delete(); fgmask.delete(); gray.delete(); lap.delete(); mean.delete(); stddev.delete();
         } catch (e) {
            console.error('OpenCV pass failed', e);
         }
      }

      // --- Phase 3: webcam tamper detection, all pure JS/Canvas — no OpenCV,
      // so this can't reintroduce the main-thread freeze that OpenCV.js
      // caused. Runs on the same cheap 32x24 downsample the brightness check
      // already used; the extra frame-diff and gradient math below are
      // trivial at 768 pixels.
      if (frameCountRef.current % 30 === 0) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        const W = 32, H = 24;
        canvas.width = W;
        canvas.height = H;
        ctx.drawImage(video, 0, 0, W, H);
        const { data } = ctx.getImageData(0, 0, W, H);

        // Grayscale conversion, reused below for both the frozen-feed diff
        // and the blur proxy.
        const gray = new Uint8ClampedArray(W * H);
        let brightnessSum = 0;
        for (let i = 0, p = 0; i < data.length; i += 4, p++) {
          const g = (data[i] + data[i + 1] + data[i + 2]) / 3;
          gray[p] = g;
          brightnessSum += g;
        }
        const avgBrightness = brightnessSum / (W * H);

        // 1. Black/covered camera.
        if (avgBrightness < 5) {
          blackFrameStreak += 1;
          if (blackFrameStreak > 2) {
            raiseViolation(V.WEBCAM_TAMPERED, 'Feed appears black/covered');
            blackFrameStreak = 0;
          }
        } else {
          blackFrameStreak = 0;
        }

        // 2. Frozen/static feed — a real camera's sensor noise means two
        // genuinely live frames are essentially never pixel-identical, even
        // when the person is sitting still. A sustained near-zero diff
        // across several samples strongly suggests a static image or paused
        // stream substituted for the camera, not just a still subject.
        if (cvStateRef.current.prevFrameGray) {
          let diffSum = 0;
          for (let p = 0; p < gray.length; p++) {
            diffSum += Math.abs(gray[p] - cvStateRef.current.prevFrameGray[p]);
          }
          if (diffSum < 150) {
            cvStateRef.current.staticFeedStreak += 1;
            if (cvStateRef.current.staticFeedStreak > 4) {
              raiseViolation(V.WEBCAM_TAMPERED, `Frozen/static feed detected (diff=${diffSum.toFixed(0)})`);
              cvStateRef.current.staticFeedStreak = 0;
            }
          } else {
            cvStateRef.current.staticFeedStreak = 0;
          }
        }
        cvStateRef.current.prevFrameGray = gray;

        // 3. Blur/defocus proxy — cheap gradient-magnitude variance, a
        // lightweight JS stand-in for OpenCV's Laplacian-variance blur
        // check. Not a fully covered lens (that's caught by #1 above), but
        // a lens smeared or heavily defocused so the face isn't actually
        // readable. Only checked when the frame is bright enough that this
        // isn't just "the room is dark" (which reads as low-gradient too).
        if (avgBrightness >= 15) {
          let gradientSum = 0;
          for (let y = 0; y < H; y++) {
            for (let x = 0; x < W - 1; x++) {
              gradientSum += Math.abs(gray[y * W + x + 1] - gray[y * W + x]);
            }
          }
          const avgGradient = gradientSum / (W * (H - 1));
          if (avgGradient < 1.5) {
            blurStreak += 1;
            if (blurStreak > 4) {
              raiseViolation(V.WEBCAM_TAMPERED, `Feed appears heavily blurred/defocused (grad=${avgGradient.toFixed(2)})`);
              blurStreak = 0;
            }
          } else {
            blurStreak = 0;
          }
        }
      }

      // 4. Belt-and-braces check for camera loss: the track's 'ended' event
      // (registered once, at setup) usually fires reliably, but polling
      // readyState here too catches any browser where it doesn't.
      if (frameCountRef.current % 60 === 0) {
        const track = videoRef.current?.srcObject?.getVideoTracks?.()?.[0];
        if (track && (track.readyState === 'ended' || !track.enabled)) {
          raiseViolation(V.WEBCAM_TAMPERED, `Camera track unavailable (readyState=${track.readyState})`);
        }
      }

      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
  }, [raiseViolation, showDebug]);
  
  useEffect(() => {
     if (systemCheckPassed && ready) {
        document.documentElement.requestFullscreen?.().catch(() => {});
        startCVLoop();
     }
  }, [systemCheckPassed, ready, startCVLoop]);

  const selectOption = (questionId, optionNumber) => {
    answersRef.current = { ...answersRef.current, [questionId]: optionNumber };
    setAnswers({ ...answersRef.current });
  };
  
  // Samples brightness every 500ms while the student is on the system-check
  // screen, so low light can be flagged before calibration — calibrating in
  // bad light bakes an unreliable gaze baseline into the whole exam. This is
  // advisory, not a hard block: see `lightingOk` below, which always allows
  // proceeding once LIGHTING_OVERRIDE_DELAY_MS has passed, regardless of the
  // reading. A gate with no way through it is worse than no gate at all.
  useEffect(() => {
    if (!ready || systemCheckPassed) return;
    const interval = setInterval(() => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      canvas.width = 32;
      canvas.height = 24;
      ctx.drawImage(video, 0, 0, 32, 24);
      const { data } = ctx.getImageData(0, 0, 32, 24);
      let sum = 0;
      for (let i = 0; i < data.length; i += 4) sum += (data[i] + data[i + 1] + data[i + 2]) / 3;
      const reading = sum / (32 * 24);
      setBrightness(reading);
      if (reading < MIN_CALIBRATION_BRIGHTNESS) {
        setLightingWarnedAt((prev) => prev ?? Date.now());
      } else {
        setLightingWarnedAt(null);
        setLightingOverride(false);
      }
    }, 500);
    return () => clearInterval(interval);
  }, [ready, systemCheckPassed]);

  // "Continue Anyway" becomes available after a few seconds of a low
  // reading, re-checked every second so the button doesn't need a page
  // interaction to appear.
  const [nowTick, setNowTick] = useState(Date.now());
  useEffect(() => {
    if (!lightingWarnedAt) return;
    const t = setInterval(() => setNowTick(Date.now()), 1000);
    return () => clearInterval(t);
  }, [lightingWarnedAt]);
  const overrideAvailable = lightingWarnedAt !== null && nowTick - lightingWarnedAt >= LIGHTING_OVERRIDE_DELAY_MS;

  const lightingOk = brightness === null || brightness >= MIN_CALIBRATION_BRIGHTNESS || lightingOverride;

  // This is the real "exam starts now" moment: creates (or resumes) the
  // session on the server, which is what timestamps started_at — deferred
  // to here specifically so that camera setup, model loading, the lighting
  // check, and calibration itself (all of which can take a while, especially
  // on a slower laptop) never eat into the exam's timed duration. Previously
  // the session was created at page load, before any of that setup, which
  // on a short test-duration exam could burn through the whole clock before
  // the student ever saw a question — causing an immediate auto-submit the
  // instant calibration finished.
  const beginExam = useCallback(async () => {
    setStartingExam(true);
    try {
      const sRes = await authedFetch('/api/session/start', {
        method: 'POST',
        body: JSON.stringify({
          examId,
          candidateName: user?.fullName,
          candidateEmail: user?.primaryEmailAddress?.emailAddress,
        }),
      });
      if (!sRes.ok) throw new Error('Failed to start session');
      const { session, exam: examData } = await sRes.json();

      let shuffledQuestions = [...rawQuestionsRef.current];
      let originalMap = {};
      if (examData?.shuffle_questions) {
        const rng = seedrandom(session.id);
        shuffledQuestions.sort(() => 0.5 - rng());
        shuffledQuestions.forEach((q) => {
          let opts = [q.option_1, q.option_2, q.option_3, q.option_4];
          let originalIndices = [1, 2, 3, 4];
          for (let i = 3; i > 0; i--) {
            const j = Math.floor(rng() * (i + 1));
            [opts[i], opts[j]] = [opts[j], opts[i]];
            [originalIndices[i], originalIndices[j]] = [originalIndices[j], originalIndices[i]];
          }
          q.option_1 = opts[0]; q.option_2 = opts[1]; q.option_3 = opts[2]; q.option_4 = opts[3];
          originalMap[q.id] = originalIndices;
        });
      } else {
        shuffledQuestions.forEach((q) => { originalMap[q.id] = [1, 2, 3, 4]; });
      }
      setOriginalQuestionsMap(originalMap);
      setQuestions(shuffledQuestions);

      sessionIdRef.current = session.id;
      if (session.status !== 'in_progress') {
        setTerminated(true);
        return;
      }
      if (session.answers) {
        answersRef.current = session.answers;
        setAnswers(session.answers);
      }
      // started_at now reflects this exact moment (or, for a resumed
      // session, the original true start) — either way, the right thing to
      // measure elapsed time against.
      const elapsedSecs = Math.floor((Date.now() - new Date(session.started_at).getTime()) / 1000);
      setSecondsLeft(Math.max(exam.duration_minutes * 60 - elapsedSecs, 0));

      setSystemCheckPassed(true);
    } catch (err) {
      console.error('beginExam failed:', err);
      setInitError(err.message || 'Failed to start the exam. Please reload and try again.');
      setStartingExam(false);
    }
  }, [authedFetch, examId, user, exam]);

  const runCalibration = useCallback(async () => {
    setCalibrating(true);
    setCalibrationProgress(0);

    let sumYaw = 0;
    let sumPitch = 0;
    let frames = 0;

    const sampleCount = 30;
    const startedAt = performance.now();

    const calibrateLoop = () => {
       const video = videoRef.current;
       if (!video || !faceLandmarkerRef.current) return requestAnimationFrame(calibrateLoop);

       const faceResult = faceLandmarkerRef.current.detectForVideo(video, performance.now());
       const faceCount = faceResult.faceLandmarks?.length || 0;

       if (faceCount === 1) {
          const matrix = faceResult.facialTransformationMatrixes?.[0]?.data;
          if (matrix) {
            const { yaw, pitch } = getYawPitchFromMatrix(matrix);
            sumYaw += yaw;
            sumPitch += pitch;
            frames++;
            setCalibrationProgress(Math.floor((frames / sampleCount) * 100));
          }
       }

       // Older/grainier laptop webcams can have trouble holding a confident
       // single-face reading long enough to collect 30 good samples. Rather
       // than calibration hanging forever waiting for a perfect run, give up
       // on reaching the full sample count after CALIBRATION_TIMEOUT_MS and
       // proceed with whatever was collected — even a partial baseline (or,
       // in the worst case, 0 relative to raw angles) is better than being
       // stuck on this screen indefinitely.
       const timedOut = performance.now() - startedAt > CALIBRATION_TIMEOUT_MS;

       if (frames < sampleCount && !timedOut) {
          requestAnimationFrame(calibrateLoop);
       } else {
          cvStateRef.current.baselineYaw = frames > 0 ? sumYaw / frames : 0;
          cvStateRef.current.baselinePitch = frames > 0 ? sumPitch / frames : 0;
          cvStateRef.current.calibrated = true;
          setCalibrating(false);
          beginExam();
       }
    };
    requestAnimationFrame(calibrateLoop);
  }, [beginExam]);

  if (terminated) {
    return (
      <div className="centered termination-screen">
        <h1>Exam Terminated</h1>
        <p>Your exam was ended automatically after repeated integrity violations.</p>
        <p>Please contact your instructor.</p>
      </div>
    );
  }

  if (submitted) {
    return (
      <Layout>
        {toast && <div className="toast toast-success">{toast}</div>}
        <div className="centered submitted-screen">
          <h1>Test is Submitted Successfully!!!</h1>
          <p>You can safely close the window now.</p>
          <button onClick={() => navigate('/dashboard')}>Go Back To Home</button>
        </div>
      </Layout>
    );
  }

  if (initError) {
    return (
      <div className="centered">
        <div className="error-banner">{initError}</div>
        <button onClick={() => window.location.reload()}>Reload and Try Again</button>
      </div>
    );
  }

  if (!exam || questions.length === 0) {
    return <div className="centered">Loading exam…</div>;
  }

  if (!systemCheckPassed) {
    return (
       <div className="centered" style={{ background: '#0f1115', color: '#fff', textAlign: 'center' }}>
          <h2>System Check & Calibration</h2>
          <p>Please sit straight and look at the screen normally.</p>
          {!ready && <p>{loadingStep}</p>}
          {ready && startingExam && <p>Starting exam…</p>}
          {ready && !calibrating && !startingExam && (
            <>
              {brightness !== null && (
                <p style={{ color: lightingOk && !lightingOverride ? '#4ade80' : '#f87171' }}>
                  {lightingOverride
                    ? `Continuing with current lighting (reading: ${brightness.toFixed(0)}). Detection accuracy may be reduced.`
                    : lightingOk
                    ? `Lighting looks good. (reading: ${brightness.toFixed(0)})`
                    : `Lighting is low (reading: ${brightness.toFixed(0)}, need ${MIN_CALIBRATION_BRIGHTNESS}+) — try facing a light source rather than having one behind you.`}
                </p>
              )}
              <button className="finish-btn" onClick={runCalibration} disabled={!lightingOk}>
                {lightingOk ? 'Start Calibration' : 'Waiting for better lighting…'}
              </button>
              {!lightingOk && overrideAvailable && (
                <div style={{ marginTop: '10px' }}>
                  <button onClick={() => setLightingOverride(true)}>Continue Anyway</button>
                </div>
              )}
            </>
          )}
          {calibrating && (
             <div>
               <p>Calibrating... {calibrationProgress}%</p>
               <progress value={calibrationProgress} max="100"></progress>
             </div>
          )}
          <div style={{ marginTop: '20px' }}>
            <video ref={videoRef} className="webcam-mini" style={{ width: '320px', maxWidth: '320px', transform: 'scaleX(-1)' }} muted playsInline />
          </div>
       </div>
    );
  }

  const currentQuestion = questions[currentIndex];
  const originalIndices = originalQuestionsMap[currentQuestion.id] || [1,2,3,4];

  return (
    <div className="exam-room">
      {modal && (
        <div className="modal-backdrop">
          <div className="modal-card">
            <div className="modal-icon">✕</div>
            <h2>{VIOLATION_LABEL[modal.type]}</h2>
            <p>Action has been Recorded</p>
            <button className="modal-ok" onClick={() => setModal(null)}>
              OK
            </button>
          </div>
        </div>
      )}

      <div className="exam-topbar">
        <span className="bell">🔔</span>
        <div className="topbar-right">
          <label style={{ marginRight: '10px', fontSize: '12px' }}>
             <input type="checkbox" checked={showDebug} onChange={e => setShowDebug(e.target.checked)} /> Show Dev/Debug View
          </label>
          <span>Hello, Student</span>
        </div>
      </div>

      <div className="exam-content">
        <div className="exam-question-panel">
          <h3>Question {currentIndex + 1}:</h3>
          <p>{currentQuestion.question_text}</p>
          <div className="option-list">
            {[currentQuestion.option_1, currentQuestion.option_2, currentQuestion.option_3, currentQuestion.option_4].map((opt, i) => (
              <label key={i} className="radio-row">
                <input
                  type="radio"
                  name={`q-${currentQuestion.id}`}
                  checked={answers[currentQuestion.id] === originalIndices[i]}
                  onChange={() => selectOption(currentQuestion.id, originalIndices[i])}
                />
                {opt}
              </label>
            ))}
          </div>
        </div>

        <div className="exam-side-panel">
          <div className="exam-meta-bar">
            <span>
              Questions: {currentIndex + 1}/{questions.length}
            </span>
            <span>Time Left: {formatTime(secondsLeft)}</span>
            <button className="finish-btn" onClick={() => submitExam(false)}>
              Finish Test
            </button>
            {(!document.fullscreenElement) && (
               <button style={{marginTop: '10px'}} onClick={() => document.documentElement.requestFullscreen?.().catch(() => {})}>Re-enter Full Screen</button>
            )}
          </div>

          <div className="question-grid">
            {questions.map((q, i) => (
              <button
                key={q.id}
                className={`question-grid-btn${i === currentIndex ? ' current' : ''}${answers[q.id] ? ' answered' : ''}`}
                onClick={() => setCurrentIndex(i)}
              >
                {i + 1}
              </button>
            ))}
          </div>

          <div style={{ position: 'relative' }}>
            <video ref={videoRef} className="webcam-mini" muted playsInline />
            {showDebug && (
               <canvas ref={debugCanvasRef} width={640} height={480} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }} />
            )}
          </div>
          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </div>
      </div>
    </div>
  );
}
