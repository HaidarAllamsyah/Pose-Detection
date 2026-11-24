// Aplikasi web deteksi gerakan: kamera -> MediaPipe Pose (JS) -> KNN (engine.js) -> tampilan.
import { PoseLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { KlasifikasiPose, StabilisatorDeteksi, badanLengkap, float16ke32 } from "./engine.js";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const AMBANG_VIS = 0.5;
const SAMBUNGAN = [
  [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32],
];
const WARNA = { hijau: "#00ff00", kuning: "#ffdc00", abu: "#c8c8c8", oranye: "#ff8c00", merah: "#ff3030" };

const kanvas = document.getElementById("kanvas");
const ctx = kanvas.getContext("2d");
const pesan = document.getElementById("pesan");
const tmbMulai = document.getElementById("tmbMulai");
const tmbCermin = document.getElementById("tmbCermin");
const video = document.createElement("video");
video.playsInline = true; video.muted = true;

let pendeteksi = null, klasifikator = null;
const facing = "user";   // kamera DEPAN saja (tidak ada pilihan kamera belakang)
let stream = null, cermin = true, jalan = false;
const stab = new StabilisatorDeteksi();
let fps = 0, tSebelum = performance.now(), videoTerakhir = -1;

function tampilPesan(teks) { pesan.style.display = teks ? "flex" : "none"; if (teks) pesan.innerHTML = teks; }

async function muatModel() {
  tampilPesan("Memuat model… (pertama kali agak lama)");
  const [meta, bufAcuan, bufLabel] = await Promise.all([
    fetch("model/meta.json").then((r) => r.json()),
    fetch("model/acuan.bin").then((r) => r.arrayBuffer()),
    fetch("model/label.bin").then((r) => r.arrayBuffer()),
  ]);
  klasifikator = new KlasifikasiPose(meta, float16ke32(new Uint16Array(bufAcuan)), new Uint8Array(bufLabel));

  const fileset = await FilesetResolver.forVisionTasks(WASM);
  const buat = (delegate) => PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "models/pose_landmarker_lite.task", delegate },
    runningMode: "VIDEO", numPoses: 1,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  try { pendeteksi = await buat("GPU"); } catch { pendeteksi = await buat("CPU"); }
}

async function bukaKamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: facing, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false,
  });
  video.srcObject = stream;
  await video.play();
  kanvas.width = video.videoWidth; kanvas.height = video.videoHeight;
  document.getElementById("panggung").style.aspectRatio = `${video.videoWidth}/${video.videoHeight}`;
  stab.reset(); klasifikator.reset();
}

// ---------- Gambar ----------
function px(p, w, h) { return [(cermin ? 1 - p.x : p.x) * w, p.y * h]; }

function gambarKepala(lm, w, h) {
  const ok = (i) => (lm[i].visibility ?? 1) >= AMBANG_VIS;
  const P = (i) => px(lm[i], w, h);
  let lebarBahu = null;
  if (ok(11) && ok(12)) { const a = P(11), b = P(12); lebarBahu = Math.hypot(a[0] - b[0], a[1] - b[1]); }
  let cx, cy, r;
  if (ok(7) && ok(8)) {
    const a = P(7), b = P(8);
    cx = (a[0] + b[0]) / 2; cy = (a[1] + b[1]) / 2;
    r = 0.75 * Math.hypot(a[0] - b[0], a[1] - b[1]);
    if (lebarBahu) r = Math.max(r, 0.28 * lebarBahu);
  } else if (ok(0) && lebarBahu) {
    [cx, cy] = P(0); r = 0.30 * lebarBahu;
  } else return;
  ctx.strokeStyle = WARNA.hijau; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(cx, cy - 0.15 * r, r, 0, Math.PI * 2); ctx.stroke();
}

function gambarKerangka(lm, w, h) {
  ctx.strokeStyle = WARNA.hijau; ctx.lineWidth = 3;
  for (const [a, b] of SAMBUNGAN) {
    if ((lm[a].visibility ?? 1) >= AMBANG_VIS && (lm[b].visibility ?? 1) >= AMBANG_VIS) {
      const [xa, ya] = px(lm[a], w, h), [xb, yb] = px(lm[b], w, h);
      ctx.beginPath(); ctx.moveTo(xa, ya); ctx.lineTo(xb, yb); ctx.stroke();
    }
  }
  gambarKepala(lm, w, h);
  ctx.fillStyle = "#ff0000";
  for (let i = 11; i < lm.length; i++) {
    if ((lm[i].visibility ?? 1) >= AMBANG_VIS) {
      const [x, y] = px(lm[i], w, h);
      ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill();
    }
  }
}

function tulis(teks, x, y, warna, ukuran) {
  ctx.font = `600 ${ukuran}px system-ui, sans-serif`;
  ctx.lineWidth = 4; ctx.strokeStyle = "#000"; ctx.strokeText(teks, x, y);
  ctx.fillStyle = warna; ctx.fillText(teks, x, y);
}

// ---------- Loop ----------
function loop() {
  if (!jalan) return;
  requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === videoTerakhir) return;
  videoTerakhir = video.currentTime;

  const w = kanvas.width, h = kanvas.height;
  ctx.save();
  if (cermin) { ctx.translate(w, 0); ctx.scale(-1, 1); }
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();

  const hasil = pendeteksi.detectForVideo(video, performance.now());
  let status, warna = WARNA.merah;
  if (hasil.landmarks && hasil.landmarks.length) {
    const lm = hasil.landmarks[0];
    gambarKerangka(lm, w, h);
    if (!badanLengkap(lm)) {
      status = "MUNDUR: seluruh badan harus terlihat"; warna = WARNA.oranye;
      stab.reset(); klasifikator.reset();
    } else {
      const [label, conf] = klasifikator.prediksiFrame(lm);
      const [stabil, kode] = stab.perbarui(label, conf);
      warna = WARNA[kode] || WARNA.abu;
      if (stabil === "Menganalisis..." || stabil === "Transisi...") status = stabil;
      else if (stabil === "SIAP / STANDBY") { status = "POSISI: SIAP / STANDBY"; warna = WARNA.kuning; }
      else status = `Gerakan: ${stabil} (${Math.round(conf * 100)}%)`;
    }
  } else {
    status = "Tidak ada orang terdeteksi";
    stab.reset(); klasifikator.reset();
  }

  const sekarang = performance.now();
  const f = 1000 / Math.max(sekarang - tSebelum, 1);
  tSebelum = sekarang; fps = fps === 0 ? f : 0.9 * fps + 0.1 * f;

  const u = Math.max(16, Math.round(w / 28));
  tulis(`FPS: ${fps.toFixed(1)}`, 10, u + 6, "#00e5ff", u);
  tulis(status, 10, u * 2 + 14, warna, Math.round(u * 0.95));
}

// ---------- Tombol ----------
async function mulai() {
  tmbMulai.disabled = true;
  try {
    if (!pendeteksi) await muatModel();
    tampilPesan("Membuka kamera…");
    await bukaKamera();
    tampilPesan("");
    jalan = true; tmbMulai.textContent = "Berhenti"; tmbMulai.disabled = false;
    tmbCermin.disabled = false;
    requestAnimationFrame(loop);
  } catch (e) {
    console.error(e);
    tmbMulai.disabled = false;
    const izin = e && (e.name === "NotAllowedError" || e.name === "SecurityError");
    tampilPesan(izin
      ? "Akses kamera ditolak. Izinkan kamera di pengaturan browser, lalu coba lagi."
      : `Gagal: ${e && e.message ? e.message : e}`);
  }
}

function berhenti() {
  jalan = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  ctx.clearRect(0, 0, kanvas.width, kanvas.height);
  tmbMulai.textContent = "Mulai kamera"; tmbCermin.disabled = true;
  tampilPesan("Kamera dihentikan.");
}

tmbMulai.onclick = () => (jalan ? berhenti() : mulai());
tmbCermin.onclick = () => {
  cermin = !cermin;
  tmbCermin.textContent = `Cermin: ${cermin ? "aktif" : "mati"}`;
};
