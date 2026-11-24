// Mesin klasifikasi gerakan (port dari src/fitur.py, src/klasifikasi.py, tes_pose.py).
// Tanpa DOM, sehingga bisa diuji di Node dan dipakai di browser.
// Landmark: array 33 objek {x, y, z, visibility} (koordinat ternormalisasi MediaPipe).

export const JENDELA_DEFAULT = 15;
const N_STATIS = 26;
const N_DASAR = 28;
const IDX_SUDUT_UNIK = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11];
const IDX_GERAK = [16, 17, 18, 19, 14, 15, 26, 27, 10, 11, 12, 13];
export const AMBANG_DIAM = 0.012;
export const MIN_FRAME_GERAK = 8;

const SUDUT_SENDI = [
  [15, 13, 11], [16, 14, 12], [13, 11, 23], [14, 12, 24],
  [11, 23, 25], [12, 24, 26], [23, 25, 27], [24, 26, 28],
  [11, 23, 25], [12, 24, 26], [12, 11, 23], [23, 24, 26],
];
const JARAK_SENDI = [[11, 12], [15, 23], [16, 24], [27, 28], [25, 26], [15, 16]];

const rad2deg = (r) => (r * 180) / Math.PI;

function sudutTigaTitik(a, p, b) {
  const vax = a.x - p.x, vay = a.y - p.y, vbx = b.x - p.x, vby = b.y - p.y;
  const na = Math.hypot(vax, vay), nb = Math.hypot(vbx, vby);
  if (na < 1e-8 || nb < 1e-8) return NaN;
  const c = Math.max(-1, Math.min(1, (vax * vbx + vay * vby) / (na * nb)));
  return rad2deg(Math.acos(c));
}

export function hitungSudut(lm) {
  return SUDUT_SENDI.map(([a, p, b]) => sudutTigaTitik(lm[a], lm[p], lm[b]));
}

function tinggiBadan(lm) {
  const bahuY = (lm[11].y + lm[12].y) / 2;
  const kakiY = (lm[27].y + lm[28].y) / 2;
  const t = Math.abs(kakiY - bahuY);
  return t > 1e-4 ? t : 1.0;
}

export function hitungJarak(lm) {
  const t = tinggiBadan(lm);
  return JARAK_SENDI.map(([a, b]) => Math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y) / t);
}

// Fitur dasar per frame (28): 26 statis + posisi pinggul x/y.
export function hitungFiturDasar(lm) {
  const mid = (i, j) => ({ x: (lm[i].x + lm[j].x) / 2, y: (lm[i].y + lm[j].y) / 2 });
  const bahu = mid(11, 12), pinggul = mid(23, 24);
  const tx = bahu.x - pinggul.x, ty = bahu.y - pinggul.y;
  const torso = Math.hypot(tx, ty);
  if (!Number.isFinite(torso) || torso < 1e-4) return new Float64Array(N_DASAR);

  const sd = hitungSudut(lm);
  const sudut = IDX_SUDUT_UNIK.map((i) => sd[i]);
  const rel = (i, ac) => [(lm[i].x - ac.x) / torso, (lm[i].y - ac.y) / torso];
  const tKi = rel(15, bahu), tKa = rel(16, bahu);
  const lKi = rel(25, pinggul), lKa = rel(26, pinggul);
  const kKi = rel(27, pinggul), kKa = rel(28, pinggul);
  const dist = (i, j) => Math.hypot(lm[i].x - lm[j].x, lm[i].y - lm[j].y) / torso;

  const v = [
    ...sudut,
    tKi[0], tKi[1], tKa[0], tKa[1],
    lKi[1], lKa[1],
    kKi[0], kKi[1], kKa[0], kKa[1],
    dist(11, 12), dist(23, 24),
    rad2deg(Math.atan2(Math.abs(tx), Math.abs(ty))),
    dist(15, 16), dist(13, 26), dist(14, 25),
    pinggul.x / torso, pinggul.y / torso,
  ];
  return Float64Array.from(v, (x) => (Number.isFinite(x) ? x : 0));
}

// Fitur model (110) dari jendela fitur dasar (array of Float64Array(28)).
export function fiturDariJendela(jendela) {
  const n = jendela.length;
  const out = new Float64Array(N_STATIS + 3 * N_DASAR);
  const akhir = jendela[n - 1];
  for (let j = 0; j < N_STATIS; j++) out[j] = akhir[j];
  for (let j = 0; j < N_DASAR; j++) {
    let sum = 0, mn = Infinity, mx = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = jendela[i][j];
      sum += v; if (v < mn) mn = v; if (v > mx) mx = v;
    }
    const mean = sum / n;
    let ss = 0;
    for (let i = 0; i < n; i++) ss += (jendela[i][j] - mean) ** 2;
    let vel = 0;
    for (let i = 1; i < n; i++) vel += Math.abs(jendela[i][j] - jendela[i - 1][j]);
    out[N_STATIS + j] = Math.sqrt(ss / n);
    out[N_STATIS + N_DASAR + j] = mx - mn;
    out[N_STATIS + 2 * N_DASAR + j] = n > 1 ? vel / (n - 1) : 0;
  }
  return out;
}

export function gerakTubuh(jendela) {
  const n = jendela.length;
  if (n < 2) return 0;
  let total = 0;
  for (let i = 1; i < n; i++)
    for (const j of IDX_GERAK) total += Math.abs(jendela[i][j] - jendela[i - 1][j]);
  return total / ((n - 1) * IDX_GERAK.length);
}

export function float16ke32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * 2 ** -14 * (f / 1024)
      : e === 31 ? (f ? NaN : s * Infinity)
        : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return out;
}

const JEDA_MAKS = 0.5;   // detik; jeda lebih lama -> buffer dimulai ulang

export class KlasifikasiPose {
  constructor(meta, acuan, label) {
    this.meta = meta;
    this.classes = meta.classes;
    this.k = meta.k;
    this.d = meta.d;
    this.n = meta.n;
    this.acuan = acuan;     // Float32Array n*d (sudah distandarisasi)
    this.label = label;     // Uint8Array n
    this.jendela = meta.jendela || JENDELA_DEFAULT;
    // Model dilatih pada bingkai setara 16:9 dan 30 fps (lihat src/fitur.py)
    this.pakaiBingkai = !!meta.samakan_bingkai;
    this.rasioAcuan = meta.rasio_acuan || 16 / 9;
    this.fpsAcuan = meta.fps_acuan || 30;
    this.faktorX = 1;
    this.buffer = [];
    this._dist = null;
    this.reset();
  }

  reset() {
    this.buffer.length = 0;
    this._lmLalu = null; this._tLalu = null; this._tBerikut = null;
  }

  // Ukuran gambar kamera (piksel). Kamera depan HP tegak (mis. 480x640) disetarakan ke 16:9.
  aturBingkai(lebar, tinggi) {
    this.faktorX = this.pakaiBingkai && lebar && tinggi ? (lebar / tinggi) / this.rasioAcuan : 1;
  }

  _siapkan(lm) {
    const f = this.faktorX;
    return lm.map((p) => ({ x: p.x * f, y: p.y, z: p.z, visibility: p.visibility ?? 1 }));
  }

  _push(dasar) {
    this.buffer.push(dasar);
    if (this.buffer.length > this.jendela) this.buffer.shift();
  }

  // Buffer diisi sampel tiap 1/30 detik (interpolasi antar frame kamera), sehingga
  // HP 10 fps menghasilkan fitur yang setara dengan video latih 30 fps.
  _masukkan(lm, waktu) {
    if (waktu == null || !this.pakaiBingkai) { this._push(hitungFiturDasar(lm)); return; }
    const langkah = 1 / this.fpsAcuan;
    const dt = this._lmLalu ? waktu - this._tLalu : -1;
    if (!this._lmLalu || dt < 0 || dt > JEDA_MAKS) {
      this.buffer.length = 0;
      this._push(hitungFiturDasar(lm));
      this._tBerikut = waktu + langkah;
    } else {
      const t0 = this._tLalu, lm0 = this._lmLalu;
      while (dt > 0 && this._tBerikut <= waktu + 1e-9) {
        const a = (this._tBerikut - t0) / dt, b = 1 - a;
        this._push(hitungFiturDasar(lm.map((p, i) => ({
          x: lm0[i].x * b + p.x * a, y: lm0[i].y * b + p.y * a,
          z: lm0[i].z * b + p.z * a, visibility: lm0[i].visibility * b + p.visibility * a,
        }))));
        this._tBerikut += langkah;
      }
    }
    this._lmLalu = lm; this._tLalu = waktu;
  }

  cekStandby(lm) {
    const s = hitungSudut(lm), j = hitungJarak(lm);
    const [bahuKi, bahuKa, pingKi, pingKa, lutKi, lutKa] = [s[2], s[3], s[4], s[5], s[6], s[7]];
    const lebarKaki = j[3];
    if ([bahuKi, bahuKa, pingKi, pingKa, lutKi, lutKa, lebarKaki].some(Number.isNaN)) return false;
    const diam = this.buffer.length < MIN_FRAME_GERAK || gerakTubuh(this.buffer) < AMBANG_DIAM;
    return diam && bahuKi < 25 && bahuKa < 25 && lutKi > 155 && lutKa > 155 &&
      pingKi > 155 && pingKa > 155 && lebarKaki > 0.06 && lebarKaki < 0.18;
  }

  // KNN (k tetangga terdekat, suara terbanyak; seri -> indeks kelas terkecil seperti sklearn)
  _knn(fitur) {
    const { d, n, acuan, k, meta } = this;
    const q = new Float32Array(d);
    for (let i = 0; i < d; i++) q[i] = (fitur[i] - meta.mean[i]) / meta.scale[i];
    // Jarak dihitung sambil menyimpan k terdekat. Perhitungan satu titik dihentikan lebih
    // awal bila jaraknya sudah melebihi tetangga ke-k (hasil tetap persis sama, jauh lebih cepat).
    const idx = new Int32Array(k).fill(-1), dk = new Float64Array(k).fill(Infinity);
    for (let r = 0, o = 0; r < n; r++, o += d) {
      const batas = dk[k - 1];
      let s = 0, i = 0;
      while (i < d) {
        const akhir = Math.min(i + 16, d);
        for (; i < akhir; i++) { const t = acuan[o + i] - q[i]; s += t * t; }
        if (s >= batas) break;
      }
      if (s >= batas) continue;
      let p = k - 1;
      while (p > 0 && dk[p - 1] > s) { dk[p] = dk[p - 1]; idx[p] = idx[p - 1]; p--; }
      dk[p] = s; idx[p] = r;
    }
    const suara = new Array(this.classes.length).fill(0);
    for (let i = 0; i < k; i++) suara[this.label[idx[i]]]++;
    let best = 0;
    for (let c = 1; c < suara.length; c++) if (suara[c] > suara[best]) best = c;
    return [this.classes[best], suara[best] / k];
  }

  // waktu: detik saat frame diambil (null = tiap frame dianggap 1/30 detik).
  // Mengembalikan [label, keyakinan]
  prediksiFrame(lmMentah, waktu = null) {
    const lm = this._siapkan(lmMentah);
    this._masukkan(lm, waktu);
    const fitur = fiturDariJendela(this.buffer);
    if (this.cekStandby(lm)) return ["standby", 0.95];
    return this._knn(fitur);
  }
}

// ---- Stabilisator (port StabilisatorDeteksi di tes_pose.py, versi BERBASIS WAKTU) ----
// Versi PC menghitung frame (riwayat 20, minimal 5, kunci 8, lepas 12) dan berjalan ~16 fps.
// HP hanya ~10 fps, sehingga hitungan frame membuat label lebih lambat berganti. Di sini
// semua dihitung dalam detik, disetel agar terasa sama dengan PC pada 16 fps.
const FPS_PC = 16;
const RIWAYAT_DETIK = 20 / FPS_PC;   // 1,25 dtk
const MIN_DETIK = 5 / FPS_PC;        // 0,31 dtk
const KUNCI_DETIK = 8 / FPS_PC;      // 0,50 dtk
const LEPAS_DETIK = 12 / FPS_PC;     // 0,75 dtk

export function formatNama(label) {
  if (["standby", "tidak_yakin", "Transisi...", "Menganalisis..."].includes(label)) return label;
  return label.replaceAll("_", " ").toUpperCase();
}

export class StabilisatorDeteksi {
  constructor() { this.reset(); }
  reset() {
    this.riwayat = [];        // {t, label}
    this.terkunci = null;
    this.durasi = 0;          // detik kandidat konsisten (pengganti hitungan frame)
    this.kandidat = null;
    this.tMulai = null;
    this.tLalu = null;
  }
  _terbanyak() {
    const m = new Map();
    for (const e of this.riwayat) m.set(e.label, (m.get(e.label) || 0) + 1);
    let best = null, bc = 0;
    for (const [l, c] of m) if (c > bc) { best = l; bc = c; }
    return [best, bc];
  }
  // t: detik (performance.now()/1000). Mengembalikan [teks, kodeWarna]
  perbarui(label, conf, t) {
    if (this.tMulai === null) this.tMulai = t;
    const dt = this.tLalu === null ? 1 / FPS_PC : Math.min(Math.max(t - this.tLalu, 0), 0.25);
    this.tLalu = t;
    if (conf >= 0.5) this.riwayat.push({ t, label });
    while (this.riwayat.length && this.riwayat[0].t < t - RIWAYAT_DETIK) this.riwayat.shift();
    if (!this.riwayat.length || t - this.tMulai < MIN_DETIK) return ["Menganalisis...", "abu"];
    const [sering, jumlah] = this._terbanyak();
    const rasio = jumlah / this.riwayat.length;

    if (this.terkunci === null) {
      if (sering === "standby") return ["SIAP / STANDBY", "kuning"];
      if (rasio >= 0.55) {
        this.durasi += dt;
        if (this.durasi >= KUNCI_DETIK) { this.terkunci = sering; this.durasi = 0; this.kandidat = null; }
        return [formatNama(sering), "abu"];
      }
      this.durasi = 0;
      return ["Transisi...", "abu"];
    }

    if (sering !== this.terkunci) {
      if (sering === "standby") {
        this.terkunci = null; this.durasi = 0;
        return ["SIAP / STANDBY", "kuning"];
      }
      if (sering === this.kandidat) this.durasi += dt;
      else { this.kandidat = sering; this.durasi = dt; }
      if (this.durasi >= LEPAS_DETIK) { this.terkunci = sering; this.kandidat = null; this.durasi = 0; }
    } else {
      this.kandidat = null; this.durasi = 0;
    }
    return [formatNama(this.terkunci), "hijau"];
  }
}

export function badanLengkap(lm) {
  for (const i of [11, 12, 23, 24, 25, 26]) {
    const p = lm[i], vis = p.visibility ?? 1.0;
    if (vis < 0.45 || p.y < -0.1 || p.y > 1.05) return false;
  }
  return true;
}
