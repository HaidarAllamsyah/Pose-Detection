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
    this.buffer = [];
    this._dist = new Float32Array(this.n);
  }

  reset() { this.buffer.length = 0; }

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
    const dist = this._dist;
    for (let r = 0, o = 0; r < n; r++, o += d) {
      let s = 0;
      for (let i = 0; i < d; i++) { const t = acuan[o + i] - q[i]; s += t * t; }
      dist[r] = s;
    }
    const idx = new Int32Array(k).fill(-1), dk = new Float32Array(k).fill(Infinity);
    for (let r = 0; r < n; r++) {
      const v = dist[r];
      if (v >= dk[k - 1]) continue;
      let p = k - 1;
      while (p > 0 && dk[p - 1] > v) { dk[p] = dk[p - 1]; idx[p] = idx[p - 1]; p--; }
      dk[p] = v; idx[p] = r;
    }
    const suara = new Array(this.classes.length).fill(0);
    for (let i = 0; i < k; i++) suara[this.label[idx[i]]]++;
    let best = 0;
    for (let c = 1; c < suara.length; c++) if (suara[c] > suara[best]) best = c;
    return [this.classes[best], suara[best] / k];
  }

  // Mengembalikan [label, keyakinan]
  prediksiFrame(lm) {
    this.buffer.push(hitungFiturDasar(lm));
    if (this.buffer.length > this.jendela) this.buffer.shift();
    const fitur = fiturDariJendela(this.buffer);
    if (this.cekStandby(lm)) return ["standby", 0.95];
    return this._knn(fitur);
  }
}

// ---- Stabilisator (port StabilisatorDeteksi di tes_pose.py) ----
const FRAMES_UNTUK_KUNCI = 8;
const FRAMES_UNTUK_UNLOCK = 12;

export function formatNama(label) {
  if (["standby", "tidak_yakin", "Transisi...", "Menganalisis..."].includes(label)) return label;
  return label.replaceAll("_", " ").toUpperCase();
}

export class StabilisatorDeteksi {
  constructor() { this.reset(); }
  reset() {
    this.riwayat = [];
    this.terkunci = null;
    this.hitung = 0;
    this.kandidat = null;
  }
  _terbanyak() {
    const m = new Map();
    for (const l of this.riwayat) m.set(l, (m.get(l) || 0) + 1);
    let best = null, bc = 0;
    for (const [l, c] of m) if (c > bc) { best = l; bc = c; }
    return [best, bc];
  }
  // Mengembalikan [teks, kodeWarna]
  perbarui(label, conf) {
    if (conf >= 0.5) {
      this.riwayat.push(label);
      if (this.riwayat.length > 20) this.riwayat.shift();
    }
    if (this.riwayat.length < 5) return ["Menganalisis...", "abu"];
    const [sering, jumlah] = this._terbanyak();
    const rasio = jumlah / this.riwayat.length;

    if (this.terkunci === null) {
      if (sering === "standby") return ["SIAP / STANDBY", "kuning"];
      if (rasio >= 0.55) {
        this.hitung++;
        if (this.hitung >= FRAMES_UNTUK_KUNCI) { this.terkunci = sering; this.hitung = 0; this.kandidat = null; }
        return [formatNama(sering), "abu"];
      }
      this.hitung = 0;
      return ["Transisi...", "abu"];
    }

    if (sering !== this.terkunci) {
      if (sering === "standby") {
        this.terkunci = null; this.hitung = 0;
        return ["SIAP / STANDBY", "kuning"];
      }
      if (sering === this.kandidat) this.hitung++;
      else { this.kandidat = sering; this.hitung = 1; }
      if (this.hitung >= FRAMES_UNTUK_UNLOCK) { this.terkunci = sering; this.kandidat = null; this.hitung = 0; }
    } else {
      this.kandidat = null; this.hitung = 0;
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
