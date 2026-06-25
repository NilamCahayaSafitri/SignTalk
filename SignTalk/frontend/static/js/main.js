/* ================================================================
   main.js — SignSense Frontend
   Pakai Flask biasa (HTTP Polling), bukan SocketIO
   Mode kata sekarang pakai BiGRU PyTorch (sekuens 30 frame)
================================================================ */

// ===============================================================
// DATA
// ===============================================================
const LABEL_KATA = [
  "berasal","berpikir","maaf","makan","mandi",
  "nama","perkenalkan","saya","tidur","tolong",
  "tunggu","umur","berteman","sementara","santai aja","bareng"
];
const MODE_DESC = {
  huruf: "Deteksi huruf A–Z dari gerakan dua tangan.",
  angka: "Deteksi angka 0–9 dari posisi jari.",
  kata:  "Deteksi kata BISINDO (sekuens 30 frame, dua tangan + wajah)."
};
const HERO_WORDS = ["Instantly.", "Akurat.", "Mudah.", "Real-Time.", "Sekarang."];

// Mode kata butuh buffer 30 frame penuh sebelum bisa prediksi (BiGRU)
const KATA_BUFFER_TARGET = 30;

// ===============================================================
// STATE
// ===============================================================
let currentMode   = "huruf";
let cameraOn      = false;
let stream        = null;
let pollInterval  = null;
let isSending     = false;
let frameCount    = 0;
let fpsCounter    = 0;
let lastFpsTime   = Date.now();
let heroWordIdx   = 0;

// ===============================================================
// DOM
// ===============================================================
const video          = document.getElementById("video");
const camOverlay    = document.getElementById("camOverlay");
const livePill      = document.getElementById("livePill");
const onCamPred     = document.getElementById("onCamPred");
const onCamText     = document.getElementById("onCamText");
const btnStart      = document.getElementById("btnStart");
const btnStop       = document.getElementById("btnStop");
const btnClear      = document.getElementById("btnClear");
const predMain      = document.getElementById("predMain");
const predRaw       = document.getElementById("predRaw");
const confFill      = document.getElementById("confFill");
const confNum       = document.getElementById("confNum");
const fpsVal        = document.getElementById("fpsVal");
const frameVal      = document.getElementById("frameVal");
const histChips     = document.getElementById("histChips");
const histBadge     = document.getElementById("histBadge");
const modeActiveName = document.getElementById("modeActiveName");
const modeActiveDesc = document.getElementById("modeActiveDesc");
const srvDot        = document.getElementById("srvDot");
const srvText       = document.getElementById("srvText");
const vocabCard     = document.getElementById("vocabCard");
const vocabWrap     = document.getElementById("vocabWrap");
const heroAccent    = document.getElementById("heroAccentWord");
const navbar        = document.getElementById("navbar");

const globalHistChips = document.getElementById("globalHistChips");
const globalHistBadge = document.getElementById("globalHistBadge");
const btnGlobalClear  = document.getElementById("btnGlobalClear");

// Elemen indikator buffer sekuens BiGRU (opsional, dicek dulu sebelum dipakai)
const bufferProgressWrap = document.getElementById("bufferProgressWrap");
const bufferProgressFill = document.getElementById("bufferProgressFill");
const bufferProgressText = document.getElementById("bufferProgressText");

// Array lokal untuk menampung riwayat gabungan secara real-time
let globalHistoryList = [];

// ===============================================================
// NAVBAR SCROLL EFFECT (SINKRONISASI BOOTSTRAP)
// ===============================================================
window.addEventListener("scroll", () => {
  if (navbar) {
    navbar.classList.toggle("scrolled", window.scrollY > 30);
  }

  const sections = ["home", "detect", "referensi"];
  for (const id of sections) {
    const el = document.getElementById(id);
    if (!el) continue;
    const rect = el.getBoundingClientRect();
    if (rect.top <= 90 && rect.bottom > 90) {
      document.querySelectorAll(".custom-nav-link").forEach(l => {
        l.classList.toggle("active", l.getAttribute("href") === "#" + id);
      });
      break;
    }
  }
});

// ===============================================================
// HERO — Rotating accent word
// ===============================================================
function rotateHeroWord() {
  if (!heroAccent) return;
  heroAccent.style.opacity = "0";
  heroAccent.style.transform = "translateY(-10px)";
  setTimeout(() => {
    heroWordIdx = (heroWordIdx + 1) % HERO_WORDS.length;
    heroAccent.textContent = HERO_WORDS[heroWordIdx];
    heroAccent.style.transition = "all 0.4s ease";
    heroAccent.style.opacity = "1";
    heroAccent.style.transform = "translateY(0)";
  }, 350);
}
setInterval(rotateHeroWord, 2500);
if (heroAccent) {
  heroAccent.style.transition = "all 0.4s ease";
  heroAccent.style.display = "inline-block";
}

// ===============================================================
// LANDMARK DOTS — Animated hand points on camera card
// ===============================================================
function spawnLandmarkDots() {
  const container = document.getElementById("landmarkDots");
  if (!container) return;

  const points = [
    [0.45,0.78],[0.40,0.60],[0.38,0.48],[0.37,0.38],[0.36,0.28],
    [0.50,0.55],[0.51,0.40],[0.52,0.30],[0.52,0.22],
    [0.58,0.57],[0.59,0.40],[0.60,0.30],[0.60,0.22],
    [0.65,0.60],[0.66,0.45],[0.67,0.35],[0.67,0.28],
    [0.72,0.65],[0.73,0.53],[0.74,0.43],[0.74,0.36],
  ];

  points.forEach(([rx, ry], i) => {
    const dot = document.createElement("div");
    dot.className = "lm-dot";
    dot.style.left = (rx * 100) + "%";
    dot.style.top  = (ry * 100) + "%";
    dot.style.animationDelay = (i * 0.1) + "s";
    container.appendChild(dot);
  });
}
spawnLandmarkDots();

// ===============================================================
// SERVER STATUS CHECK
// ===============================================================
async function checkStatus() {
  try {
    const res  = await fetch("/api/status");
    const data = await res.json();
    if (srvDot && srvText) {
      srvDot.className = "status-dot-sm online";
      srvText.textContent = "Server online";
    }
    const missing = [];
    if (!data.model_huruf) missing.push("huruf");
    if (!data.model_angka) missing.push("angka");
    if (!data.model_kata)  missing.push("kata");
    if (missing.length) {
      console.warn("Model belum ada:", missing);
    }
  } catch {
    if (srvDot && srvText) {
      srvDot.className = "status-dot-sm offline";
      srvText.textContent = "Server offline";
    }
  }
}
checkStatus();

// ===============================================================
// KAMERA
// ===============================================================
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false
    });
    video.srcObject = stream;
    await video.play();

    cameraOn = true;
    camOverlay.classList.add("hidden");
    livePill.classList.remove("hidden");
    onCamPred.classList.remove("hidden");
    btnStart.classList.add("hidden");
    btnStop.classList.remove("hidden");

    pollInterval = setInterval(sendFrame, 40);

  } catch (err) {
    alert("Gagal akses kamera: " + err.message);
  }
}

function stopCamera() {
  cameraOn = false;
  clearInterval(pollInterval);

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;

  camOverlay.classList.remove("hidden");
  livePill.classList.add("hidden");
  onCamPred.classList.add("hidden");
  btnStart.classList.remove("hidden");
  btnStop.classList.add("hidden");

  predMain.textContent = "—";
  predRaw.textContent  = "—";
  onCamText.textContent = "-";
  confFill.style.width = "0%";
  confNum.textContent  = "—%";
  if (fpsVal) fpsVal.textContent = "—";
  frameCount = 0;
  if (frameVal) frameVal.textContent = "0";
  hideBufferProgress();
}

// ===============================================================
// KIRIM FRAME → Flask /predict (HTTP POST)
// ===============================================================
async function sendFrame() {
  if (!cameraOn || video.readyState < 2 || isSending) return;
  isSending = true;

  try {
    const canvas = document.createElement("canvas");
    canvas.width = 640; canvas.height = 480; // Ubah ke resolusi standar
    canvas.getContext("2d").drawImage(video, 0, 0, 640, 480);
    const b64 = canvas.toDataURL("image/jpeg", 0.9);

    const res  = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: b64, mode: currentMode })
    });

    if (res.ok) {
      const data = await res.json();
      updateDisplay(data);
    }
  } catch (err) {
    console.warn("Predict error:", err.message);
  } finally {
    isSending = false;
  }

  // FPS & Frame Tracker
  fpsCounter++;
  frameCount++;
  if (frameVal) frameVal.textContent = frameCount;
  const now = Date.now();
  if (now - lastFpsTime >= 1000) {
    if (fpsVal) fpsVal.textContent = fpsCounter;
    fpsCounter = 0;
    lastFpsTime = now;
  }
}

// ===============================================================
// INDIKATOR BUFFER SEKUENS (khusus mode kata / BiGRU)
// ===============================================================
function updateBufferProgress(progress) {
  if (currentMode !== "kata" || progress === null || progress === undefined) {
    hideBufferProgress();
    return;
  }
  if (!bufferProgressWrap) return;

  bufferProgressWrap.classList.remove("hidden");
  const pct = Math.min(100, (progress / KATA_BUFFER_TARGET) * 100);
  if (bufferProgressFill) bufferProgressFill.style.width = pct + "%";

  if (bufferProgressText) {
    if (progress >= KATA_BUFFER_TARGET) {
      bufferProgressText.textContent = "Menganalisis gerakan...";
    } else {
      bufferProgressText.textContent = `Mengumpulkan gerakan ${progress}/${KATA_BUFFER_TARGET}`;
    }
  }
}

function hideBufferProgress() {
  if (bufferProgressWrap) bufferProgressWrap.classList.add("hidden");
}

// ===============================================================
// UPDATE TAMPILAN
// ===============================================================
function updateDisplay({ raw, stable, confidence, history, buffer_progress }) {
  // Indikator buffer sekuens (hanya relevan di mode kata)
  updateBufferProgress(buffer_progress);

  // Prediksi utama
  if (stable && stable !== "-" && predMain.textContent !== stable) {
    predMain.textContent = stable;
    predMain.classList.remove("pop");
    void predMain.offsetWidth;
    predMain.classList.add("pop");
    onCamText.textContent = stable;
  }

  predRaw.textContent = raw || "-";

  // Confidence bar
  if (confidence !== null && confidence !== undefined) {
    const pct = Math.min(100, confidence);
    confFill.style.width = pct + "%";
    confNum.textContent  = pct.toFixed(1) + "%";
    if (pct >= 80)      confFill.style.background = "#10b981";
    else if (pct >= 50) confFill.style.background = "#f59e0b";
    else                confFill.style.background = "#ef4444";
  } else {
    confFill.style.width = "0%";
    confNum.textContent  = "—%";
  }

  // [1] UPDATE RIWAYAT MODE AKTIF
  if (history && history.length) {
    histBadge.textContent = history.length;
    histChips.innerHTML = "";
    history.slice().reverse().forEach(item => {
      const chip = document.createElement("span");
      chip.className = "hist-chip";
      chip.textContent = item;
      histChips.appendChild(chip);
    });
  }

  // [2] UPDATE RIWAYAT GLOBAL
  if (stable && stable !== "-") {
    const kataTerbaru = stable;

    if (globalHistoryList.length === 0 || globalHistoryList[globalHistoryList.length - 1].teks !== kataTerbaru) {

      globalHistoryList.push({
        teks: kataTerbaru,
        mode: currentMode === "huruf" ? "h" : currentMode === "angka" ? "a" : "k"
      });

      if (globalHistoryList.length > 20) {
        globalHistoryList.shift();
      }

      globalHistBadge.textContent = globalHistoryList.length;
      globalHistChips.innerHTML = "";

      globalHistoryList.forEach(item => {
        const chip = document.createElement("span");
        chip.className = `hist-chip global-chip-${item.mode}`;
        chip.textContent = item.teks;
        globalHistChips.appendChild(chip);
      });

      globalHistChips.scrollLeft = globalHistChips.scrollWidth;
    }
  }

  // Highlight vocab
  if (currentMode === "kata" && stable && stable !== "-") {
    document.querySelectorAll(".vocab-pill").forEach(p => {
      p.classList.toggle("active", p.dataset.kata === stable);
    });
  }
}

// ===============================================================
// MODE SWITCH
// ===============================================================
function setMode(mode) {
  currentMode = mode;

  document.querySelectorAll(".mode-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.mode === mode);
  });

  modeActiveName.textContent = mode.toUpperCase();
  modeActiveDesc.textContent = MODE_DESC[mode];

  if (mode === "kata") {
    vocabCard.classList.remove("hidden");
    buildVocab();
  } else {
    vocabCard.classList.add("hidden");
    hideBufferProgress();
  }

  // Reset display
  predMain.textContent = "—";
  predRaw.textContent  = "—";
  onCamText.textContent = "-";
  confFill.style.width = "0%";
  confNum.textContent  = "—%";
  histChips.innerHTML  = '<span class="hist-empty">Belum ada deteksi...</span>';
  histBadge.textContent = "0";

  fetch("/api/change_mode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode })
  }).catch(() => {});
}

function buildVocab() {
  vocabWrap.innerHTML = "";
  LABEL_KATA.forEach(kata => {
    const pill = document.createElement("span");
    pill.className   = "vocab-pill";
    pill.textContent = kata;
    pill.dataset.kata = kata;
    vocabWrap.appendChild(pill);
  });
}

// ===============================================================
// CLEAR HISTORY SELEKTIF
// ===============================================================
async function clearHistory() {
  try {
    await fetch("/api/history/clear", { method: "POST" });

    histChips.innerHTML = '<span class="hist-empty">Belum ada deteksi...</span>';
    histBadge.textContent = "0";
    hideBufferProgress();

    let modeAktifSaatIni = currentMode === "huruf" ? "h" : currentMode === "angka" ? "a" : "k";
    globalHistoryList = globalHistoryList.filter(item => item.mode !== modeAktifSaatIni);

    globalHistBadge.textContent = globalHistoryList.length;
    globalHistChips.innerHTML = "";

    if (globalHistoryList.length === 0) {
      globalHistChips.innerHTML = '<span class="hist-empty">Belum ada riwayat gabungan...</span>';
    } else {
      globalHistoryList.forEach(item => {
        const chip = document.createElement("span");
        chip.className = `hist-chip global-chip-${item.mode}`;
        chip.textContent = item.teks;
        globalHistChips.appendChild(chip);
      });
      globalHistChips.scrollLeft = globalHistChips.scrollWidth;
    }

  } catch (error) {
    console.error("Gagal menghapus history:", error);
    alert("Gagal menghapus history.");
  }
}

// ===============================================================
// EVENT LISTENERS utama
// ===============================================================
if (btnStart) btnStart.addEventListener("click", startCamera);
if (btnStop)  btnStop.addEventListener("click", stopCamera);
if (btnClear) btnClear.addEventListener("click", clearHistory);

document.querySelectorAll(".mode-btn").forEach(btn => {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
});

// Smooth scroll & Otomatis Tutup Menu Hamburger Bootstrap di HP jika diklik
document.querySelectorAll('.custom-nav-link, a[href^="#"]').forEach(a => {
  a.addEventListener("click", e => {
    e.preventDefault();
    const targetId = a.getAttribute("href");
    const target = document.querySelector(targetId);
    if (target) {
      const targetPosition = target.getBoundingClientRect().top + window.scrollY - 80;
      window.scrollTo({ top: targetPosition, behavior: "smooth" });
    }

    const navbarCollapse = document.getElementById("navbarNav");
    if (navbarCollapse && navbarCollapse.classList.contains("show")) {
      const bsCollapse = bootstrap.Collapse.getInstance(navbarCollapse);
      if (bsCollapse) bsCollapse.hide();
    }
  });
});

if (btnGlobalClear) {
  btnGlobalClear.addEventListener("click", () => {
    globalHistoryList = [];
    globalHistBadge.textContent = "0";
    globalHistChips.innerHTML = '<span class="hist-empty">Belum ada riwayat gabungan...</span>';
  });
}

// ===============================================================
// DATA REFERENSI / PANDUAN GESTUR
// ===============================================================
const REF_DATA = {
  huruf: [
    { label: "A", tip: "Buat kepalan penuh, ibu jari menempel di sisi telunjuk." },
    { label: "B", tip: "Rapatkan keempat jari, ibu jari dilipat ke telapak." },
    { label: "C", tip: "Buka tangan setengah seperti memegang gelas." },
    { label: "D", tip: "Telunjuk tegak lurus, jari-jari lain menyentuh ibu jari." },
    { label: "E", tip: "Tekuk semua jari ke arah telapak, ibu jari ke dalam." },
    { label: "F", tip: "Ibu jari & telunjuk menyentuh, tiga jari lainnya tegak." },
    { label: "G", tip: "Seperti pistol mengarah ke samping." },
    { label: "H", tip: "Dua jari rapat mengarah horizontal." },
    { label: "I", tip: "Hanya kelingking yang tegak." },
    { label: "J", tip: "Mulai dari atas, lengkungkan ke bawah dan ke kiri." },
    { label: "K", tip: "Ibu jari di antara telunjuk dan jari tengah." },
    { label: "L", tip: "Ibu jari horizontal, telunjuk vertikal." },
    { label: "M", tip: "Tiga jari (telunjuk-tengah-manis) tekuk di atas ibu jari." },
    { label: "N", tip: "Telunjuk dan jari tengah tekuk di atas ibu jari." },
    { label: "O", tip: "Rapatkan ujung semua jari ke ibu jari membentuk O." },
    { label: "P", tip: "Posisi K diputar ke bawah." },
    { label: "Q", tip: "Posisi G diputar ke bawah." },
    { label: "R", tip: "Silangkan jari tengah di atas telunjuk." },
    { label: "S", tip: "Kepalan penuh, ibu jari menutupi keempat jari." },
    { label: "T", tip: "Ibu jari keluar di antara telunjuk dan jari tengah." },
    { label: "U", tip: "Dua jari rapat berdiri tegak." },
    { label: "V", tip: "Dua jari terbuka membentuk tanda V." },
    { label: "W", tip: "Telunjuk, jari tengah, dan jari manis terbuka." },
    { label: "X", tip: "Hanya telunjuk yang sedikit ditekuk." },
    { label: "Y", tip: "Hanya ibu jari dan kelingking yang terbuka (shaka)." },
    { label: "Z", tip: "Gerakkan telunjuk dari kiri-atas ke kanan, lalu diagonal ke bawah-kiri, lalu kanan." },
  ],
  angka: [
    { label: "0", tip: "Sama seperti huruf O." },
    { label: "1", tip: "Kepalan, hanya telunjuk yang tegak." },
    { label: "2", tip: "Dua jari tegak, jari lain mengepal." },
    { label: "3", tip: "Tiga jari terbuka — ibu jari, telunjuk, jari tengah." },
    { label: "4", tip: "Keempat jari tegak, ibu jari dilipat ke telapak." },
    { label: "5", tip: "Buka kelima jari selebar mungkin." },
    { label: "6", tip: "Ibu jari menyentuh kelingking, tiga jari tegak." },
    { label: "7", tip: "Ibu jari menyentuh jari manis, tiga jari tegak." },
    { label: "8", tip: "Ibu jari menyentuh jari tengah." },
    { label: "9", tip: "Ibu jari menyentuh telunjuk seperti huruf F." },
  ],
  kata: [
    { label: "berasal" },
    { label: "berpikir" },
    { label: "berteman" },
    { label: "maaf" },
    { label: "makan" },
    { label: "mandi" },
    { label: "nama" },
    { label: "perkenalkan" },
    { label: "saya" },
    { label: "tidur" },
    { label: "tolong" },
    { label: "tunggu" },
    { label: "umur" },
    { label: "sementara" },
    { label: "santai aja" },
    { label: "bareng" },
  ]
};

const REF_EXT = { huruf: "jpg", angka: "jpeg", kata: "jpg" };
let currentRefTab = "huruf";

// ===============================================================
// BANGUN GRID KARTU
// ===============================================================
function buildRefGrid(tab) {
  const grid = document.getElementById("refGrid");
  if (!grid) return;
  grid.innerHTML = "";

  const items = REF_DATA[tab];
  if (!items) return;

  items.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = `ref-card ${tab}`;
    card.dataset.idx = idx;
    card.dataset.tab = tab;

    const img = document.createElement("img");
    img.className = "ref-card-img";
    img.alt = item.label;
    img.loading = "lazy";

    const placeholder = document.createElement("div");
    placeholder.className = "ref-card-placeholder";
    placeholder.textContent = tab === "huruf" ? "🤟" : tab === "angka" ? "✋" : "👐";

    img.onload  = () => { placeholder.style.display = "none"; img.style.display = "block"; };
    img.onerror = () => { img.style.display = "none"; placeholder.style.display = "flex"; };
    img.style.display = "none";
    img.src = `/static/img/referensi/${tab}/${encodeURIComponent(item.label)}.${REF_EXT[tab]}`;

    const label = document.createElement("div");
    label.className = "ref-card-label";
    label.textContent = item.label.toUpperCase();

    card.appendChild(placeholder);
    card.appendChild(img);
    card.appendChild(label);

    card.addEventListener("click", () => openRefModal(tab, idx));
    grid.appendChild(card);
  });
}

// ===============================================================
// BUKA MODAL REFERENSI (PROTEKSI UNDEFINED VALUE)
// ===============================================================
function openRefModal(tab, idx) {
  const item     = REF_DATA[tab][idx];
  const backdrop = document.getElementById("refModalBackdrop");
  const modalImg = document.getElementById("refModalImg");
  const noImg    = document.getElementById("refModalNoImg");
  const pathEl   = document.getElementById("refModalPath");

  if (!backdrop || !item) return;

  const deskripsiTxt = item.desc ? item.desc : `Panduan visual gerakan untuk ${tab} ${item.label.toUpperCase()}.`;
  const tipTxt       = item.tip ? item.tip : "Pastikan tangan dan wajah terlihat utuh, pencahayaan cukup terang di depan webcam.";

  document.getElementById("refModalLabel").textContent = item.label.toUpperCase();
  document.getElementById("refModalDesc").textContent  = deskripsiTxt;
  document.getElementById("refModalTip").textContent   = tipTxt;

  modalImg.classList.add("hidden");
  noImg.classList.remove("hidden");

  const ext  = REF_EXT[tab];
  const path = `/static/img/referensi/${tab}/${encodeURIComponent(item.label)}.${ext}`;
  if (pathEl) pathEl.textContent = `static/img/referensi/${tab}/${item.label}.${ext}`;

  const testImg = new Image();
  testImg.onload = () => {
    modalImg.src = path;
    modalImg.classList.remove("hidden");
    noImg.classList.add("hidden");
  };
  testImg.onerror = () => {
    modalImg.classList.add("hidden");
    noImg.classList.remove("hidden");
  };
  testImg.src = path;

  backdrop.classList.add("open");
  document.body.style.overflow = "hidden";
}

function closeRefModal() {
  const backdrop = document.getElementById("refModalBackdrop");
  if (!backdrop) return;
  backdrop.classList.remove("open");
  document.body.style.overflow = "";
}

// ===============================================================
// EVENT LISTENERS REFERENSI
// ===============================================================
document.querySelectorAll(".ref-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".ref-tab").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentRefTab = btn.dataset.ref;
    buildRefGrid(currentRefTab);
  });
});

const refClose    = document.getElementById("refModalClose");
const refBackdrop = document.getElementById("refModalBackdrop");

if (refClose)    refClose.addEventListener("click", closeRefModal);
if (refBackdrop) refBackdrop.addEventListener("click", e => { if (e.target === refBackdrop) closeRefModal(); });

document.addEventListener("keydown", e => {
  if (e.key === "Escape") closeRefModal();
});

// ===============================================================
// HERO SLIDESHOW INITIALIZATION
// ===============================================================
const slides = document.querySelectorAll(".slide");
const dotsWrap = document.getElementById("slideDots");
let currentSlide = 0;

if (slides.length && dotsWrap) {
  dotsWrap.innerHTML = "";
  slides.forEach((_, i) => {
    const dot = document.createElement("span");
    dot.classList.add("dot");
    if (i === 0) dot.classList.add("active");
    dotsWrap.appendChild(dot);
  });

  const dots = document.querySelectorAll(".dot");

  function showSlide(index) {
    slides.forEach(slide => slide.classList.remove("active"));
    dots.forEach(dot => dot.classList.remove("active"));

    if (slides[index] && dots[index]) {
      slides[index].classList.add("active");
      dots[index].classList.add("active");
    }
  }

  setInterval(() => {
    currentSlide++;
    if (currentSlide >= slides.length) {
      currentSlide = 0;
    }
    showSlide(currentSlide);
  }, 3000);
}

// ===============================================================
// INIT RUNNING INITIALIZER
// ===============================================================
setMode("huruf");
buildRefGrid("huruf");