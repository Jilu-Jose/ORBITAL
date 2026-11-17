/* ORBITAL frontend. Vanilla JS, no build step. Talks to /api/* (see README section 6). */
(() => {
  "use strict";

  // Set this to your repository URL.
  const REPO_URL = "https://github.com/";

  const API = location.protocol === "file:" ? "http://localhost:8000" : "";
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const nf = (n, d = 0) => Number(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const TAU = Math.PI * 2;

  const OFFLINE_MSG = "BACKEND OFFLINE. Start the API from the project root:  uvicorn backend.app.main:app --port 8000";
  function showOffline(slot) {
    const el = $(`[data-offline="${slot}"]`);
    if (el) el.innerHTML = `<div class="offline" role="alert">${OFFLINE_MSG}</div>`;
  }
  async function api(path, opts) {
    const r = await fetch(API + path, opts);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }

  /* ---------------------------------------------------------------- misc UI */
  $("#repoLink").href = REPO_URL;

  function tickClock() {
    const d = new Date(), p = (n) => String(n).padStart(2, "0");
    $("#clock").textContent = `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())} UTC`;
  }
  tickClock();
  setInterval(tickClock, 1000);

  // scroll reveal (fade + 20px slide)
  const revealIO = "IntersectionObserver" in window
    ? new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); revealIO.unobserve(e.target); } }), { threshold: 0.08 })
    : null;
  $$(".reveal").forEach((el) => (revealIO ? revealIO.observe(el) : el.classList.add("in")));

  // active nav link
  const sectionIds = ["top", "orbit-map", "clusters", "risk", "performance", "scalability", "pipeline"];
  function updateNav() {
    const hdr = parseInt(getComputedStyle(document.documentElement).getPropertyValue("--hdr")) || 56;
    let current = "top";
    for (const id of sectionIds) {
      const el = document.getElementById(id);
      if (el && el.getBoundingClientRect().top <= hdr + 80) current = id;
    }
    $$(".navlink").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + current));
  }
  addEventListener("scroll", updateNav, { passive: true });
  updateNav();

  // starfield in hero (static, seeded)
  (function stars() {
    const cv = $("#stars"), ctx = cv.getContext("2d");
    function seeded(a) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
    function paint() {
      const r = cv.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
      cv.width = r.width * dpr; cv.height = r.height * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const rnd = seeded(11);
      for (let i = 0; i < 170; i++) {
        ctx.fillStyle = `rgba(255,255,255,${0.15 + rnd() * 0.6})`;
        const s = rnd() < 0.1 ? 1.6 : 1;
        ctx.fillRect(rnd() * r.width, rnd() * r.height * 0.8, s, s);
      }
    }
    paint();
    addEventListener("resize", paint);
  })();

  /* ---------------------------------------------------------------- stats */
  function countUp(el, target, decimals) {
    const dur = REDUCED ? 0 : 900, t0 = performance.now();
    (function step(now) {
      const p = dur ? Math.min(1, (now - t0) / dur) : 1, e = 1 - Math.pow(1 - p, 3);
      el.textContent = nf(target * e, decimals);
      if (p < 1) requestAnimationFrame(step);
    })(t0);
  }
  function initStats(stats) {
    $("#tkCatalog").textContent = `${nf(stats.tracked)} OBJECTS`;
    const els = $$("[data-stat]");
    const run = () => els.forEach((el) => countUp(el, Number(stats[el.dataset.stat]) || 0, Number(el.dataset.decimals) || 0));
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { run(); io.disconnect(); } }, { threshold: 0.3 });
      io.observe($("#stats"));
    } else run();
  }

  /* ---------------------------------------------------------------- orbit globe */
  const radOf = (alt) => 1 + 0.55 * Math.log(1 + alt / 800); // compressed altitude scale (Earth radius = 1)

  function initGlobe(objs) {
    const cv = $("#globe"), box = $("#globeBox"), tip = $("#tip"), ctx = cv.getContext("2d");
    const STYLE = { "PAYLOAD": "#ffffff", "ROCKET BODY": "#c6c6c6", "DEBRIS": "#8e9192" };
    const sats = objs.map((o) => {
      const inc = (o.inc * Math.PI) / 180;
      return { o, rad: radOf(o.alt), ci: Math.cos(inc), si: Math.sin(inc), co: Math.cos(o.raan), so: Math.sin(o.raan), ph: o.phase, spd: o.spd, vis: true, shown: false, sx: 0, sy: 0 };
    });
    const f = { types: new Set(["PAYLOAD", "ROCKET BODY", "DEBRIS"]), maxAlt: Infinity, regime: "ALL" };
    let W = 0, H = 0, R = 100, yaw = 0.6, tilt = 0.45, T = 0, paused = false, dragging = false, hover = null;

    const counts = { PAYLOAD: 0, "ROCKET BODY": 0, DEBRIS: 0 };
    objs.forEach((o) => { if (o.type in counts) counts[o.type]++; });
    $$("[data-count]").forEach((el) => (el.textContent = nf(counts[el.dataset.count] || 0)));

    function applyFilters() {
      let n = 0;
      for (const s of sats) {
        s.vis = f.types.has(s.o.type) && s.o.alt <= f.maxAlt && (f.regime === "ALL" || s.o.regime === f.regime);
        if (s.vis) n++;
      }
      $("#projCount").textContent = `${nf(n)} nodes`;
      if (REDUCED) draw();
    }

    function resize() {
      const r = box.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
      W = r.width; H = r.height;
      cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      R = Math.min(W, H) * 0.15;
      draw();
    }

    // scene coords (unit = Earth radius, z = north) -> screen. depth > 0 means behind the Earth.
    function proj(x, y, z) {
      const cyw = Math.cos(yaw), syw = Math.sin(yaw), ct = Math.cos(tilt), st = Math.sin(tilt);
      const X = x * cyw - y * syw, Y = x * syw + y * cyw;
      return [W / 2 + X * R, H / 2 - (Y * st + z * ct) * R, Y * ct - z * st];
    }

    function draw() {
      if (!W) return;
      ctx.clearRect(0, 0, W, H);
      const cx = W / 2, cy = H / 2;

      // Background gridlines (light colored, clipped so they never overlap the central sphere)
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, W, H);
      ctx.arc(cx, cy, R + 2, 0, TAU);
      ctx.clip("evenodd");

      const grid = new Path2D();
      const step = 48;
      const startX = ((cx % step) + step) % step;
      const startY = ((cy % step) + step) % step;
      for (let x = startX; x <= W; x += step) {
        const px = Math.round(x) + 0.5;
        grid.moveTo(px, 0);
        grid.lineTo(px, H);
      }
      for (let y = startY; y <= H; y += step) {
        const py = Math.round(y) + 0.5;
        grid.moveTo(0, py);
        grid.lineTo(W, py);
      }
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.20)";
      ctx.stroke(grid);
      ctx.restore();

      // Earth disc + wireframe
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.fillStyle = "#131313"; ctx.fill();
      const front = new Path2D(), back = new Path2D();
      const ring = (fn, s) => {
        let prev = null;
        for (let i = 0; i <= 72; i++) {
          const p = fn((i / 72) * TAU), q = proj(p[0] * s, p[1] * s, p[2] * s);
          if (prev) { const pa = (prev[2] + q[2]) / 2 <= 0 ? front : back; pa.moveTo(prev[0], prev[1]); pa.lineTo(q[0], q[1]); }
          prev = q;
        }
      };
      for (const la of [-1.0472, -0.5236, 0, 0.5236, 1.0472]) ring((t) => [Math.cos(la) * Math.cos(t), Math.cos(la) * Math.sin(t), Math.sin(la)], 1);
      for (let k = 0; k < 6; k++) { const lo = (k * Math.PI) / 6; ring((t) => [Math.cos(t) * Math.cos(lo), Math.cos(t) * Math.sin(lo), Math.sin(t)], 1); }
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(255,255,255,.10)"; ctx.stroke(back);
      ctx.strokeStyle = "rgba(255,255,255,.42)"; ctx.stroke(front);
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.strokeStyle = "rgba(255,255,255,.7)"; ctx.stroke();

      // reference shells (equatorial) with labels
      ctx.setLineDash([2, 5]); ctx.strokeStyle = "rgba(198,198,198,.28)";
      ctx.font = '500 9px "JetBrains Mono",monospace'; ctx.fillStyle = "#8e9192"; ctx.textAlign = "left";
      for (const [lab, alt, ang] of [["LEO 550 km", 550, 2.3], ["MEO 20,200 km", 20200, 0.9], ["GEO 35,786 km", 35786, -0.35]]) {
        const s = radOf(alt), pa = new Path2D();
        let prev = null;
        for (let i = 0; i <= 96; i++) { const t = (i / 96) * TAU, q = proj(Math.cos(t) * s, Math.sin(t) * s, 0); if (prev) { pa.moveTo(prev[0], prev[1]); pa.lineTo(q[0], q[1]); } prev = q; }
        ctx.stroke(pa);
        const lp = proj(Math.cos(ang) * s, Math.sin(ang) * s, 0); ctx.fillText(lab, lp[0] + 5, lp[1] - 4);
      }
      ctx.setLineDash([]);

      // objects
      const pPay = new Path2D(), pDeb = new Path2D(), pRb = new Path2D();
      for (const s of sats) {
        s.shown = false;
        if (!s.vis) continue;
        const u = s.ph + s.spd * T, x0 = Math.cos(u), y0 = Math.sin(u) * s.ci, z0 = Math.sin(u) * s.si;
        const p = proj((x0 * s.co - y0 * s.so) * s.rad, (x0 * s.so + y0 * s.co) * s.rad, z0 * s.rad);
        if (p[2] > 0) { const dx = p[0] - cx, dy = p[1] - cy; if (dx * dx + dy * dy < R * R) continue; } // hidden by Earth
        s.sx = p[0]; s.sy = p[1]; s.shown = true;
        const t = s.o.type;
        if (t === "PAYLOAD") { pPay.moveTo(p[0] + 2.2, p[1]); pPay.arc(p[0], p[1], 2.2, 0, TAU); }
        else if (t === "ROCKET BODY") pRb.rect(p[0] - 2, p[1] - 2, 4, 4);
        else { pDeb.moveTo(p[0] + 1.2, p[1]); pDeb.arc(p[0], p[1], 1.2, 0, TAU); }
      }
      ctx.fillStyle = STYLE.DEBRIS; ctx.fill(pDeb);
      ctx.fillStyle = STYLE["ROCKET BODY"]; ctx.fill(pRb);
      ctx.fillStyle = STYLE.PAYLOAD; ctx.fill(pPay);

      if (hover && hover.shown) {
        ctx.beginPath(); ctx.arc(hover.sx, hover.sy, 7, 0, TAU); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.stroke();
      }
    }

    // animation loop (pauses when off-screen or tab hidden)
    let onScreen = true, rafId = 0, last = performance.now();
    function frame(now) {
      rafId = 0;
      if (!onScreen || document.hidden) return;
      const dt = Math.min(0.05, (now - last) / 1000); last = now;
      if (!REDUCED) { if (!paused && !dragging) yaw += 0.0012 * dt * 60; T += dt * 60; }
      draw();
      if (!REDUCED) rafId = requestAnimationFrame(frame);
    }
    function kick() { if (!rafId) { last = performance.now(); rafId = requestAnimationFrame(frame); } }
    if ("IntersectionObserver" in window) new IntersectionObserver((es) => { onScreen = es[0].isIntersecting; if (onScreen) kick(); }).observe(box);
    document.addEventListener("visibilitychange", kick);
    new ResizeObserver(resize).observe(box);

    // drag to rotate
    let lx = 0, ly = 0;
    cv.addEventListener("pointerdown", (e) => { dragging = true; lx = e.clientX; ly = e.clientY; cv.setPointerCapture(e.pointerId); cv.style.cursor = "grabbing"; });
    cv.addEventListener("pointerup", (e) => { dragging = false; cv.style.cursor = "grab"; try { cv.releasePointerCapture(e.pointerId); } catch (_) {} });
    cv.addEventListener("pointermove", (e) => {
      const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
      if (dragging) {
        yaw += (e.clientX - lx) * 0.006; tilt = clamp(tilt + (e.clientY - ly) * 0.006, -1.4, 1.4);
        lx = e.clientX; ly = e.clientY; hover = null; tip.style.display = "none";
        if (REDUCED) draw();
        return;
      }
      let best = null, bd = 100; // 10px radius
      for (const s of sats) { if (!s.shown) continue; const dx = s.sx - px, dy = s.sy - py, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = s; } }
      hover = best;
      if (best) {
        const o = best.o;
        tip.innerHTML = `NORAD ${o.norad}<br>${String(o.name).replace(/</g, "&lt;")}<br>ALT ${nf(o.alt)} km · INC ${nf(o.inc, 1)}°`;
        tip.style.display = "block";
        tip.style.left = clamp(px + 14, 4, W - 190) + "px"; tip.style.top = clamp(py + 14, 4, H - 60) + "px";
      } else tip.style.display = "none";
      if (REDUCED) draw();
    });
    cv.addEventListener("pointerleave", () => { hover = null; tip.style.display = "none"; });

    // filters
    $$("input[data-type]").forEach((cb) => cb.addEventListener("change", () => { cb.checked ? f.types.add(cb.dataset.type) : f.types.delete(cb.dataset.type); applyFilters(); }));
    const alt = $("#altCut");
    function setAlt() {
      const v = Number(alt.value);
      alt.style.setProperty("--p", v + "%");
      f.maxAlt = v >= 100 ? Infinity : 200 * Math.pow(200, v / 100);
      $("#altCutVal").textContent = (f.maxAlt === Infinity ? 40000 : Math.round(f.maxAlt / 10) * 10).toLocaleString("en-US") + " km";
      applyFilters();
    }
    alt.addEventListener("input", setAlt);
    $$("#regimeBtns .seg").forEach((b) => b.addEventListener("click", () => {
      f.regime = b.dataset.regime;
      $$("#regimeBtns .seg").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      applyFilters();
    }));
    $("#spinBtn").addEventListener("click", (e) => {
      paused = !paused; e.currentTarget.textContent = paused ? "Resume spin" : "Pause spin"; e.currentTarget.setAttribute("aria-pressed", String(paused));
    });

    setAlt();
    resize();
    kick();
  }

  /* ---------------------------------------------------------------- regime scatter */
  function shapeOf(o) {
    if (o.regime === "HEO") return o.inc < 35 && o.ecc > 0.3 ? "GTO" : "HEO";
    return o.regime;
  }
  const LEGEND = [["LEO", "LEO"], ["SSO", "SSO"], ["MEO", "MEO"], ["GEO", "GEO"], ["GTO", "GTO"], ["HEO", "HEO / MOLNIYA"]];
  function legendSvg(k) {
    const s = { LEO: '<circle cx="7" cy="7" r="3" fill="#fff"/>', SSO: '<circle cx="7" cy="7" r="3.5" fill="none" stroke="#c6c6c6"/>',
      MEO: '<rect x="3.5" y="3.5" width="7" height="7" fill="#8e9192"/>', GEO: '<path d="M7 2 12 7 7 12 2 7Z" fill="none" stroke="#fff"/>',
      GTO: '<path d="M7 2v10M2 7h10" stroke="#c6c6c6"/>', HEO: '<path d="M7 2 12 12H2Z" fill="none" stroke="#fff"/>' }[k];
    return `<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">${s}</svg>`;
  }

  function initScatter(objs) {
    const cv = $("#scatter"), ctx = cv.getContext("2d");
    const pts = objs.map((o) => ({ o, k: shapeOf(o) }));
    $("#scatterLegend").innerHTML = LEGEND.map(([k, t]) => `<span class="flex items-center gap-2">${legendSvg(k)}${t}</span>`).join("");
    const M = { l: 62, r: 18, t: 16, b: 44 };
    let W = 0, H = 0, progress = 0;
    const LA = Math.log10(200), LB = Math.log10(40000);
    const X = (inc) => M.l + (clamp(inc, 0, 120) / 120) * (W - M.l - M.r);
    const Y = (alt) => M.t + (1 - (Math.log10(clamp(alt, 200, 40000)) - LA) / (LB - LA)) * (H - M.t - M.b);

    function draw() {
      if (!W) return;
      ctx.clearRect(0, 0, W, H);
      ctx.font = '400 10px "JetBrains Mono",monospace'; ctx.fillStyle = "#8e9192"; ctx.strokeStyle = "#2a2a2a"; ctx.lineWidth = 1;
      ctx.textAlign = "center";
      for (let v = 0; v <= 120; v += 20) { const x = X(v); ctx.beginPath(); ctx.moveTo(x, M.t); ctx.lineTo(x, H - M.b); ctx.stroke(); ctx.fillText(v + "°", x, H - M.b + 16); }
      ctx.textAlign = "right";
      for (const a of [200, 500, 1000, 2000, 5000, 10000, 20000, 40000]) { const y = Y(a); ctx.beginPath(); ctx.moveTo(M.l, y); ctx.lineTo(W - M.r, y); ctx.stroke(); ctx.fillText(nf(a), M.l - 8, y + 3); }
      ctx.textAlign = "center"; ctx.fillStyle = "#8e9192";
      ctx.fillText("INCLINATION (DEG)", (M.l + W - M.r) / 2, H - 6);
      ctx.save(); ctx.translate(12, (M.t + H - M.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("SEMI-MAJOR ALTITUDE (KM, LOG)", 0, 0); ctx.restore();
      // GEO reference
      ctx.setLineDash([3, 4]); ctx.strokeStyle = "#444748"; ctx.beginPath(); ctx.moveTo(M.l, Y(35786)); ctx.lineTo(W - M.r, Y(35786)); ctx.stroke(); ctx.setLineDash([]);
      ctx.textAlign = "right"; ctx.fillStyle = "#8e9192"; ctx.fillText("GEO 35,786 km", W - M.r - 4, Y(35786) - 5);

      const e = 1 - Math.pow(1 - progress, 3), n = Math.floor(pts.length * e);
      const P = { LEO: new Path2D(), SSO: new Path2D(), MEO: new Path2D(), GEO: new Path2D(), GTO: new Path2D(), HEO: new Path2D() };
      for (let i = 0; i < n; i++) {
        const { o, k } = pts[i], x = X(o.inc), y = Y(o.alt), p = P[k];
        if (!p) continue;
        if (k === "LEO") { p.moveTo(x + 2.2, y); p.arc(x, y, 2.2, 0, TAU); }
        else if (k === "SSO") { p.moveTo(x + 3, y); p.arc(x, y, 3, 0, TAU); }
        else if (k === "MEO") p.rect(x - 2.5, y - 2.5, 5, 5);
        else if (k === "GEO") { p.moveTo(x, y - 4); p.lineTo(x + 4, y); p.lineTo(x, y + 4); p.lineTo(x - 4, y); p.closePath(); }
        else if (k === "GTO") { p.moveTo(x - 4, y); p.lineTo(x + 4, y); p.moveTo(x, y - 4); p.lineTo(x, y + 4); }
        else { p.moveTo(x, y - 4); p.lineTo(x + 4, y + 3.5); p.lineTo(x - 4, y + 3.5); p.closePath(); }
      }
      ctx.lineWidth = 1;
      ctx.fillStyle = "rgba(255,255,255,.72)"; ctx.fill(P.LEO);
      ctx.fillStyle = "#8e9192"; ctx.fill(P.MEO);
      ctx.strokeStyle = "#c6c6c6"; ctx.stroke(P.SSO); ctx.stroke(P.GTO);
      ctx.strokeStyle = "#fff"; ctx.stroke(P.GEO); ctx.stroke(P.HEO);
    }
    function resize() {
      const r = cv.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
      W = r.width; H = r.height; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
    }
    new ResizeObserver(resize).observe(cv);
    resize();

    function play() {
      if (REDUCED) { progress = 1; draw(); return; }
      const t0 = performance.now();
      (function step(now) { progress = Math.min(1, (now - t0) / 800); draw(); if (progress < 1) requestAnimationFrame(step); })(t0);
    }
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((es) => { if (es[0].isIntersecting) { play(); io.disconnect(); } }, { threshold: 0.2 });
      io.observe(cv);
    } else { progress = 1; draw(); }

    // summary tiles computed from the data
    const byReg = {};
    objs.forEach((o) => (byReg[o.regime] = (byReg[o.regime] || 0) + 1));
    const bins = {};
    objs.forEach((o) => { const b = Math.floor(o.alt / 50) * 50; bins[b] = (bins[b] || 0) + 1; });
    const peak = Object.entries(bins).sort((a, b) => b[1] - a[1])[0] || [0, 0];
    const incs = objs.map((o) => o.inc).sort((a, b) => a - b), med = incs.length ? incs[Math.floor(incs.length / 2)] : 0;
    const top = Object.entries(byReg).sort((a, b) => b[1] - a[1])[0] || ["--", 0];
    const tile = (l, v, s) => `<div class="panel p-4"><div class="label">${l}</div><div class="mono text-white text-[20px] mt-1">${v}</div><div class="label mt-1" style="text-transform:none">${s}</div></div>`;
    $("#clusterStats").innerHTML =
      tile("Objects plotted", nf(objs.length), "Sample used for the map") +
      tile("Dominant regime", top[0], `${nf((top[1] / objs.length) * 100, 0)}% of sample`) +
      tile("Densest 50 km shell", `${nf(peak[0])} km`, `${nf(peak[1])} objects in sample`) +
      tile("Median inclination", `${nf(med, 1)}°`, "All plotted objects");
  }

  /* ---------------------------------------------------------------- predictor */
  const sciHTML = (v) => { const [m, e] = v.toExponential(2).split("e"); return `${m} x 10<sup style="font-size:.45em;vertical-align:super;letter-spacing:0">${parseInt(e, 10)}</sup>`; };

  function initPredictor() {
    const S = { miss: $("#s_miss"), vel: $("#s_vel"), inc: $("#s_inc"), bstar: $("#s_bstar") };
    const fmt = { miss: (v) => `${v.toFixed(2)} km`, vel: (v) => `${v.toFixed(1)} km/s`, inc: (v) => `${v.toFixed(1)}°`, bstar: (v) => `${v.toFixed(2)}e-4` };
    function sync(k) {
      const el = S[k], pct = ((el.value - el.min) / (el.max - el.min)) * 100;
      el.style.setProperty("--p", pct + "%");
      $("#v_" + k).textContent = fmt[k](Number(el.value));
    }
    Object.keys(S).forEach((k) => { sync(k); S[k].addEventListener("input", () => sync(k)); });

    function render(r) {
      const modelMode = r.pc === null || r.pc === undefined;
      $("#resLabel").textContent = modelMode ? "Model probability of high risk, P(Pc > 1e-4)" : "Derived probability of collision (Pc)";
      $("#resPc").innerHTML = modelMode ? `${nf(r.prob * 100, 1)}<span style="font-size:.5em">%</span>` : sciHTML(r.pc);
      $("#resThr").textContent = modelMode ? "Decision threshold: 50%" : "Threshold: 1.0e-4";
      $("#gaugeTicks").innerHTML = (modelMode ? ["0%", "25%", "50%", "75%", "100%"] : ["1e-6", "1e-5", "1e-4", "1e-3"]).map((t) => `<span>${t}</span>`).join("");
      const sev = r.high_risk ? "HIGH" : /MODERATE/.test(r.level) ? "MODERATE" : "LOW";
      const chip = $("#resChip");
      chip.className = `mono text-[11px] px-2 py-0.5 border rounded-sm sev-${sev}`;
      chip.textContent = (sev === "HIGH" ? "▲ " : "") + r.chip;
      $("#resLevel").textContent = r.level;
      $("#resMiss").textContent = `${nf(r.miss_m)} m`;
      $("#resVel").textContent = `${nf(r.rel_vel, 1)} km/s`;
      $("#resEngine").textContent = r.engine;
      $("#resAction").textContent = r.action;
      const note = $("#resNote"); note.classList.toggle("hidden", !r.note); note.textContent = r.note || "";
      $("#gauge").setAttribute("aria-valuenow", r.score);
      requestAnimationFrame(() => ($("#gaugeFill").style.width = clamp(r.score, 0, 1) * 100 + "%"));
      $("#engineChip").textContent = "ENGINE: " + r.engine;
      $("#tkEngine").textContent = r.engine; $("#ftEngine").textContent = r.engine;
    }

    async function run() {
      const btn = $("#calcBtn");
      btn.disabled = true;
      try {
        const r = await api("/api/predict", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ miss_km: +S.miss.value, rel_vel: +S.vel.value, inc_div: +S.inc.value, bstar: +S.bstar.value }),
        });
        $('[data-offline="predict"]').innerHTML = "";
        render(r);
      } catch (e) { showOffline("predict"); }
      btn.disabled = false;
    }
    $("#predictForm").addEventListener("submit", (e) => { e.preventDefault(); run(); });
    run(); // fill the card on load
  }

  /* ---------------------------------------------------------------- metrics */
  const pretty = (s) => String(s).replace(/_/g, " ").toUpperCase();

  function initMetrics(m) {
    // models table
    $("#modelTable tbody").innerHTML = m.models.map((r) => `
      <tr class="${r.selected ? "sel" : ""}"><td class="${r.selected ? "text-white font-semibold" : "text-sec"}">${r.name}</td>
      <td>${nf(r.pr_auc, 3)}</td><td>${nf(r.recall, 3)}</td><td>${nf(r.f1, 3)}</td>
      <td>${r.train_s ? nf(r.train_s, 1) + "s" : "--"}</td>
      <td>${r.selected ? '<span class="mono text-[10px] px-2 py-0.5 bg-white text-s0 rounded-sm font-semibold" style="letter-spacing:.1em">SELECTED</span>' : '<span class="label">candidate</span>'}</td></tr>`).join("");

    // importance
    const imp = (m.importance || []).slice(0, 8), sum = imp.reduce((a, b) => a + b[1], 0) || 1;
    $("#importance").innerHTML = imp.map(([n, v], i) => `
      <div><div class="flex justify-between mb-1.5"><span class="mono text-[12px] text-white">${String(i + 1).padStart(2, "0")}. ${pretty(n)}</span><span class="mono text-[12px] text-sec">${nf((v / sum) * 100, 1)}%</span></div>
      <div class="gauge" style="height:6px"><div class="bar-fill" data-w="${(v / sum) * 100}" style="background:${i === 0 ? "#fff" : "#8e9192"}"></div></div></div>`).join("");
    const grow = () => $$("#importance .bar-fill").forEach((b) => (b.style.width = b.dataset.w + "%"));
    if ("IntersectionObserver" in window) { const io = new IntersectionObserver((es) => { if (es[0].isIntersecting) { grow(); io.disconnect(); } }, { threshold: 0.3 }); io.observe($("#importance")); } else grow();

    // confusion matrix
    const c = m.confusion, tot = c.tn + c.fp + c.fn + c.tp;
    const cell = (id, label, v) => ($("#" + id).innerHTML = `<div class="label" style="color:inherit;opacity:.7">${label}</div><div class="mono text-[20px] font-medium mt-1">${nf(v)}</div>`);
    cell("cm_tn", "True neg.", c.tn); cell("cm_fp", "False pos.", c.fp); cell("cm_fn", "False neg.", c.fn); cell("cm_tp", "True pos.", c.tp);
    $("#cmTotal").textContent = `N = ${nf(tot)}`;
    const ratio = (a, b) => (b ? a / b : 0);
    const stat = (l, v) => `<div><div class="label">${l}</div><div class="mono text-white text-[16px]">${nf(v * 100, 2)}%</div></div>`;
    $("#cmStats").innerHTML = stat("Precision", ratio(c.tp, c.tp + c.fp)) + stat("Recall", ratio(c.tp, c.tp + c.fn)) + stat("Specificity", ratio(c.tn, c.tn + c.fp));
  }

  /* ---------------------------------------------------------------- scalability */
  function initScalability(sc) {
    const cores = sc.cores, sp = sc.spark_s, sk = sc.sklearn_s || [];
    const W = 640, H = 300, M = { l: 54, r: 20, t: 20, b: 42 };
    const ymax = Math.max(...sp, ...sk, 1) * 1.1, niceStep = Math.pow(10, Math.floor(Math.log10(ymax / 4))) * ([1, 2, 5, 10].find((m) => (Math.pow(10, Math.floor(Math.log10(ymax / 4))) * m) * 4 >= ymax) || 10);
    const top = Math.ceil(ymax / niceStep) * niceStep;
    const x = (i) => M.l + (cores.length > 1 ? (i / (cores.length - 1)) * (W - M.l - M.r) : (W - M.l - M.r) / 2);
    const y = (v) => M.t + (1 - v / top) * (H - M.t - M.b);
    const line = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    let g = "";
    for (let v = 0; v <= top + 1e-9; v += niceStep) g += `<line x1="${M.l}" x2="${W - M.r}" y1="${y(v)}" y2="${y(v)}" stroke="#2a2a2a"/><text x="${M.l - 8}" y="${y(v) + 3}" text-anchor="end" fill="#8e9192" font-size="10" font-family="'JetBrains Mono',monospace">${nf(v)}</text>`;
    cores.forEach((c, i) => (g += `<text x="${x(i)}" y="${H - M.b + 18}" text-anchor="middle" fill="#8e9192" font-size="10" font-family="'JetBrains Mono',monospace">${c}</text>`));
    g += `<text x="${(M.l + W - M.r) / 2}" y="${H - 6}" text-anchor="middle" fill="#8e9192" font-size="10" font-family="'JetBrains Mono',monospace">SPARK CORES</text>`;
    g += `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle" fill="#8e9192" font-size="10" font-family="'JetBrains Mono',monospace">RUNTIME (S)</text>`;
    if (sk.length) {
      g += `<path id="skLine" d="${line(sk)}" fill="none" stroke="#8e9192" stroke-width="1.5" stroke-dasharray="5 5" style="opacity:0;transition:opacity .8s"/>`;
      sk.forEach((v, i) => (g += `<rect class="skDot" x="${x(i) - 3}" y="${y(v) - 3}" width="6" height="6" fill="#0e0e0e" stroke="#8e9192" style="opacity:0;transition:opacity .8s"/>`));
    }
    g += `<path id="spLine" d="${line(sp)}" pathLength="1" fill="none" stroke="#fff" stroke-width="2" style="stroke-dasharray:1;stroke-dashoffset:1;transition:stroke-dashoffset .8s ease"/>`;
    sp.forEach((v, i) => (g += `<circle class="spDot" cx="${x(i)}" cy="${y(v)}" r="4" fill="#fff" style="opacity:0;transition:opacity .8s"/>`));
    $("#scalChart").innerHTML = `
      <div class="flex gap-6 label mb-3"><span class="flex items-center gap-2"><i class="inline-block w-5 h-0.5 bg-white"></i>Apache Spark (distributed)</span><span class="flex items-center gap-2"><i class="inline-block w-5 border-t border-dashed border-mute"></i>scikit-learn (single node)</span></div>
      <svg viewBox="0 0 ${W} ${H}" class="w-full h-auto" role="img" aria-label="Line chart of training runtime in seconds against number of Spark cores for Spark and scikit-learn">${g}</svg>`;
    const go = () => {
      const sl = $("#spLine"); sl.style.strokeDashoffset = 0;
      $$(".spDot,.skDot,#skLine", $("#scalChart")).forEach((e) => (e.style.opacity = 1));
    };
    if (REDUCED || !("IntersectionObserver" in window)) go();
    else { const io = new IntersectionObserver((es) => { if (es[0].isIntersecting) { go(); io.disconnect(); } }, { threshold: 0.3 }); io.observe($("#scalChart")); }

    const last = sp.length - 1, speed = sp[0] / sp[last], eff = cores[last] / cores[0] ? speed / (cores[last] / cores[0]) : 0;
    const cross = sk.length ? cores.find((c, i) => sp[i] < sk[i]) : undefined;
    const st = (l, v, s) => `<div><div class="label">${l}</div><div class="mono text-white text-[26px] leading-tight">${v}</div><div class="label" style="text-transform:none">${s}</div></div>`;
    $("#scalStats").innerHTML =
      st("Speedup", `${nf(speed, 1)}x`, `${cores[0]} to ${cores[last]} cores`) +
      st("Parallel efficiency", `${nf(eff * 100, 0)}%`, "Speedup / core ratio") +
      st("Beats scikit-learn at", cross !== undefined ? `${cross} cores` : "never", sk.length ? `Baseline ${nf(sk[0], 0)} s on one node` : "No baseline supplied");
  }

  /* ---------------------------------------------------------------- bootstrap */
  async function main() {
    const [st, me, ob] = await Promise.allSettled([api("/api/stats"), api("/api/metrics"), api("/api/objects")]);
    const stats = st.status === "fulfilled" ? st.value : null, metrics = me.status === "fulfilled" ? me.value : null, objs = ob.status === "fulfilled" ? ob.value : null;
    const online = !!(stats || metrics || objs);

    const demo = !!((stats && stats.demo) || (metrics && metrics.demo));
    document.documentElement.dataset.demo = String(demo);
    $("#statusText").textContent = !online ? "[SYS: BACKEND OFFLINE]" : demo ? "[SYS: NOMINAL // DEMO MODE]" : "[SYS: NOMINAL // LIVE DATA]";
    $("#tkApi").textContent = online ? "ONLINE" : "OFFLINE";
    $("#tkMode").textContent = !online ? "--" : demo ? "DEMO (PLACEHOLDER)" : "TRAINED MODEL DATA";

    if (stats) initStats(stats); else showOffline("stats");

    if (objs && objs.length) {
      try { initGlobe(objs); } catch (e) { console.error(e); }
      try { initScatter(objs); } catch (e) { console.error(e); }
    } else { showOffline("objects"); showOffline("scatter"); }

    initPredictor();

    if (metrics) {
      try { initMetrics(metrics); } catch (e) { console.error(e); }
      if (metrics.scalability && metrics.scalability.cores && metrics.scalability.cores.length) {
        try { initScalability(metrics.scalability); } catch (e) { console.error(e); }
      } else { $("#scalability").remove(); $("#navScal").remove(); $("#navScalM").remove(); }
    } else showOffline("models");
  }

  main();
})();
