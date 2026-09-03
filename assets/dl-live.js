/* dl-live.js - a real neural network, trained in your browser.
   Nothing scripted: the multilayer perceptron below runs genuine forward and
   backward passes (full-batch gradient descent, real chain rule) every time
   you press train. The decision-boundary heatmap is the live model queried on
   a pixel grid. Tabular mode trains the SAME architecture on the SAME seeded
   300-customer Mango Lane sample the ensemble-methods course uses, so the
   tree-vs-net comparison is measured, not asserted. */
(function () {
  "use strict";

  function rng(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  /* ---------- a tiny real MLP: arrays, forward, backprop ---------- */
  function makeNet(sizes, seed) {
    var r = rng(seed || 7), W = [], B = [];
    for (var l = 0; l < sizes.length - 1; l++) {
      var w = [], scale = Math.sqrt(2 / sizes[l]);
      for (var i = 0; i < sizes[l + 1]; i++) {
        var row = [];
        for (var j = 0; j < sizes[l]; j++) row.push((r() * 2 - 1) * scale);
        w.push(row);
      }
      W.push(w); B.push(new Array(sizes[l + 1]).fill(0));
    }
    return { sizes: sizes, W: W, B: B };
  }
  function forward(net, x, act) {
    var as = [x], zs = [];
    for (var l = 0; l < net.W.length; l++) {
      var z = [], a = [];
      for (var i = 0; i < net.W[l].length; i++) {
        var s = net.B[l][i];
        for (var j = 0; j < x.length; j++) s += net.W[l][i][j] * x[j];
        z.push(s);
        var last = (l === net.W.length - 1);
        a.push(last ? sigmoid(s) : (act === "relu" ? Math.max(0, s) : Math.tanh(s)));
      }
      zs.push(z); as.push(a); x = a;
    }
    return { as: as, zs: zs, out: x[0] };
  }
  /* one full-batch gradient step; returns mean BCE loss */
  function trainStep(net, X, Y, lr, act) {
    var gW = net.W.map(function (w) { return w.map(function (r2) { return r2.map(function () { return 0; }); }); });
    var gB = net.B.map(function (b) { return b.map(function () { return 0; }); });
    var n = X.length, loss = 0;
    for (var s = 0; s < n; s++) {
      var f = forward(net, X[s], act), p = f.out;
      loss += -(Y[s] * Math.log(p + 1e-9) + (1 - Y[s]) * Math.log(1 - p + 1e-9));
      /* delta at output: p - y (BCE + sigmoid) */
      var delta = [p - Y[s]];
      for (var l = net.W.length - 1; l >= 0; l--) {
        var aPrev = f.as[l];
        for (var i = 0; i < net.W[l].length; i++) {
          gB[l][i] += delta[i];
          for (var j = 0; j < aPrev.length; j++) gW[l][i][j] += delta[i] * aPrev[j];
        }
        if (l > 0) {
          var nd = new Array(net.W[l][0].length).fill(0);
          for (var j2 = 0; j2 < nd.length; j2++) {
            var acc = 0;
            for (var i2 = 0; i2 < net.W[l].length; i2++) acc += net.W[l][i2][j2] * delta[i2];
            var zv = f.zs[l - 1][j2];
            var dact = (act === "relu") ? (zv > 0 ? 1 : 0) : (1 - Math.tanh(zv) * Math.tanh(zv));
            nd[j2] = acc * dact;
          }
          delta = nd;
        }
      }
    }
    for (var l2 = 0; l2 < net.W.length; l2++) {
      for (var i3 = 0; i3 < net.W[l2].length; i3++) {
        net.B[l2][i3] -= lr * gB[l2][i3] / n;
        for (var j3 = 0; j3 < net.W[l2][i3].length; j3++) net.W[l2][i3][j3] -= lr * gW[l2][i3][j3] / n;
      }
    }
    return loss / n;
  }
  function accuracy(net, X, Y, act) {
    var ok = 0;
    for (var s = 0; s < X.length; s++) if ((forward(net, X[s], act).out > 0.5 ? 1 : 0) === Y[s]) ok++;
    return Math.round(1000 * ok / X.length) / 10;
  }

  /* ---------- 2D toy datasets (deterministic) ---------- */
  function dataset(name) {
    var r = rng(name === "xor" ? 11 : name === "circle" ? 22 : name === "spiral" ? 33 : 44);
    var X = [], Y = [];
    for (var i = 0; i < 240; i++) {
      var x, y, lab;
      if (name === "xor") {
        x = r() * 2 - 1; y = r() * 2 - 1;
        lab = ((x > 0) !== (y > 0)) ? 1 : 0;
        x += (r() - 0.5) * 0.14; y += (r() - 0.5) * 0.14;
      } else if (name === "circle") {
        var a = r() * 6.283, rad = (i % 2 === 0) ? r() * 0.42 : 0.62 + r() * 0.34;
        x = Math.cos(a) * rad; y = Math.sin(a) * rad; lab = (i % 2 === 0) ? 1 : 0;
      } else if (name === "spiral") {
        var t = r() * 7.2 + 0.5, arm = i % 2;
        x = Math.cos(t + arm * Math.PI) * t / 8.5 + (r() - 0.5) * 0.06;
        y = Math.sin(t + arm * Math.PI) * t / 8.5 + (r() - 0.5) * 0.06;
        lab = arm;
      } else { /* moons */
        var th = r() * Math.PI, m = i % 2;
        x = Math.cos(th) * 0.7 * (m ? 1 : -1) + (m ? 0.35 : -0.35) + (r() - 0.5) * 0.12;
        y = (m ? -1 : 1) * Math.sin(th) * 0.7 - (m ? -0.18 : 0.18) + (r() - 0.5) * 0.12;
        lab = m;
      }
      X.push([x, y]); Y.push(lab);
    }
    return { X: X.slice(0, 180), Y: Y.slice(0, 180), Xt: X.slice(180), Yt: Y.slice(180) };
  }

  /* ---------- Mango Lane sample: byte-identical generator to ensemble-live.js ---------- */
  var R2 = rng(20260903);
  var MX = [], MY = [];
  for (var mi = 0; mi < 300; mi++) {
    var days = Math.round(2 + R2() * 88);
    var orders = Math.round(R2() * 9);
    var aov = Math.round(30 + R2() * 90 + (R2() < 0.15 ? 60 : 0));
    var tickets = Math.round(R2() * 4.4);
    var disc = Math.round(R2() * 100) / 100;
    var sess = Math.round(R2() * 14);
    var z = 1.7 * (days / 45 - 1)
      - 1.1 * Math.log(1 + orders) / Math.log(7)
      + 0.9 * (tickets >= 3 ? 1 : 0)
      + 0.7 * (disc > 0.55 ? 1 : 0)
      + 1.2 * ((days > 30 && sess < 4) ? 1 : 0)
      - 0.4;
    var yv = (R2() < sigmoid(1.4 * z)) ? 1 : 0;
    if (R2() < 0.10) yv = 1 - yv;
    MX.push([days, orders, aov, tickets, disc, sess]);
    MY.push(yv);
  }
  /* standardize features on the training split */
  function standardized() {
    var mu = [0,0,0,0,0,0], sd = [0,0,0,0,0,0];
    for (var f = 0; f < 6; f++) {
      for (var i = 0; i < 200; i++) mu[f] += MX[i][f];
      mu[f] /= 200;
      for (var i2 = 0; i2 < 200; i2++) sd[f] += Math.pow(MX[i2][f] - mu[f], 2);
      sd[f] = Math.sqrt(sd[f] / 200) || 1;
    }
    return MX.map(function (row) { return row.map(function (v, f2) { return (v - mu[f2]) / sd[f2]; }); });
  }
  function tabularRun(sizes, epochs, lr, seed) {
    var Xs = standardized();
    var net = makeNet([6].concat(sizes).concat([1]), seed || 7);
    for (var e = 0; e < epochs; e++) trainStep(net, Xs.slice(0, 200), MY.slice(0, 200), lr, "tanh");
    var okT = 0, okH = 0;
    for (var i = 0; i < 200; i++) if ((forward(net, Xs[i], "tanh").out > 0.5 ? 1 : 0) === MY[i]) okT++;
    for (var j = 200; j < 300; j++) if ((forward(net, Xs[j], "tanh").out > 0.5 ? 1 : 0) === MY[j]) okH++;
    return { train: Math.round(1000 * okT / 200) / 10, holdout: Math.round(1000 * okH / 100) / 10 };
  }

  window.DLLIVE = { makeNet: makeNet, forward: forward, trainStep: trainStep, accuracy: accuracy,
    dataset: dataset, tabularRun: tabularRun };

  /* ---------- UI: boundary mode ---------- */
  var mountB = document.getElementById("dl-boundary");
  if (mountB) {
    var box = document.createElement("div");
    box.className = "dl-wrap";
    box.innerHTML =
      '<div class="dl-honesty">This is a real network: every press of train runs genuine forward and backward passes in your browser, and the heatmap is the live model queried on a pixel grid. Nothing is pre-rendered.</div>' +
      '<div class="dl-ctrl">' +
      '<label>data <select id="dl-data"><option>moons</option><option>xor</option><option>circle</option><option selected>spiral</option></select></label>' +
      '<label>hidden layers <select id="dl-layers"><option>0</option><option>1</option><option selected>2</option></select></label>' +
      '<label>width <select id="dl-width"><option>4</option><option selected>8</option><option>16</option></select></label>' +
      '<label>rate <select id="dl-lr"><option>0.03</option><option selected>0.3</option><option>1.5</option></select></label>' +
      '<button class="btn primary" id="dl-train" type="button">Train 200 epochs</button>' +
      '<button class="btn" id="dl-reset" type="button">Reset</button>' +
      '<span id="dl-stat" class="dl-stat">untrained</span></div>' +
      '<div class="dl-canvasrow"><canvas id="dl-canvas" width="360" height="360"></canvas>' +
      '<div id="dl-read" class="dl-read"></div></div>';
    mountB.appendChild(box);

    var net = null, act = "tanh", epochs = 0, D = dataset("spiral");
    function arch() {
      var L = parseInt(document.getElementById("dl-layers").value, 10);
      var Wd = parseInt(document.getElementById("dl-width").value, 10);
      var sizes = [2];
      for (var i = 0; i < L; i++) sizes.push(Wd);
      sizes.push(1);
      return sizes;
    }
    function reset() {
      D = dataset(document.getElementById("dl-data").value);
      net = makeNet(arch(), 7); epochs = 0; draw();
    }
    function draw() {
      var cv = document.getElementById("dl-canvas"), ctx = cv.getContext("2d");
      var img = ctx.createImageData(90, 90);
      for (var py = 0; py < 90; py++) for (var px = 0; px < 90; px++) {
        var x = px / 45 - 1, y = 1 - py / 45;
        var p = forward(net, [x, y], act).out;
        var idx = (py * 90 + px) * 4;
        /* cyan for class 1, amber for class 0, honest confidence in alpha */
        if (p > 0.5) { img.data[idx] = 14; img.data[idx+1] = 116; img.data[idx+2] = 144; img.data[idx+3] = Math.round(60 + 150 * (p - 0.5) * 2); }
        else { img.data[idx] = 245; img.data[idx+1] = 158; img.data[idx+2] = 11; img.data[idx+3] = Math.round(60 + 150 * (0.5 - p) * 2); }
      }
      var off = document.createElement("canvas"); off.width = 90; off.height = 90;
      off.getContext("2d").putImageData(img, 0, 0);
      ctx.clearRect(0, 0, 360, 360);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, 360, 360);
      for (var s = 0; s < D.X.length; s++) {
        ctx.beginPath();
        ctx.arc((D.X[s][0] + 1) * 180, (1 - D.X[s][1]) * 180, 3.2, 0, 6.283);
        ctx.fillStyle = D.Y[s] ? "#155E75" : "#B45309";
        ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 0.8; ctx.stroke();
      }
      var trA = accuracy(net, D.X, D.Y, act), teA = accuracy(net, D.Xt, D.Yt, act);
      document.getElementById("dl-stat").textContent = epochs + " epochs";
      document.getElementById("dl-read").innerHTML =
        '<div class="dl-num"><b>' + trA.toFixed(1) + '%</b><span>train (180 pts)</span></div>' +
        '<div class="dl-num"><b>' + teA.toFixed(1) + '%</b><span>test (60 pts)</span></div>' +
        '<div class="dl-note">' + (arch().length === 2 ?
          "no hidden layer: this model IS a straight line, whatever the data needs" :
          arch().length - 2 + " hidden layer(s) of " + arch()[1]) + '</div>';
    }
    document.getElementById("dl-train").addEventListener("click", function () {
      if (!net) reset();
      var lr = parseFloat(document.getElementById("dl-lr").value);
      var btn = this; btn.disabled = true;
      var done = 0;
      (function chunk() {
        for (var k = 0; k < 25; k++) trainStep(net, D.X, D.Y, lr, act);
        done += 25; epochs += 25; draw();
        if (done < 200) setTimeout(chunk, 0); else btn.disabled = false;
      })();
    });
    document.getElementById("dl-reset").addEventListener("click", reset);
    ["dl-data", "dl-layers", "dl-width"].forEach(function (id) {
      document.getElementById(id).addEventListener("change", reset);
    });
    reset();
  }

  /* ---------- UI: tabular showdown mode ---------- */
  var mountT = document.getElementById("dl-tabular");
  if (mountT) {
    var rows = [
      { name: "Logistic regression (no hidden layer)", run: function () { return tabularRun([], 400, 0.5); } },
      { name: "MLP · 1 hidden layer of 8, tanh", run: function () { return tabularRun([8], 600, 0.5); } },
      { name: "MLP · 2 hidden layers of 16, tanh", run: function () { return tabularRun([16, 16], 600, 0.5); } },
      { name: "MLP · 3x32, overparameterized, run long", run: function () { return tabularRun([32, 32, 32], 1500, 0.5); } }
    ];
    var tb = document.createElement("div");
    tb.className = "dl-wrap";
    tb.innerHTML =
      '<div class="dl-honesty">Same 300 customers, same seed, same 200/100 split as the ensemble-methods course - the generator is copied byte for byte, so the comparison is real. The tuned gradient booster scored 73.0 holdout on this exact data. Each row trains its network live when you press run. One honest wrinkle: the deep 3x32 run is chaotic enough that different browsers\' floating-point libraries land it between 68 and 70 - determinism has layers.</div>' +
      '<div class="dl-board">' + rows.map(function (r3, i5) {
        return '<div class="dl-row" data-i="' + i5 + '"><button class="btn" type="button">Run</button>' +
          '<span class="dl-name">' + r3.name + '</span><span class="dl-res">-</span></div>';
      }).join("") +
      '<div class="dl-row dl-bench"><span class="dl-mark">◆</span><span class="dl-name">Tuned gradient booster (ensemble course, same data)</span><span class="dl-res">train <b>100.0%</b> · holdout <b>73.0%</b></span></div>' +
      '</div>';
    mountT.appendChild(tb);
    tb.addEventListener("click", function (e) {
      var rowEl = e.target.closest(".dl-row");
      if (!rowEl || e.target.tagName !== "BUTTON") return;
      var r4 = rows[parseInt(rowEl.getAttribute("data-i"), 10)];
      e.target.textContent = "...";
      setTimeout(function () {
        var res = r4.run();
        rowEl.querySelector(".dl-res").innerHTML =
          "train <b>" + res.train.toFixed(1) + "%</b> · holdout <b>" + res.holdout.toFixed(1) + "%</b>";
        e.target.textContent = "Run";
      }, 30);
    });
  }
})();
