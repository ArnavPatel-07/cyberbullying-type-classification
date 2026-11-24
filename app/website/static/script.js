// ---- Results data (from results/final_comparison.csv) ----
// [model, group, macro-F1 test, macro-F1 validation]; group: tf=transformer, dl=deep learning, ml=classical
const DATA = [
    ["BERTweet", "tf", 0.8917, 0.8987],
    ["RoBERTa-base", "tf", 0.8919, 0.8939],
    ["DistilBERT (deployed)", "tf", 0.8896, 0.8930],
    ["BERT-base", "tf", 0.8869, 0.8969],
    ["LinearSVM (word+char)", "ml", 0.8801, 0.8753],
    ["LogReg (word+char)", "ml", 0.8756, 0.8786],
    ["BiLSTM + Attention", "dl", 0.8695, 0.8743],
    ["VotingEnsemble (word)", "ml", 0.8664, 0.8644],
    ["LinearSVM (word)", "ml", 0.8652, 0.8646],
    ["XGBoost (word)", "ml", 0.8626, 0.8637],
    ["LogReg (word)", "ml", 0.8622, 0.8630],
    ["RandomForest (word)", "ml", 0.8425, 0.8518],
    ["NaiveBayes (word+char)", "ml", 0.8038, 0.8006],
    ["NaiveBayes (word)", "ml", 0.7665, 0.7700]
];

const $ = s => document.querySelector(s);

// ---- Chart ----
function drawChart(mode) {
    const idx = mode === "val" ? 3 : 2;
    const rows = [...DATA].sort((a, b) => b[idx] - a[idx]);
    const lo = 0.74, hi = 0.91;   // axis range so differences are visible
    $("#chart").innerHTML = rows.map(r =>
        `<div class="bar"><span>${r[0]}</span><div class="tr"><div class="fl ${r[1]}" data-w="${((r[idx] - lo) / (hi - lo) * 100).toFixed(1)}"></div></div><span class="v">${r[idx].toFixed(3)}</span></div>`
    ).join("") +
        `<div class="legend"><span><i style="background:#7b7bff"></i>Transformer</span><span><i style="background:#e58bff"></i>Deep learning</span><span><i style="background:#4a5b8c"></i>Classical ML</span><span>Bars start at 0.74, so gaps look larger than they are.</span></div>`;
    requestAnimationFrame(() => requestAnimationFrame(() =>
        document.querySelectorAll("#chart .fl").forEach(el => el.style.width = el.dataset.w + "%")));
}
document.querySelectorAll(".tab").forEach(b => b.onclick = () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("on"));
    b.classList.add("on"); drawChart(b.dataset.m);
});
drawChart("test");

// ---- Animated counters ----
function count(el) {
    const end = +el.dataset.count, suf = el.dataset.suffix || "", t0 = performance.now();
    (function tick(t) {
        const p = Math.min((t - t0) / 1400, 1);
        el.textContent = Math.round(end * (1 - Math.pow(1 - p, 3))).toLocaleString() + suf;
        if (p < 1) requestAnimationFrame(tick);
    })(t0);
}
document.querySelectorAll("[data-count]").forEach(count);

// ---- Scroll reveal ----
const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) e.target.classList.add("in"); }), { threshold: .08 });
document.querySelectorAll(".reveal").forEach(el => io.observe(el));

// ---- Live demo ----
document.querySelectorAll("[data-ex]").forEach(a => a.onclick = () => { $("#t").value = a.dataset.ex; });
const pretty = s => s.replace(/_/g, " ");
$("#go").onclick = async () => {
    const text = $("#t").value.trim(), out = $("#out"), btn = $("#go");
    if (!text) return;
    btn.disabled = true; out.textContent = "Thinking...";
    try {
        const res = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
        const d = await res.json();
        if (!res.ok || d.error || !d.scores) { out.innerHTML = `<span class="err">${d.error || "Request failed (" + res.status + ")"}</span>`; return; }
        const good = d.prediction === "not_cyberbullying";
        out.innerHTML =
            `<div class="pred ${good ? "good" : "bad"}">${pretty(d.prediction)}<small>${(d.scores[0].score * 100).toFixed(1)}% confident</small></div>` +
            d.scores.map(s => `<div class="row"><span>${pretty(s.label)}</span><div class="tr"><div class="fl" data-w="${s.score * 100}"></div></div><span class="pc">${(s.score * 100).toFixed(1)}%</span></div>`).join("");
        requestAnimationFrame(() => requestAnimationFrame(() =>
            out.querySelectorAll(".fl").forEach(el => el.style.width = el.dataset.w + "%")));
    } catch (e) {
        out.innerHTML = `<span class="err">Could not reach the server: ${e}</span>`;
    } finally { btn.disabled = false; }
};