document.addEventListener("DOMContentLoaded", () => {
  // 1. Theme toggle
  const root = document.documentElement;
  const themeBtn = document.getElementById("themeToggle");
  const savedTheme = localStorage.getItem("tennis-blog-theme");
  if (savedTheme === "dark") {
    root.setAttribute("data-theme", "dark");
    if (themeBtn) themeBtn.textContent = "Light Mode";
  }

  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const isDark = root.getAttribute("data-theme") === "dark";
      if (isDark) {
        root.removeAttribute("data-theme");
        localStorage.setItem("tennis-blog-theme", "light");
        themeBtn.textContent = "Dark Mode";
      } else {
        root.setAttribute("data-theme", "dark");
        localStorage.setItem("tennis-blog-theme", "dark");
        themeBtn.textContent = "Light Mode";
      }
    });
  }

  // 2. Reading progress bar
  const progressBar = document.getElementById("progressBar");
  const updateProgress = () => {
    if (!progressBar) return;
    const scrollTop = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const pct = docHeight > 0 ? Math.min(100, Math.max(0, (scrollTop / docHeight) * 100)) : 0;
    progressBar.style.width = `${pct.toFixed(1)}%`;
  };
  window.addEventListener("scroll", updateProgress, { passive: true });
  updateProgress();

  // 3. Scrollspy for TOC
  const tocLinks = Array.from(document.querySelectorAll(".toc-list a"));
  const sections = tocLinks
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  if (sections.length > 0 && "IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const id = entry.target.getAttribute("id");
            tocLinks.forEach((l) => {
              l.classList.toggle("active", l.getAttribute("href") === `#${id}`);
            });
          }
        });
      },
      { rootMargin: "-20% 0px -65% 0px" }
    );
    sections.forEach((sec) => observer.observe(sec));
  }

  // 4. Interactive Validation vs. Holdout Test metrics switcher
  const metricsData = {
    test: [
      { model: "Logistic Regression", type: "Linear (C=0.01)", acc: "65.57%", auc: "0.7187", loss: "0.6136", brier: "0.2134", best: true },
      { model: "HistGradientBoosting", type: "Gradient Boosting (depth=3)", acc: "65.57%", auc: "0.7156", loss: "0.6168", brier: "0.2145", best: false },
      { model: "MLP Neural Net", type: "Neural Network (64-32)", acc: "64.83%", auc: "0.7173", loss: "0.6159", brier: "0.2143", best: false },
      { model: "Random Forest", type: "Bagged Trees (depth=8)", acc: "64.97%", auc: "0.7138", loss: "0.6174", brier: "0.2149", best: false },
      { model: "Baseline: Pure Elo", type: "Rating Baseline", acc: "63.94%", auc: "0.6968", loss: "0.6317", brier: "0.2209", best: false },
      { model: "Baseline: ATP Rank", type: "Heuristic Baseline", acc: "63.98%", auc: "0.6970", loss: "0.6455", brier: "0.2249", best: false },
    ],
    val: [
      { model: "Logistic Regression", type: "Linear (C=0.01)", acc: "65.78%", auc: "0.7267", loss: "0.6061", brier: "0.2103", best: true },
      { model: "HistGradientBoosting", type: "Gradient Boosting (depth=3)", acc: "65.68%", auc: "0.7260", loss: "0.6082", brier: "0.2107", best: false },
      { model: "Random Forest", type: "Bagged Trees (depth=8)", acc: "65.55%", auc: "0.7241", loss: "0.6086", brier: "0.2113", best: false },
      { model: "MLP Neural Net", type: "Neural Network (64-32)", acc: "65.55%", auc: "0.7238", loss: "0.6088", brier: "0.2117", best: false },
      { model: "Baseline: Pure Elo", type: "Rating Baseline", acc: "64.08%", auc: "0.7025", loss: "0.6237", brier: "0.2181", best: false },
      { model: "Baseline: ATP Rank", type: "Heuristic Baseline", acc: "63.98%", auc: "0.6932", loss: "0.6463", brier: "0.2260", best: false },
    ],
  };

  const metricsBody = document.getElementById("metricsTableBody");
  const splitButtons = document.querySelectorAll("[data-split-btn]");
  const splitCaption = document.getElementById("splitCaption");

  const renderMetrics = (splitKey) => {
    if (!metricsBody) return;
    const rows = metricsData[splitKey] || metricsData.test;
    metricsBody.innerHTML = rows
      .map(
        (r) => `
      <tr class="${r.best ? "row-best" : ""}">
        <td>${r.model}${r.best ? ' <span class="pill pill-hit">Best</span>' : ""}</td>
        <td>${r.type}</td>
        <td>${r.acc}</td>
        <td>${r.auc}</td>
        <td>${r.loss}</td>
        <td>${r.brier}</td>
      </tr>`
      )
      .join("");
    if (splitCaption) {
      splitCaption.textContent =
        splitKey === "test"
          ? "Showing 2025–2026 Holdout Test Set (4,359 played ATP matches)"
          : "Showing 2024 Validation Set (3,054 played ATP matches)";
    }
  };

  splitButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      splitButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderMetrics(btn.getAttribute("data-split-btn"));
    });
  });
  renderMetrics("test");

  // 5. Lightbox for figures
  const lightbox = document.getElementById("lightbox");
  const lightboxImg = document.getElementById("lightboxImg");
  const lightboxCaption = document.getElementById("lightboxCaption");
  const lightboxClose = document.getElementById("lightboxClose");

  document.querySelectorAll(".figure-card img").forEach((img) => {
    img.addEventListener("click", () => {
      if (!lightbox || !lightboxImg || !lightboxCaption) return;
      lightboxImg.src = img.src;
      lightboxImg.alt = img.alt;
      const figCap = img.closest(".figure-card")?.querySelector(".figure-caption");
      lightboxCaption.textContent = figCap ? figCap.textContent : img.alt;
      lightbox.classList.add("open");
    });
  });

  const closeLightbox = () => {
    if (lightbox) lightbox.classList.remove("open");
  };

  if (lightboxClose) lightboxClose.addEventListener("click", closeLightbox);
  if (lightbox) {
    lightbox.addEventListener("click", (e) => {
      if (e.target === lightbox) closeLightbox();
    });
  }
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeLightbox();
  });
});
