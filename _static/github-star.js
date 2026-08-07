(() => {
  "use strict";

  const REPOSITORY_URL = "https://github.com/SavvasRaptis/helio-data-methods";

  function addStarLink() {
    if (document.querySelector("[data-helio-github-star]")) {
      return;
    }

    const header = document.querySelector("#pst-header .bd-header__inner");
    if (!header) {
      return;
    }

    const link = document.createElement("a");
    link.href = REPOSITORY_URL;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.className = "helio-github-star";
    link.dataset.helioGithubStar = "";
    link.setAttribute("aria-label", "Star helio-data-methods on GitHub");
    link.title = "Star on GitHub";
    link.innerHTML = [
      '<span class="fa-regular fa-star" aria-hidden="true"></span>',
      '<span class="helio-github-star__label">Star on GitHub</span>',
    ].join("");

    const secondaryToggle = header.querySelector(".secondary-toggle");
    header.insertBefore(link, secondaryToggle || null);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", addStarLink, { once: true });
  } else {
    addStarLink();
  }
})();
