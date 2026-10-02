// Dark-mode toggle shared by every page. The saved preference is applied
// early by an inline snippet in base.html to avoid a flash of the wrong theme.
(function () {
    const root = document.documentElement;
    const toggle = document.getElementById('theme-toggle');
    if (!toggle) return;

    function render() {
        const dark = root.getAttribute('data-bs-theme') === 'dark';
        toggle.innerHTML = dark ? '<i class="bi bi-sun"></i>' : '<i class="bi bi-moon-stars"></i>';
        toggle.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
    }

    toggle.addEventListener('click', function () {
        const dark = root.getAttribute('data-bs-theme') !== 'dark';
        root.setAttribute('data-bs-theme', dark ? 'dark' : 'light');
        try {
            localStorage.setItem('darkMode', dark);
        } catch (e) { /* storage unavailable: preference lasts for this page only */ }
        render();
    });

    render();
})();
