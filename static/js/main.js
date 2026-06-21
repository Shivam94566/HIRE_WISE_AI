document.addEventListener('DOMContentLoaded', () => {
    // 1. Theme Toggle logic
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const currentTheme = localStorage.getItem('theme') || 'light';
    
    // Set initial theme
    document.documentElement.setAttribute('data-theme', currentTheme);
    updateThemeButton(currentTheme);
    
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            let theme = document.documentElement.getAttribute('data-theme');
            let newTheme = theme === 'dark' ? 'light' : 'dark';
            
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateThemeButton(newTheme);
        });
    }
    
    function updateThemeButton(theme) {
        if (!themeToggleBtn) return;
        const icon = themeToggleBtn.querySelector('i');
        const text = themeToggleBtn.querySelector('span');
        
        if (theme === 'dark') {
            if (icon) icon.className = 'bi bi-sun-fill';
            if (text) text.textContent = 'Light Mode';
        } else {
            if (icon) icon.className = 'bi bi-moon-stars-fill';
            if (text) text.textContent = 'Dark Mode';
        }
    }
    
    // 2. Flash message automatic dismissal
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.classList.add('fade');
            setTimeout(() => {
                alert.remove();
            }, 150);
        }, 5000);
    });
});
