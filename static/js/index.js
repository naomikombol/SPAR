window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
    }

	// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();
    
    // Setup video autoplay for carousel
    setupVideoCarouselAutoplay();

})

document.addEventListener("DOMContentLoaded", () => {
  const container = document.querySelector('.image-fade-container');
  const frontImg = document.querySelector('.img-front');
  const questionText = document.querySelector('.fade-in-question');
  const sparText = document.querySelector('.fade-in-spar');

  if (!container || !frontImg) {
    console.log("Elements not found");
    return;
  }

  function clamp01(value) {
    return Math.max(0, Math.min(1, value));
  }

  function updateMotivationFade() {
    const rect = container.getBoundingClientRect();
    const windowHeight = window.innerHeight;

    const start = windowHeight * 0.3;
    const fadeDistance = windowHeight * 0.23;

    const progress = clamp01((start - rect.top) / fadeDistance);

    frontImg.style.opacity = progress;

    if (questionText) {
      const questionProgress = clamp01((progress - 0.45) / 0.4);
      questionText.style.opacity = questionProgress;
      questionText.style.transform = `translateY(${(1 - questionProgress) * 60}px)`;
    }

    if (sparText) {
      const sparProgress = clamp01((progress - 0.72) / 0.28);
      sparText.style.opacity = sparProgress;
      sparText.style.transform = `translateY(${(1 - sparProgress) * 40}px)`;
    }
  }

  window.addEventListener('scroll', updateMotivationFade, { passive: true });
  updateMotivationFade();
});

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".bal-container-small").forEach((container) => {
    const before = container.querySelector(".bal-before");
    const handle = container.querySelector(".bal-handle");

    if (!before || !handle) {
      return;
    }

    const start = Number(container.dataset.start) || 50;

    function setPercent(percent) {
      const clamped = Math.max(0, Math.min(100, percent));
      container.style.setProperty("--bal-pos", clamped + "%");
    }

    function setFromClientX(clientX) {
      const rect = container.getBoundingClientRect();
      const percent = ((clientX - rect.left) / rect.width) * 100;
      setPercent(percent);
    }

    setPercent(start);
    container.querySelectorAll("img").forEach((img) => {
      img.setAttribute("draggable", "false");
    });

    container.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      container.setPointerCapture(event.pointerId);
      setFromClientX(event.clientX);
    });

    container.addEventListener("pointermove", (event) => {
      if (!container.hasPointerCapture(event.pointerId)) {
        return;
      }
      setFromClientX(event.clientX);
    });

    container.addEventListener("pointerup", (event) => {
      if (container.hasPointerCapture(event.pointerId)) {
        container.releasePointerCapture(event.pointerId);
      }
    });

    container.addEventListener("pointercancel", (event) => {
      if (container.hasPointerCapture(event.pointerId)) {
        container.releasePointerCapture(event.pointerId);
      }
    });
  });
});
