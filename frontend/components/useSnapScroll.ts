import { useEffect, useRef } from 'react';

export const useSnapScroll = () => {
  const scrollTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastScrollYRef = useRef(0);
  const isScrollingRef = useRef(false);

  useEffect(() => {
    const sections = document.querySelectorAll('.snap-section') as NodeListOf<HTMLElement>;
    if (sections.length === 0) return;

    const handleScroll = () => {
      if (isScrollingRef.current) return; // Ignore scroll events while animating

      lastScrollYRef.current = window.scrollY;

      // Clear existing timeout
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }

      // Set a timeout to detect when scrolling has stopped
      scrollTimeoutRef.current = setTimeout(() => {
        const currentScroll = window.scrollY;
        const viewportHeight = window.innerHeight;

        // Find which section we're closest to
        let closestSection = sections[0];
        let closestDistance = Math.abs(closestSection.offsetTop - currentScroll);

        sections.forEach((section) => {
          const distance = Math.abs(section.offsetTop - currentScroll);
          if (distance < closestDistance) {
            closestDistance = distance;
            closestSection = section;
          }
        });

        // Calculate the threshold - snap if we've scrolled more than 30% of viewport
        const scrollDelta = Math.abs(currentScroll - closestSection.offsetTop);
        const threshold = viewportHeight * 0.3;

        // Only snap if we've scrolled past the threshold
        if (scrollDelta > threshold) {
          // Find the next or previous section based on scroll direction
          const currentIndex = Array.from(sections).indexOf(closestSection);
          let targetSection = closestSection;

          if (currentScroll > closestSection.offsetTop) {
            // Scrolling down - snap to next section
            if (currentIndex < sections.length - 1) {
              targetSection = sections[currentIndex + 1];
            }
          } else {
            // Scrolling up - snap to previous section
            if (currentIndex > 0) {
              targetSection = sections[currentIndex - 1];
            }
          }

          // Dispatch fade event before snapping
          const fadeEvent = new CustomEvent('snapFade', {
            detail: {
              fadeOutElement: closestSection,
              fadeInElement: targetSection,
            },
          });
          window.dispatchEvent(fadeEvent);

          // Snap to the target section
          isScrollingRef.current = true;
          window.scrollTo({
            top: targetSection.offsetTop,
            behavior: 'smooth',
          });

          // Re-enable scroll after animation
          setTimeout(() => {
            isScrollingRef.current = false;
          }, 400);
        } else {
          // Snap back to closest section
          isScrollingRef.current = true;
          window.scrollTo({
            top: closestSection.offsetTop,
            behavior: 'smooth',
          });

          setTimeout(() => {
            isScrollingRef.current = false;
          }, 400);
        }
      }, 50); // Wait 50ms after scroll stops
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', handleScroll);
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
    };
  }, []);
};
