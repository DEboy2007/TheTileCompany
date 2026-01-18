import { useEffect, useRef, useState } from 'react';

export const useScrollFade = () => {
  const ref = useRef<HTMLDivElement>(null);
  const [opacity, setOpacity] = useState(1);

  useEffect(() => {
    // Listen for snap fade events
    const handleSnapFade = (e: Event) => {
      const event = e as CustomEvent;
      if (event.detail.fadeOutElement === ref.current) {
        setOpacity(0);
      } else {
        setOpacity(1);
      }
    };

    window.addEventListener('snapFade', handleSnapFade);
    return () => window.removeEventListener('snapFade', handleSnapFade);
  }, []);

  return { ref, opacity };
};

export const useScrollFadeIn = () => {
  const ref = useRef<HTMLDivElement>(null);
  const [opacity, setOpacity] = useState(0);

  useEffect(() => {
    // Listen for snap fade events
    const handleSnapFade = (e: Event) => {
      const event = e as CustomEvent;
      // Show if this is the target, hide if this was the source being faded out
      if (event.detail.fadeInElement === ref.current) {
        setOpacity(1);
      } else if (event.detail.fadeOutElement === ref.current) {
        setOpacity(0);
      }
    };

    window.addEventListener('snapFade', handleSnapFade);
    return () => window.removeEventListener('snapFade', handleSnapFade);
  }, []);

  return { ref, opacity };
};
