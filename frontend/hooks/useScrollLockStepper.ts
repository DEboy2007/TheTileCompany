import { useEffect, useRef, useCallback } from 'react';

interface UseScrollLockStepperProps {
  onStepChange: (index: number) => void;
  onLockChange?: (isLocked: boolean) => void;
  totalSteps: number;
  currentStep: number;
  isAnimating: boolean;
  containerRef: React.RefObject<HTMLDivElement | null>;
}

export const useScrollLockStepper = ({
  onStepChange,
  onLockChange,
  totalSteps,
  currentStep,
  isAnimating,
  containerRef,
}: UseScrollLockStepperProps) => {
  const scrollDeltaRef = useRef(0);
  const isLockedRef = useRef(false);
  const scrollTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Intersection Observer to detect when section is in view
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        // Lock is ON when the sentinel is visible (entered the viewport)
        const shouldLock = entry.isIntersecting;
        isLockedRef.current = shouldLock;
        onLockChange?.(shouldLock);
      },
      { threshold: 0 }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, [containerRef]);

  // Wheel event handler
  const handleWheel = useCallback(
    (e: WheelEvent) => {
      if (!isLockedRef.current || isAnimating) {
        // Reset delta when stepper is not locked
        scrollDeltaRef.current = 0;
        return;
      }

      e.preventDefault();
      scrollDeltaRef.current += e.deltaY;

      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }

      const threshold = 80;

      if (scrollDeltaRef.current > threshold) {
        const nextStep = Math.min(currentStep + 1, totalSteps - 1);
        if (nextStep !== currentStep) {
          onStepChange(nextStep);
        }
        scrollDeltaRef.current = 0;
      } else if (scrollDeltaRef.current < -threshold) {
        const prevStep = Math.max(currentStep - 1, 0);
        if (prevStep !== currentStep) {
          onStepChange(prevStep);
        }
        scrollDeltaRef.current = 0;
      } else {
        scrollTimeoutRef.current = setTimeout(() => {
          scrollDeltaRef.current = 0;
        }, 150);
      }
    },
    [isAnimating, currentStep, totalSteps, onStepChange]
  );

  // Touch event handler
  const touchStartRef = useRef({ y: 0 });
  const handleTouchStart = useCallback((e: TouchEvent) => {
    if (!isLockedRef.current) return;
    touchStartRef.current.y = e.touches[0].clientY;
  }, []);

  const handleTouchEnd = useCallback(
    (e: TouchEvent) => {
      if (!isLockedRef.current || isAnimating) return;

      const deltaY = touchStartRef.current.y - e.changedTouches[0].clientY;
      const threshold = 80;

      if (Math.abs(deltaY) > threshold) {
        if (deltaY > 0) {
          const nextStep = Math.min(currentStep + 1, totalSteps - 1);
          if (nextStep !== currentStep) {
            onStepChange(nextStep);
          }
        } else {
          const prevStep = Math.max(currentStep - 1, 0);
          if (prevStep !== currentStep) {
            onStepChange(prevStep);
          }
        }
      }
    },
    [isAnimating, currentStep, totalSteps, onStepChange]
  );

  // Keyboard event handler
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!isLockedRef.current || isAnimating) return;

      if (e.key === 'ArrowDown' || e.key === 'PageDown') {
        e.preventDefault();
        const nextStep = Math.min(currentStep + 1, totalSteps - 1);
        if (nextStep !== currentStep) {
          onStepChange(nextStep);
        }
      } else if (e.key === 'ArrowUp' || e.key === 'PageUp') {
        e.preventDefault();
        const prevStep = Math.max(currentStep - 1, 0);
        if (prevStep !== currentStep) {
          onStepChange(prevStep);
        }
      }
    },
    [isAnimating, currentStep, totalSteps, onStepChange]
  );

  // Attach event listeners
  useEffect(() => {
    // Attach wheel to window to intercept globally when locked
    window.addEventListener('wheel', handleWheel, { passive: false });
    window.addEventListener('touchstart', handleTouchStart, { passive: true });
    window.addEventListener('touchend', handleTouchEnd, { passive: true });
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('wheel', handleWheel);
      window.removeEventListener('touchstart', handleTouchStart);
      window.removeEventListener('touchend', handleTouchEnd);
      window.removeEventListener('keydown', handleKeyDown);
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
    };
  }, [handleWheel, handleTouchStart, handleTouchEnd, handleKeyDown]);
};
