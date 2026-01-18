'use client';

import React, { useEffect, useRef, useState, ReactNode, useCallback } from 'react';

export interface StepData {
  id: string;
  label: string;
  title: string;
  content: ReactNode;
  image?: string;
  imageAlt?: string;
}

interface ScrollLockedStepperProps {
  steps: StepData[];
  className?: string;
  containerClassName?: string;
}

export const ScrollLockedStepper: React.FC<ScrollLockedStepperProps> = ({
  steps,
  className = '',
  containerClassName = '',
}) => {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [isLocked, setIsLocked] = useState(false);
  const [direction, setDirection] = useState<'next' | 'prev' | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const scrollDeltaRef = useRef(0);
  const scrollTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const animationTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const prefersReducedMotion = useRef(false);

  // Check for reduced motion preference
  useEffect(() => {
    prefersReducedMotion.current = window.matchMedia(
      '(prefers-reduced-motion: reduce)'
    ).matches;
  }, []);

  // Intersection Observer to detect when section is 60%+ visible
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsLocked(entry.intersectionRatio >= 0.6);
      },
      { threshold: 0.6 }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, []);

  // Handle advancing to next step
  const advanceStep = useCallback(
    (dir: 'next' | 'prev') => {
      if (isAnimating) return;

      const newIndex =
        dir === 'next'
          ? Math.min(currentStepIndex + 1, steps.length - 1)
          : Math.max(currentStepIndex - 1, 0);

      // If we're already at the edge, release lock
      if (newIndex === currentStepIndex) {
        scrollDeltaRef.current = 0;
        return;
      }

      setIsAnimating(true);
      setDirection(dir);
      setCurrentStepIndex(newIndex);
      scrollDeltaRef.current = 0;

      // Animation duration
      const duration = prefersReducedMotion.current ? 0 : 400;
      animationTimeoutRef.current = setTimeout(() => {
        setIsAnimating(false);
        setDirection(null);
      }, duration);
    },
    [currentStepIndex, isAnimating, steps.length, prefersReducedMotion]
  );

  // Wheel event handler
  const handleWheel = useCallback(
    (e: WheelEvent) => {
      if (!isLocked || isAnimating) return;

      e.preventDefault();

      scrollDeltaRef.current += e.deltaY;

      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }

      const threshold = 120;

      if (scrollDeltaRef.current > threshold) {
        advanceStep('next');
      } else if (scrollDeltaRef.current < -threshold) {
        advanceStep('prev');
      } else {
        scrollTimeoutRef.current = setTimeout(() => {
          scrollDeltaRef.current = 0;
        }, 150);
      }
    },
    [isLocked, isAnimating, advanceStep]
  );

  // Touch event handler
  const touchStartRef = useRef({ y: 0 });
  const handleTouchStart = useCallback((e: TouchEvent) => {
    if (!isLocked) return;
    touchStartRef.current.y = e.touches[0].clientY;
  }, [isLocked]);

  const handleTouchEnd = useCallback(
    (e: TouchEvent) => {
      if (!isLocked || isAnimating) return;

      const deltaY = touchStartRef.current.y - e.changedTouches[0].clientY;
      const threshold = 80;

      if (Math.abs(deltaY) > threshold) {
        if (deltaY > 0) {
          advanceStep('next');
        } else {
          advanceStep('prev');
        }
      }
    },
    [isLocked, isAnimating, advanceStep]
  );

  // Keyboard event handler
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!isLocked || isAnimating) return;

      if (e.key === 'ArrowDown' || e.key === 'PageDown') {
        e.preventDefault();
        advanceStep('next');
      } else if (e.key === 'ArrowUp' || e.key === 'PageUp') {
        e.preventDefault();
        advanceStep('prev');
      }
    },
    [isLocked, isAnimating, advanceStep]
  );

  // Attach event listeners
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.addEventListener('wheel', handleWheel, { passive: false });
    container.addEventListener('touchstart', handleTouchStart, {
      passive: true,
    });
    container.addEventListener('touchend', handleTouchEnd, { passive: true });
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      container.removeEventListener('wheel', handleWheel);
      container.removeEventListener('touchstart', handleTouchStart);
      container.removeEventListener('touchend', handleTouchEnd);
      window.removeEventListener('keydown', handleKeyDown);
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
      if (animationTimeoutRef.current) {
        clearTimeout(animationTimeoutRef.current);
      }
    };
  }, [handleWheel, handleTouchStart, handleTouchEnd, handleKeyDown]);

  const currentStep = steps[currentStepIndex];

  return (
    <div ref={containerRef} className={`relative w-full h-screen overflow-hidden ${className}`}>
      <style>{`
        @keyframes slideUp {
          from {
            opacity: 1;
            transform: translateY(0);
          }
          to {
            opacity: 0;
            transform: translateY(-40px);
          }
        }

        @keyframes slideDown {
          from {
            opacity: 0;
            transform: translateY(40px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .step-exit {
          animation: slideUp 0.2s ease-out forwards;
        }

        .step-enter {
          animation: slideDown 0.2s ease-out forwards;
        }

        .step-content {
          transition: none;
        }

        @media (prefers-reduced-motion: reduce) {
          .step-exit,
          .step-enter {
            animation: none;
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>

      {/* Steps container */}
      <div className="relative w-full h-full flex items-center justify-center">
        <div
          key={`step-${currentStep.id}`}
          className={`absolute inset-0 flex items-center justify-center step-content ${
            isAnimating && direction === 'next' ? 'step-exit' : ''
          } ${isAnimating && direction === 'prev' ? 'step-exit' : ''} ${
            !isAnimating && direction === null ? 'step-enter' : ''
          } ${containerClassName}`}
        >
          {/* Content grid */}
          <div className="max-w-6xl mx-auto w-full grid grid-cols-1 md:grid-cols-2 gap-12 items-center px-6 py-8">
            {/* Text content */}
            <div className="space-y-8">
              <div className="space-y-6">
                <p className="section-label">{currentStep.label}</p>
                <h2 className="text-5xl text-[var(--color-dark)] font-serif leading-tight">
                  {currentStep.title}
                </h2>
                <div className="text-lg text-gray-600 leading-relaxed space-y-4">
                  {currentStep.content}
                </div>
              </div>
            </div>

            {/* Image/Visual content */}
            {currentStep.image && (
              <div className="flex items-center justify-center">
                <div className="relative">
                  <img
                    src={currentStep.image}
                    alt={currentStep.imageAlt || currentStep.title}
                    className="w-full rounded-lg shadow-lg border border-gray-300"
                  />
                </div>
              </div>
            )}

            {/* Placeholder if no image */}
            {!currentStep.image && (
              <div className="flex items-center justify-center">
                <div className="bg-white rounded-lg shadow-lg border border-gray-300 p-8">
                  <div className="space-y-6 text-center">
                    <div className="text-4xl font-bold text-gray-300">Visual</div>
                    <div className="text-gray-500 text-sm">
                      Step {currentStepIndex + 1} visualization
                    </div>
                    <div className="text-4xl font-bold text-gray-300">Preview</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Step indicators - vertical on the right */}
      <div className="absolute right-8 top-1/2 transform -translate-y-1/2 flex flex-col gap-4">
        {steps.map((_, idx) => (
          <button
            key={idx}
            onClick={() => {
              if (!isAnimating) {
                setCurrentStepIndex(idx);
              }
            }}
            className={`transition-all duration-300 rounded-full ${
              idx === currentStepIndex
                ? 'h-8 w-2 bg-[var(--color-dark)]'
                : 'h-2 w-2 bg-gray-400 hover:bg-gray-600'
            }`}
            aria-label={`Go to step ${idx + 1}`}
          />
        ))}
      </div>
    </div>
  );
};
