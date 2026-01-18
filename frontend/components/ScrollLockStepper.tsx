'use client';

import React, { useState, useRef, ReactNode, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useScrollLockStepper } from '@/hooks/useScrollLockStepper';

export interface StepData {
  id: string;
  label: string;
  title: string;
  content: ReactNode;
  image?: string;
  imageAlt?: string;
  customRender?: ReactNode;
}

interface ScrollLockStepperProps {
  steps: StepData[];
  uploadBoxRef?: React.RefObject<any>;
}

export const ScrollLockStepper: React.FC<ScrollLockStepperProps> = ({ steps, uploadBoxRef }) => {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [isLocked, setIsLocked] = useState(false);
  const [selectedImage, setSelectedImage] = useState<{ src: string; alt: string } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);

  // Handle step changes with animation lock
  const handleStepChange = (newIndex: number) => {
    if (isAnimating || newIndex === currentStepIndex) return;
    setIsAnimating(true);
    setCurrentStepIndex(newIndex);
  };

  // Disable page scroll when carousel is locked
  useEffect(() => {
    if (isLocked) {
      document.documentElement.style.overflow = 'hidden';
      document.body.style.overflow = 'hidden';
    } else {
      document.documentElement.style.overflow = '';
      document.body.style.overflow = '';
    }
    return () => {
      document.documentElement.style.overflow = '';
      document.body.style.overflow = '';
    };
  }, [isLocked]);

  // Attach scroll lock stepper - use sentinel for intersection detection
  useScrollLockStepper({
    onStepChange: handleStepChange,
    onLockChange: setIsLocked,
    totalSteps: steps.length,
    currentStep: currentStepIndex,
    isAnimating,
    containerRef: sentinelRef,
  });

  const currentStep = steps[currentStepIndex];

  // Framer Motion variants
  const exitVariants = {
    exit: {
      opacity: 0,
      y: -40,
      transition: { duration: 0.2 },
    },
  };

  const enterVariants = {
    initial: { opacity: 0, y: 40 },
    animate: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.2 },
    },
    exit: {
      opacity: 0,
      y: -40,
      transition: { duration: 0.2 },
    },
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full bg-[#FAF9F5]"
    >
      {/* Sentinel at start to detect when carousel enters view */}
      <div ref={sentinelRef} className="h-0" />

      {/* Fixed stage container - covers full viewport */}
      <div className="fixed top-0 left-0 w-full h-screen pointer-events-none">
        <AnimatePresence mode="wait">
          <motion.div
            key={`step-${currentStep.id}`}
            initial="initial"
            animate="animate"
            exit="exit"
            variants={enterVariants}
            onAnimationComplete={() => setIsAnimating(false)}
            className="w-full h-full flex items-center justify-center pointer-events-auto"
          >
            {/* Custom render or standard grid */}
            {currentStep.customRender ? (
              <div className="w-full">{currentStep.customRender}</div>
            ) : (
              <div className="max-w-6xl mx-auto w-full grid grid-cols-1 md:grid-cols-2 gap-12 items-center px-6">
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
                    <div className="relative cursor-pointer group">
                      <img
                        src={currentStep.image}
                        alt={currentStep.imageAlt || currentStep.title}
                        className="w-full rounded-lg shadow-lg border border-gray-300 group-hover:shadow-xl transition-shadow cursor-pointer"
                        onClick={() => setSelectedImage({ src: currentStep.image!, alt: currentStep.imageAlt || currentStep.title })}
                      />
                      <div className="absolute inset-0 rounded-lg bg-black opacity-0 group-hover:opacity-10 transition-opacity duration-200 pointer-events-none" />
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
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Step indicators - vertical on the right */}
      <div className="fixed right-8 top-1/2 transform -translate-y-1/2 flex flex-col gap-4 pointer-events-none z-40">
        {steps.map((step, idx) => (
          <motion.button
            key={idx}
            onClick={() => {
              if (!isAnimating) {
                setCurrentStepIndex(idx);
              }
            }}
            className={`rounded-full pointer-events-auto transition-colors ${
              idx === currentStepIndex
                ? 'bg-[var(--color-dark)]'
                : 'bg-gray-400 hover:bg-gray-600'
            }`}
            animate={{
              height: idx === currentStepIndex ? 32 : 8,
              width: 8,
            }}
            aria-label={`Go to ${step.label}`}
          />
        ))}
      </div>

      {/* Image Modal */}
      <AnimatePresence>
        {selectedImage && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={() => setSelectedImage(null)}
            className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4 pointer-events-auto"
          >
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              transition={{ duration: 0.3, type: 'spring', stiffness: 300, damping: 30 }}
              onClick={(e) => e.stopPropagation()}
              className="relative w-full h-[90vh] max-w-6xl"
            >
              <img
                src={selectedImage.src}
                alt={selectedImage.alt}
                className="w-full h-full object-contain rounded-lg"
              />
              <button
                onClick={() => setSelectedImage(null)}
                className="absolute top-4 right-4 bg-white text-black rounded-full w-10 h-10 flex items-center justify-center hover:bg-gray-200 transition-colors"
              >
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Spacer to allow scrolling past the carousel */}
      <div className="h-screen bg-[#FAF9F5]" />
    </div>
  );
};
