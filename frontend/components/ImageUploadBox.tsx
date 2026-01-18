'use client';

import { useState, useRef, forwardRef, useImperativeHandle } from 'react';

export interface ImageUploadBoxHandle {
  triggerUpload: () => void;
}

interface ImageUploadBoxProps {
  onImageUpload?: (base64Image: string) => void;
}

const ImageUploadBox = forwardRef<ImageUploadBoxHandle, ImageUploadBoxProps>(({ onImageUpload }, ref) => {
  const [image, setImage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const result = event.target?.result as string;
        setImage(result);
        if (onImageUpload) {
          onImageUpload(result);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  useImperativeHandle(ref, () => ({
    triggerUpload: () => {
      fileInputRef.current?.click();
    },
  }));

  return (
    <div
      className={`w-full bg-white rounded-2xl p-8 transition-all border-2 border-gray-300 ${!image ? 'cursor-pointer hover:shadow-lg' : ''}`}
      style={{
        boxShadow: 'inset 0 2px 8px rgba(0, 0, 0, 0.06)',
      }}
      onClick={() => !image && fileInputRef.current?.click()}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="hidden"
      />

      {!image ? (
        <div className="flex flex-col items-center justify-center py-16">
          <div className="p-3 bg-gray-100 rounded-lg">
            <svg className="w-12 h-12 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
            </svg>
          </div>
        </div>
      ) : (
        <div className="w-full space-y-4">
          <div className="w-full flex items-center justify-center p-6" style={{
            animation: 'zoomIn 0.5s ease-out forwards',
          }}>
            <img
              src={image}
              alt="Uploaded preview"
              className="max-w-full max-h-80 rounded-lg object-cover"
            />
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
            className="w-full py-2 px-4 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
          >
            Change Image
          </button>
        </div>
      )}
      <style>{`
        @keyframes zoomIn {
          from {
            opacity: 0;
            transform: scale(0.92);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }
      `}</style>
    </div>
  );
});

ImageUploadBox.displayName = 'ImageUploadBox';
export default ImageUploadBox;
