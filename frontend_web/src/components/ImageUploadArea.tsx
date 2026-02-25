import { useCallback, useRef, useState } from 'react';
import { ImagePlus, X } from 'lucide-react';
import { Button } from '@/components/ui/button';

export interface ImageUploadAreaProps {
  onImageSelect: (file: File) => void;
  onImageClear: () => void;
  selectedImage: File | null;
  /** URL to display for a previously analyzed image (loaded from history) */
  historicalImageUrl?: string | null;
  disabled?: boolean;
}

export function ImageUploadArea({
  onImageSelect,
  onImageClear,
  selectedImage,
  historicalImageUrl,
  disabled = false,
}: ImageUploadAreaProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith('image/')) return;
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      onImageSelect(file);
    },
    [onImageSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile, disabled]
  );

  const handleDragOver = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (!disabled) setIsDragOver(true);
    },
    [disabled]
  );

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleClick = () => {
    if (!disabled) fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = '';
  };

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    onImageClear();
  };

  return (
    <div className="image-upload-area">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={handleFileChange}
        className="hidden"
      />

      {selectedImage && previewUrl ? (
        <div className="image-preview-container">
          <img
            src={previewUrl}
            alt="Selected shelf image"
            className="image-preview"
          />
          <div className="image-preview-overlay">
            <Button
              variant="secondary"
              size="sm"
              onClick={handleClear}
              disabled={disabled}
            >
              <X className="w-4 h-4 mr-1" />
              画像を変更
            </Button>
          </div>
          <div className="image-filename">
            {selectedImage.name}
          </div>
        </div>
      ) : historicalImageUrl ? (
        <div className="image-preview-container">
          <img
            src={historicalImageUrl}
            alt="Analyzed shelf image"
            className="image-preview"
          />
          <div className="image-filename" style={{ opacity: 0.7 }}>
            過去の分析画像
          </div>
        </div>
      ) : (
        <div
          className={`upload-dropzone ${isDragOver ? 'drag-over' : ''} ${disabled ? 'disabled' : ''}`}
          onClick={handleClick}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <ImagePlus className="w-12 h-12 text-muted-foreground mb-3" />
          <p className="text-base font-medium text-foreground">
            棚画像をアップロード
          </p>
          <p className="text-sm text-muted-foreground mt-1">
            ドラッグ&ドロップ、またはクリックしてファイルを選択
          </p>
          <p className="text-xs text-muted-foreground mt-2">
            JPEG, PNG, WebP (最大 20MB)
          </p>
        </div>
      )}
    </div>
  );
}
