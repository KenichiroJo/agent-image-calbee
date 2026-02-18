import { CheckCircle2, ImageIcon, Brain, FileText, Loader2 } from 'lucide-react';

export interface AnalysisProgressProps {
  currentStep: 'idle' | 'uploading' | 'analyzing' | 'summarizing' | 'done' | 'error';
  errorMessage?: string;
}

const steps = [
  { key: 'uploading', label: '画像読込', icon: ImageIcon },
  { key: 'analyzing', label: 'AI分析中（2モデル並列）', icon: Brain },
  { key: 'summarizing', label: 'レポート統合', icon: FileText },
] as const;

function getStepStatus(
  stepKey: string,
  currentStep: AnalysisProgressProps['currentStep']
): 'pending' | 'active' | 'done' {
  const stepOrder = ['uploading', 'analyzing', 'summarizing', 'done'];
  const currentIndex = stepOrder.indexOf(currentStep);
  const stepIndex = stepOrder.indexOf(stepKey);

  if (currentStep === 'idle' || currentStep === 'error') return 'pending';
  if (stepIndex < currentIndex) return 'done';
  if (stepIndex === currentIndex) return 'active';
  return 'pending';
}

export function AnalysisProgress({ currentStep, errorMessage }: AnalysisProgressProps) {
  if (currentStep === 'idle') return null;

  return (
    <div className="analysis-progress">
      <div className="flex items-center gap-2 mb-2">
        {currentStep === 'error' ? (
          <span className="text-sm text-destructive font-medium">
            {errorMessage || 'エラーが発生しました'}
          </span>
        ) : currentStep === 'done' ? (
          <span className="text-sm text-green-600 font-medium flex items-center gap-1">
            <CheckCircle2 className="w-4 h-4" />
            分析完了
          </span>
        ) : (
          <span className="text-sm text-muted-foreground">分析を実行中...</span>
        )}
      </div>

      <div className="flex items-center gap-1">
        {steps.map((step, index) => {
          const status = getStepStatus(step.key, currentStep);
          const Icon = step.icon;

          return (
            <div key={step.key} className="flex items-center gap-1">
              <div
                className={`progress-step ${
                  status === 'done'
                    ? 'step-done'
                    : status === 'active'
                      ? 'step-active'
                      : 'step-pending'
                }`}
              >
                {status === 'done' ? (
                  <CheckCircle2 className="w-3.5 h-3.5" />
                ) : status === 'active' ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Icon className="w-3.5 h-3.5" />
                )}
                <span className="text-xs">{step.label}</span>
              </div>
              {index < steps.length - 1 && (
                <div
                  className={`progress-connector ${
                    status === 'done' ? 'connector-done' : 'connector-pending'
                  }`}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
