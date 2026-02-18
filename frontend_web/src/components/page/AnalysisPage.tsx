import { PropsWithChildren, useCallback, useEffect, useMemo, useState } from 'react';
import { v4 as uuid } from 'uuid';
import { Loader2, Search } from 'lucide-react';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';
import { useChatContext } from '@/hooks/use-chat-context';
import { useChatList } from '@/hooks/use-chat-list';
import { ChatProvider } from '@/components/ChatProvider';
import { ChatSidebar } from '@/components/ChatSidebar';
import { ImageUploadArea } from '@/components/ImageUploadArea';
import { AnalysisResult } from '@/components/AnalysisResult';
import { AnalysisProgress, type AnalysisProgressProps } from '@/components/AnalysisProgress';
import { isMessageStateEvent } from '@/types/events';
import { uploadImage } from '@/api/chat/requests';
import { type MessageResponse } from '@/api/chat/types';

const initialMessages: MessageResponse[] = [
  {
    id: uuid(),
    role: 'assistant',
    content: {
      format: 2,
      parts: [
        {
          type: 'text',
          text: '棚画像をアップロードして分析を開始してください。',
        },
      ],
    },
    createdAt: new Date(),
    type: 'initial',
  },
];

export function AnalysisPage({
  chatId,
  setChatId,
}: {
  chatId: string;
  setChatId: (id: string) => void;
}) {
  const {
    hasChat,
    isNewChat,
    chats,
    isLoadingChats,
    addChatHandler,
    deleteChatHandler,
    isLoadingDeleteChat,
  } = useChatList({
    chatId,
    setChatId,
    showStartChat: false,
  });

  return (
    <div className="analysis-layout">
      <ChatSidebar
        isLoading={isLoadingChats}
        chatId={chatId}
        chats={chats}
        onChatCreate={addChatHandler}
        onChatSelect={setChatId}
        onChatDelete={deleteChatHandler}
        isLoadingDeleteChat={isLoadingDeleteChat}
      />

      <Loading isLoading={isLoadingChats}>
        {hasChat ? (
          <ChatProvider chatId={chatId} runInBackground={true} isNewChat={isNewChat}>
            <AnalysisImplementation />
          </ChatProvider>
        ) : (
          <div className="analysis-main">
            <div className="analysis-header">
              <h1 className="text-xl font-bold">カルビー 棚画像分析</h1>
            </div>
            <div className="flex items-center justify-center flex-1 text-muted-foreground">
              新しい分析を開始してください
            </div>
          </div>
        )}
      </Loading>
    </div>
  );
}

function Loading({ isLoading, children }: { isLoading: boolean } & PropsWithChildren) {
  if (isLoading) {
    return (
      <div className="flex flex-1 flex-col w-full p-4 space-y-4">
        <Skeleton className="h-20 w-full" />
        <Skeleton className="h-20 w-full" />
        <Skeleton className="h-20 w-full" />
      </div>
    );
  }
  return children;
}

function AnalysisImplementation() {
  const {
    sendMessage,
    combinedEvents,
    isAgentRunning,
    setInitialMessages,
    message,
  } = useChatContext();

  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [progressStep, setProgressStep] = useState<AnalysisProgressProps['currentStep']>('idle');
  const [errorMessage, setErrorMessage] = useState<string | undefined>();

  // Set initial messages on mount
  useEffect(() => {
    setInitialMessages(initialMessages);
  }, []);

  // Track progress based on agent running state
  useEffect(() => {
    if (!isAgentRunning && progressStep === 'analyzing') {
      setProgressStep('done');
    }
  }, [isAgentRunning, progressStep]);

  // Extract streaming content from the currently streaming message
  const streamingContent = useMemo(() => {
    if (message?.content?.parts) {
      const textParts = message.content.parts
        .filter((p: any) => p.type === 'text' && p.text)
        .map((p: any) => p.text);
      return textParts.join('');
    }
    return null;
  }, [message]);

  const handleAnalyze = useCallback(async () => {
    if (!selectedImage || isAgentRunning || isUploading) return;

    setErrorMessage(undefined);
    setIsUploading(true);
    setProgressStep('uploading');

    try {
      // Upload image
      const result = await uploadImage(selectedImage);

      setProgressStep('analyzing');
      setIsUploading(false);

      // Send message with image path to agent
      const msg = `[IMAGE:${result.path}] この店舗棚画像を詳細に分析してください。`;
      await sendMessage(msg);

      setProgressStep('done');
    } catch (error: any) {
      console.error('Analysis failed:', error);
      setProgressStep('error');
      setErrorMessage(error?.message || '分析中にエラーが発生しました');
      setIsUploading(false);
    }
  }, [selectedImage, isAgentRunning, isUploading, sendMessage]);

  return (
    <div className="analysis-main">
      <div className="analysis-header">
        <h1 className="text-xl font-bold">カルビー 棚画像分析</h1>
        <p className="text-sm text-muted-foreground">
          店舗棚画像をAIが分析し、棚割り・価格・販促の改善ポイントをレポートします
        </p>
      </div>

      <div className="analysis-content">
        <div className="analysis-upload-section">
          <ImageUploadArea
            selectedImage={selectedImage}
            onImageSelect={setSelectedImage}
            onImageClear={() => {
              setSelectedImage(null);
              setProgressStep('idle');
            }}
            disabled={isAgentRunning || isUploading}
          />

          <Button
            className="analysis-start-btn"
            size="lg"
            onClick={handleAnalyze}
            disabled={!selectedImage || isAgentRunning || isUploading}
          >
            {isAgentRunning || isUploading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin mr-2" />
                分析中...
              </>
            ) : (
              <>
                <Search className="w-4 h-4 mr-2" />
                分析開始
              </>
            )}
          </Button>

          <AnalysisProgress
            currentStep={progressStep}
            errorMessage={errorMessage}
          />
        </div>

        <AnalysisResult
          events={combinedEvents || []}
          isRunning={isAgentRunning}
          streamingContent={streamingContent}
        />
      </div>
    </div>
  );
}
