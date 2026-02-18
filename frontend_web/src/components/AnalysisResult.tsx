import { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import { Loader2 } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent } from '@/components/ui/card';
import { type ChatStateEvent, isMessageStateEvent } from '@/types/events';

export interface AnalysisResultProps {
  events: ChatStateEvent[];
  isRunning: boolean;
  streamingContent: string | null;
}

interface ParsedResults {
  analyzerA: string;
  analyzerB: string;
  summary: string;
}

function parseAnalysisContent(fullText: string): ParsedResults {
  const result: ParsedResults = {
    analyzerA: '',
    analyzerB: '',
    summary: '',
  };

  // Split by analyzer headers
  const analyzerAMatch = fullText.indexOf('[Analyzer-A分析結果]');
  const analyzerBMatch = fullText.indexOf('[Analyzer-B分析結果]');

  // Extract sections based on header positions
  const sections: { start: number; key: keyof ParsedResults }[] = [];

  if (analyzerAMatch !== -1) {
    sections.push({ start: analyzerAMatch, key: 'analyzerA' });
  }
  if (analyzerBMatch !== -1) {
    sections.push({ start: analyzerBMatch, key: 'analyzerB' });
  }

  // Sort by position
  sections.sort((a, b) => a.start - b.start);

  if (sections.length === 0) {
    // No analyzer headers found - treat everything as summary
    result.summary = fullText;
  } else {
    // Text before first analyzer header is summary (if any)
    const beforeFirstHeader = fullText.substring(0, sections[0].start).trim();
    if (beforeFirstHeader) {
      result.summary = beforeFirstHeader;
    }

    // Extract each section
    for (let i = 0; i < sections.length; i++) {
      const sectionStart = sections[i].start;
      const sectionEnd = i + 1 < sections.length ? sections[i + 1].start : fullText.length;
      const content = fullText.substring(sectionStart, sectionEnd).trim();

      // Remove the header from content
      const headerEnd = content.indexOf('\n');
      const cleanContent = headerEnd !== -1 ? content.substring(headerEnd + 1).trim() : '';
      result[sections[i].key] = cleanContent;
    }

    // Text after all analyzer sections that doesn't have an analyzer header
    // Check if there's content after the last analyzer section that looks like summary
    const lastSection = sections[sections.length - 1];
    const afterLastAnalyzer = fullText.substring(
      fullText.indexOf('\n', lastSection.start) !== -1
        ? fullText.length
        : lastSection.start
    );

    // Look for summary-like content after the last analyzer
    // The summarizer output typically comes after both analyzers
    const summaryPatterns = ['総合評価', '改善提案', '## ', '### '];
    const remainingText = fullText.substring(
      sections[sections.length - 1].start
    );
    const lastAnalyzerContent = result[lastSection.key];

    // If there are clear summary markers in content that appears after analyzers
    // This is handled naturally since the summarizer output is a separate message
  }

  return result;
}

function MarkdownContent({ content }: { content: string }) {
  if (!content) return null;
  return (
    <div className="analysis-markdown prose prose-sm max-w-none">
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  );
}

function LoadingPlaceholder() {
  return (
    <div className="flex items-center justify-center py-12 text-muted-foreground">
      <Loader2 className="w-6 h-6 animate-spin mr-3" />
      <span>分析中...</span>
    </div>
  );
}

function EmptyPlaceholder() {
  return (
    <div className="flex items-center justify-center py-12 text-muted-foreground">
      <span>画像をアップロードして分析を開始してください</span>
    </div>
  );
}

export function AnalysisResult({ events, isRunning, streamingContent }: AnalysisResultProps) {
  // Collect all message content from events
  const fullContent = useMemo(() => {
    const messageParts: string[] = [];

    for (const event of events) {
      if (isMessageStateEvent(event) && event.value.role === 'assistant') {
        const content = event.value.content;
        if (content?.parts) {
          for (const part of content.parts) {
            if (part.type === 'text' && part.text) {
              messageParts.push(part.text);
            }
          }
        }
      }
    }

    // Append currently streaming content
    if (streamingContent) {
      messageParts.push(streamingContent);
    }

    return messageParts.join('\n\n');
  }, [events, streamingContent]);

  const parsed = useMemo(() => parseAnalysisContent(fullContent), [fullContent]);

  const hasAnyContent = parsed.analyzerA || parsed.analyzerB || parsed.summary;

  if (!hasAnyContent && !isRunning) {
    return <EmptyPlaceholder />;
  }

  return (
    <Card className="analysis-result-card">
      <CardContent className="p-4">
        <Tabs defaultValue="summary" className="w-full">
          <TabsList className="w-full grid grid-cols-3">
            <TabsTrigger value="summary">
              統合レポート
              {isRunning && parsed.summary && (
                <Loader2 className="w-3 h-3 animate-spin ml-1" />
              )}
            </TabsTrigger>
            <TabsTrigger value="analyzer-a">
              Analyzer A (GPT)
              {isRunning && !parsed.analyzerA && (
                <Loader2 className="w-3 h-3 animate-spin ml-1" />
              )}
            </TabsTrigger>
            <TabsTrigger value="analyzer-b">
              Analyzer B (Gemini)
              {isRunning && !parsed.analyzerB && (
                <Loader2 className="w-3 h-3 animate-spin ml-1" />
              )}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="summary" className="analysis-tab-content">
            {parsed.summary ? (
              <MarkdownContent content={parsed.summary} />
            ) : isRunning ? (
              <LoadingPlaceholder />
            ) : (
              <div className="text-center py-8 text-muted-foreground text-sm">
                両方のAnalyzerの分析完了後に統合レポートが生成されます
              </div>
            )}
          </TabsContent>

          <TabsContent value="analyzer-a" className="analysis-tab-content">
            {parsed.analyzerA ? (
              <MarkdownContent content={parsed.analyzerA} />
            ) : isRunning ? (
              <LoadingPlaceholder />
            ) : (
              <div className="text-center py-8 text-muted-foreground text-sm">
                Analyzer A の分析結果はまだありません
              </div>
            )}
          </TabsContent>

          <TabsContent value="analyzer-b" className="analysis-tab-content">
            {parsed.analyzerB ? (
              <MarkdownContent content={parsed.analyzerB} />
            ) : isRunning ? (
              <LoadingPlaceholder />
            ) : (
              <div className="text-center py-8 text-muted-foreground text-sm">
                Analyzer B の分析結果はまだありません
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
