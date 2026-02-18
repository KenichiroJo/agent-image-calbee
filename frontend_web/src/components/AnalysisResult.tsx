import { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import { Loader2 } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent } from '@/components/ui/card';
import { type ChatStateEvent, isMessageStateEvent } from '@/types/events';
import { isToolInvocationPart } from '@/types/message';

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

function parseResults(
  events: ChatStateEvent[],
  streamingContent: string | null
): ParsedResults {
  const result: ParsedResults = {
    analyzerA: '',
    analyzerB: '',
    summary: '',
  };

  const textParts: string[] = [];

  for (const event of events) {
    if (isMessageStateEvent(event) && event.value.role === 'assistant') {
      const content = event.value.content;
      if (content?.parts) {
        for (const part of content.parts) {
          // Extract tool invocation results for individual analyzer tabs
          if (isToolInvocationPart(part)) {
            const toolData = part.toolInvocation;
            if (toolData?.state === 'result' && toolData.result) {
              const resultStr = String(toolData.result);
              if (toolData.toolName === 'analyze_with_gpt') {
                result.analyzerA = resultStr;
              } else if (toolData.toolName === 'analyze_with_gemini') {
                result.analyzerB = resultStr;
              }
            }
          }
          // Collect text parts for the summary (agent's final response)
          if (part.type === 'text' && part.text) {
            textParts.push(part.text);
          }
        }
      }
    }
  }

  // Append currently streaming content
  if (streamingContent) {
    textParts.push(streamingContent);
  }

  const fullText = textParts.join('\n\n');

  // The agent's final text response is the integrated summary
  // Also check if analyzer headers are in the text (fallback parsing)
  if (fullText) {
    const analyzerAMatch = fullText.indexOf('[Analyzer-A分析結果]');
    const analyzerBMatch = fullText.indexOf('[Analyzer-B分析結果]');

    if (analyzerAMatch !== -1 || analyzerBMatch !== -1) {
      // Parse analyzer sections from text (fallback if tool results not available)
      const sections: { start: number; key: 'analyzerA' | 'analyzerB' }[] = [];
      if (analyzerAMatch !== -1) sections.push({ start: analyzerAMatch, key: 'analyzerA' });
      if (analyzerBMatch !== -1) sections.push({ start: analyzerBMatch, key: 'analyzerB' });
      sections.sort((a, b) => a.start - b.start);

      // Text before first analyzer header is summary
      const beforeFirst = fullText.substring(0, sections[0].start).trim();
      if (beforeFirst) result.summary = beforeFirst;

      for (let i = 0; i < sections.length; i++) {
        const sectionStart = sections[i].start;
        const sectionEnd = i + 1 < sections.length ? sections[i + 1].start : fullText.length;
        const content = fullText.substring(sectionStart, sectionEnd).trim();
        const headerEnd = content.indexOf('\n');
        const cleanContent = headerEnd !== -1 ? content.substring(headerEnd + 1).trim() : '';
        if (!result[sections[i].key]) {
          result[sections[i].key] = cleanContent;
        }
      }

      // If no summary yet but there's text after the last analyzer section
      if (!result.summary) {
        // The remaining text is the summary
        const lastEnd = sections[sections.length - 1].start;
        const lastContent = fullText.substring(lastEnd);
        const afterLastAnalyzer = lastContent.substring(lastContent.indexOf('\n') + 1);
        // Look for summary content after both analyzers
      }
    } else {
      // No analyzer headers - the full text is the summary/integrated report
      result.summary = fullText;
    }
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
  const parsed = useMemo(
    () => parseResults(events, streamingContent),
    [events, streamingContent]
  );

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
              {isRunning && !parsed.summary && (
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
