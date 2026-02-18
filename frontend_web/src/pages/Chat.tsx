import React, { useState } from 'react';
import { AnalysisPage } from '@/components/page/AnalysisPage';

export const ChatPage: React.FC = () => {
  const [chatId, setChatId] = useState<string>(() => window.location.hash?.substring(1));

  const setChatIdHandler = (id: string) => {
    setChatId(id);
    window.location.hash = id;
  };

  return <AnalysisPage chatId={chatId} setChatId={setChatIdHandler} />;
};
