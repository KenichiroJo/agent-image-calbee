import { APIChat, APIChatWithMessages } from './types';
import apiClient from '../apiClient';

export async function getChats({ signal }: { signal: AbortSignal }) {
  return apiClient.get<APIChat[]>('v1/chat', { signal });
}

export async function deleteChat({ chatId }: any): Promise<void> {
  await apiClient.delete(`v1/chat/${chatId}`);
}

export async function updateChat({
  chatId,
  name,
}: {
  chatId: string;
  name: string;
}): Promise<void> {
  await apiClient.patch(`v1/chat/${chatId}`, { name });
}

export async function getChatHistory({ signal, chatId }: { signal: AbortSignal; chatId: string }) {
  return await apiClient.get<APIChatWithMessages>(`v1/chat/${chatId}`, { signal });
}

export interface UploadImageResponse {
  filename: string;      // UUID-prefixed safe filename (for IMAGE_FILE token)
  original_name: string; // Original filename (for display)
}

export async function uploadImage(file: File): Promise<UploadImageResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const response = await apiClient.post<UploadImageResponse>('v1/chat/upload', formData, {
    headers: { 'Content-type': 'multipart/form-data' },
  });
  return response.data;
}
