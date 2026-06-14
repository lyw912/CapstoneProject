import { useRef } from 'react';
import { apiJson, displayLog, reportSeedHtml, isSensitiveInputError, showSensitiveInputModal } from '../utils/helpers';

export default function useSSE() {
  const streamRef = useRef(null);

  const clearStream = () => {
    if (streamRef.current) {
      streamRef.current.close();
      streamRef.current = null;
    }
  };

  const startReportStream = async (query, output, onReportTask, onEvents, onReportHtml) => {
    clearStream();
    const topic = query.trim() || 'Intelligent Public Opinion Report';
    try {
      const data = await apiJson('/api/report/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: topic })
      });
      onReportTask(data.task);
      onEvents([{ type: 'status', message: 'Report generation started' }]);

      const stream = new EventSource(data.stream_url);
      streamRef.current = stream;

      const handleEvent = (event) => {
        const payload = event.payload || {};
        onEvents((items) => [{ type: event.type, message: displayLog(payload.message || payload.line || event.type), time: event.timestamp }, ...items].slice(0, 60));
        if (payload.task) onReportTask(payload.task);
        if (event.type === 'completed' || event.type === 'html_ready') {
          const taskId = event.task_id;
          apiJson(`/api/report/result/${taskId}/json`).then((result) => {
            onReportHtml(result.html_content || reportSeedHtml(output));
          }).catch(() => {});
        }
      };

      stream.addEventListener('message', (event) => handleEvent(JSON.parse(event.data)));
      ['status', 'stage', 'progress', 'warning', 'html_ready', 'completed', 'error', 'log'].forEach((name) => {
        stream.addEventListener(name, (event) => handleEvent(JSON.parse(event.data)));
      });
      stream.onerror = () => {
        stream.close();
      };

      return data;
    } catch (error) {
      if (isSensitiveInputError(error)) {
        showSensitiveInputModal();
        return null;
      }
      const { message: msg } = await import('antd');
      msg.error(error.message || 'Report generation failed to start');
      return null;
    }
  };

  return { streamRef, startReportStream, clearStream };
}
