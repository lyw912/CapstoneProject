import { useRef } from 'react';
import { apiJson } from '../utils/helpers';

export default function usePolling() {
  const pollRef = useRef(null);

  const clearPoll = () => {
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const startPoll = (taskId, onUpdate) => {
    clearPoll();
    pollRef.current = window.setInterval(async () => {
      try {
        const taskData = await apiJson(`/api/coordinator/task/${taskId}`);
        onUpdate(taskData.task);
        if (['completed', 'error'].includes(taskData.task.status)) {
          clearPoll();
        }
      } catch (error) {
        clearPoll();
        const { message: msg } = await import('antd');
        msg.error(error.message || 'Analysis status is pending');
      }
    }, 1800);
  };

  return { pollRef, startPoll, clearPoll };
}
